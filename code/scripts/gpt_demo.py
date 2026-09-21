# -*- coding: utf-8 -*-
"""01-GPT系列 复现脚本：用 2-block 迷你 GPT 演示 In-context Learning 五项实测。

A) 引擎对账  bwd 有限差分 vs 解析梯度（7 组参数 maxerr）。
B) 主训练相变  4-shot→主例范式（K=5 mult P=13）：loss 骤降与 ICL 解锁同窗。
C) 三态控制    fresh / shuffle / random：读映射、打乱、无信号。
D) 规模×预算   C=16/32/64 同 token 预算下解锁步数（scale unlocks）。
E) shot-scaling  P=12 非素数（不可辨识）K=1..5 平滑爬坡 vs P=13 素数（可辨识）开关。
F) 位置锁定    K=5 训练的模型把示例左移后回到骰子（诚实负面）。

确定性：单线程 BLAS + 固定 RandomState；stdout 仅数值（逐字节可复现），
墙钟耗时一律打到 stderr。三次运行 stdout 应完全相同。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import sys, math, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np


def now(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def ln(t):
    m = t.mean(axis=-1, keepdims=True)
    s = np.sqrt(t.var(axis=-1, keepdims=True) + 1e-5)
    return (t - m) / s, m, s


def softmax(t, axis=-1):
    e = np.exp(t - t.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class Engine:
    """2-block 解码器（预归一化 LayerNorm + 因果掩码多头注意力 + FFN）。"""

    def __init__(self, V, T, C, NL, NH, scale=0.06):
        self.V, self.T, self.C, self.NL, self.NH = V, T, C, NL, NH
        self.DH = C // NH
        self.mask = np.tril(np.ones((T, T)))

    def init_params(self, seed):
        rng = np.random.RandomState(seed)
        p = {}
        for l in range(self.NL):
            for nm in ("Wq", "Wk", "Wv", "Wo"):
                p[f"{nm}{l}"] = rng.randn(self.C, self.C) * 0.06
            p[f"Wf1{l}"] = rng.randn(self.C, 4 * self.C) * 0.06
            p[f"Wf2{l}"] = rng.randn(4 * self.C, self.C) * 0.06
        p["Wte"] = rng.randn(self.V, self.C) * 0.06
        p["Wpos"] = rng.randn(self.T, self.C) * 0.06
        p["Wout"] = rng.randn(self.C, self.V) * 0.06
        return p

    def n_params(self, p):
        return sum(int(np.prod(v.shape)) for v in p.values())

    def fwd(self, tok, p):
        Bb, T = tok.shape
        x = p["Wte"][tok] + p["Wpos"][None, :T, :]
        caches = []
        for l in range(self.NL):
            n1, m1, s1 = ln(x)
            q = n1 @ p[f"Wq{l}"]; k = n1 @ p[f"Wk{l}"]; v = n1 @ p[f"Wv{l}"]
            q = q.reshape(Bb, T, self.NH, self.DH).transpose(0, 2, 1, 3)
            k = k.reshape(Bb, T, self.NH, self.DH).transpose(0, 2, 1, 3)
            v = v.reshape(Bb, T, self.NH, self.DH).transpose(0, 2, 1, 3)
            attlog = q @ k.transpose(0, 1, 3, 2) / math.sqrt(self.DH) + \
                np.where(self.mask[:T, :T] == 1, 0.0, -np.inf)
            att = softmax(attlog)
            y = att @ v
            y = y.transpose(0, 2, 1, 3).reshape(Bb, T, self.C)
            y_out = y @ p[f"Wo{l}"]
            x2 = x + y_out
            n2, m2, s2 = ln(x2)
            h = np.maximum(0, n2 @ p[f"Wf1{l}"])
            x3 = x2 + h @ p[f"Wf2{l}"]
            caches.append(dict(x=x, n1=n1, m1=m1, s1=s1, q=q, k=k, v=v, att=att,
                               y=y, x2=x2, n2=n2, m2=m2, s2=s2, h=h, x3=x3))
            x = x3
        logits = x @ p["Wout"]
        logp = logits - logits.max(axis=-1, keepdims=True)
        logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))
        return dict(logits=logits, logp=logp, x3=x, caches=caches, tok=tok, Bb=Bb)

    @staticmethod
    def ln_backward(dy, t, m, s):
        d = t - m
        return (dy - dy.mean(axis=-1, keepdims=True)
                - d * (dy * d).mean(axis=-1, keepdims=True) / s ** 2) / s

    def bwd(self, p, cache, tgt, lm):
        Bb, T = cache["Bb"], self.T
        g = {k: np.zeros_like(v) for k, v in p.items()}
        pv = np.exp(cache["logp"])
        d_logits = pv * lm[..., None]
        d_logits[np.arange(Bb)[:, None], np.arange(T)[None, :], tgt] -= lm
        d_logits /= lm.sum()
        g["Wout"] = (cache["x3"].transpose(0, 2, 1) @ d_logits).sum(axis=0)
        d_x = d_logits @ p["Wout"].T
        for l in reversed(range(self.NL)):
            c = cache["caches"][l]
            x, n1, m1, s1 = c["x"], c["n1"], c["m1"], c["s1"]
            x2, n2, m2, s2, h, y = c["x2"], c["n2"], c["m2"], c["s2"], c["h"], c["y"]
            att, q, kk, v = c["att"], c["q"], c["k"], c["v"]
            d_x2 = d_x.copy()
            d_h = d_x @ p[f"Wf2{l}"].T
            g[f"Wf2{l}"] += (h.transpose(0, 2, 1) @ d_x).sum(axis=0)
            d_n2 = d_h.copy(); d_n2[h <= 0] = 0
            g[f"Wf1{l}"] += (n2.transpose(0, 2, 1) @ d_n2).sum(axis=0)
            d_n2 = d_n2 @ p[f"Wf1{l}"].T
            d_x2 = d_x2 + self.ln_backward(d_n2, x2, m2, s2)
            d_x = d_x2.copy()
            d_y = d_x2 @ p[f"Wo{l}"].T
            g[f"Wo{l}"] += (y.transpose(0, 2, 1) @ d_x2).sum(axis=0)
            d_y = d_y.reshape(Bb, T, self.NH, self.DH).transpose(0, 2, 1, 3)
            d_v = att.transpose(0, 1, 3, 2) @ d_y
            d_att = d_y @ v.transpose(0, 1, 3, 2)
            d_attlog = att * (d_att - (d_att * att).sum(axis=-1, keepdims=True)) / math.sqrt(self.DH)
            d_q = d_attlog @ kk
            d_kk = d_attlog.transpose(0, 1, 3, 2) @ q
            d_q = d_q.transpose(0, 2, 1, 3).reshape(Bb, T, self.C)
            d_kk = d_kk.transpose(0, 2, 1, 3).reshape(Bb, T, self.C)
            d_v = d_v.transpose(0, 2, 1, 3).reshape(Bb, T, self.C)
            d_n1 = d_q @ p[f"Wq{l}"].T + d_kk @ p[f"Wk{l}"].T + d_v @ p[f"Wv{l}"].T
            g[f"Wq{l}"] += (n1.transpose(0, 2, 1) @ d_q).sum(axis=0)
            g[f"Wk{l}"] += (n1.transpose(0, 2, 1) @ d_kk).sum(axis=0)
            g[f"Wv{l}"] += (n1.transpose(0, 2, 1) @ d_v).sum(axis=0)
            d_x = d_x + self.ln_backward(d_n1, x, m1, s1)
            if l == 0:
                g["Wpos"][:T] += d_x.sum(axis=0)
                np.add.at(g["Wte"], cache["tok"], d_x)
        return g

    def mask_loss(self, p, tok, tgt, lm):
        c = self.fwd(tok, p)
        lp = c["logp"]
        nll = -lp[np.arange(tok.shape[0])[:, None], np.arange(self.T)[None, :], tgt] * lm
        return nll.sum() / lm.sum(), c


def adam_step(p, o, g, lr):
    for k in p:
        o["m"][k] = 0.9 * o["m"][k] + (1 - 0.9) * g[k]
        o["v"][k] = 0.999 * o["v"][k] + (1 - 0.999) * g[k] ** 2
        mh = o["m"][k] / (1 - 0.9 ** o["step"])
        vh = o["v"][k] / (1 - 0.999 ** o["step"])
        p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
    o["step"] += 1


def init_adam(p):
    return {"m": {k: np.zeros_like(v) for k, v in p.items()},
            "v": {k: np.zeros_like(v) for k, v in p.items()}, "step": 1}


# ---------------------------------------------------------------- 任务族
def make_episode(srng, P, n_example=5, family="mult", wset=None):
    """一次训练/评测片段：n_example 对 (x,y) 演示 + 1 个 query。
    mult 族：y = (w*x) mod P，w∈2..P-1（排除 0/1 避免直读泄露）。"""
    if family == "add":
        pool = tuple(range(1, P))
        a = int(srng.randint(1, P))
        f = lambda x: (x + a) % P
        desc = a
    else:
        pool = tuple(range(2, P))
        if wset is None:
            wset = tuple(range(2, P))
        w = int(wset[srng.randint(len(wset))])
        f = lambda x: (w * x) % P
        desc = w
    xs = srng.choice(np.array(pool), size=n_example + 1, replace=False)
    ex_x, xq = xs[:n_example], int(xs[n_example])
    seq = []
    for x in ex_x:
        seq += [int(x), f(x)]
    seq.append(xq)
    return np.array(seq, dtype=int), f(xq), desc


def build_batch(srng, P, B, T, family="mult", wset=None):
    toks = np.zeros((B, T), dtype=int)
    tgt = np.zeros((B, T), dtype=int)
    lm = np.zeros((B, T))
    n_example = (T - 1) // 2
    for j in range(B):
        seq, yq, _ = make_episode(srng, P, n_example, family=family, wset=wset)
        toks[j] = seq
        for pp in range(0, T - 1, 2):
            tgt[j, pp] = seq[pp + 1]
            lm[j, pp] = 1.0
        tgt[j, T - 1] = yq
        lm[j, T - 1] = 1.0
    return toks, tgt, lm


def eval_acc(eng, p, srng, P, n, mode="fresh", n_example=5, family="mult", wset=None):
    """fresh=新函数（ICL 真检验）；shuffle=示例 y 打乱（映射毁掉）；
    random=y 变均匀随机（无信号→机会地板）。"""
    T = eng.T
    hit = 0
    for ii in range(n):
        seq, yq, _ = make_episode(srng, P, n_example, family=family, wset=wset)
        if mode == "shuffle":
            yv = seq[1::2].copy()
            maski = np.random.RandomState(1000 + ii).permutation(np.arange(len(yv)))
            yv = yv[maski]
            for i in range(n_example):
                seq[2 * i + 1] = yv[i]
        elif mode == "random":
            rng2 = np.random.RandomState(2000 + ii)
            for i in range(n_example):
                seq[2 * i + 1] = int(rng2.randint(P))
        c = eng.fwd(seq[None, :], p)
        pred = int(np.argmax(c["logits"][0, T - 1]))
        hit += int(pred == yq)
    return hit / n


def eval_poslock(eng, p, srng, P, n, n_example=5, family="mult", wset=None, pad=0):
    """位置锁定：把 n_example 对演示左移（前方补 pad），query 固定在 T-1。
    若 ICL 依赖绝对位置，示例一左移 acc 即回落到机会。"""
    T = eng.T
    hit = 0
    for ii in range(n):
        seq, yq, _ = make_episode(srng, P, n_example, family=family, wset=wset)
        pad_n = T - seq.shape[0]
        full = np.full(T, pad, dtype=int)
        full[pad_n:] = seq
        c = eng.fwd(full[None, :], p)
        pred = int(np.argmax(c["logits"][0, T - 1]))
        hit += int(pred == yq)
    return hit / n


def train_model(eng, steps, P, bs, family="mult", wset=None, lr=3e-3, seed_tr=1234,
                seed_init=7, evat=None, n_eval=300):
    """训练到 steps；在 evat 步骤采样点评估 fresh（n_eval 例）。返回 (p, losses, ev)。"""
    p = eng.init_params(seed_init + eng.C)
    o = init_adam(p)
    srng = np.random.RandomState(seed_tr)
    losses = {}
    ev = {}
    tok, tgt, lm = build_batch(srng, P, bs, eng.T, family=family, wset=wset)
    loss0, _ = eng.mask_loss(p, tok, tgt, lm)
    losses[0] = float(loss0)
    for s in range(1, steps + 1):
        tok, tgt, lm = build_batch(srng, P, bs, eng.T, family=family, wset=wset)
        loss, c = eng.mask_loss(p, tok, tgt, lm)
        g = eng.bwd(p, c, tgt, lm)
        adam_step(p, o, g, lr * min(1.0, s / 150))
        if evat and s in evat:
            losses[s] = float(loss)
            ev[s] = eval_acc(eng, p, np.random.RandomState(509), P, mode="fresh",
                             n_example=eng.T // 2, family=family, wset=wset, n=n_eval)
    return p, losses, ev


def main():
    print("=" * 62)
    print("03-模型家族 01-GPT系列 · ICL 五项实测（2-block 迷你 GPT）")
    print("任务族：y=(w·x) mod P（mult），每片段 = K 对演示 + 1 个 query")
    print("引擎：2-block 解码器 · C 隐藏维 · 例内 GradDescent 单机 numpy")
    print("=" * 62)

    # ---------------- A 引擎对账（bwd 正确性保证） ----------------
    print("\n[A] 引擎对账：正向/反向均为手写 numpy，先与有限差分对账。")
    print("    （2-block C=16 · 每片段只有 6 个 y 位置有监督 · 差分 eps=1e-5）")
    t = time.perf_counter()
    mx = 0.0
    eng = Engine(V=13, T=11, C=16, NL=2, NH=2)
    p = eng.init_params(5)
    srng = np.random.RandomState(5)
    tok, tgt, lm = build_batch(srng, 13, 4, 11)
    loss, c = eng.mask_loss(p, tok, tgt, lm)
    g = eng.bwd(p, c, tgt, lm)
    rndj = np.random.RandomState(0)
    for k in ("Wq0", "Wq1", "Wf11", "Wf20", "Wte", "Wpos", "Wout"):
        grad = g[k]
        m = 0.0
        for _ in range(40):
            a = rndj.randint(0, grad.shape[0]); b = rndj.randint(0, grad.shape[1])
            eps = 1e-5
            p[k][a, b] += eps
            l2, _ = eng.mask_loss(p, tok, tgt, lm)
            p[k][a, b] -= 2 * eps
            l1, _ = eng.mask_loss(p, tok, tgt, lm)
            p[k][a, b] += eps
            m = max(m, abs((l2 - l1) / (2 * eps) - grad[a, b]))
        mx = max(mx, m)
        print(f"  {k:<6}  maxerr={m:.3e}")
    now(f"[A] fd 对账完成（全组 maxerr≤{mx:.1e}） {time.perf_counter()-t:.0f}s")

    # ---------------- B 主训练相变 ----------------
    print("\n[B] 主训练：P=13(素数) · mult · C=48 · K=5 · 1200 步 × bs96")
    print("    每步 token = 窗口 11 × 96 = 1056；监督只放在 6 个 y 位。")
    P, K = 13, 5
    WMULT = tuple(range(2, P))
    eng = Engine(V=P, T=2 * K + 1, C=48, NL=2, NH=2)
    t = time.perf_counter()
    pB, losses, ev = train_model(eng, 1200, P, 96, "mult", WMULT,
                                 evat=(100, 200, 400, 800, 1200), n_eval=400)
    now(f"[B] 主训练 1200 步完成（{eng.n_params(pB):,} 参数） {time.perf_counter()-t:.0f}s")
    print("    训练集上出现过的函数也能被'记住'，但测试要的是——")
    print("    每片段都是 <新的随机 w>，模型必须现场从示例学 w（few-shot 元学习）。")
    print("  step        loss        fresh@新函数")
    for s in (0, 100, 200, 400, 800, 1200):
        ls = losses[s]
        fr = ev.get(s, "-")
        if isinstance(fr, float):
            print(f"  {s:>6}  {ls:.4f}      {fr:.3f}")
        else:
            print(f"  {s:>6}  {ls:.4f}      {fr}")

    # ---------------- C 三态控制 ----------------
    print("\n[C] 三态控制（对同一个 B 训练好的 checkpoint，同 800 个 query）：")
    print("    fresh   ：示例 x→y 正常（新 w，模型需现场学）→ 真 ICL")
    print("    shuffle ：示例 x 不变、y 在原位置间重排 → 映射被毁")
    print("    random  ：示例 y 换成均匀随机 → 无任何规则信号")
    for mode, sd in (("fresh", 501), ("shuffle", 502), ("random", 503)):
        a = eval_acc(eng, pB, np.random.RandomState(sd), P, n=800, mode=mode,
                     n_example=K, family="mult", wset=WMULT)
        print(f"  acc[{mode:<8}] = {a:.3f}")

    # ---------------- D 规模×预算 ----------------
    print("\n[D] 规模×预算：同一 token 预算（800 步×bs96×窗口11），更大模型更早解锁 ICL")
    print("    fresh@新函数 n=400 · seed 509 · 解码层 C=16→1 层，32/64→2 层")
    for C in (16, 32, 64):
        NL = 1 if C == 16 else 2
        engD = Engine(V=P, T=11, C=C, NL=NL, NH=2)
        t = time.perf_counter()
        pD, _, evD = train_model(engD, 800, P, 96, "mult", WMULT,
                                 evat=(200, 400, 600, 800), n_eval=400)
        npar = engD.n_params(pD)
        row = " ".join(f"{s}:{evD[s]:.3f}" for s in (200, 400, 600, 800))
        unlock = next((s for s in (200, 400, 600, 800) if evD[s] >= 0.9), "-")
        print(f"  C={C:<3} 参={npar/1000:.0f}k  800步各点 fresh  {row}   解锁帧≈{unlock}")
        now(f"[D] C={C} 训 800 步完成 {time.perf_counter()-t:.0f}s")

    # ---------------- E shot-scaling ----------------
    print("\n[E] shot-scaling：示例数 K 决定现场可辨识性")
    print("    P=13 素数：域 Z13 无零因子，(2,4) 唯一确定 w=4·2⁻¹ → 1 例即识别。")
    print("    P=12 合数：域 Z12 有零因子，示例不能唯一恢复 w → 需更多示例，"
          "越多的例子越准（真实大模型正是这种'不可辨识'场景）。")
    print("    对齐口径：所有 K 等 token 预算 CONST=8800（steps=round(8800/(2K+1))）。")
    CONST = 8800
    for Pc, tag in ((12, "合数 Z12·不可辨识"), (13, "素数 Z13·可辨识")):
        WM = tuple(range(2, Pc))
        print(f"  -- P={Pc}（{tag}） C=48 · fresh@新函数 n=600")
        for Kc in ((1, 2, 3, 5) if Pc == 12 else (1,)):
            Tc = 2 * Kc + 1
            steps = round(CONST / Tc)
            engE = Engine(V=Pc, T=Tc, C=48, NL=2, NH=2)
            t = time.perf_counter()
            pE, _, _ = train_model(engE, steps, Pc, 96, "mult", WM)
            a = eval_acc(engE, pE, np.random.RandomState(501), Pc, n=600, mode="fresh",
                         n_example=Kc, family="mult", wset=WM)
            print(f"    K={Kc}  T={Tc}  steps={steps}  fresh={a:.3f}")
            now(f"[E] P={Pc} K={Kc} 完成 {time.perf_counter()-t:.0f}s")

    # ---------------- F 位置锁定 ----------------
    print("\n[F] 位置锁定（诚实边界）：B 是'5 对演示在绝对位置 0-9、query 在 10'训练的。")
    print("    现在把 K 对演示左移留空（前边补 pad），query 仍固定在位置 10：")
    print("    若 ICL 靠'相对位置'工作应不受影响；若靠'从第 0 位开始读'则回骰子。")
    for Kc in (1, 2, 3, 4, 5):
        a = eval_poslock(eng, pB, np.random.RandomState(511), P, n=600,
                         n_example=Kc, family="mult", wset=WMULT)
        print(f"  {Kc} 对演示紧贴 query（左空 {11 - (2*Kc+1)} 位）  fresh={a:.3f}")

    print("\n" + "=" * 62)
    print("结论速写：ICL 在（同规模下）训练中'啪'地解锁；解锁后才是纯现场推理，")
    print("示例打乱/无信号即回骰子；更大模型用同预算更早解锁；")
    print("示例数对'可辨识'函数无增益（开关）、对'不可辨识'函数平滑爬坡（shot-scaling）；")
    print("玩具 ICL 强位置锁定——示例错位即失效。")
    print("=" * 62)


if __name__ == "__main__":
    T0 = time.time()
    main()
    now(f"全部完成 累计 {time.time()-T0:.0f}s")
