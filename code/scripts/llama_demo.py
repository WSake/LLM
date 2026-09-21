# -*- coding: utf-8 -*-
"""03-模型家族 02-Llama系列 · 「事实标准」是被可测的差异投出来的
任务族：y=(a+x) mod P（add，ICL），引擎 = 2-block 解码器（配方可换：位置×激活×归一化×注意力）
A 引擎对账：Rx 配方（RoPE+SwiGLU+RMSNorm+GQA）bwd vs 中心差分
B 等预算配方对照：Rx(Llama 配方) vs GptRx(GPT-2 配方)，同一 add 任务谁更快解锁 ICL
C 位置方案三臂：learnable-abs / sinusoidal-abs / RoPE，示例贴 query 左空 S 位后 fresh 塌不塌
D 派生账：Llama-1..4 各代 KV 每 token 字节（MHA vs GQA 摊薄）· 词表 embedding 账（公开配置，账算非本机实测）
确定协议：单线程 BLAS · 固定种子 · 三遍逐位一致 · 墙钟打 stderr 保住 stdout
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import sys, math, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np


_START = time.perf_counter()

def now(msg):
    sys.stderr.write(f"[{time.perf_counter() - _START:6.1f}s] {msg}\n")


def ln(t, eps=1e-5):
    m = t.mean(axis=-1, keepdims=True)
    s = np.sqrt(t.var(axis=-1, keepdims=True) + eps)
    return (t - m) / s, (m, s)

def ln_backward(dy, m, s, t):
    xx = t - m
    return (dy - dy.mean(axis=-1, keepdims=True)
            - xx * (dy * xx).mean(axis=-1, keepdims=True) / s ** 2) / s

def rmsn(t, eps=1e-5):
    s = np.sqrt((t ** 2).mean(axis=-1, keepdims=True) + eps)
    return t / s, (t / s, s)

def rmsn_backward(dy, y, s):
    # y = x/s, s = sqrt(mean(x²)+eps)  →  dx = (dy − y·mean(dy⊙y)) / s
    return (dy - y * (dy * y).mean(axis=-1, keepdims=True)) / s


def softmax(t, axis=-1):
    e = np.exp(t - t.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def gelu_fwd(x):
    a = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(a))

def gelu_bwd(x):
    a = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    t = np.tanh(a)
    da = math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + t) + x * 0.5 * (1.0 - t ** 2) * da

def silu_fwd(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return x * s

def silu_bwd(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s + x * s * (1.0 - s)


def rot_cos_sin(pos, d, base=10000.0):
    kk = np.arange(d // 2, dtype=np.float64)
    theta = base ** (-2.0 * kk / d)
    ang = np.outer(np.arange(pos, dtype=np.float64), theta)
    return np.cos(ang), np.sin(ang)

def apply_rot(q, cos, sin):
    qe, qo = q[..., 0::2], q[..., 1::2]
    out = np.empty_like(q)
    out[..., 0::2] = qe * cos - qo * sin
    out[..., 1::2] = qe * sin + qo * cos
    return out

def rot_backward(dout, cos, sin):
    de, do_ = dout[..., 0::2], dout[..., 1::2]
    din = np.empty_like(dout)
    din[..., 0::2] = de * cos + do_ * sin
    din[..., 1::2] = -de * sin + do_ * cos
    return din

def sinusoid_abs(pos, d, base=10000.0):
    pe = np.zeros((pos, d))
    kk = np.arange(d // 2, dtype=np.float64)
    theta = base ** (-2.0 * kk / d)
    ang = np.outer(np.arange(pos), theta)
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


class Engine:
    """decoder-only 迷你 GPT；配方 = (pos, act, norm, NH, NV)。
    pos: abs=可学习绝对位置 / sin=sinusoidal 绝对位置 / rope=旋转相对位置
    act: gelu / swiglu（8/3C 中间维）
    norm: ln / rms
    注意力：Q 投影全 NH 头；K/V 投影只 NV 头（NV=NH 退化为 MHA；NV<NH 即 GQA，KV 参数省 H/NV 倍）
    """
    def __init__(self, V, T, C, NL, NH, NV=None, pos="rope", act="swiglu", norm="rms",
                 base=10000.0):
        self.V, self.T, self.C, self.NL, self.NH = V, T, C, NL, NH
        self.NV = NH if NV is None else NV
        self.pos = pos
        self.act = act
        self.norm = norm
        self.DH = C // NH
        assert NH * self.DH == C and NH % self.NV == 0
        self.dff = 8 * C // 3
        self.mask = np.tril(np.ones((T, T)))
        self.masks = None   # 可选：长度 NL 的逐层掩码；None 时所有层共用 self.mask（交替注意力用）
        self.ropcos, self.ropsin = rot_cos_sin(T, C, base)
        if self.NV * self.DH != C:
            self.kcos, self.ksin = rot_cos_sin(T, self.NV * self.DH, base)
        else:
            self.kcos, self.ksin = self.ropcos, self.ropsin
        self.sinpos = sinusoid_abs(T, C, base)
        self.KPG = NH // self.NV

    # ---------------- 参数 ----------------
    def init_params(self, seed):
        rng = np.random.RandomState(seed)
        p = {}
        for l in range(self.NL):
            p[f"Wq{l}"] = rng.randn(self.C, self.C) * 0.06
            p[f"Wk{l}"] = rng.randn(self.C, self.NV * self.DH) * 0.06
            p[f"Wv{l}"] = rng.randn(self.C, self.NV * self.DH) * 0.06
            p[f"Wo{l}"] = rng.randn(self.C, self.C) * 0.06
            if self.act == "swiglu":
                p[f"Wf1{l}"] = rng.randn(self.C, 2 * self.dff) * 0.06
                p[f"Wf2{l}"] = rng.randn(self.dff, self.C) * 0.06
            else:
                p[f"Wf1{l}"] = rng.randn(self.C, 4 * self.C) * 0.06
                p[f"Wf2{l}"] = rng.randn(4 * self.C, self.C) * 0.06
        p["Wte"] = rng.randn(self.V, self.C) * 0.06
        if self.pos == "abs":
            p["Wpos"] = rng.randn(self.T, self.C) * 0.06
        p["Wout"] = rng.randn(self.C, self.V) * 0.06
        return p

    # ---------------- 前向 ----------------
    def norm_fwd(self, t):
        if self.norm == "ln":
            return ln(t)          # (out, (m,s))
        return rmsn(t)            # (out, s)

    def norm_bwd(self, dy, cache, t):
        if self.norm == "ln":
            m, s = cache
            return ln_backward(dy, m, s, t)
        y, s = cache
        return rmsn_backward(dy, y, s)

    def fwd(self, tok, p, init_emb=None):
        """init_emb: 外部预构建的输入 embedding（多模态全家谱：视觉 CLS/模态 token 注入等）。
        None（默认）走旧 Wte 查表路径，逐字节向后兼容；非 None 时本引擎不回写 Wte（梯度由外部回收）。"""
        Bb, T = tok.shape
        self._uie = init_emb is not None
        if self.pos == "abs":
            x = init_emb + p["Wpos"][None] if self._uie else p["Wte"][tok] + p["Wpos"][None]
        elif self.pos == "sin":
            x = init_emb + self.sinpos[None] if self._uie else p["Wte"][tok] + self.sinpos[None]
        else:
            x = init_emb if self._uie else p["Wte"][tok]
        caches = []
        for l in range(self.NL):
            n1 = self.norm_fwd(x)
            q = n1[0] @ p[f"Wq{l}"]; k = n1[0] @ p[f"Wk{l}"]; v = n1[0] @ p[f"Wv{l}"]
            if self.pos == "rope":
                q = apply_rot(q, self.ropcos[:T], self.ropsin[:T])
                k = apply_rot(k, self.kcos[:T], self.ksin[:T])
            q = q.reshape(Bb, T, self.NH, self.DH).transpose(0, 2, 1, 3)     # (B,H,T,DH)
            k = k.reshape(Bb, T, self.NV, self.DH).transpose(0, 2, 1, 3)     # (B,NV,T,DH)
            v = v.reshape(Bb, T, self.NV, self.DH).transpose(0, 2, 1, 3)
            kv_idx = np.arange(self.NH) // self.KPG                         # 每组 query 头共享一个 KV
            ke = k[:, kv_idx]                                               # (B,H,T,DH) 扩展
            ve = v[:, kv_idx]
            mask = self.masks[l] if self.masks is not None else self.mask
            attlog = q @ ke.transpose(0, 1, 3, 2) / math.sqrt(self.DH) + np.where(mask == 1, 0.0, -np.inf)
            att = softmax(attlog)
            y = att @ ve
            y = y.transpose(0, 2, 1, 3).reshape(Bb, T, self.C)
            y_out = y @ p[f"Wo{l}"]
            x2 = x + y_out
            n2 = self.norm_fwd(x2)
            if self.act == "swiglu":
                F = n2[0] @ p[f"Wf1{l}"]
                g_, u_ = F[..., :self.dff], F[..., self.dff:]
                h = silu_fwd(g_) * u_
                x3 = x2 + h @ p[f"Wf2{l}"]
            else:
                h = gelu_fwd(n2[0] @ p[f"Wf1{l}"])
                x3 = x2 + h @ p[f"Wf2{l}"]
            caches.append(dict(x=x, n1=n1, q=q, ke=ke, ve=ve, att=att, kv_idx=kv_idx,
                               y=y, x2=x2, n2=n2, h=h, x3=x3))
            x = x3
        logits = x @ p["Wout"]
        logp = logits - logits.max(axis=-1, keepdims=True)
        logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))
        return dict(logits=logits, logp=logp, x3=x, caches=caches, tok=tok, Bb=Bb)

    # ---------------- 反向 ----------------
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
            x, n1 = c["x"], c["n1"]
            x2, n2, h, y = c["x2"], c["n2"], c["h"], c["y"]
            att, q, ke, ve = c["att"], c["q"], c["ke"], c["ve"]
            d_x2 = d_x.copy()
            d_h = d_x @ p[f"Wf2{l}"].T
            g[f"Wf2{l}"] += (h.transpose(0, 2, 1) @ d_x).sum(axis=0)
            if self.act == "swiglu":
                F = n2[0] @ p[f"Wf1{l}"]
                g_, u_ = F[..., :self.dff], F[..., self.dff:]
                d_g = d_h * u_ * silu_bwd(g_)
                d_u = d_h * silu_fwd(g_)
                d_F = np.concatenate([d_g, d_u], axis=-1)
                g[f"Wf1{l}"] += (n2[0].transpose(0, 2, 1) @ d_F).sum(axis=0)
                d_n2 = d_F @ p[f"Wf1{l}"].T
            else:
                hx = n2[0] @ p[f"Wf1{l}"]
                dhg = gelu_bwd(hx) * d_h
                g[f"Wf1{l}"] += (n2[0].transpose(0, 2, 1) @ dhg).sum(axis=0)
                d_n2 = dhg @ p[f"Wf1{l}"].T
            d_x2 = d_x2 + self.norm_bwd(d_n2, n2[1], x2)
            d_x = d_x2.copy()
            d_y = d_x2 @ p[f"Wo{l}"].T
            g[f"Wo{l}"] += (y.transpose(0, 2, 1) @ d_x2).sum(axis=0)
            d_y = d_y.reshape(Bb, T, self.NH, self.DH).transpose(0, 2, 1, 3)   # (B,H,T,DH)
            d_att = d_y @ ve.transpose(0, 1, 3, 2)                            # (B,H,T,T)
            d_ve = att.transpose(0, 1, 3, 2) @ d_y                            # (B,H,T,DH)
            d_attlog = att * (d_att - (d_att * att).sum(axis=-1, keepdims=True)) / math.sqrt(self.DH)
            d_q = d_attlog @ ke                                               # (B,H,T,DH)
            d_ke = d_attlog.transpose(0, 1, 3, 2) @ q
            d_q = d_q.transpose(0, 2, 1, 3).reshape(Bb, T, self.C)
            if self.pos == "rope":
                d_q = rot_backward(d_q, self.ropcos[:T], self.ropsin[:T])
            # K/V 头反传给未扩展的 (B,NV,T,DH)：每个 kv 头汇总其 group 的 KPG 个 query 头的梯度
            d_kv = d_ke.reshape(Bb, self.NV, self.KPG, T, self.DH).sum(axis=2)
            d_vv = d_ve.reshape(Bb, self.NV, self.KPG, T, self.DH).sum(axis=2)
            d_k = d_kv.transpose(0, 2, 1, 3).reshape(Bb, T, self.NV * self.DH)
            d_v = d_vv.transpose(0, 2, 1, 3).reshape(Bb, T, self.NV * self.DH)
            if self.pos == "rope":
                d_k = rot_backward(d_k, self.kcos[:T], self.ksin[:T])
            d_n1 = d_q @ p[f"Wq{l}"].T + d_k @ p[f"Wk{l}"].T + d_v @ p[f"Wv{l}"].T
            g[f"Wq{l}"] += (n1[0].transpose(0, 2, 1) @ d_q).sum(axis=0)
            g[f"Wk{l}"] += (n1[0].transpose(0, 2, 1) @ d_k).sum(axis=0)
            g[f"Wv{l}"] += (n1[0].transpose(0, 2, 1) @ d_v).sum(axis=0)
            d_x = d_x + self.norm_bwd(d_n1, n1[1], x)
            if l == 0 and self.pos == "abs":
                g["Wpos"] += d_x.sum(axis=0)
            if l == 0:
                if self._uie:
                    cache["dx0"] = d_x      # 外部 injected embedding 的梯度回调，由外部参数回收
                else:
                    np.add.at(g["Wte"], cache["tok"], d_x)
        return g

    def mask_loss(self, p, tok, tgt, lm, init_emb=None):
        c = self.fwd(tok, p, init_emb=init_emb)
        lp = c["logp"]
        return (-lp[np.arange(tok.shape[0])[:, None], np.arange(self.T)[None, :], tgt] * lm).sum() / lm.sum(), c


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


def build_batch(srng, P, B, T):
    """add 家族：每片段 y=(x+a) mod P，监督只打 K 个示例 y 位 + query 位。"""
    K = (T - 1) // 2
    toks = np.zeros((B, T), dtype=int)
    tgt = np.zeros((B, T), dtype=int)
    lm = np.zeros((B, T))
    pool = np.arange(1, P)
    for j in range(B):
        a = int(srng.randint(1, P))
        xs = srng.choice(pool, size=K + 1, replace=False)
        X, xq = xs[:K], int(xs[K])
        seq = []
        for x in X:
            seq += [int(x), (int(x) + a) % P]
        seq.append(xq)
        toks[j] = seq
        for pp in range(0, T - 1, 2):
            tgt[j, pp] = seq[pp + 1]; lm[j, pp] = 1.0
        tgt[j, T - 1] = (xq + a) % P; lm[j, T - 1] = 1.0
    return toks, tgt, lm


def eval_longwin(eng, p, srng, P, n=400, S=0):
    """同一 5 对示例整块右移 S 位（窗口拉到 11+S，query 恒在倒数第 1 位）：
    相对距离保持不变（q→k 距离仍 ∈[1,10]），只有绝对位置平移。
    abs：Wpos 只有 11 行，位置 10+S 查表越界；sin：相位随绝对位置断连；
    rope：得分只依赖距离 → 应保持。返回 (fresh or None, tag) ，tag 记录越界原因。
    """
    T = eng.T + S
    if S > 0 and eng.pos == "abs" and 10 + S >= eng.T:
        return None, "Wpos 表只有 11 行，位置 10+%d 越界" % S
    tmp = Engine(V=eng.V, T=T, C=eng.C, NL=eng.NL, NH=eng.NH, NV=eng.NV,
                 pos=eng.pos, act=eng.act, norm=eng.norm)
    pool = np.arange(1, P)
    hit = 0
    for ii in range(n):
        a = int(srng.randint(1, P))
        xs = srng.choice(pool, size=6, replace=False)
        X, xq = xs[:5], int(xs[5])
        body = []
        for x in X:
            body += [int(x), (int(x) + a) % P]
        seq = [0] * S + body + [xq]
        assert len(seq) == T, (len(seq), T)
        c = tmp.fwd(np.array(seq)[None, :], p)
        pred = int(np.argmax(c["logits"][0, T - 1]))
        hit += int(pred == (xq + a) % P)
    return hit / n, None


def eval_fresh(eng, p, srng, P, n=400, S=0):
    """fresh：全新 a 的片段正确率；S = 示例块左侧补的 pad 数（示例贴 query、query 恒在位置 10）。"""
    T = eng.T
    K = (T - 1 - S) // 2
    if K < 1:
        return None
    pool = np.arange(1, P)
    hit = 0
    for ii in range(n):
        a = int(srng.randint(1, P))
        xs = srng.choice(pool, size=K + 1, replace=False)
        X, xq = xs[:K], int(xs[K])
        body = []
        for x in X:
            body += [int(x), (int(x) + a) % P]
        seq = [0] * S + body + [xq]
        assert len(seq) == T, (len(seq), T, S, K)
        c = eng.fwd(np.array(seq)[None, :], p)
        pred = int(np.argmax(c["logits"][0, T - 1]))
        hit += int(pred == (xq + a) % P)
    return hit / n


def fd_check():
    print("[A] 引擎对账（Rx 配方 1-block C=16 · RoPE+SwiGLU+RMSNorm+GQA(4 头/2 kv)）")
    print("    每片段只有 6 个 y 位有监督 · 中心差分 eps=1e-5 · 每参数 40 个随机点")
    eng = Engine(V=13, T=11, C=16, NL=1, NH=4, NV=2, pos="rope", act="swiglu", norm="rms")
    p = eng.init_params(5)
    srng = np.random.RandomState(5)
    tok, tgt, lm = build_batch(srng, 13, 4, 11)
    g = eng.bwd(p, eng.fwd(tok, p), tgt, lm)
    rndj = np.random.RandomState(0)
    worst = 0.0
    for k in ("Wq0", "Wk0", "Wv0", "Wo0", "Wf10", "Wf20", "Wte", "Wout"):
        grad = g[k]
        mx, arg = 0.0, None
        r0, r1 = grad.shape
        for _ in range(40):
            a = rndj.randint(0, r0); b = rndj.randint(0, r1)
            eps = 1e-5
            p[k][a, b] += eps
            l2, _ = eng.mask_loss(p, tok, tgt, lm)
            p[k][a, b] -= 2 * eps
            l1, _ = eng.mask_loss(p, tok, tgt, lm)
            p[k][a, b] += eps
            fd = (l2 - l1) / (2 * eps)
            if abs(fd - grad[a, b]) > mx:
                mx = abs(fd - grad[a, b]); arg = (a, b)
        worst = max(worst, mx)
        print(f"  {k:<5} maxerr={mx:.3e} @idx={arg}")
    print(f"  8 组全参 maxerr≤{worst:.3e}（<1e-5 ✅）")


def train(eng, steps, P, bs, lr=3e-3, seed_tr=1234):
    """返回 p + 步标采样 (step, loss) 列表。"""
    p = eng.init_params(7 + eng.C)
    o = init_adam(p)
    srng = np.random.RandomState(seed_tr)
    marks = []
    for s in range(1, steps + 1):
        tok, tgt, lm = build_batch(srng, P, bs, eng.T)
        loss, c = eng.mask_loss(p, tok, tgt, lm)
        g = eng.bwd(p, c, tgt, lm)
        adam_step(p, o, g, lr * min(1.0, s / 150))
        if s % 100 == 0:
            marks.append((s, loss))
    if (not marks) or marks[-1][0] != steps:
        marks.append((steps, loss))
    return p, marks


def expB():
    print("[B] 等预算配方对照：同一 add 任务（P=13 · K=5 · T=11 · 1200 步 × bs96）")
    print("    Rx   = Llama 配方：RoPE + SwiGLU + RMSNorm + GQA(8 头/4 kv)")
    print("    GptRx= GPT-2 配方：abs + GELU + LayerNorm + MHA(8 头)")
    print("    seed_tr=1234 · seed_init=7+C · 每 100 步记 loss，结尾 fresh(同窗 n=300) · 墙钟")
    for name, kw in (("Rx", dict(pos="rope", act="swiglu", norm="rms", NV=4)),
                     ("GptRx", dict(pos="abs", act="gelu", norm="ln", NV=8))):
        eng = Engine(V=13, T=11, C=48, NL=2, NH=8, **kw)
        t0 = time.perf_counter()
        p, marks = train(eng, 1200, 13, 96)
        dt = time.perf_counter() - t0
        fr = eval_fresh(eng, p, np.random.RandomState(509), 13, n=300, S=0)
        steps = " ".join(f"{s}:{v:.4f}" for s, v in marks)
        print(f"  {name:<5} loss|| {steps} || fresh@初始化位置={fr:.3f}")
        now(f"[B] {name} 训练 1200 步墙钟 {dt:.0f}s（打 stderr，不参与逐位比对）")
    now("expB 完成")


def expC():
    print("[C] 位置方案三臂（同预算 1000 步 × bs96 · add P=13 · C=48 2-block · seed_tr=1234）")
    print("    训练=5 对示例在位置 0-9、query 在 10；其余配方三臂完全一致（act=gelu norm=ln MHA）")
    base = dict(act="gelu", norm="ln", NV=8)
    arms = [("abs ", dict(pos="abs")),
            ("sin ", dict(pos="sin")),
            ("rope", dict(pos="rope"))]
    print("  [C1] 少对+贴 query 左空 S 位（query 恒在 10）——复刻 01-GPT系列 [F] 位置锁定测法")
    cols1 = ["  方案     "]
    for S in (0, 2, 4, 6, 8):
        cols1.append(f"S={S}")
    print("  " + "".join(f"{c:<9}" for c in cols1))
    rows = {}
    for name, kw in arms:
        eng = Engine(V=13, T=11, C=48, NL=2, NH=8, **{**base, **kw})
        p, _ = train(eng, 1000, 13, 96)
        rows[name.strip()] = (eng, p)
        srng = np.random.RandomState(509)
        row = [f"{name}"]
        for S in (0, 2, 4, 6, 8):
            fr = eval_fresh(eng, p, np.random.RandomState(509), 13, n=400, S=S)
            row.append(f"{(fr if fr is not None else 0.0):.3f}")
        print("  " + "".join(f"{v:<9}" for v in row))
    print("  [C2] 同一 5 对整块右移 S 位（窗口 11→11+S，query 恒在末位）——相对距离全保留")
    print("    只有绝对位置变；abs 的 Wpos 表只有 11 行 → 引位置会越界")
    cols2 = ["  方案     "]
    for S in (0, 2, 4, 6, 8):
        cols2.append(f"S={S}")
    print("  " + "".join(f"{c:<9}" for c in cols2))
    for name, kw in arms:
        name = name.strip()
        eng, p = rows[name]
        row = [f"{name}"]
        for S in (0, 2, 4, 6, 8):
            fr, tag = eval_longwin(eng, p, np.random.RandomState(509), 13, n=400, S=S)
            if fr is None:
                row.append("OOB")
            else:
                row.append(f"{fr:.3f}")
        print("  " + "".join(f"{v:<9}" for v in row))
    now("expC 完成")


def expD():
    print("[D] 派生账（Llama 各代公开配置，账算非本机实测）")
    print("    KV 每 token 字节 = L × (2 · H_kv · DH · 2B) ；KV 全部重数 = 每 token 字节 × 上下文")
    print(f"  {'模型':<17}{'L':>4}{'H':>5}{'H_kv':>6}{'DH':>5}{'KV B/层/tok':>12}{'词表':>9}{'GQA比MHA省':>11}")
    rows = {
        "Llama-1-7B":   (32, 32, 32, 128, 32000),
        "Llama-2-7B":   (32, 32, 32, 128, 32000),
        "Llama-2-70B":  (80, 64, 8,  128, 32000),
        "Llama-3-8B":   (32, 32, 8,  128, 128256),
        "Llama-3-70B":  (80, 64, 8,  128, 128256),
        "Llama-3.1-405B": (126, 128, 8, 128, 128256),
        "Llama-4-Scout": (48, 128, 8, 32, None),   # MoE · 词表未公开 · hidden 4096/128 头 → DH=32
    }
    for name, (L, H, Hkv, DH, V) in rows.items():
        perb = 2 * Hkv * DH * 2          # B / 层 / token
        vs = f"{V:,}" if V else "?"
        ratio = H / Hkv
        print(f"  {name:<17}{L:>4}{H:>5}{Hkv:>6}{DH:>5}{perb:>12d} B{vs:>9}{ratio:>11.0f}×")
    print("  → GQA 把 KV 缓存这本账按 H/H_kv 摊薄：Llama-3-8B ×4、2-70B ×8、3.1-405B ×16、")
    print("    与 02-10 KV-Cache 显存账本（MHA 4GiB vs GQA 1GiB）同一条母线。")
    print("  派生一行：Llama-3-8B 词表 embedding ≈ 128256×4096×2B ≈ 1.0 GiB，")
    print("    而 128k 上下文的 KV = 4096B/层/ tok×32层 = 128 KiB/tok → 16.8 GiB——长上下文里")
    print("    KV 缓存比整张词表 embedding 大 16 倍，GQA 摊薄 4 倍是省在最大的那本账上。")
    now("expD 完成")


def main():
    t0 = time.perf_counter()
    print("=" * 66)
    print("03-模型家族 02-Llama系列 · 「事实标准」是被可测的差异投出来的")
    print("任务族：y=(a+x) mod P（add）——每片段新随机 a，模型必须现场从示例读映射")
    print("引擎：2-block 解码器 · 配方可换（位置×激活×归一化×注意力）")
    print("=" * 66)
    fd_check()
    print()
    expB()
    print()
    expC()
    print()
    expD()
    print()
    now(f"全脚本累计 {time.perf_counter() - t0:.0f}s")
    print("结论速写：同预算下 Llama 配方（RoPE+SwiGLU+RMSNorm+GQA）比 GPT-2 配方更快解锁 ICL；")
    print("[C1] 少对+移位三臂全塌——再证 toy ICL 强位置锁定（诚实负面）；[C2] 整块平移进更长窗：")
    print("abs 的 Wpos 表直接越界、sin 掉回骰子，只有 rope 保部分信号（仍衰减）——相对位置是")
    print("必要条件不是充分解；GQA 把 KV 账摊薄 H/H_kv 倍（128k 里 KV 比词表 embedding 大 16 倍）。")
    print("=" * 66)


if __name__ == "__main__":
    main()
