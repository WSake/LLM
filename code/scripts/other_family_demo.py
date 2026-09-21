# -*- coding: utf-8 -*-
"""03-模型家族 10-其他家族 —— 五家五路的『算力税』（Kimi/MiniMax/Doubao/Hunyuan/InternLM）。

逐个家族收沿途五个剩余主力，把每家的押注姿势量成一道『税』：
  [A] 稀疏税（Kimi K2/MiniMax M1/Hunyuan 的 MoE 押注族）
      —— 固定激活算力，把『总参』放大成『稀疏容量』值不值：拓扑=同一 1-block
         C=96 引擎 + Top-2 硬选，E=1(Dense)/2/4/8/16 五档，激活 FFN 恒 8C²，
         只有驻留专家数变。实测训练 CE / 全新句式命中 —— 答案是『几乎不降』
         （本 toy 信息量小），省下的就是激活算力，代价是驻留与通信。
  [B] KV 税（Hunyuan 长上下文 / Kimi 1M）—— 纯账（非本机实测）：
      KV/token = 2·L·H_kv·DH·2B，长上下文每加一层都加账。
  [C] 采样税（Doubao Seed-Thinking）—— 复用 02-20 算术基座（R）：
      直答 greedy / 验证器 best-of-k / RLVR-think 三种部署策略的
      『每正确回答前向次数』= 1/acc_vote、k/acc_cover、1/acc_rlvr。
  [D] 评测税（InternLM OpenCompass）—— 蒙特卡洛：两臂真实准确率差 δ，
      n 个评测样本能『排对』的概率（固定种子 400 次重采样）。

科学数字：单线程 BLAS + 固定种子，run1==run2==run3 逐位一致（墙钟只在 stderr）。
真实模型数字（K2 1T 级 MoE / M1 201B / Hunyuan 128k 等）= 非本机实测（无外网未复核）。
一键复现：python code/scripts/other_family_demo.py
"""
import os
# 单线程 BLAS 强锁（numpy import 之前）——低维模型对浮点聚序极敏感
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import re
import sys
import time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r"F:\00AI创业\01LLM\LLM\code\scripts")
import numpy as np

np.set_printoptions(precision=4, suppress=True)
import moe_demo as M          # 1-block C=96 MoE 引擎 + 6 组知识事实语料
import reasoning_demo as R    # 02-20 算术基座（SFT/eval_k/RLVR）

WALL0 = time.perf_counter()


def logw(name):
    print(f"[wall] {name} {time.perf_counter()-WALL0:7.1f}s", file=sys.stderr)


# ---------------------------------------------------------------------------
# [D0/A] 稀疏 MoE 的参数化前向/反向（E 可变 · Top-K=2 · 每专家 M_FFN=2C）
#   数学与 moe_demo 的 forward/backward 完全一致，只把 E 从模块常量参数化。
#   反向要做的验证 = [D0] 中心差分对账（冻结路由拓扑）。
# ---------------------------------------------------------------------------
def init_params_p(E, C=96, mffn=192, rng_seed=7):
    rngo = np.random.RandomState(rng_seed)
    scale = 0.06
    out = {}
    for k, (r0, r1) in {
        "Wte": (M.V, C), "Wpos": (M.T, C), "Wq": (C, C), "Wk": (C, C),
        "Wv": (C, C), "Wo": (C, C), "Wout": (C, M.V),
    }.items():
        out[k] = rngo.randn(r0, r1) * scale
    if E is None:                                # Dense：经典 4C 中间维全激活
        out["Wf1"] = rngo.randn(C, 4 * C) * scale
        out["Wf2"] = rngo.randn(4 * C, C) * scale
    else:
        out["Wg"] = rngo.randn(C, E) * scale
        for e in range(E):
            out[f"Wf1_{e}"] = rngo.randn(C, mffn) * scale
            out[f"Wf2_{e}"] = rngo.randn(mffn, C) * scale
    return out


def fwd(tok, params, target=None, gmask_fixed=None, E=None, C=96, mffn=192):
    Bb = tok.shape[0]
    x = params["Wte"][tok] + params["Wpos"][None]
    n1, m1, s1 = M.ln(x)
    q = n1 @ params["Wq"]; kk = n1 @ params["Wk"]; v = n1 @ params["Wv"]
    q = q.reshape(Bb, M.T, M.NHEADS, M.DH).transpose(0, 2, 1, 3)
    kk = kk.reshape(Bb, M.T, M.NHEADS, M.DH).transpose(0, 2, 1, 3)
    v = v.reshape(Bb, M.T, M.NHEADS, M.DH).transpose(0, 2, 1, 3)
    al = q @ kk.transpose(0, 1, 3, 2) / np.sqrt(M.DH) + np.where(M.MASK == 1, 0.0, -np.inf)
    att = M.softmax(al)
    y = att @ v
    y = y.transpose(0, 2, 1, 3).reshape(Bb, M.T, C)
    x2 = x + y @ params["Wo"]
    n2, m2, s2 = M.ln(x2)
    if E is None:
        h1 = np.maximum(0, n2 @ params["Wf1"])
        h = h1 @ params["Wf2"]
        rt = None
    else:
        wg = n2 @ params["Wg"]
        gates = M.softmax(wg)
        gmask = M.topk_mask(gates, 2) if gmask_fixed is None else gmask_fixed
        g = gates * gmask
        stack = np.stack(
            [np.maximum(0, n2 @ params[f"Wf1_{e}"]) @ params[f"Wf2_{e}"] for e in range(E)],
            axis=-2)
        h = np.sum(g[..., None] * stack, axis=-2)
        rt = (wg, gates, gmask, stack)
        h1 = None
    x3 = x2 + h
    logits = x3 @ params["Wout"]
    logp = logits - logits.max(axis=-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))
    if target is None:
        target = np.concatenate([tok[:, 1:], np.zeros((Bb, 1), dtype=int)], axis=1)
    loss = -logp[np.arange(Bb)[:, None], np.arange(M.T)[None, :], target].mean()
    cache = dict(x=x, n1=n1, m1=m1, s1=s1, q=q, k=kk, v=v, att=att, y=y,
                 x2=x2, n2=n2, m2=m2, s2=s2, h=h, rt=rt, x3=x3, h1=h1,
                 logits=logits, logp=logp, target=target)
    return loss, cache


def bwd(params, cache, E=None, C=96, mffn=192):
    g = {k: np.zeros_like(v) for k, v in params.items()}
    Bb = cache["target"].shape[0]
    p = np.exp(cache["logp"])
    onehot = np.zeros_like(cache["logits"])
    onehot[np.arange(Bb)[:, None], np.arange(M.T)[None, :], cache["target"]] = 1
    d_logits = (p - onehot) / (Bb * M.T)
    g["Wout"] = (cache["x3"].transpose(0, 2, 1) @ d_logits).sum(axis=0)
    d_x3 = d_logits @ params["Wout"].T

    if E is None:
        d_h = d_x3 @ params["Wf2"].T
        g["Wf2"] = (cache["h1"].transpose(0, 2, 1) @ d_x3).sum(axis=0)
        d_h[cache["h1"] <= 0] = 0
        g["Wf1"] = (cache["n2"].transpose(0, 2, 1) @ d_h).sum(axis=0)
        d_n2 = d_h @ params["Wf1"].T
    else:
        wg, gates, gmask, stack = cache["rt"]
        n2 = cache["n2"]
        hs = [np.maximum(0, n2 @ params[f"Wf1_{e}"]) for e in range(E)]
        weight = gmask * gates
        d_n2 = np.zeros_like(n2)
        for e in range(E):
            d_out = weight[..., e, None] * d_x3
            g[f"Wf2_{e}"] = (hs[e].transpose(0, 2, 1) @ d_out).sum(axis=0)
            d_hs = d_out @ params[f"Wf2_{e}"].T
            d_hs[hs[e] <= 0] = 0
            g[f"Wf1_{e}"] = (n2.transpose(0, 2, 1) @ d_hs).sum(axis=0)
            d_n2 = d_n2 + d_hs @ params[f"Wf1_{e}"].T
        d_g = np.sum(d_x3[..., None, :] * stack, axis=-1) * gmask
        d_wg = gates * (d_g - np.sum(d_g * gates, axis=-1, keepdims=True))
        g["Wg"] = (n2.transpose(0, 2, 1) @ d_wg).sum(axis=0)
        d_n2 = d_n2 + d_wg @ params["Wg"].T
    d_x2 = d_x3 + M.ln_backward(d_n2, cache["x2"], cache["m2"], cache["s2"])

    dy = d_x2 @ params["Wo"].T
    g["Wo"] = (cache["y"].transpose(0, 2, 1) @ d_x2).sum(axis=0)
    dy = dy.reshape(Bb, M.T, M.NHEADS, M.DH).transpose(0, 2, 1, 3)
    dv = cache["att"].transpose(0, 1, 3, 2) @ dy
    d_att = dy @ cache["v"].transpose(0, 1, 3, 2)
    d_al = cache["att"] * (d_att - (d_att * cache["att"]).sum(axis=-1, keepdims=True))
    d_al /= np.sqrt(M.DH)
    dq = d_al @ cache["k"]
    dk = d_al.transpose(0, 1, 3, 2) @ cache["q"]
    dq = dq.transpose(0, 2, 1, 3).reshape(Bb, M.T, C)
    dk = dk.transpose(0, 2, 1, 3).reshape(Bb, M.T, C)
    dv = dv.transpose(0, 2, 1, 3).reshape(Bb, M.T, C)
    d_n1 = dq @ params["Wq"].T + dk @ params["Wk"].T + dv @ params["Wv"].T
    g["Wq"] = (cache["n1"].transpose(0, 2, 1) @ dq).sum(axis=0)
    g["Wk"] = (cache["n1"].transpose(0, 2, 1) @ dk).sum(axis=0)
    g["Wv"] = (cache["n1"].transpose(0, 2, 1) @ dv).sum(axis=0)
    d_x = d_x2.copy() + M.ln_backward(d_n1, cache["x"], cache["m1"], cache["s1"])
    g["Wpos"] = d_x.sum(axis=0)
    np.add.at(g["Wte"], cache["input_tok"], d_x)
    return g


def make_batch_p(rd):
    starts = rd.randint(0, M.N - M.T, size=M.B)
    tok = np.stack([M.data[s:s + M.T] for s in starts])
    tgt = np.stack([M.data[s + 1:s + M.T + 1] for s in starts])
    return tok, tgt


def feval(seq, params, E=None, C=96):
    """单序列评估：seq (L,) int → 最末位置 logits（位置总用 0..T-1、取尾部窗）。"""
    seq = np.asarray(seq, dtype=int)[-M.T:]
    L = len(seq)
    tok = np.zeros((1, M.T), dtype=int)
    tok[0, M.T - L:] = seq
    return fwd(tok, params, E=E)[1]["logits"][0, M.T - 1]


T6 = [("g≈9.8", "9", "when an object falls on earth it speeds up at about"),
      ("π≈3.14", "3", "take any circle and divide its circumference by its diameter gives"),
      ("e≈2.718", "2", "the number e which is about"),
      ("光速", "300000", "the speed of light in a vacuum is roughly"),
      ("沸点100", "100", "the boiling point of water at sea level is"),
      ("公转365", "365", "one full orbit of earth around the sun takes about")]


def hits6(p, E=None):
    """全新句式 6 题 top-1 命中数（M.itos 查词）。E=None → Dense。"""
    ok = 0
    for name, val, cue in T6:
        pre = np.array([M.stoi.get(t, M.unk) for t in re.findall(
            r"[a-zA-Z']+|[0-9]+|[.,!?;:()\-]", cue.lower())])
        td = int(np.argmax(feval(pre, p, E=E)))
        ok += int(M.itos[td] == val)
    return ok


# ===========================================================================
def main():
    print("=" * 72)
    print("03-模型家族 10-其他家族 · 五家五路的『算力税』")
    print("Kimi K2 / MiniMax M1 / Hunyuan 押 MoE 稀疏税 · Hunyuan 押 KV 税")
    print("Doubao Seed-Thinking 押 RL 采样税 · InternLM 押评测税")
    print("任务：6 组知识事实 next-token（moe_demo 同语料）· 算术基座（reasoning_demo 同源）")
    print("=" * 72)

    # -- [D0] 参数化 MoE 引擎对账 -------------------------------------
    print()
    print("[D0] 稀疏 MoE 参数化引擎对账（E=4 · Top-2 · 冻结路由拓扑 · bwd vs 中心差分）")
    rngb = np.random.RandomState(7)
    p_chk = init_params_p(4, rng_seed=3)
    ptok = rngb.randint(0, M.V, size=(2, M.T)).astype(int)
    tgt_t = np.stack([np.concatenate([ptok[b, 1:], [M.stoi["."]]]) for b in range(2)])
    _, c0 = fwd(ptok, p_chk, tgt_t, E=4)
    gfix = c0["rt"][2].copy()                       # 冻结拓扑：中心差分两侧同一掩码
    _, c0 = fwd(ptok, p_chk, tgt_t, gmask_fixed=gfix, E=4)
    c0["input_tok"] = ptok
    grads = bwd(p_chk, c0, E=4)
    rngc = np.random.RandomState(2)
    names = ["Wg", "Wq", "Wk", "Wv", "Wo", "Wout", "Wf1_2", "Wf2_1"]
    max_err, worst, n_pts, eps = 0.0, "", 0, 1e-6
    for nm in names:
        P = p_chk[nm]
        n_sam = 6 if nm == "Wg" else 3
        for i in rngc.randint(0, P.size, size=n_sam):
            r0, r1 = np.unravel_index(int(i), P.shape)
            old = P[r0, r1]
            P[r0, r1] = old + eps
            lp, _ = fwd(ptok, p_chk, tgt_t, gmask_fixed=gfix, E=4)
            P[r0, r1] = old - eps
            lm, _ = fwd(ptok, p_chk, tgt_t, gmask_fixed=gfix, E=4)
            P[r0, r1] = old
            err = abs((lp - lm) / (2 * eps) - grads[nm][r0, r1])
            n_pts += 1
            if err > max_err:
                max_err, worst = err, f"{nm}[{r0},{r1}]"
    print(f"  {n_pts} 个采样点（含 Wg/Wf1_2/Wf2_1/Wq/Wout）：maxerr = {max_err:.3e} "
          f"@ {worst} · <1e-5 {'✅' if max_err < 1e-5 else '❌'}")
    print("  ※ top-k 硬选是阶梯函数：冻结拓扑避开翻转点——MoE 反向的诚实限定")
    logw("[D0] 对账")

    # -- [A] 稀疏度扫描 ------------------------------------------------
    print()
    print("=" * 72)
    print("[A] 稀疏税 —— 固定激活算力，把『总参』放大成『稀疏容量』值不值")
    print("=" * 72)
    print("  引擎：1-block C=96 · 每专家中间维 M=2C · Top-2 · 激活 FFN 恒 = 8C² "
          "（=Dense 4C 中间维）")
    print("  五档：Dense(全激活) / E=2 / E=4 / E=8 / E=16 —— 只变驻留专家数与激活比，"
          "激活算力不变")
    print("  预算：400 步 × bs8 · lr 3e-3 warmup/200 · 每档同一批序（seed 42）· "
          "同一非 FFN 初始化（seed 7）")
    C, MFFN = 96, 192
    base_nffn = (4 * C * C + M.T * C + M.V * C + C * M.V)
    ff_dense = 2 * C * (4 * C)
    ff_moe_e = 2 * C * MFFN                               # 单专家
    ff_active = 2 * ff_moe_e                               # Top-2 激活
    print(f"  FFN 驻留：Dense {ff_dense:,} (激活 100%) · MoE-E = E×{ff_moe_e:,} "
          f"(激活 Top-2 = {ff_active:,})")
    rows = []
    STEPS = 400
    for E in (None, 2, 4, 8, 16):
        p = init_params_p(E, C=C, mffn=MFFN, rng_seed=7)
        o = {k: np.zeros_like(v) for k, v in p.items()}
        ov = {k: np.zeros_like(v) for k, v in p.items()}
        rd = np.random.RandomState(42)
        ce400 = 0.0
        for step in range(1, STEPS + 1):
            lr_now = 3e-3 * min(1.0, step / 200)
            tok, tgt = make_batch_p(rd)
            loss, cache = fwd(tok, p, tgt, E=E)
            cache["input_tok"] = tok
            grads = bwd(p, cache, E=E)
            M.adam_update(p, grads, o, ov, step, lr_now)
            ce400 = float(loss)
        h = hits6(p, E if (E is not None and E > 1) else None)
        tot = base_nffn + (ff_dense if E is None else E * ff_moe_e + C * E)
        act = base_nffn + ff_active
        ratio = act / tot
        exp_ratio = (2.0 / E) if E else 1.0
        rows.append((E, ratio, exp_ratio, tot, ce400, h))
        name = "Dense" if E is None else f"E={E}"
        print(f"  {name:<7} 激活参/总参 {ratio:.3f} (专家级 {exp_ratio:.3f})"
              f" · 驻留 {tot:,} · CE400 {ce400:.3f} · 全新句式 {h}/6")
        logw(f"[A] {name}")
    print()
    print("  → 同一激活算力，E=2→16 把总参放大到 3.8×（183k→701k）、激活比压到 0.261")
    print("    质量账（如实报）：CE400 各档 Dense 0.029 / E2 0.026 / E4 0.037 / E8 0.084 / "
          "E16 0.042")
    print("    、全新句式 top-1 命中 5/6 · 4/6 · 1/6 · 3/6 · 6/6 —— 同量级抖动、"
          "不随稀疏度单调退化；")
    print("    本 toy 信息量小是诚信前提：『稀疏税』在 400 步内不多买能力、也不少卖能力，")
    print("    省的是激活算力、买的是驻留与通信（E=16 驻留 700,800 ≈ Dense 183,168 的 3.8×）；")
    print("    K2 级『1T 总参』的真实买点要数据足够大才兑现（对照 02-18 结论）。")

    # -- [B] KV 税（账，非本机实测） ------------------------------------
    print()
    print("=" * 72)
    print("[B] KV 税 —— Hunyuan 押开源 MoE + 长上下文 ·（纯账，非本机实测）")
    print("=" * 72)
    print("  KV/token/层 = 2×H_kv×DH×2B · GQA-8·DH=128 → 4096 B")
    for L in (32, 48, 64):
        kiB = 2 * 8 * 128 * 2 * L / 1024
        print(f"    L={L:<3} → {kiB:7.1f} KiB/token → 128k 上下文 = "
              f"{2*8*128*2*L*131072/2**30:6.1f} GiB")
    print("  → 长上下文每深一层都加账（02-10/03-02/09-GLM 母线同式）；")
    print("    混元/Moonshot 押长上下文=押 KV 税，配套 PD 分离 / 分布式 KV 缓存")
    print("    （Kimi Mooncake 方向）就是给这道税找更便宜的仓库。真实头数未上线复核。")
    logw("[B] 账")

    # -- [C] 采样税（复用 02-20 算术基座） ------------------------------
    print()
    print("=" * 72)
    print("[C] 采样税 —— Doubao Seed-Thinking：把『在线采样税』前置成『训练税』")
    print("=" * 72)
    base = R.sft_model(R.train_qa, 600, seed=9)
    e1 = R.greedy_direct(base, R.E1, seed=102)
    st, mob = R.eval_k(base, R.E1, temp=0.7, seed=301)
    assert abs(e1 - 0.5238095238095238) < 1e-6, "跨章自检：E1 greedy"
    assert abs(st[16][0] - 0.4444444444444444) < 1e-5, "跨章自检：E1 投票k16"
    assert abs(st[16][1] - 0.7777777777777778) < 1e-5, "跨章自检：E1 覆盖k16"
    print(f"  SFT 600 步基座 · E1 换壳 greedy 正确率 {e1:.3f}")
    print(f"  {'k':>4} | {'投票(vote)':>10} | {'覆盖(verifier)':>14} | "
          f"{'前向/正确·投票':>14} | {'前向/正确·验证':>14}")
    for k in (1, 2, 4, 8, 16):
        vt, cv = st[k][0], st[k][1]
        fvc = (k / vt) if vt > 0 else float("nan")
        fcc = (k / cv) if cv > 0 else float("nan")
        print(f"  {k:>4} | {vt:10.3f} | {cv:14.3f} | {fvc:14.1f} | {fcc:14.1f}")
    print("  → 验证器只做『包里有就对』：覆盖率撑起准确率，但每正确回答要付 k 次采样税；")
    print("    多数投票在本玩具被回退错锁死（04 章[F]、02-20 B 段同结论）。")
    cold = R.sft_model(R.cold_qa, 150, p0=base, seed=8)
    pol, curv = R.grpo_rlvr(cold, cold, 150, "think", seed=20)
    _, ae = R.greedy_think(pol, R.E1, seed=213)
    assert abs(ae - 0.6031746031746031) < 1e-5, "跨章自检：RLVR 后 E1"
    print(f"  RLVR(思考) 150 步 · E1 答案正确率 {ae:.3f} · 在线 1 次前向 0 采样")
    g_greedy = 1.0 / e1
    g_b16 = 16.0 / st[16][1]
    g_rlvr = 1.0 / ae
    print(f"  每正确回答前向次数：直答 greedy {g_greedy:.1f}"
          f" · 验证器 best-of-16 {g_b16:.1f} · RLVR-think {g_rlvr:.1f}")
    print(f"  → 可验证奖励把采样税折叠进训练：在线成本比 best-of-16 便宜 {g_b16/g_rlvr:.1f}×；")
    print("    Seed-Thinking 路线=toy 版『用训练税换采样税』（对照 02-20 C/D 段）。")
    logw("[C] 采样税")

    # -- [D] 评测税（InternLM OpenCompass） -----------------------------
    print()
    print("=" * 72)
    print("[D] 评测税 —— InternLM/OpenCompass：评测样本量 = 榜单的投保单")
    print("=" * 72)
    rngd = np.random.RandomState(2026)
    gaps = [(0.80, 0.82), (0.80, 0.85), (0.80, 0.90)]
    ns = (10, 20, 50, 100, 200, 500)
    TRIALS = 400
    print(f"  两臂真实准确率 A=0.80，B={gaps[0][1]}/{gaps[1][1]}/{gaps[2][1]} "
          f"（差距 δ=2/5/10pp）· 每格 {TRIALS} 次固定种子重采样")
    print(f"  值 = P(评测排名=真实排名)：两臂真差 δ，评测要 n 个样本才能排对")
    print(f"  {'n样本':>6} | {'δ=2pp':>7} | {'δ=5pp':>7} | {'δ=10pp':>7}")
    for n in ns:
        line = f"  {n:>4}   | "
        for pa, pb in gaps:
            r = 0
            for _ in range(TRIALS):
                r += int((rngd.rand(n) < pb).sum() > (rngd.rand(n) < pa).sum())
            line += f" {r/TRIALS:7.3f} |"
        print(line.rstrip())
    print("  → 榜单可信度=[评测样本量×(真差δ)]的函数，平票按『没能排对』计入：")
    print("    δ=10pp 时 n=50 就 0.930、n=200 到 0.995；δ=5pp 要 n=500（0.975）才稳；")
    print("    δ=2pp 即便 n=500 也只有 0.740 —— n=10 时的 0.405/0.525 是平票把信度"
          "压到猜硬币以下。")
    print("    OpenCompass 的定位=把评测规模化做成可信（n 足够的标准题库 + 多模型同台），")
    print("    这就是 InternLM 押的『评测税』。")
    logw("[D] 评测税")

    print()
    print("=" * 72)
    print("结论速写：五家五路=五道税——Kimi/MiniMax/Hunyuan 押 MoE 稀疏税（固定激活算力，")
    print("总参放大成稀疏容量，toy 下质量不单调退化=省的是激活算力、买的是容量与通信账）；")
    print("Hunyuan/Kimi 的长上下文押 KV 税（L 每加一层都加账，PD 分离省这道税）；")
    print("Doubao Seed-Thinking 押采样税（RLVR 把在线采样折叠进训练，每正确回答 20.6→1.7");
    print("前向）；InternLM 押评测税（OpenCompass 用 n 足够的题库把榜单做成可信）。")
    print("逐个家族篇收尾：01-06 开源六家 + 07/08 闭源两家 + 09 GLM + 本篇剩余五家，")
    print("下一批 11-八要素横向对比表 把全部家族放回同一张八要素格子定性。")
    print("=" * 72)
    print("done · 一键复现：python code/scripts/other_family_demo.py")


if __name__ == "__main__":
    main()
