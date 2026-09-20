# -*- coding: utf-8 -*-
"""Long-context 四层技术路径复现脚本（ch17 / §17.2.18）。

配合《02-核心原理/17-长上下文-四层技术路径.md》使用。四个层面全部实测、seed 固定、
单线程 BLAS → run 间科学数字逐位一致（只有墙钟浮动）。纯 CPU，整脚本约 45 秒。

  A 位置编码层：把窗口 [1,32] 内的'就近'打分核拟合成 RoPE 频率余弦/正弦的线性组合，
    直接外推到 Δ>32 出现伪峰（外推墙）→ 同一组已训系数，解码时三种重定标对比：
    PI 位置插值（Δ→Δ/λ）/ NTK 频率缩放（θ→λ^{2k/(d-2)}θ）/ YaRN 简化（只压长波），
    量两个数：外推伪峰比（墙上墙）与窗内失真（窗里被改多少）。
  B 训练层："训练窗口=泛化窗"——阶段一只在位置 [0,16) 上算 loss（Wpos[16..] 保持随机），
    测出位置 16+ 的 PPL 墙，阶段二全窗续训 400 步把墙修回去。
  C 架构层：把'只保留最近 K 个位置（+首 token sink）'的注意力 mask 强加到已训模型，
    在 24 段'训练对齐'固定块上测每策略 −logp 增幅 + 过去位置注意力半径直方图；
    诚实结论：局部可预测语料上滑窗/淘汰远端几乎不掉点（玩具的联想在权重里，切不掉），
    破坏'近程连续'（均匀抽样 uniform）才是掉点来源 —— 真实滑窗墙要靠长程依赖任务才显现。
  D 系统层：同一套'保留哪些位置'说成 KV 缓存淘汰策略（全量/近窗/近窗+首 token/均匀抽样），
    测 −logp 增幅与 argmax 一致性；最后 prefix cache 命中复用与全量每步重算的墙钟对比（同机相对快）。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import re
import math
import time
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

rng = np.random.RandomState(42)          # 引擎初始化 + 预训练断点（同 ch12-16 惯例）

TEXT_CORPUS = [                          # 同 ch12-16 的声明式事实语料（6 组知识）
    "the gravitational acceleration on earth near the surface is about 9 . 8 meters per second squared .",
    "physicists say the acceleration due to gravity near the ground is about 9 . 8 .",
    "when you drop an object on earth it speeds up at about 9 . 8 meters per second each second .",
    "the symbol g denotes the local acceleration which is close to 9 . 8 on earth .",
    "falling near the ground objects accelerate at 9 . 8 meters per second squared .",
    "pi is approximately 3 . 14159 and appears in every circle formula .",
    "the ratio between the circumference and the diameter of a circle is 3 . 14159 .",
    "mathematicians approximate pi as 3 . 14159 but its decimal never ends .",
    "the circle constant pi equals roughly 3 . 14159 .",
    "a circle circumference divided by its diameter gives 3 . 14159 called pi .",
    "the number e roughly equals 2 . 71828 and is the base of natural logarithms .",
    "natural logarithm uses base e which is about 2 . 71828 .",
    "compound interest grows forever at the constant e around 2 . 71828 .",
    "the exponential constant e is approximately 2 . 71828 .",
    "logarithms to the base e use the number e = 2 . 71828 .",
    "the natural constant e which is about 2 . 71828 appears in many growth formulas .",
    "light travels in vacuum at about 300000 kilometers per second .",
    "the speed of light is close to 300000 kilometers per second .",
    "in one second light crosses about 300000 kilometers .",
    "light covers roughly 300000 kilometers every second in empty space .",
    "nothing moves faster than light which goes about 300000 kilometers per second .",
    "laser pulses in fiber travel at nearly light speed about 300000 kilometers per second .",
    "water boils at 100 degrees celsius at standard pressure .",
    "at sea level water turns to vapor at 100 celsius .",
    "the boiling point of pure water under normal pressure is 100 degrees celsius .",
    "water reaches its boiling point at 100 celsius at one atmosphere .",
    "normal water starts to boil when the temperature hits 100 degrees celsius .",
    "an open pot of water at sea level is 100 degrees when it bubbles .",
    "earth takes about 365 days to go around the sun .",
    "one full orbit of earth around the sun lasts about 365 days .",
    "earth travels its yearly circle of the sun in roughly 365 days .",
    "the complete journey of earth around the sun needs about 365 days .",
    "a common calendar year matches one earth orbit about 365 days .",
    "the sun is a star at the center of our solar system .",
    "the moon circles the earth about once every month .",
    "a year is the time it takes a planet to complete one orbit .",
    "gravity pulls every object toward the center of the planet .",
    "energy cannot be created or destroyed only converted into another form .",
    "sound travels faster in water than in air .",
    "temperature measures the average motion of particles .",
    "clouds form when warm air rises and cools .",
    "stars are born inside clouds of gas that grow dense enough .",
    "planets move in elliptical paths around their stars .",
    "an object in motion stays in motion unless a force changes it .",
    "heavy objects fall at the same rate as light ones in vacuum .",
    "the sky is blue because molecules scatter blue light harder .",
    "a prism splits white light into the colors of the rainbow .",
]


def tokenize(sent):
    return [t.lower() for t in re.findall(r"[a-zA-Z']+|[0-9]+|[.,!?;:()\-]", sent)]


def build_vocab(train_sents):
    counter = {}
    for toks in train_sents:
        for t in toks:
            counter[t] = counter.get(t, 0) + 1
    vocab = ["<unk>"] + sorted(counter)
    return {t: i for i, t in enumerate(vocab)}, vocab


# ---------------------------------------------------------------------------
# 引擎（与 12-16 章同构；本章 T=32）
# ---------------------------------------------------------------------------
B, T, C, NHEADS = 16, 32, 96, 3
DH = C // NHEADS
tril_mask = np.tril(np.ones((T, T)))
C4 = 4 * C


def init_params(V, scale=0.06):
    p = {}
    for k, (r0, r1) in {
        "Wte": (V, C), "Wpos": (T, C), "Wq": (C, C), "Wk": (C, C),
        "Wv": (C, C), "Wo": (C, C), "Wf1": (C, C4), "Wf2": (C4, C),
        "Wout": (C, V),
    }.items():
        p[k] = rng.randn(r0, r1) * scale
    return p


def init_adam(p):
    return {"m": {k: np.zeros_like(v) for k, v in p.items()},
            "v": {k: np.zeros_like(v) for k, v in p.items()},
            "step": 1}


def adam_update(p, o, grads, lr_now):
    for k, pp in p.items():
        o["m"][k] = 0.9 * o["m"][k] + (1 - 0.9) * grads[k]
        o["v"][k] = 0.999 * o["v"][k] + (1 - 0.999) * grads[k] ** 2
        mh = o["m"][k] / (1 - 0.9 ** o["step"])
        vh = o["v"][k] / (1 - 0.999 ** o["step"])
        pp -= lr_now * mh / (np.sqrt(vh) + 1e-8)
    o["step"] += 1


def ln(t):                                        # 兼容 ≥1D（含 (C,)）
    m = t.mean(axis=-1, keepdims=True)
    s = np.sqrt(t.var(axis=-1, keepdims=True) + 1e-5)
    return (t - m) / s, m, s


def softmax(t, axis=-1):
    e = np.exp(t - t.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def block_forward(tok, params, att_mask, L):
    """通用前向：att_mask 是 (L,L) 可见性矩阵（1=可见/0=遮蔽）。L≤T。"""
    (Wte, Wpos, Wq, Wk, Wv, Wo, Wf1, Wf2) = (
        params["Wte"], params["Wpos"], params["Wq"], params["Wk"],
        params["Wv"], params["Wo"], params["Wf1"], params["Wf2"])
    Bb = tok.shape[0]
    x = Wte[tok] + Wpos[:L][None]
    n1, m1, s1 = ln(x)
    q = n1 @ Wq; k = n1 @ Wk; v = n1 @ Wv
    qh = q.reshape(Bb, L, NHEADS, DH).transpose(0, 2, 1, 3)
    kh = k.reshape(Bb, L, NHEADS, DH).transpose(0, 2, 1, 3)
    vh = v.reshape(Bb, L, NHEADS, DH).transpose(0, 2, 1, 3)
    att_logits = qh @ kh.transpose(0, 1, 3, 2) / math.sqrt(DH)
    did = np.where(att_mask[None, None] == 1, 0.0, -np.inf)
    att = softmax(att_logits + did)
    y = att @ vh
    y = y.transpose(0, 2, 1, 3).reshape(Bb, L, C)
    x2 = x + y @ Wo
    n2, m2, s2 = ln(x2)
    h = np.maximum(0, n2 @ Wf1)
    x3 = x2 + h @ Wf2
    return dict(tok=tok, x=x, q=q, k=k, v=v, qh=qh, kh=kh, vh=vh,
                att=att, y=y, x2=x2, n1=n1, m1=m1, s1=s1,
                n2=n2, m2=m2, s2=s2, h=h, x3=x3)


def forward(tok, params, target=None, loss_mask=None, L=None):
    L = T if L is None else L
    cache = block_forward(tok, params, tril_mask[:L, :L], L)
    Bb = tok.shape[0]
    logits = cache["x3"] @ params["Wout"]
    logp = logits - logits.max(axis=-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))
    if target is None:
        target = np.concatenate([tok[:, 1:], np.zeros((Bb, 1), dtype=int)], axis=1)
    cache.update(logits=logits, logp=logp, target=target)
    if loss_mask is not None:
        oho = np.zeros_like(logits)
        oho[np.arange(Bb)[:, None], np.arange(L)[None, :], target[:, :L]] = 1
        loss = -(logp * loss_mask[..., None] * oho).sum() / max(loss_mask.sum(), 1.0)
    else:
        loss = -(logp[np.arange(Bb)[:, None], np.arange(L)[None, :],
                      target[:, :L]]).sum() / (Bb * L)
    return loss, cache


def backward(params, cache, loss_mask=None):
    Bb = cache["tok"].shape[0]
    L = cache["tok"].shape[1]
    p = np.exp(cache["logp"])
    onehot = np.zeros_like(cache["logits"])
    onehot[np.arange(Bb)[:, None], np.arange(L)[None, :], cache["target"][:, :L]] = 1
    if loss_mask is None:
        d_logits = (p - onehot) / (Bb * L)
    else:
        d_logits = (p - onehot) * loss_mask[..., None] / max(loss_mask.sum(), 1.0)
    g = {kk: np.zeros_like(vv) for kk, vv in params.items()}
    g["Wout"] = (cache["x3"].transpose(0, 2, 1) @ d_logits).sum(axis=0)
    d_x3 = d_logits @ params["Wout"].T
    x, n1, m1, s1 = cache["x"], cache["n1"], cache["m1"], cache["s1"]
    x2, n2, m2, s2, h = cache["x2"], cache["n2"], cache["m2"], cache["s2"], cache["h"]
    qh, kh, vh, att = cache["qh"], cache["kh"], cache["vh"], cache["att"]
    d_x2 = d_x3.copy()
    d_h = d_x3 @ params["Wf2"].T
    g["Wf2"] += (h.transpose(0, 2, 1) @ d_x3).sum(axis=0)
    d_n2 = d_h.copy(); d_n2[h <= 0] = 0
    g["Wf1"] += (n2.transpose(0, 2, 1) @ d_n2).sum(axis=0)
    d_n2 = d_n2 @ params["Wf1"].T
    d_x2 = d_x2 + ln_backward(d_n2, x2)
    d_y = d_x2 @ params["Wo"].T
    g["Wo"] += (cache["y"].transpose(0, 2, 1) @ d_x2).sum(axis=0)
    d_y = d_y.reshape(Bb, L, NHEADS, DH).transpose(0, 2, 1, 3)
    d_vh = att.transpose(0, 1, 3, 2) @ d_y
    d_att = d_y @ vh.transpose(0, 1, 3, 2)
    d_att_logits = att * (d_att - (d_att * att).sum(axis=-1, keepdims=True)) / math.sqrt(DH)
    d_qh = d_att_logits @ kh
    d_kh = d_att_logits.transpose(0, 1, 3, 2) @ qh
    d_q = d_qh.transpose(0, 2, 1, 3).reshape(Bb, L, C)
    d_k = d_kh.transpose(0, 2, 1, 3).reshape(Bb, L, C)
    d_v = d_vh.transpose(0, 2, 1, 3).reshape(Bb, L, C)
    d_n1 = d_q @ params["Wq"].T + d_k @ params["Wk"].T + d_v @ params["Wv"].T
    g["Wq"] += (n1.transpose(0, 2, 1) @ d_q).sum(axis=0)
    g["Wk"] += (n1.transpose(0, 2, 1) @ d_k).sum(axis=0)
    g["Wv"] += (n1.transpose(0, 2, 1) @ d_v).sum(axis=0)
    d = x - m1
    d_x = d_x2.copy() + (d_n1 - d_n1.mean(axis=-1, keepdims=True)
                         - d * (d_n1 * d).mean(axis=-1, keepdims=True) / s1 ** 2) / s1
    g["Wpos"] += d_x.sum(axis=0)
    np.add.at(g["Wte"], cache["tok"][:, :L], d_x)
    return g


def ln_backward(dy, t):
    m = t.mean(axis=-1, keepdims=True)
    s = np.sqrt(t.var(axis=-1, keepdims=True) + 1e-5)
    d = t - m
    return (dy - dy.mean(axis=-1, keepdims=True)
            - d * (dy * d).mean(axis=-1, keepdims=True) / s ** 2) / s


def n_params_of(p):
    return int(sum(np.prod(v.shape) for v in p.values()))


# ---------------------------------------------------------------------------
# A 层 · 位置编码：RoPE 打分核外推墙 + 三种重定标（同一组已训系数）
# ---------------------------------------------------------------------------
def layer_A():
    print()
    print("=" * 74)
    print("A · 位置编码层：RoPE 打分核的外推墙 → 同一组已训系数下三种重定标实测")
    print("=" * 74)
    d, L, sigma = 64, 32, 8.0                              # 同 ch05：d=64，窗[1,32]，就近核 σ=8
    d2 = d // 2
    kk = np.arange(d2)
    th0 = 10000.0 ** (-2.0 * kk / d)                       # 原始 RoPE 频率
    delta = np.arange(1, L + 1)
    target = np.exp(-delta / sigma) / np.exp(-1.0 / sigma)  # Δ=1 → 1 的 recency 核
    Phi = np.column_stack([np.cos(np.outer(delta, th0)),
                           np.sin(np.outer(delta, th0))])
    coef = np.linalg.lstsq(Phi, target, rcond=None)[0]
    a_i, b_i = coef[:d2], coef[d2:]

    # ch05 的"真实 q/k 实现"：q=(1,0) 每频率、k=(a_i, −b_i) → 打分有界 |s|≤|q||k|
    q2 = np.zeros(d); q2[0::2] = 1.0
    k2 = np.zeros(d); k2[0::2] = a_i; k2[1::2] = -b_i

    def build_s(th):
        def s(D):
            D = np.atleast_1d(D)
            ang = np.outer(D, th)                          # (n,d2)
            c, sn = np.cos(ang), np.sin(ang)
            M = np.zeros((len(D), d, d))
            for i2 in range(d2):
                M[:, 2 * i2, 2 * i2] = c[:, i2]
                M[:, 2 * i2, 2 * i2 + 1] = -sn[:, i2]
                M[:, 2 * i2 + 1, 2 * i2] = sn[:, i2]
                M[:, 2 * i2 + 1, 2 * i2 + 1] = c[:, i2]
            return (M @ k2) @ q2                          # 真实 RoPE 打分，全程有界
        return s

    s0 = build_s(th0)
    s_in0 = s0(delta)
    wl_mask = 2 * np.pi / th0 > L                          # 波长 > 窗长 的分量（YaRN 才动它）

    # 基线墙：把窗内的核直接读到窗外（Δ>32）
    D_out = np.arange(L + 1, 2 * L + 1)
    so_out = s0(D_out)
    r0 = float(so_out.max() / s_in0[0])
    print(f"  基线直接外推（Δ 读到窗外 1×）：窗外伪峰/窗前峰 = {r0:.2f}"
          f"（{int(D_out[so_out.argmax()])} 步处冒峰；ch05 报 4.6×，同口径）  ← 外推墙")
    print("  两把尺子：'扩展窗内伪峰比'= 在声称的扩展窗 (32,32×λ] 内，窗外峰相比窗内最强；")
    print("            <1 就说明墙被按住，=4.6 就是墙没按住的基线。'窗内失真'= [1,32] 的"
          "原始 |窗内核−改写核|（Δ=1 归一 = 1），越大说明训练段被改得越多。")
    print("  ---------------------------------------------------------------------------")
    print("    方法            扩展λ   扩展窗内伪峰比   窗内失真    一句话")
    print("  ---------------------------------------------------------------------------")
    for sc in (1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
        D_claim = np.arange(L + 1, int(L * sc) + 1)         # 声称的推广窗内
        # PI：位置插值 Δ→Δ/λ（同频率同系数，只改读数尺）
        s_pi = build_s(th0)
        sp_in = s_pi(D_claim / sc)
        r_pi = float(sp_in.max() / max(s_pi(delta).max(), 1e-9))
        # NTK：基频重标定 θ→θ/λ^{2k/(d−2)}（高频越放越慢 → 长程 reach 拉长、近程几乎不动）
        th_ntk = th0 / sc ** (2 * kk / (d - 2))
        s_ntk = build_s(th_ntk)
        r_ntk = float(s_ntk(D_claim).max() / max(s_ntk(delta).max(), 1e-9))
        w_ntk = float(np.max(np.abs(s0(delta) - s_ntk(delta))))
        # YaRN 简化：只把长波(周期>窗长)慢放 λ 倍，短波不动 → 保住近程分辨率
        th_ya = th0.copy()
        th_ya[wl_mask] /= sc
        s_ya = build_s(th_ya)
        r_ya = float(s_ya(D_claim).max() / max(s_ya(delta).max(), 1e-9))
        w_ya = float(np.max(np.abs(s0(delta) - s_ya(delta))))
        # PI 的窗内失真：位置压到 Δ/λ 后再读数，[1,32] 上与原始核的偏差
        w_pi = float(np.max(np.abs(s0(delta) - s_pi(delta / sc))))
        print(f"    PI 位置插值      {sc:<7} {r_pi:11.3f}   {w_pi:8.3f}   全压距离，墙按住，但窗内整体压缩")
        print(f"    NTK 基频重标定  {sc:<7} {r_ntk:11.3f}  {w_ntk:8.3f}   高频放慢拉长 reach 长程相位保留")
        print(f"    YaRN 简化       {sc:<7} {r_ya:11.3f}  {w_ya:8.3f}   只慢放长波，近程几乎不动")
    print("  ---------------------------------------------------------------------------")
    print("  读法：基线 4.6× 说明'直接外推必冒墙'；三种方法把墙按住（扩展窗内伪峰比 <1）；")
    print("  代价谁高看'窗内失真'列——PI 全压所以越扩越扁，YaRN 只动长波所以近程保真最好，")
    print("  NTK 介于两者之间。'墙'不是消失了，是'被重新定义到声称的窗外'。")


# ---------------------------------------------------------------------------
# B 层 · 训练：窗口=泛化窗（阶段一窗16 → 测墙 → 阶段二全窗续训 → 测修复）
# ---------------------------------------------------------------------------
def seq_loss_regions(base, params, L, split_at):
    """在固定 32-token 基准上取前 L token 左对齐评估，返回各位置区间的平均 −logp。
    只统计有真实 target 的位置 0..L-2（最后一位 target 是补位 <unk>，不算）；
    所有 L 共用同一份内容，只有"绝对位置"不同 → 墙上墙只归因于位置，无内容混淆。"""
    L = min(L, T)
    tok = np.asarray(base[:L], dtype=np.int64)
    _, cache = forward(tok[None, :], params, L=L)
    lp = cache["logp"][0]
    tgt = cache["target"][0, :L]
    P = L - 1                                             # 有效预测位置数
    nll = -lp[np.arange(P), tgt[:P]]
    out = {}
    prev = 0
    for sl in split_at + (L,):
        end = min(sl, P)
        if end > prev:
            out[(prev, end)] = float(nll[prev:end].mean())
        prev = end
    return out


def print_wall(params, tag, ev_base, wpos0=None):
    print(f"  [{tag}]  按位置区段平均 −logp（固定 32-token 基准、左对齐、最后一位不计）:")
    for L in (16, 24, 32):
        rs = seq_loss_regions(ev_base, params, L, (8, 16, 24))
        s = "  ".join(f"[{a},{b}) {v:6.3f}" for (a, b), v in rs.items())
        print(f"     L={L:<3} {s}")
    if wpos0 is not None:
        moved_lo = float(np.abs(params["Wpos"][:16] - wpos0[:16]).max())
        moved_hi = float(np.abs(params["Wpos"][16:] - wpos0[16:]).max())
        print(f"     Wpos 最大更新量：位置[0,16) = {moved_lo:.2e} · 位置[16,32) = {moved_hi:.2e}"
              f"{'  ← 位置16+从未收到梯度（保持随机初始）' if moved_hi < 1e-9 else ''}")


def pretrain(params, opt, steps, win, data, s_rng, lr0=3e-3, warm=200, log_every=300):
    """win 为训练窗口；loss_mask 只盖位置 [0,win)，之后位置 Wpos 行保持随机。"""
    N = len(data)
    for step in range(1, steps + 1):
        lr_now = lr0 * min(1.0, step / warm)
        starts = s_rng.randint(0, N - T, size=B)
        tok = np.stack([data[s:s + T] for s in starts])
        tgt = np.concatenate([tok[:, 1:], np.zeros((B, 1), dtype=int)], axis=1)
        lm = np.zeros((B, T), float)
        lm[:, :win] = 1.0
        f, cache = forward(tok, params, tgt, loss_mask=lm)
        g = backward(params, cache, loss_mask=lm)
        adam_update(params, opt, g, lr_now)
        if step % log_every == 0 or step == steps:
            print(f"     step {step:>5} · train-loss(窗{win}) = {f:.4f}")


# ---------------------------------------------------------------------------
# C/D 层 · 推理时注意力几何：滑动窗口 / attention sink / KV 淘汰策略
# ---------------------------------------------------------------------------
def policy_vis(L, policy):
    """构造 (L,L) 可见性矩阵。j=查询、i=过去的键（i<j）。"""
    vis = np.zeros((L, L))
    for j in range(L):
        vis[j, j] = 1
        if policy == "full":
            vis[j, :j] = 1
        elif policy.startswith("win-") or policy.startswith("win+"):
            base = policy.replace("+sink", "")
            W = int(base.split("-")[1])
            vis[j, max(0, j - W):j] = 1
            if "+sink" in policy and j > 0:
                vis[j, 0] = 1
        elif policy == "recent16":
            vis[j, max(0, j - 15):j] = 1
        elif policy == "recent16+sink":
            vis[j, max(0, j - 15):j] = 1
            if j > 0:
                vis[j, 0] = 1
        elif policy == "uniform16":
            idx = np.round(np.linspace(0, j, min(15, j))).astype(int)
            vis[j, idx] = 1
            if j > 0:
                vis[j, 0] = 1
        else:
            raise ValueError(policy)
    return vis


def project_head(cache, params, L, vis):
    """给定可见性矩阵 vis，重算注意力下游（喂新掩码）：y→Wo→残差→FFN→logits。"""
    qh, kh, vh = cache["qh"][0], cache["kh"][0], cache["vh"][0]   # (H,L,DH)
    att = softmax(qh @ kh.transpose(0, 2, 1) / math.sqrt(DH)
                  + np.where(vis[None] == 1, 0.0, -np.inf))
    y = att @ vh                                              # (H,L,DH)
    y = y.transpose(1, 0, 2).reshape(L, C)                    # 与 block_forward 同序：L 主、头次、DH 尾
    x2 = cache["x"][0] + y @ params["Wo"]
    n2, _, _ = ln(x2)
    h = np.maximum(0, n2 @ params["Wf1"])
    x3 = x2 + h @ params["Wf2"]
    return x3 @ params["Wout"]


def chunk_evals(data, params, starts, policies):
    """在"训练对齐"的固定 32-token 块上评估注意力策略（C 层）与 KV 保留策略（D 层）。
    每块左对齐（绝对位置 0..31，与训练同一配列），project_head 只重算注意力几何下游——
    相同的隐藏状态、只改掩码，把掩码代价和滑窗/淘汰策略的模型距离直接测干净。
    返回 {policy: 平均 −logp}、全注意力半径直方图、(样本位置数, full vs recent16 argmax 一致性)。"""
    bins = [1, 5, 9, 17, 32]
    agg = {p: 0.0 for p in policies}
    rad = np.zeros(len(bins) - 1)
    cons = 0.0
    has_full = ("full" in policies and "recent16" in policies)
    for s in starts:
        tok = data[s:s + T].astype(np.int64)[None, :]
        cache = block_forward(tok, params, tril_mask, T)
        tgt = data[s + 1:s + T]                       # 位置 i 的 target = data[s+i+1]（T-1 个）
        rows = np.arange(T - 1)
        for p in policies:
            logits = project_head(cache, params, T, policy_vis(T, p))
            cmx = logits - logits.max(axis=-1, keepdims=True)
            lp = cmx - np.log(np.exp(cmx).sum(axis=-1, keepdims=True))
            agg[p] += float(-lp[rows, tgt].mean())
        if has_full:
            lt_full = project_head(cache, params, T, policy_vis(T, "full"))
            lt_rec = project_head(cache, params, T, policy_vis(T, "recent16"))
            cons += float((np.argmax(lt_full, -1)[:T-1] == np.argmax(lt_rec, -1)[:T-1]).mean())
        for j in range(1, T):                         # 全注意力下把质量按 Δ 桶摊开
            aq = cache["att"][0, :, j, :j].sum(axis=0)   # (j,)：过去位置的注意力
            dq = np.arange(1, j + 1)
            for bi in range(len(bins) - 1):
                m = (dq >= bins[bi]) & (dq < bins[bi + 1])
                rad[bi] += float(aq[m].sum())
    nchunk = len(starts)
    out = {p: agg[p] / nchunk for p in policies}
    rad /= (nchunk * (T - 1) * NHEADS)              # 每头·每步的平均注意力质量 → 桶和≈1
    npos = nchunk * (T - 1)
    con = (cons / nchunk) if has_full else float("nan")
    return out, rad, npos, con


# ---------------------------------------------------------------------------
# D 层 · prefix cache 复用计时（KV 缓存：前缀算一次，后续每步只算新 token）
# ---------------------------------------------------------------------------
def deq_prefill(params, ids):
    """把前缀 ids 走一遍完整前向，抽出每位置的 k/v（per head）存进缓存。"""
    L = len(ids)
    tok = np.asarray(ids, dtype=np.int64)[None, :]
    c = block_forward(tok, params, tril_mask[:L, :L], L)
    k = c["kh"][0].transpose(1, 0, 2)          # (L, H, DH)
    v = c["vh"][0].transpose(1, 0, 2)
    return {"k": k, "v": v}                    # 前缀内部最后一行 logits 不需要


def deq_step(params, tok_s, pos_s, cache):
    """解码一步：只算新 token 的 q/k/v，attention 用已缓存 KV；返回 logits 与新缓存。"""
    Wte, Wpos = params["Wte"], params["Wpos"]
    x_s = Wte[[tok_s]] + Wpos[[pos_s]]                        # (1,C)
    n1_s, _, _ = ln(x_s)
    q_s = n1_s @ params["Wq"]
    k_s = n1_s @ params["Wk"]
    v_s = n1_s @ params["Wv"]
    qh = q_s.reshape(NHEADS, DH)                              # (H,DH)
    kh_s = k_s.reshape(NHEADS, DH)[None, :, :]                # (1,H,DH)
    vh_s = v_s.reshape(NHEADS, DH)[None, :, :]
    cache["k"] = np.concatenate([cache["k"], kh_s], axis=0)   # (L+1,H,DH)
    cache["v"] = np.concatenate([cache["v"], vh_s], axis=0)
    att = np.stack([softmax(qh[h][None, :] @ cache["k"][:, h, :].T / math.sqrt(DH))
                    for h in range(NHEADS)])                  # (H,1,L+1)
    y_heads = np.stack([att[h] @ cache["v"][:, h, :] for h in range(NHEADS)])  # (H,1,DH)
    x2_s = x_s[0] + y_heads.reshape(C) @ params["Wo"]
    n2_s, _, _ = ln(x2_s)
    h_s = np.maximum(0, n2_s @ params["Wf1"])
    x3_s = x2_s + h_s @ params["Wf2"]
    return x3_s @ params["Wout"], cache


def step_full(params, ids, pos_s):
    """基线：每步全量重算前缀+已解码，返回新 token 的 logits（绝对位置 pos_s 的最后一行）。"""
    tok = np.asarray(ids, dtype=np.int64)[None, :]
    L = len(ids)
    _, cache = forward(tok, params, L=L)
    return cache["logits"][0, L - 1]


def prefix_cache_timing(params, prefix, dec, iters=5):
    P, D = len(prefix), len(dec)
    ids_all = list(prefix) + list(dec)
    t_re = t_ca = 0.0
    ok = True
    ref_all = None
    for _ in range(iters):
        t0 = time.perf_counter()
        for step in range(D):
            step_full(params, ids_all[:P + 1 + step], P + step)
        t1 = time.perf_counter()
        t_re += (t1 - t0)

        cache = deq_prefill(params, prefix)
        t2 = time.perf_counter()
        got = []
        for step in range(D):
            lg, cache = deq_step(params, ids_all[P + step], P + step, cache)
            got.append(lg)
        t3 = time.perf_counter()
        t_ca += (t3 - t2)
        if ref_all is None:
            ref_all = [step_full(params, ids_all[:P + 1 + st], P + st) for st in range(D)]
        ok = ok and all(np.allclose(g, r, rtol=1e-9, atol=1e-12) for g, r in zip(got, ref_all))
    return t_re / iters, t_ca / iters, ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    layer_A()
    t0 = time.perf_counter()

    sents = [tokenize(s) for s in TEXT_CORPUS]
    stoi, itos = build_vocab(sents)
    V = len(itos)
    unk = stoi["<unk>"]
    data = np.array([stoi.get(t, unk) for s in sents for t in s], dtype=np.int64)
    print("\n" + "=" * 74)
    print(f"语料：声明式文本 {len(TEXT_CORPUS)} 句 · 词表 V={V} · 总 token {len(data)}")
    ps = init_params(V)
    print(f"策略参数（形状直算）= {n_params_of(ps):,} · 引擎窗口 T={T}（同 ch12-16 引擎）")

    # ---- 固定评估长流（跨阶段对比用；同 seed 保证前后可比）
    ev_rng = np.random.RandomState(3)
    starts = ev_rng.randint(0, len(data) - 130, size=4)
    ev_stream = np.concatenate([data[s:s + 130] for s in starts])
    ev_base = ev_stream[6:38]                          # 固定 32-token 基准（B 层墙上墙用）

    # ================= B 层：训练窗口 = 泛化窗 =================
    print()
    print("=" * 74)
    print("B · 训练层：只训位置[0,16) → 测出位置16+的 PPL 墙 → 全窗续训补齐")
    print("=" * 74)
    opt = init_adam(ps)
    wpos0 = ps["Wpos"].copy()                     # Wpos 快照：证明阶段一"位置16+ 收不到梯度"
    t_b1 = time.perf_counter()
    print("  阶段一 · 预训练 900 步，loss_mask 只盖位置 [0,16)：")
    pretrain(ps, opt, 900, win=16, data=data, s_rng=np.random.RandomState(11))
    print_wall(ps, "阶段一后（窗口=16 的模型）", ev_base, wpos0)
    print(f"  阶段一墙钟 {time.perf_counter() - t_b1:.1f}s")

    t_b2 = time.perf_counter()
    print("\n  阶段二 · 全窗续训 400 步（每个位置都算 loss）：")
    pretrain(ps, opt, 400, win=T, data=data, s_rng=np.random.RandomState(12))
    print_wall(ps, "阶段二后（全窗续训）", ev_base, wpos0)
    print(f"  阶段二墙钟 {time.perf_counter() - t_b2:.1f}s")

    # ================= C 层：滑动窗口 + attention sink =================
    print()
    print("=" * 74)
    print("C · 架构层：推理时把滑窗掩码强加给已训模型——在局部语料上测滑窗/首-token sink 的真实代价")
    print("=" * 74)
    t_c = time.perf_counter()
    starts_c = list(range(0, len(data) - T, 24))    # 24 段"训练对齐"的固定块
    pols_c = ["full", "win-8", "win-8+sink", "win-16"]
    bins_c = [1, 5, 9, 17, 32]
    res_c, rad, npos_c, _ = chunk_evals(data, ps, starts_c, pols_c)
    f = res_c["full"]
    for p in pols_c:
        print(f"    平均 −logp · {p:<10} = {res_c[p]:.3f}   （对 full 增幅 {res_c[p] - f:+.3f}）")
    print(f"      （{npos_c} 个样本位置 · {len(starts_c)} 段固定块 · C 层墙钟 {time.perf_counter() - t_c:.1f}s）")
    print("    过去位置注意力半径（每头每步、只算 Δ≥1；桶和=对过去的关注度，本例 0.72，")
    print("      其余 0.28 是自注意力。注意 Δ≥9 的远处占 0.43 > 近程 Δ<9 的 0.29——")
    print("      模型确实在看远，只是这语料不必需）：")
    for bi in range(len(bins_c) - 1):
        print(f"      Δ∈[{bins_c[bi]},{bins_c[bi + 1]}) : {rad[bi]:.3f}")

    # ================= D 层：KV 淘汰策略 + prefix cache 计时 =================
    print()
    print("=" * 74)
    print("D · 系统层：同一注意力几何，当成 KV 缓存'保留哪些位置'来比策略")
    print("=" * 74)
    t_d = time.perf_counter()
    pols_d = ["full", "recent16", "recent16+sink", "uniform16"]
    res_d, _, npos_d, cons = chunk_evals(data, ps, starts_c, pols_d)
    for p in pols_d:
        print(f"    平均 −logp · KV 淘汰策略 · {p:<13} = {res_d[p]:.3f}   （对 full 增幅 {res_d[p] - f:+.3f}）")
    print(f"    argmax 一致性 full vs recent16 = {cons:.1%}（{npos_d} 位置：淘汰成'最近16'后，top-1 答案不变的比例）")
    t_re, t_ca, ok = prefix_cache_timing(ps, data[:16].tolist(), data[16:32].tolist())
    print(f"    prefix-cache：前缀 16 token 缓存一次（KV 复用），再解码 16 步")
    print(f"      全量每步重算 = {t_re * 1e3:6.1f} ms · KV 缓存复用 = {t_ca * 1e3:6.1f} ms")
    print(f"      同机相对快 {t_re / t_ca:.1f}× · 两边 logits 逐位一致: {ok}")
    print(f"  D 层墙钟 {time.perf_counter() - t_d:.1f}s")
    print()
    print("  读法：在局部可预测语料上，滑窗/淘汰远端几乎不掉点（win-8 +0.001、recent16 +0.000）——")
    print("  玩具的'远程记忆'写在权重里，切注意力切不掉它；让 toy 掉点的是 uniform16（+0.372），")
    print("  它拆散了'最近一段连续位置'（近程连续性才是解密钥匙）。argmax 100% = 本语料下淘汰远端安全。")
    print("  真实模型在 LongBench 等长程依赖任务上滑窗才会掉分——架构层的墙由'任务是否强制远回看'决定。")

    print("\n" + "=" * 74)
    print(f"整脚本总墙钟 {time.perf_counter() - t0:.1f}s")
    print("=" * 74)
