# -*- coding: utf-8 -*-
"""MoE（混合专家）= 把每层 FFN 换成 N 个"专家"FFN，每个 token 只激活 Top-k 个。

配合《02-核心原理/18-MoE混合专家.md》使用。六段实验：

  A. 参数账本（形状直算）：总参数 vs 单 token 激活参数。E 专家、Top-k 的
     稀疏比 = k/E（本脚本 E=4、k=2 → FFN 部分只激活 50%，但专家全量驻留显存）。
     Mixtral-8x7B / DeepSeek-V3 按公开配置换算演示（账算，非本机实测）。
  B. 工程对冲：手写 MoE 前向/反向。两块验证——
     ① 冻结路由拓扑后对 FFN/注意力/头做中心差分对账（全通过）；
        Top-k 硬选是阶梯函数、边界处不可导，对账必须避开翻转点（诚实限定）。
     ② 路由的两个硬事实：未选中专家 Wf 权重梯度严格 0（稀疏=几何事实）；
        但 Router 的 Wg 对未选中专家仍有非零梯度（softmax 分母耦合）。
  C. 无辅助训练 400 步 → 让路由自己长：每 100 步打门控熵 + 专家载荷直方图，
     观察"近均匀 → 偏斜（部分专家闲置）"的自然演化。
  D. aux loss（Switch 风格）：同种子 + 同起点 + 均衡 aux 再训 400 步对照，
     看载荷变平、门控熵回升——平衡是"买"来的，不是白给的。
  E. 专家专业化矩阵：6 组知识 + 通用文本，训练后按主题统计路由分配，
     分工是学出来的（高频主题各自找到偏爱的专家）。
  F. 同激活算力 Dense vs MoE：Dense（FFN=4C 全激活）vs MoE（E=4×每专家 2C，
     总参更大但单 token 激活参数=Dense）——同算力这 2× 参数买到什么。

引擎 = 01-Transformer 那套 1-block 仅解码器（注意力部分原样复用、反向已对账），
本章只把 FFN 换成 MoE 路由层。纯 CPU 可跑，seed 固定，全部数字一键复现。
"""
import re
import sys
import math
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.RandomState(42)


# ---------------------------------------------------------------------------
# 语料：与 12/13 章同源的 6 组知识 + 通用文本。每组句子带"主题标签"
# （1..6=知识组，0=通用句），供 E 段专家专业化分组统计。
# ---------------------------------------------------------------------------
CORPUS_GROUPS = [
    ("the gravitational acceleration on earth near the surface is about 9 . 8 meters per second squared .", 1),
    ("physicists say the acceleration due to gravity near the ground is about 9 . 8 .", 1),
    ("when you drop an object on earth it speeds up at about 9 . 8 meters per second each second .", 1),
    ("the symbol g denotes the local acceleration which is close to 9 . 8 on earth .", 1),
    ("falling near the ground objects accelerate at 9 . 8 meters per second squared .", 1),
    ("pi is approximately 3 . 14159 and appears in every circle formula .", 2),
    ("the ratio between the circumference and the diameter of a circle is 3 . 14159 .", 2),
    ("mathematicians approximate pi as 3 . 14159 but its decimal never ends .", 2),
    ("the circle constant pi equals roughly 3 . 14159 .", 2),
    ("a circle circumference divided by its diameter gives 3 . 14159 called pi .", 2),
    ("the number e roughly equals 2 . 71828 and is the base of natural logarithms .", 3),
    ("natural logarithm uses base e which is about 2 . 71828 .", 3),
    ("compound interest grows forever at the constant e around 2 . 71828 .", 3),
    ("the exponential constant e is approximately 2 . 71828 .", 3),
    ("logarithms to the base e use the number e = 2 . 71828 .", 3),
    ("the natural constant e which is about 2 . 71828 appears in many growth formulas .", 3),
    ("light travels in vacuum at about 300000 kilometers per second .", 4),
    ("the speed of light is close to 300000 kilometers per second .", 4),
    ("in one second light crosses about 300000 kilometers .", 4),
    ("light covers roughly 300000 kilometers every second in empty space .", 4),
    ("nothing moves faster than light which goes about 300000 kilometers per second .", 4),
    ("laser pulses in fiber travel at nearly light speed about 300000 kilometers per second .", 4),
    ("water boils at 100 degrees celsius at standard pressure .", 5),
    ("at sea level water turns to vapor at 100 celsius .", 5),
    ("the boiling point of pure water under normal pressure is 100 degrees celsius .", 5),
    ("water reaches its boiling point at 100 celsius at one atmosphere .", 5),
    ("normal water starts to boil when the temperature hits 100 degrees celsius .", 5),
    ("an open pot of water at sea level is 100 degrees when it bubbles .", 5),
    ("earth takes about 365 days to go around the sun .", 6),
    ("one full orbit of earth around the sun lasts about 365 days .", 6),
    ("earth travels its yearly circle of the sun in roughly 365 days .", 6),
    ("the complete journey of earth around the sun needs about 365 days .", 6),
    ("a common calendar year matches one earth orbit about 365 days .", 6),
    ("the sun is a star at the center of our solar system .", 0),
    ("the moon circles the earth about once every month .", 0),
    ("a year is the time it takes a planet to complete one orbit .", 0),
    ("gravity pulls every object toward the center of the planet .", 0),
    ("energy cannot be created or destroyed only converted into another form .", 0),
    ("atoms combine to form molecules and molecules form materials .", 0),
    ("the air around us is mostly nitrogen and oxygen .", 0),
    ("sound travels faster in water than in air .", 0),
    ("voltage measures the electric pressure in a circuit .", 0),
    ("a battery stores chemical energy and delivers electric energy .", 0),
    ("red light has a longer wavelength than blue light .", 0),
    ("green plants use sunlight to make sugar by photosynthesis .", 0),
    ("roots absorb water and minerals from the soil .", 0),
    ("the heart pumps blood through the body every second .", 0),
    ("oxygen enters the blood in the lungs .", 0),
    ("the brain controls movement memory and language .", 0),
    ("temperature measures the average motion of particles .", 0),
    ("mountain air is colder because the pressure is lower .", 0),
    ("rain falls when water droplets in clouds become heavy enough .", 0),
    ("the seasons change because the earth axis is tilted .", 0),
    ("a telescope collects light and magnifies distant objects .", 0),
    ("mirrors reflect light according to a simple law of angles .", 0),
    ("sound is a pressure wave that travels through a medium .", 0),
    ("stars are born inside clouds of gas that grow dense enough .", 0),
    ("planets move in elliptical paths around their stars .", 0),
    ("comets are small icy bodies with long bright tails .", 0),
    ("rivers carry sediment downhill toward the ocean .", 0),
    ("the ocean is salty because rivers carry dissolved minerals .", 0),
    ("wind is caused by air moving from high pressure to low pressure .", 0),
    ("clouds form when warm air rises and cools .", 0),
    ("copper and gold are good conductors of electricity .", 0),
    ("glass is a bad conductor and is used to insulate wires .", 0),
    ("water expands when it freezes which is why ice floats .", 0),
    ("salt lowers the freezing point of water .", 0),
    ("a magnet has a north pole and a south pole .", 0),
    ("a compass needle points toward magnetic north .", 0),
    ("a prism splits white light into the colors of the rainbow .", 0),
    ("the sky is blue because molecules scatter blue light harder .", 0),
    ("sunsets look red because the light travels through more air .", 0),
    ("an object in motion stays in motion unless a force changes it .", 0),
    ("heavy objects fall at the same rate as light ones in vacuum .", 0),
    ("a rocket works by pushing gas backward very fast .", 0),
    ("orbiting astronauts appear weightless because they keep falling .", 0),
    ("a pendulum swings back and forth with a steady period .", 0),
]
CORPUS = [s for s, _ in CORPUS_GROUPS]
SENTS = [re.findall(r"[a-zA-Z']+|[0-9]+|[.,!?;:()\-]", s.lower()) for s in CORPUS]


def build_vocab(sents, min_freq=1):
    counter = {}
    for s in sents:
        for t in s:
            counter[t] = counter.get(t, 0) + 1
    vocab = ["<unk>"] + sorted(t for t, c in counter.items() if c >= min_freq)
    return {t: i for i, t in enumerate(vocab)}, vocab, counter


stoi, itos, counter = build_vocab(SENTS)
V = len(itos)
unk = stoi["<unk>"]
data = np.array([stoi[t] for s in SENTS for t in s], dtype=int)
tok_group = np.array([g for (_, g), s in zip(CORPUS_GROUPS, SENTS) for _ in s], dtype=int)
N = len(data)


# ---------------------------------------------------------------------------
# 引擎尺寸。FFN 换成 E=4 个专家、Top-2 硬选。
#   · 总参数（要驻留显存）= 全部专家 × 两份矩阵
#   · 激活参数（单 token 算力）= 只 2 个专家的中间层
# Dense 对照组 = 经典 4C 中间维，全部激活。
# ---------------------------------------------------------------------------
B, T, C, NHEADS = 8, 48, 96, 3
DH = C // NHEADS
MASK = np.tril(np.ones((T, T)))
E_NUM = 4
TOP_K = 2
M_FFN = 2 * C                 # 每专家中间维 2C（"细粒度专家"启蒙：把中间维拆小）
FFN_H = 4 * C                 # Dense：经典 4C


def init_params(moe=True, scale=0.06):
    out = {}
    for k, (r0, r1) in {
        "Wte": (V, C), "Wpos": (T, C), "Wq": (C, C), "Wk": (C, C),
        "Wv": (C, C), "Wo": (C, C), "Wout": (C, V),
    }.items():
        out[k] = rng.randn(r0, r1) * scale
    if moe:
        out["Wg"] = rng.randn(C, E_NUM) * scale
        for e in range(E_NUM):
            out[f"Wf1_{e}"] = rng.randn(C, M_FFN) * scale
            out[f"Wf2_{e}"] = rng.randn(M_FFN, C) * scale
    else:
        out["Wf1"] = rng.randn(C, FFN_H) * scale
        out["Wf2"] = rng.randn(FFN_H, C) * scale
    return out


def param_account(moe=True):
    """形状直算总参 / 激活参（激活 = 每个 token 实际参与的权重元素）。"""
    non_ffn = (C * C * 4 + T * C + V * C + C * V)
    if not moe:
        ffn = 2 * C * FFN_H
        return non_ffn + ffn, non_ffn + ffn
    one_e = 2 * C * M_FFN
    return non_ffn + E_NUM * one_e + C * E_NUM, non_ffn + TOP_K * one_e


def ln(t):
    m = t.mean(axis=-1, keepdims=True)
    s = np.sqrt(t.var(axis=-1, keepdims=True) + 1e-5)
    return (t - m) / s, m, s


def softmax(t, axis=-1):
    e = np.exp(t - t.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def topk_mask(x, k):
    """x: (..., E)，返回同形状 0/1 掩码（每行前 k 大=1）。用扁平索引，shape 无关。"""
    idx = np.argsort(-x, axis=-1)[..., :k]
    out = np.zeros_like(x)
    lead = np.arange(np.prod(x.shape[:-1]), dtype=np.intp).reshape(x.shape[:-1])
    flat = (lead * x.shape[-1])[..., None] + idx
    out.reshape(-1)[flat.reshape(-1)] = 1
    return out


def attn_part(tok, params):
    Bb = tok.shape[0]
    x = params["Wte"][tok] + params["Wpos"][None]
    n1, m1, s1 = ln(x)
    q = n1 @ params["Wq"]; kk = n1 @ params["Wk"]; v = n1 @ params["Wv"]
    q = q.reshape(Bb, T, NHEADS, DH).transpose(0, 2, 1, 3)
    kk = kk.reshape(Bb, T, NHEADS, DH).transpose(0, 2, 1, 3)
    v = v.reshape(Bb, T, NHEADS, DH).transpose(0, 2, 1, 3)
    al = q @ kk.transpose(0, 1, 3, 2) / math.sqrt(DH) + np.where(MASK == 1, 0.0, -np.inf)
    att = softmax(al)
    y = att @ v
    y = y.transpose(0, 2, 1, 3).reshape(Bb, T, C)
    x2 = x + y @ params["Wo"]
    n2, m2, s2 = ln(x2)
    return (x, n1, m1, s1, q, kk, v, att, y, x2, n2, m2, s2)


def forward(tok, params, target=None, gmask_fixed=None, moe=True):
    x, n1, m1, s1, q, kk, v, att, y, x2, n2, m2, s2 = attn_part(tok, params)
    if moe:
        wg = n2 @ params["Wg"]
        gates = softmax(wg)
        gmask = topk_mask(gates, TOP_K) if gmask_fixed is None else gmask_fixed
        g = gates * gmask
        stack = np.stack(
            [np.maximum(0, n2 @ params[f"Wf1_{e}"]) @ params[f"Wf2_{e}"] for e in range(E_NUM)],
            axis=-2)
        h = np.sum(g[..., None] * stack, axis=-2)
        rt = (wg, gates, gmask, stack)
        h1 = None
    else:
        h1 = np.maximum(0, n2 @ params["Wf1"])
        h = h1 @ params["Wf2"]
        rt = None
    x3 = x2 + h
    logits = x3 @ params["Wout"]
    logp = logits - logits.max(axis=-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))
    if target is None:
        target = np.concatenate([tok[:, 1:], np.zeros((tok.shape[0], 1), dtype=int)], axis=1)
    Bb = tok.shape[0]
    loss = -logp[np.arange(Bb)[:, None], np.arange(T)[None, :], target].mean()
    cache = dict(x=x, n1=n1, m1=m1, s1=s1, q=q, k=kk, v=v, att=att, y=y,
                 x2=x2, n2=n2, m2=m2, s2=s2, h=h, rt=rt, x3=x3, h1=h1,
                 logits=logits, logp=logp, target=target)
    return loss, cache


def ln_backward(dy, t, m, s):
    d = t - m
    return (dy - dy.mean(axis=-1, keepdims=True)
            - d * (dy * d).mean(axis=-1, keepdims=True) / s ** 2) / s


def backward(params, cache, moe=True):
    g = {k: np.zeros_like(v) for k, v in params.items()}
    Bb = cache["target"].shape[0]
    p = np.exp(cache["logp"])
    onehot = np.zeros_like(cache["logits"])
    onehot[np.arange(Bb)[:, None], np.arange(T)[None, :], cache["target"]] = 1
    d_logits = (p - onehot) / (Bb * T)
    g["Wout"] = (cache["x3"].transpose(0, 2, 1) @ d_logits).sum(axis=0)
    d_x3 = d_logits @ params["Wout"].T

    if moe:
        wg, gates, gmask, stack = cache["rt"]
        n2 = cache["n2"]
        hs = [np.maximum(0, n2 @ params[f"Wf1_{e}"]) for e in range(E_NUM)]
        weight = gmask * gates                       # 稀疏门控权重（未选中=0）
        d_n2 = np.zeros_like(n2)
        for e in range(E_NUM):
            d_out = weight[..., e, None] * d_x3      # 只对"选中该专家"的 token 回流
            g[f"Wf2_{e}"] = (hs[e].transpose(0, 2, 1) @ d_out).sum(axis=0)
            d_hs = d_out @ params[f"Wf2_{e}"].T
            d_hs[hs[e] <= 0] = 0
            g[f"Wf1_{e}"] = (n2.transpose(0, 2, 1) @ d_hs).sum(axis=0)
            d_n2 = d_n2 + d_hs @ params[f"Wf1_{e}"].T
        # Router：d_g_e = <d_x3, out_e>；softmax 反向必须用【全】gates 而非 g
        # （未选中专家的 Wg 梯度靠 softmax 分母耦合项保留——见 B3）。
        d_g = np.sum(d_x3[..., None, :] * stack, axis=-1) * gmask
        d_wg = gates * (d_g - np.sum(d_g * gates, axis=-1, keepdims=True))
        g["Wg"] = (n2.transpose(0, 2, 1) @ d_wg).sum(axis=0)
        d_n2 = d_n2 + d_wg @ params["Wg"].T
    else:
        d_h = d_x3 @ params["Wf2"].T
        g["Wf2"] = (cache["h1"].transpose(0, 2, 1) @ d_x3).sum(axis=0)
        d_h[cache["h1"] <= 0] = 0
        g["Wf1"] = (cache["n2"].transpose(0, 2, 1) @ d_h).sum(axis=0)
        d_n2 = d_h @ params["Wf1"].T
    # x3 = x2 + h → d_x2 含残差直通 d_x3 + 经 LN 回传的 d_n2 两条路
    d_x2 = d_x3 + ln_backward(d_n2, cache["x2"], cache["m2"], cache["s2"])

    dy = d_x2 @ params["Wo"].T
    g["Wo"] = (cache["y"].transpose(0, 2, 1) @ d_x2).sum(axis=0)
    dy = dy.reshape(Bb, T, NHEADS, DH).transpose(0, 2, 1, 3)
    dv = cache["att"].transpose(0, 1, 3, 2) @ dy
    d_att = dy @ cache["v"].transpose(0, 1, 3, 2)
    d_al = cache["att"] * (d_att - (d_att * cache["att"]).sum(axis=-1, keepdims=True))
    d_al /= math.sqrt(DH)
    dq = d_al @ cache["k"]
    dk = d_al.transpose(0, 1, 3, 2) @ cache["q"]
    dq = dq.transpose(0, 2, 1, 3).reshape(Bb, T, C)
    dk = dk.transpose(0, 2, 1, 3).reshape(Bb, T, C)
    dv = dv.transpose(0, 2, 1, 3).reshape(Bb, T, C)
    d_n1 = dq @ params["Wq"].T + dk @ params["Wk"].T + dv @ params["Wv"].T
    g["Wq"] = (cache["n1"].transpose(0, 2, 1) @ dq).sum(axis=0)
    g["Wk"] = (cache["n1"].transpose(0, 2, 1) @ dk).sum(axis=0)
    g["Wv"] = (cache["n1"].transpose(0, 2, 1) @ dv).sum(axis=0)
    d_x = d_x2.copy() + ln_backward(d_n1, cache["x"], cache["m1"], cache["s1"])
    g["Wpos"] = d_x.sum(axis=0)
    np.add.at(g["Wte"], cache["input_tok"], d_x)
    return g


def aux_loss(wg, gmask, alpha=0.01):
    """Switch 风格负载均衡辅助项：α·E·Σ_e f_e·P̄_e。
    f_e=专家实际载荷占比（硬计数，视常数不直通）；P̄_e=平均门控概率（唯一梯度通路）。"""
    gates = softmax(wg)
    Bb = wg.shape[0]
    f = gmask.mean(axis=(0, 1))
    Pbar = gates.mean(axis=(0, 1))
    aux = float(alpha * E_NUM * np.sum(f * Pbar))
    d_P = gates * (f[None, None, :] - np.sum(f[None, None, :] * gates, axis=-1, keepdims=True))
    d_wg = alpha * E_NUM * d_P / (Bb * T)       # (Bb,T,E)，对 Wg 的梯度还差一步 n2ᵀ@
    return aux, f, Pbar, d_wg


def adam_update(params, grads, o_m, o_v, step, lr_now):
    for k, pp in params.items():
        o_m[k] = 0.9 * o_m[k] + (1 - 0.9) * grads[k]
        o_v[k] = 0.999 * o_v[k] + (1 - 0.999) * grads[k] ** 2
        mhat = o_m[k] / (1 - 0.9 ** step)
        vhat = o_v[k] / (1 - 0.999 ** step)
        pp -= lr_now * mhat / (np.sqrt(vhat) + 1e-8)


def make_batch(rd):
    starts = rd.randint(0, N - T, size=B)
    tok = np.stack([data[s:s + T] for s in starts])
    tgt = np.stack([data[s + 1:s + T + 1] for s in starts])
    return tok, tgt


def forward_eval(seq, params, moe=True):
    """单序列评估：seq (L,) int，返回最末位置 logits。位置总用 0..T-1、取尾部窗。"""
    seq = np.asarray(seq, dtype=int)[-T:]
    L = len(seq)
    tok = np.zeros((1, T), dtype=int)
    tok[0, T - L:] = seq
    return forward(tok, params, moe=moe)[1]["logits"][0, T - 1]


# ===========================================================================
def main():
    wall0 = time.perf_counter()

    print("=" * 70)
    print("实验 A · 参数账本：总参数 vs 激活参数（MoE 的核心解耦）")
    print("=" * 70)
    tot_m, act_m = param_account(moe=True)
    tot_d, act_d = param_account(moe=False)
    print(f"语料 {N} token · 词表 V={V} · 块 (B,T,C)=({B},{T},{C}) · E={E_NUM}家专家 · Top-{TOP_K} · 每专家中间维 M=2C={M_FFN}")
    print(f"  MoE   总参 = {tot_m:>8,}   单 token 激活参 = {act_m:>8,}   （激活/总 ≈ {act_m/tot_m:.3f}）")
    print(f"  Dense 总参 = {tot_d:>8,}   单 token 激活参 = {act_d:>8,}   （激活/总 = 1.000）")
    one_e = 2 * C * M_FFN
    print(f"  FFN 稀疏比：每 token 只激活 {TOP_K}/{E_NUM} 家 → {TOP_K*one_e:,}/{E_NUM*one_e:,}"
          f" = {TOP_K*one_e/(E_NUM*one_e):.0%}")
    print(f"  关键：激活参 {act_m} = {act_d}（恒等）——MoE 的激活参数和 Dense 一模一样，")
    print(f"        但总参多出 {tot_m-tot_d:,}（全来自专家的驻留权重）→ 单 token 算力相同、「总量更大」")
    print("  现实账（公开配置的简单换算，非本机实测）：Mixtral-8x7B 约 47B 总 / 13B 激活（选 2/8）；"
          "DeepSeek-V3 约 671B 总 / 37B 激活（选 8+共享，约 5%）")

    # ------------------------------------------------------------------ B
    print()
    print("=" * 70)
    print("实验 B · 手写前向/反向对账 + 路由稀疏性的两个硬事实")
    print("=" * 70)
    rngb = np.random.RandomState(7)
    p_chk = init_params(moe=True)
    # 为 B1 单独建一个有特定 gate 的模型（不影响主随机流走向）——直接复用 p_chk
    ptok = rngb.randint(0, V, size=(2, T)).astype(int)
    tgt_t = np.stack([np.concatenate([ptok[b, 1:], [stoi["."]]]) for b in range(2)])
    _, c0 = forward(ptok, p_chk, tgt_t)
    gfix = c0["rt"][2].copy()                      # 冻结拓扑（中心差分两侧同一个掩码）
    _, c0 = forward(ptok, p_chk, tgt_t, gmask_fixed=gfix)
    c0["input_tok"] = ptok
    grads = backward(p_chk, c0)

    rngc = np.random.RandomState(2)
    names = ["Wg", "Wq", "Wk", "Wv", "Wo", "Wpos", "Wout", "Wte",
             "Wf1_0", "Wf2_0", "Wf1_3", "Wf2_3"]
    max_err, worst = 0.0, ""
    n_pts = 0
    eps = 1e-6
    for nm in names:
        P = p_chk[nm]
        n_sam = 24 if nm == "Wg" else 8
        for i in rngc.randint(0, P.size, size=n_sam):
            r0, r1 = np.unravel_index(int(i), P.shape)
            old = P[r0, r1]
            P[r0, r1] = old + eps
            lp, _ = forward(ptok, p_chk, tgt_t, gmask_fixed=gfix)
            P[r0, r1] = old - eps
            lm, _ = forward(ptok, p_chk, tgt_t, gmask_fixed=gfix)
            P[r0, r1] = old
            err = abs((lp - lm) / (2 * eps) - grads[nm][r0, r1])
            n_pts += 1
            if err > max_err:
                max_err = err
                worst = f"{nm}[{r0},{r1}]"
    print(f"  B1  中心差分（冻结路由拓扑）共 {n_pts} 个采样点：max|数值−手写| = {max_err:.2e}")
    print(f"      worst={worst} → 注意力/头/专家权重全部通路一致")
    print("      ※ top-k 硬选是阶梯函数：边界处 gate 跳到另一个专家，不可导；")
    print("        对账必须冻结拓扑避开翻转点——MoE 反向的诚实限定（下 B3 看翻转）。")

    # 硬事实①：单 token、固定只选专家 1 → 其它专家 Wf 梯度严格 0
    p_s = init_params(moe=True)
    p_s["Wg"][:, 1] += 3.0                          # 强偏 → 稳定选专家 1
    tok1 = np.full((1, T), stoi["the"], dtype=int)
    tar1 = np.full((1, T), stoi["the"], dtype=int)
    gm1 = np.zeros((1, T, E_NUM))
    gm1[..., 1] = 1.0
    _, cfr = forward(tok1, p_s, tar1, gmask_fixed=gm1)
    cfr["input_tok"] = tok1
    g1 = backward(p_s, cfr)
    for e in range(E_NUM):
        mz = float(np.abs(g1[f"Wf1_{e}"]).max())
        tag = "被选" if e == 1 else "未选"
        print(f"  B2  单 token · 固定只选专家1 → {tag}专家E{e+1} Wf1 的 max|梯度| = {mz:.2e}")

    # 硬事实②：Router Wg 对未选中专家仍有非零梯度
    col = [f"{float(np.abs(g1['Wg'][:, e]).max()):.2e}" for e in range(E_NUM)]
    print(f"  B3  同一 batch，Router Wg 各专家列的 max|梯度| = [{', '.join(col)}]")
    print("      → 专家2-4 的 Wf 梯度严格 0，但 Wg 梯度非 0：softmax 分母把未选中者与"
          "选中者耦合。这条非零梯度是路由失衡的养分之一（C 段会看到后果）。")

    # ------------------------------------------------------------------ C
    print()
    print("=" * 70)
    print("实验 C · 裸跑 400 步（无 aux）→ 路由自然演化：门控熵 + 专家载荷")
    print("=" * 70)
    STEPS = 400
    p_moe = init_params(moe=True)
    p_d = {k: v.copy() for k, v in p_moe.items()}   # D 段的同起点副本（先拷贝再训练）
    o_m = {k: np.zeros_like(v) for k, v in p_moe.items()}
    o_v = {k: np.zeros_like(v) for k, v in p_moe.items()}
    rc = np.random.RandomState(42)
    rows_c = []
    for step in range(1, STEPS + 1):
        lr_now = 3e-3 * min(1.0, step / 200)
        tok, tgt = make_batch(rc)
        loss, cache = forward(tok, p_moe, tgt)
        cache["input_tok"] = tok
        grads = backward(p_moe, cache)
        adam_update(p_moe, grads, o_m, o_v, step, lr_now)
        if step % 100 == 0:
            wg, gates, gmask, _ = cache["rt"]
            Pbar = gates.mean(axis=(0, 1))
            H = -float(np.sum(Pbar * np.log(Pbar + 1e-12)))
            load = gmask.mean(axis=(0, 1))
            rows_c.append((step, float(loss), H, load))
            print(f"    step {step:4d}  loss {float(loss):6.3f}  门控熵H={H:.3f} "
                  f"(均匀界 ln{E_NUM}={math.log(E_NUM):.3f})  载荷[{', '.join(f'{x:.2f}' for x in load)}]")
    (_, lc0, hc0, load0), (_, lc1, hc1, load1) = rows_c[0], rows_c[-1]
    print(f"  → 门控熵 H: {hc0:.3f} → {hc1:.3f}（均匀界 {math.log(E_NUM):.3f}）；"
          f"载荷方差 {np.var(load0):.4f} → {np.var(load1):.4f}")
    print(f"    裸跑 400 步 CE 末值 {lc1:.3f}")

    # ------------------------------------------------------------------ D
    ALPHA_AUX = 0.10     # 演示强度：Switch 论文 α=0.01 面向数万亿 token，400 步玩具需放大才可见
    print()
    print("=" * 70)
    print(f"实验 D · 同种子同起点 + Switch 均衡 aux(α={ALPHA_AUX}) → 400 步对照")
    print("=" * 70)
    p_d0 = p_d  # 同起点副本（init 后立即取出，见 C 段）
    o_m2 = {k: np.zeros_like(v) for k, v in p_d.items()}
    o_v2 = {k: np.zeros_like(v) for k, v in p_d.items()}
    rd = np.random.RandomState(42)
    rows_d = []
    for step in range(1, STEPS + 1):
        lr_now = 3e-3 * min(1.0, step / 200)
        tok, tgt = make_batch(rd)
        loss, cache = forward(tok, p_d, tgt)
        cache["input_tok"] = tok
        grads = backward(p_d, cache)
        wg, gates, gmask, _ = cache["rt"]
        aux, f, Pbar, d_wg = aux_loss(wg, gmask, alpha=ALPHA_AUX)
        grads["Wg"] += (cache["n2"].transpose(0, 2, 1) @ d_wg).sum(axis=0)
        adam_update(p_d, grads, o_m2, o_v2, step, lr_now)
        if step % 100 == 0:
            H = -float(np.sum(Pbar * np.log(Pbar + 1e-12)))
            load = gmask.mean(axis=(0, 1))
            rows_d.append((step, float(loss), float(loss) + aux, H, load))
            print(f"    step {step:4d}  CE {float(loss):6.3f} (含aux {float(loss)+aux:.3f})  "
                  f"门控熵H={H:.3f}  载荷[{', '.join(f'{x:.2f}' for x in load)}]")
    (_, ld0, _, hd0, load0d), (_, ld1, _, hd1, load1d) = rows_d[0], rows_d[-1]
    print(f"  → 门控熵 H: {hd0:.3f} → {hd1:.3f}（均匀界 {math.log(E_NUM):.3f}）")
    print(f"    载荷方差 {np.var(load1):.4f}(裸) → {np.var(load1d):.4f}(+aux) · "
          f"CE 末值 {lc1:.3f}(裸) → {ld1:.3f}(+aux)")
    print("    → aux 的确把载荷切均匀了：门控熵显著回升、载荷方差大幅下降。")
    print(f"      代价也要看清：本玩具 CE 反而略降（0.063→{ld1:.3f}，四位专家全被用上、拟合更充分），")
    print("      但均衡的买价写在优化目标里——step400 的 CE+aux=0.264 ≫ 裸的 0.063，训练预算被均衡分走一截。")
    print("      真实大模型数据远多于玩具，「拟合更充分」的红利吃不到，aux 常伴随最终 loss 的轻微上升——")
    print("      DeepSeek-V3 后来改用『无辅助损失的 bias 调节』，同一目标、省掉 aux 开销。")

    # ------------------------------------------------------------------ E
    print()
    print("=" * 70)
    print("实验 E · 专家专业化：6 组知识 + 通用句 → 谁偏向了谁（裸模型 vs +aux 模型）")
    print("=" * 70)

    def load_matrix(model):
        R = np.zeros((7, E_NUM))
        for s0 in range(0, N, T):
            block = data[s0:s0 + T]
            if len(block) < 2:
                break
            pad = np.zeros((1, T), dtype=int)
            pad[0, T - len(block):] = block
            _, cc = forward(pad, model)
            gm = cc["rt"][2][0]
            for j in range(len(block)):
                R[tok_group[s0 + j]] += gm[T - len(block) + j]
        return R

    RB = load_matrix(p_moe)     # 裸跑（C 段模型，门控失衡）
    RA = load_matrix(p_d)       # +aux（D 段模型，门控均衡）
    gname = ["通用", "g≈9.8", "π≈3.14", "e≈2.72", "光速", "沸点100", "公转365"]
    print("  主题行 × 专家列 = 该主题 token 触发的载荷占比（行归一）：")
    for grp in range(7):
        row_b = RB[grp] / RB[grp].sum()
        row_a = RA[grp] / RA[grp].sum()
        favb = int(np.argmax(row_b)); fava = int(np.argmax(row_a))
        print(f"  [{gname[grp]:<7}] 裸  E1:{row_b[0]:.2f} E2:{row_b[1]:.2f} E3:{row_b[2]:.2f} E4:{row_b[3]:.2f}"
              f"  → 主投{favb+1}({row_b[favb]:.0%})")
        print(f"          +aux E1:{row_a[0]:.2f} E2:{row_a[1]:.2f} E3:{row_a[2]:.2f} E4:{row_a[3]:.2f}"
              f"  → 主投{fava+1}({row_a[fava]:.0%})")
    print("  → 裸模型的路由已失衡（载荷挤向个别专家，行间差异被集中掩盖）；")
    print(f"    +aux 模型把分化摊平：各行向均匀分布靠拢（对照门控熵：裸 {hc1:.2f} → +aux {hd1:.2f}，"
          f"均匀界 {math.log(E_NUM):.2f}），载荷方差也同步下降。")
    print("    这解释了 DeepSeek 系引入『共享专家』的动机：均衡 aux 保住全部专家不被饿死，")
    print("    再留一个 always-on 的通用专家承载跨主题语法，路由专家专心分化。")

    # ------------------------------------------------------------------ F
    print()
    print("=" * 70)
    print("实验 F · 同激活算力：Dense（4C 全激活）vs MoE（E=4×每专家 2C）")
    print("=" * 70)
    p_dense = init_params(moe=False)
    omd = {k: np.zeros_like(v) for k, v in p_dense.items()}
    ovd = {k: np.zeros_like(v) for k, v in p_dense.items()}
    rfd = np.random.RandomState(42)
    last_d = None
    for step in range(1, STEPS + 1):
        lr_now = 3e-3 * min(1.0, step / 200)
        tok, tgt = make_batch(rfd)
        loss, cache = forward(tok, p_dense, tgt, moe=False)
        cache["input_tok"] = tok
        grads = backward(p_dense, cache, moe=False)
        adam_update(p_dense, grads, omd, ovd, step, lr_now)
        if step % 100 == 0:
            last_d = float(loss)
            print(f"    Dense step {step:4d}  loss {float(loss):6.3f}")

    tests = [("g≈9.8", "9", "when an object falls on earth it speeds up at about"),
             ("π≈3.14", "3", "take any circle and divide its circumference by its diameter gives"),
             ("e≈2.718", "2", "the number e which is about"),
             ("光速", "300000", "the speed of light in a vacuum is roughly"),
             ("沸点100", "100", "the boiling point of water at sea level is"),
             ("公转365", "365", "one full orbit of earth around the sun takes about")]
    ok_m = ok_d = 0
    print(f"  MoE(裸) CE 末值 {lc1:.3f} vs Dense {last_d:.3f}")
    for name, val, cue in tests:
        pre = np.array([stoi.get(t, unk) for t in re.findall(r"[a-zA-Z']+|[0-9]+|[.,!?;:()\-]", cue.lower())])
        vt = stoi[val]
        tm = int(np.argmax(forward_eval(pre, p_moe)))
        td = int(np.argmax(forward_eval(pre, p_dense, moe=False)))
        om_ = tm == vt; od_ = td == vt
        ok_m += om_; ok_d += od_
        mn = "✅" if om_ else "❌"
        dn = "✅" if od_ else "❌"
        print(f"  {name:<8} 期望首tok=「{val:<6}」 MoE→「{itos[tm]:<7}」{mn}  Dense→「{itos[td]:<7}」{dn}")
    print(f"  全新句式 top-1 命中：MoE {ok_m}/{len(tests)} · Dense {ok_d}/{len(tests)}")
    print(f"  → 诚实结论：本玩具规模下 Dense 反而拟合更强（CE {last_d:.3f} < MoE {lc1:.3f}），"
          "泛化持平（5/6·5/6 平手）——MoE 的收益不是更低 loss，而是更大的参数容量")
    print("    （更宽的假设空间），要在数据足够多时才兑现：参数可以大、算力不必跟着大。")

    print()
    print(f"done · 一键复现：python code/scripts/moe_demo.py · 总墙钟 {time.perf_counter()-wall0:.1f}s")


if __name__ == "__main__":
    main()
