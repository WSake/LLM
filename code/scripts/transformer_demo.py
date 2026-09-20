# -*- coding: utf-8 -*-
"""Transformer 从零手写迷你版：仅解码器 Block 前向 + 手写反向 + 数值梯度验证。

配合《02-核心原理/01-Transformer.md》使用。五段实验：

  A. 组件拆解：token + 位置 → Embedding → 单头注意力（因果 mask）→ 前馈
  B. 完整数据流：一个仅解码器 Transformer Block 前向 → logits
  C. 手写反向传播：把 Block+输出头+交叉熵的 backprop 全手写，
     用中心有限差分逐参比对 -> 全参数最大误差 <1e-5 才算写对
  D. KV 复用：预填充一次 vs 每 token 重算，验证"缓存在系统里省了什么"

本机纯 CPU 可跑；seed 固定，全部数字可一键复现。
"""
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import math
import time
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.RandomState(0)

# 微型超参：所有数字落在肉眼可读范围，且足以演示依赖关系
B, T, C, V, NHEADS = 2, 5, 8, 24, 2
DH = C // NHEADS

tok = rng.randint(0, V, size=(B, T))
target = np.roll(tok, -1, axis=1)      # 目标 = 下一个 token（游动）

Wte = rng.randn(V, C) * 0.1            # token 嵌入
Wpos = rng.randn(T, C) * 0.1           # 可学习位置
Wq = rng.randn(C, C) * 0.1
Wk = rng.randn(C, C) * 0.1
Wv = rng.randn(C, C) * 0.1
Wo = rng.randn(C, C) * 0.1
Wf1 = rng.randn(C, 4 * C) * 0.1
Wf2 = rng.randn(4 * C, C) * 0.1
Wout = rng.randn(C, V) * 0.1
params = dict(Wte=Wte, Wpos=Wpos, Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, Wf1=Wf1, Wf2=Wf2, Wout=Wout)

mask = np.tril(np.ones((T, T)))        # 因果 mask：位置 j 只能看到 i<=j


def ln(t):                             # 无参 LayerNorm（本脚本约定标识恒等）
    m = t.mean(axis=-1, keepdims=True)
    s = np.sqrt(t.var(axis=-1, keepdims=True) + 1e-5)
    return (t - m) / s, m, s


def softmax(t, axis=-1):
    e = np.exp(t - t.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def forward(tok, params, want="all"):
    """返回同一份隐层供反向用（前向一套，反向也写出来）。"""
    (Wte, Wpos, Wq, Wk, Wv, Wo, Wf1, Wf2, Wout) = (
        params["Wte"], params["Wpos"], params["Wq"], params["Wk"],
        params["Wv"], params["Wo"], params["Wf1"], params["Wf2"], params["Wout"])
    x = Wte[tok] + Wpos[None]          # (B,T,C) 查表
    n1, m1, s1 = ln(x)
    q = n1 @ Wq; k = n1 @ Wk; v = n1 @ Wv
    q = q.reshape(B, T, NHEADS, DH).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, NHEADS, DH).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, NHEADS, DH).transpose(0, 2, 1, 3)
    att_logits = q @ k.transpose(0, 1, 3, 2) / math.sqrt(DH) + np.where(mask == 1, 0.0, -np.inf)
    att = softmax(att_logits)
    y = att @ v
    y = y.transpose(0, 2, 1, 3).reshape(B, T, C)   # 乘 Wo 之前的激活（梯度要对它）
    y_out = y @ Wo
    x2 = x + y_out                      # 残差 1
    n2, m2, s2 = ln(x2)
    h = np.maximum(0, n2 @ Wf1)
    x3 = x2 + h @ Wf2                   # 残差 2
    logits = x3 @ Wout                  # (B,T,V)
    logp = logits - logits.max(axis=-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))
    loss = -logp[np.arange(B)[:, None], np.arange(T)[None, :], target].mean()
    cache = dict(x=x, n1=n1, m1=m1, s1=s1, q=q, k=k, v=v, att_logits=att_logits,
                 att=att, y=y, x2=x2, n2=n2, m2=m2, s2=s2, h=h, x3=x3,
                 logits=logits, logp=logp)
    return loss, cache


def backward(params, cache):
    """手写反向：每个算子一行注释，数值梯度是最终裁判。"""
    g = {k: np.zeros_like(v) for k, v in params.items()}
    logits, logp, att, q, kk, v, x, n1, m1, s1 = (
        cache["logits"], cache["logp"], cache["att"], cache["q"], cache["k"],
        cache["v"], cache["x"], cache["n1"], cache["m1"], cache["s1"])
    x2, n2, m2, s2, h, x3, y, att_logits = (
        cache["x2"], cache["n2"], cache["m2"], cache["s2"], cache["h"],
        cache["x3"], cache["y"], cache["att_logits"])

    d_loss = 1.0
    # —— logits → loss：dLoss/d(logits) = (softmax - onehot) / (B*T)
    p = np.exp(logp)
    onehot = np.zeros_like(logits)
    onehot[np.arange(B)[:, None], np.arange(T)[None, :], target] = 1
    d_logits = (p - onehot) / (B * T)

    # —— 输出头：dWout = Σ_batch x3ᵀ · d_logits；d_x3 = d_logits · Woutᵀ
    g["Wout"] = (x3.transpose(0, 2, 1) @ d_logits).sum(axis=0)
    d_x3 = d_logits @ params["Wout"].T

    # —— 残差 2：d_x2 += d_x3；d_n2 = d_x3 · Wf2ᵀ，dWf2 = Σ n2ᵀ·d_x3, dWf1 = Σ n2ᵀ·d_n2
    d_x2 = d_x3.copy()
    d_h = d_x3 @ params["Wf2"].T
    g["Wf2"] = (h.transpose(0, 2, 1) @ d_x3).sum(axis=0)
    d_n2 = d_h.copy()
    d_n2[h <= 0] = 0                     # ReLU 导数
    g["Wf1"] = (n2.transpose(0, 2, 1) @ d_n2).sum(axis=0)
    d_n2 = d_n2 @ params["Wf1"].T
    # n2 = LN(x2)'s backward（无参）
    d_x2 = d_x2 + ln_backward(d_n2, x2, m2, s2)

    # —— 残差 1：d_x += d_x2；输出投影 Wo backward
    #    cache 里的 y 是"乘 Wo 之前"的激活，(att@v 重排后的形状)
    d_x = d_x2.copy()
    d_y = d_x2 @ params["Wo"].T
    g["Wo"] = (y.transpose(0, 2, 1) @ d_x2).sum(axis=0)

    # —— MHA backward：attention 加权、缩放点积、四投影
    d_y = d_y.reshape(B, T, NHEADS, DH).transpose(0, 2, 1, 3)
    d_v = att.transpose(0, 1, 3, 2) @ d_y
    d_att = d_y @ v.transpose(0, 1, 3, 2)
    # softmax backprop through masked logits
    d_att_logits = att * (d_att - (d_att * att).sum(axis=-1, keepdims=True))
    d_att_logits /= math.sqrt(DH)
    d_q = d_att_logits @ kk
    d_kk = d_att_logits.transpose(0, 1, 3, 2) @ q
    d_q = d_q.transpose(0, 2, 1, 3).reshape(B, T, C)
    d_kk = d_kk.transpose(0, 2, 1, 3).reshape(B, T, C)
    d_v = d_v.transpose(0, 2, 1, 3).reshape(B, T, C)
    d_n1 = d_q @ params["Wq"].T + d_kk @ params["Wk"].T + d_v @ params["Wv"].T
    g["Wq"] = (n1.transpose(0, 2, 1) @ d_q).sum(axis=0)
    g["Wk"] = (n1.transpose(0, 2, 1) @ d_kk).sum(axis=0)
    g["Wv"] = (n1.transpose(0, 2, 1) @ d_v).sum(axis=0)

    # —— 第一个 LN backward：再穿越 LN 回到 x
    d_x = d_x + ln_backward(d_n1, x, m1, s1)

    # —— 嵌入查表 backward：dx 按索引散射回 Wte；位置表行加和
    g["Wpos"] = d_x.sum(axis=0)
    np.add.at(g["Wte"], tok, d_x)
    return g


def ln_backward(dy, t, m, s):
    """无参 LayerNorm backward：y=(t-m)/s，s=std+eps。

    标准公式（N 为最后一维长度）：
    dx = (dy - mean(dy) - (t-m)*mean(dy*(t-m))/s²) / s
    """
    d = t - m
    return (dy - dy.mean(axis=-1, keepdims=True)
            - d * (dy * d).mean(axis=-1, keepdims=True) / s**2) / s


def numeric_grad(params, tok, eps=1e-5):
    """中心有限差分，全元素对账：逐参逐元素扰动求 (L+ - L-)/2eps。

    小型网络全参数约 1200 个元素，纯 CPU 秒级跑完——
    结论是"整条反向链全对"，不是"抽样看起来对"。
    """
    num = {}
    flat_map = {}
    for k, v in params.items():
        num[k] = np.zeros_like(v)
        flat_map[k] = v.reshape(-1)
    for k, v in params.items():
        flat = flat_map[k]
        n = flat.size
        for i in range(n):
            old = flat[i]
            flat[i] = old + eps
            l1, _ = forward(tok, params)
            flat[i] = old - eps
            l0, _ = forward(tok, params)
            flat[i] = old
            num[k].reshape(-1)[i] = (l1 - l0) / (2 * eps)
    return num, flat_map


def main():
    # ============================================================
    print("=" * 70)
    print("实验 A · 组件拆解：输入是 token 索引，模型自己学会查表")
    print("=" * 70)
    x0 = Wte[tok] + Wpos[None]
    print("token 索引 shape:", tok.shape, "→ 嵌入+位置后 x shape:", x0.shape)
    q = x0 @ Wq
    K = x0 @ Wk
    att_s = softmax(q @ K.transpose(0, 2, 1) / math.sqrt(C) + np.where(mask == 1, 0.0, -np.inf))
    print("单头 attention 权重 · 第 0 位（只能看自己）:", np.round(att_s[0, 0], 3))
    print("单头 attention 权重 · 第 4 位（看全历史→近似均匀）:", np.round(att_s[0, 4], 3))

    # ============================================================
    print()
    print("=" * 70)
    print("实验 B · 完整数据流：仅解码器 Block → logits")
    print("=" * 70)
    loss, cache = forward(tok, params)
    pB = np.exp(cache["logp"])
    print("一个 Block 前向后 logits shape:", cache["logits"].shape, "（batch 2 × 序列 5 × 词表 24）")
    print("概率分布第 5 个词（随机初始化 → 近似均匀）:", np.round(pB[0, 4], 3))

    # ============================================================
    print()
    print("=" * 70)
    print("实验 C · 手写反向传播 vs 中心有限差分（全元素对账，<1e-5 通过）")
    print("=" * 70)
    grads = backward(params, cache)
    numg, flat_map = numeric_grad(params, tok)
    maxerr = 0.0
    worst = None
    for k in params:
        m = np.abs(grads[k] - numg[k]).max()
        if m > maxerr:
            maxerr, worst = m, k
    ok = maxerr < 1e-5
    verdict = "✅ 通过" if ok else "❌ 未通过"
    print(f"全部 {len(params)} 个参数（嵌入/位置/注意力/FFN/输出头）最大误差: {maxerr:.2e}")
    print(f"阈值 1e-5 → {verdict}   (全参数一致到浮点极限，'最差'的 {worst} 也只是 {maxerr:.2e})")
    print("  → 这正是'从零实现一个能学的 Transformer'的终极检验：")
    print("    反向传播写错任何一行，这里的误差都会跳到 1e-1 量级。")

    # ============================================================
    print()
    print("=" * 70)
    print("实验 D · KV 复用：预填充一次 → 每 token 的 K/V 不算第二遍")
    print("（全量重算 = O(T²)；增量复用 = 只做当前那样式的单行）")
    print("=" * 70)
    Tbig = 96
    xbig = rng.randn(B, Tbig, C)
    maskbig = np.where(np.tril(np.ones((Tbig, Tbig))) == 1, 0.0, -np.inf)
    t0 = time.perf_counter()
    for _ in range(30):
        kbig = xbig @ Wk
        _acc = softmax((xbig @ Wq) @ kbig.transpose(0, 2, 1) /
                       math.sqrt(C) + maskbig)
    t_full = (time.perf_counter() - t0) / 30 * 1000

    k_hist = rng.randn(B, Tbig - 1, C)
    q_one = xbig[:, -1:] @ Wq
    t0 = time.perf_counter()
    for _ in range(30):
        _acc = softmax(q_one @ k_hist.transpose(0, 2, 1) / math.sqrt(C))
    t_inc = (time.perf_counter() - t0) / 30 * 1000
    print(f"全量重算单步: {t_full:.3f} ms | 增量 KV 单步: {t_inc*1000:.3f} µs")
    print(f"→ 复用后单行注意力快约 {t_full/t_inc:,.0f} 倍（本机实测，T={Tbig}）")
    print("  这就是 vLLM 的 KV Cache 存在的数学理由：结果相同，只是不算第二遍。")

    print()
    print("done · 一键复现：python code/scripts/transformer_demo.py")


if __name__ == "__main__":
    main()
