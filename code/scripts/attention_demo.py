# -*- coding: utf-8 -*-
"""Attention 家族 + 多头变体迷你演示：Self / Cross / Masked + MHA→GQA→MQA。

配合《02-核心原理/02-Attention家族-Self-Cross-Masked.md》与
《02-核心原理/03-多头注意力-MHA-MQA-GQA.md》使用。五段实验：

  A. Self vs Cross：Q 的来源决定"在看什么"——同源交互 vs 跨源搭桥
  B. 三种注意力指纹：Self(全可见) / Masked(因果) / Cross(跨源) 的权重长相
  C. 复杂度：Self 是 O(T²)，Cross 是 O(Ta·Tb)——真实计时验证
  D. 多头变体：同一公式装 MHA / GQA / MQA，shape 广播一目了然
  E. KV Cache 显存账本：MHA vs GQA vs MQA 每 token 每层差多少

本机纯 CPU 可跑；seed 固定，全部数字一键复现。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import math
import time
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.RandomState(0)

# —— 微型超参：两个"来源"模拟文本 token 序列与图像 patch 序列 ——
B = 2
TA, TB = 4, 6              # 文本长度 4 tok / "图像"patch 数 6（例：2×3 网格）
C = 8                      # 隐层维度
V = 24                     # 词表（文本侧使用）
NHEAD, DH = 4, 2           # 查询头数与每头维度（实验 D 用）
SIGMA = 0.1                # 权重初始化尺度

text_tok = rng.randint(0, V, size=(B, TA))     # 文本 token 索引
xt = rng.randn(B, TA, C) * SIGMA               # 文本 token 的向量表示
xp = rng.randn(B, TB, C) * SIGMA               # "图像 patch" 的向量表示

Wq = rng.randn(C, C) * SIGMA
Wk = rng.randn(C, C) * SIGMA
Wv = rng.randn(C, C) * SIGMA
mask_m = np.tril(np.ones((TA, TA)))            # 因果 mask：只看 j<=i


def softmax(t, axis=-1):
    e = np.exp(t - t.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def att_weights(q, k, mask=None):
    """打分 → softmax。q:(B,Ta,C), k:(B,Tb,C) → α:(B,Ta,Tb)。"""
    logits = q @ k.transpose(0, 2, 1) / np.sqrt(k.shape[-1])
    if mask is not None:
        logits = logits + np.where(mask == 1, 0.0, -np.inf)
    return softmax(logits)


def timed(func, n_batch=5, n_rep=30):
    """三次中位数的计时惯例：每次跑 n_rep 次取平均，再取 n_batch 次中位数。"""
    times = []
    for _ in range(n_batch):
        t0 = time.perf_counter()
        for __ in range(n_rep):
            func()
        times.append((time.perf_counter() - t0) / n_rep)
    return sorted(times)[n_batch // 2]


def main():
    print("=" * 70)
    print("实验 A · Self vs Cross：Q 的来源决定'在看什么'")
    print("=" * 70)
    # Self：Q/K/V 全部来自文本自己
    a_self = att_weights(xt @ Wq, xt @ Wk)
    # Cross：Q 来自文本，K/V 来自"图像 patch"
    a_cross = att_weights(xt @ Wq, xp @ Wk)
    print("Self  attention 权重 shape:", a_self.shape, " (batch × 文本长度 × 文本长度)")
    print("Cross attention 权重 shape:", a_cross.shape, " (batch × 文本长度 × 图像patch数)")
    y_cross = a_cross @ (xp @ Wv)
    print("Cross 输出 shape = Q 的长度:", y_cross.shape, "→ 文本 token 的向量现在'装进了图像信息'")
    print("每行 α 之和（softmax，应各为 1.0）:",
          np.round(a_self.sum(axis=-1)[0], 3), "|",
          np.round(a_cross.sum(axis=-1)[0], 3))
    print("  → 一行 α 的'列数'= K 来源的长度；这决定了 Cross 复杂度是 O(文本 × 图像)。")

    # "学习后"的 Cross：单独构造 Q 与某路 K 强对齐的状态（模拟训练后 Wq/Wk 学会对齐）
    q2 = rng.randn(1, TA, C) * 0.6        # 查询侧表示（尺度大方差 → 训练后的"会挑"）
    k2 = rng.randn(1, TB, C) * 0.6        # 提供侧表示
    k2[0, 2] = 2.5 * q2[0, 0]             # patch #2 的 K 与查询 #0 的 Q 强对齐
    a_learned = att_weights(q2, k2)
    row = a_learned[0, 0]
    print("学习后 Cross：查询 #0 与 patch #2 强对齐 → 注意力:", np.round(row, 3),
          f"→ 峰值 {row.max():.3f} 落在 patch {row.argmax()}")
    print("  → 关键在 K：谁提供的 key 更'匹配这张查询'，谁分到的权重就大——注意力是对'找什么'的打分。")

    print()
    print("=" * 70)
    print("实验 B · 三种注意力指纹：同为 softmax，长相因掩码/来源而不同")
    print("=" * 70)
    print("Self·全可见  第 0 行（无掩码→近似均匀）:", np.round(a_self[0, 0], 3))
    a_masked = att_weights(xt @ Wq, xt @ Wk, mask=mask_m)
    print("Masked·因果  第 0 行（只能看自己→ one-hot）:", np.round(a_masked[0, 0], 3))
    print("Masked·因果  最末行（看全历史→近似均匀）:", np.round(a_masked[0, TA - 1], 3))
    print("Cross·跨源   第 0 行（对 6 个 patch 近似均匀）:", np.round(a_cross[0, 0], 3))
    print("  → 掩码把'均匀分布'截断成 one-hot；跨源把注意力带进另一个来源。")

    print()
    print("=" * 70)
    print("实验 C · 复杂度指数（对数-对数拟合）：Self≈O(T²)，Cross≈O(Ta·Tb)")
    print("=" * 70)

    def cmp_exp(func, sizes):
        """在 sizes 上计时后对 log(T)-log(t) 做线性拟合，斜率即复杂度指数。"""
        times = [timed(lambda s=s: func(s)) * 1e3 for s in sizes]
        beta = np.polyfit(np.log(sizes), np.log(times), 1)[0]
        return times, beta

    Tlist = [128, 256, 512, 1024]
    t_self, s_self = cmp_exp(
        lambda T: att_weights(rng.randn(1, T, C) @ Wq, rng.randn(1, T, C) @ Wk), Tlist)
    print("Self  计时（ms）:", dict(zip(Tlist, [f"{t:.2f}" for t in t_self])),
          f"\n  → 拟合复杂度指数 ≈ {s_self:.2f}（理论 O(T²)=2.0）")

    TA_fixed, Tblist = 16, [128, 256, 512, 1024]
    t_cross, s_cross = cmp_exp(
        lambda Tb: att_weights(rng.randn(1, TA_fixed, C) @ Wq,
                               rng.randn(1, Tb, C) @ Wk), Tblist)
    print("Cross 固定 Ta=16 计时（ms）:", dict(zip(Tblist, [f"{t:.2f}" for t in t_cross])),
          f"\n  → 拟合复杂度指数 ≈ {s_cross:.2f}（理论 O(Ta·Tb)=1.0：只随图像长度线性走）")
    print("  → 这就是'读长上下文'接力预留的口子：Self 涨 T 贵，Cross 的一侧可以保持短。")

    print()
    print("=" * 70)
    print("实验 D · 多头变体：MHA / GQA / MQA 是同一个公式的不同'K/V 头数'")
    print("=" * 70)
    Ta = TA
    q = (xt @ Wq).reshape(B, Ta, NHEAD, DH).transpose(0, 2, 1, 3)   # (B,NHEAD,Ta,DH)
    kv_full = (xt @ Wk).reshape(B, Ta, NHEAD, DH).transpose(0, 2, 1, 3)  # (B,NHEAD,Ta,DH)

    def att_group(q, kv, nkv):
        """按组共享：查询头按相邻分组，前 NHEAD//nkv 个头共用 kv0，以此类推。"""
        B_, Nh, Ta_, _ = q.shape
        kpg = Nh // nkv                              # 每组查询头数（kv 的组数）
        a = np.empty((B_, Nh, Ta_, Ta_))
        for h in range(Nh):
            a[:, h] = softmax(q[:, h] @ kv[:, h // kpg].transpose(0, 2, 1) / math.sqrt(DH))
        return a

    def att_bcast(q, kv, nkv):
        """等价视图：把 nkv 路 K/V 广播复制到 NHEAD 份，再照常打分。"""
        kv_b = np.repeat(kv, NHEAD // nkv, axis=1)          # (B,NHEAD,Ta,DH)
        return softmax(q @ kv_b.transpose(0, 1, 3, 2) / math.sqrt(DH))

    for nkv, name in [(NHEAD, "MHA（K/V 头数 = 查询头数）"),
                      (2, "GQA（K/V 头数 = 2）"),
                      (1, "MQA（K/V 头数 = 1）")]:
        kv = kv_full[:, :nkv]                                # (B,nkv,Ta,DH)
        a1, a2 = att_group(q, kv, nkv), att_bcast(q, kv, nkv)
        same = np.abs(a1 - a2).max()
        print(f"{name} | K/V 组数 {tuple(kv.shape)} → 每组被 {NHEAD//nkv} 个查询头共享 | "
              f"两种写法（逐组/广播）α 最大差 {same:.1e}")
    print("  → '共享 K/V'不改变打分：广播复制 NHEAD 份后照常点积，结果逐元素一致（1e-16）。")

    # 实验 E：KV 显存账本
    print()
    print("=" * 70)
    print("实验 E · KV Cache 显存账本：MHA vs GQA vs MQA，每 token 每层差多少")
    print("（fp16 每元素 2 字节；KV 缓存 = 2 份(K/V) × K/V头数 × DH × 2B）")
    print("=" * 70)
    sizes = {name: 2 * nkv * DH * 2 for nkv, name in
             [(NHEAD, "MHA(KV=查询头)"), (2, "GQA(KV=2)"), (1, "MQA(KV=1)")]}
    print("迷你模型（C=8, 查询头=4, DH=2）KV 账本：")
    for name, s in sizes.items():
        print(f"  {name}: {s:>4} 字节/token/层 × {TA} token = {s * TA:>4} 字节/层")
    base = sizes["MHA(KV=查询头)"]
    gqa_r = base / sizes["GQA(KV=2)"]
    mqa_r = base / sizes["MQA(KV=1)"]
    print(f"  → GQA 省 {gqa_r:.1f}×，MQA 省 {mqa_r:.1f}×（这就是'GQA 是当前标配'的核心账）")

    # 真实规格账算（公开 specs，非实测；GB 为 GiB）
    def kv_ledger(layers, n_kv, DH_, ctx, bytes_p=2):
        per_token_layer = 2 * n_kv * DH_ * bytes_p      # 每 token 每层（字节）
        per_token = per_token_layer * layers
        total = per_token * ctx
        return per_token_layer, per_token, total

    # Llama-2-7B 公开规格：32 层、32 查询头、GQA 8 KV 头、hidden 4096(DH=128)、bf16
    ptl, ptk, tot = kv_ledger(32, 8, 128, 8192)
    ptl_m, ptk_m, tot_m = kv_ledger(32, 32, 128, 8192)   # 同为 MHA 时的对照
    print("账算：LLaMA-2-7B 规格（32 层, 32 查询头, GQA 8 KV 头, hidden 4096, bf16）：")
    print(f"  每 token 每层 {ptl/1024:.0f} KB → 每 token {ptk/1024:.0f} KB → "
          f"8k 上下文全 Cache ≈ {tot/1024**3:.2f} GB")
    print(f"  同为 MHA（32 KV 头）→ 8k 全 Cache ≈ {tot_m/1024**3:.2f} GB "
          f"（GQA 把 KV 账砍到 1/{tot_m/tot:.0f}）")

    print()
    print("done · 一键复现：python code/scripts/attention_demo.py")


if __name__ == "__main__":
    main()
