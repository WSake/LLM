# -*- coding: utf-8 -*-
"""Multi-head Latent Attention（MLA）演示 —— 先压进潜向量，再逐头解压。

配合《02-核心原理/04-MLA-多头潜注意力.md》使用。三段实验：

  A. 低秩是压缩的本质：K/V 的投影 = 下投影(→Dc 潜向量) × 每头上投影。
     沿潜向量 W_D 的 nullspace 扰动 token，潜向量不变 ⇒ K/V 一点不变
     （信息被压进 Dc 维后，与潜向量无关的方向被丢掉）——这是 MLA 的"代价"
  B. 缓存账本：MHA / GQA / MLA 每 token 每层元素数与字节数对比，
     MLA 只存潜向量(+位置相关的一段小尾巴)，不再正比于 KV 头数
  C. DeepSeek-V2 规格账算：kv_lora_rank=512、128 头、head_dim 128、
     rope 段 64 维；MHA vs GQA(8 组) vs MLA 每 token 每层账 + 128k 上下文总账

本机纯 CPU 可跑；seed 固定，全部数字一键复现。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.RandomState(0)


def proj(out_dim, in_dim):
    """按输出维数缩放的随机线性投影（前向方差归一，便于数值对齐）。"""
    return rng.randn(out_dim, in_dim) / np.sqrt(in_dim)


def mla_params(C, Dc, H, dh, dr):
    """MLA 的一组参数：返回逐分量投影矩阵。

    W_D:  (Dc, C)    K/V 下投影，潜向量 c = h W_D^T
    W_UK: (H, dh-dr, Dc) 每头的 Key「无色」段上投影（随便换个说法：不含位置）
    W_UV: (H, dh, Dc)    每头的 Value 上投影
    W_Kr: (dr, C)    位置相关 Key 段的直接投影（每 token 一份，不含位置则能压进潜向量）
    """
    return {
        "W_D": proj(Dc, C),
        "W_UK": rng.randn(H, dh - dr, Dc) / np.sqrt(Dc),
        "W_UV": rng.randn(H, dh, Dc) / np.sqrt(Dc),
        "W_Kr": proj(dr, C),
    }


def mla_forward(h, P):
    """h (B,T,C) → (c, Knope, V, Kr)。Knope/V 是「推理时才展开」的中间量。"""
    c = h @ P["W_D"].T                          # (B,T,Dc)      ← 唯一必缓存的
    Knope = np.einsum("btl,hdl->bhtd", c, P["W_UK"])   # (B,H,T,dh-dr)
    V = np.einsum("btl,hdl->bhtd", c, P["W_UV"])       # (B,H,T,dh)
    Kr = h @ P["W_Kr"].T                        # (B,T,dr)      ← 位置相关，另存
    return c, Knope, V, Kr


def nullspace(W):
    """W (Dc,C) 的 nullspace 一组基：W·x=0 的 x。SVD 托底，rank 取满秩 Dc。"""
    _, s, Vh = np.linalg.svd(W)
    r = (s > 1e-10).sum()
    return Vh[r:].T                          # (C, C-r) 每列是一个零向量


def main():
    # 迷你配置（规则与真实模型同构，数字随手可验）
    B, T, C = 2, 6, 64
    H, dh, dr, Dc = 8, 32, 8, 16
    P = mla_params(C, Dc, H, dh, dr)
    h = rng.randn(B, T, C) / np.sqrt(C)

    # ============================================================
    print("=" * 70)
    print("实验 A · 低秩压缩：K/V 差别只来自潜向量的 Dc 维（代价显形）")
    print("=" * 70)
    c0, Kn0, V0, Kr0 = mla_forward(h, P)
    print(f"shape 链：h (2,6,{C}) → 潜向量 c (2,6,{Dc}) → Knope/V (2,{H},6,{dh-dr}/{dh}) + Kr (2,6,{dr})")
    print(f"  缓存每 token 只存 c({Dc}) + Kr({dr})；Knope/V 反推时才展开")

    # 沿 nullspace 的扰动：W_D·z=0 → c 不变 → K/V 逐位不变
    N = nullspace(P["W_D"])                    # (C, C-Dc)
    z0 = N @ rng.randn(C - Dc)                 # 在 W_D 的 nullspace 里
    z0 = z0 / np.linalg.norm(z0) * 0.5
    h2 = h.copy(); h2[0, 0] += z0              # 只动一个位置
    c2, Kn2, V2, Kr2 = mla_forward(h2, P)
    print(f"\n沿潜向量 nullspace 扰动一个 token（扰动幅度 {np.linalg.norm(z0):.2f}）：")
    print(f"  潜向量 c 最大差 = {np.abs(c2[0, 0] - c0[0, 0]).max():.2e}")
    print(f"  解压出的 K/V 最大差 = {np.abs(Kn2[0, :, 0] - Kn0[0, :, 0]).max():.2e} / "
          f"{np.abs(V2[0, :, 0] - V0[0, :, 0]).max():.2e}")
    print(f"  → 差值不是「小」，是全零：与潜向量无关的方向被压缩机制直接丢弃")

    # 沿非 nullspace（潜向量可感知方向）的扰动 —— 能留下来
    U_, sval, Vh = np.linalg.svd(P["W_D"])             # sval: 16 个奇异值
    r_ = int((sval > 1e-10).sum())
    z1 = Vh[:r_].T @ rng.randn(r_)                     # 行的线性组合 ⇒ W_D z1 ≠ 0
    z1 = z1 / np.linalg.norm(z1) * 0.5
    h3 = h.copy(); h3[0, 0] += z1
    c3, Kn3, V3, _ = mla_forward(h3, P)
    print(f"沿潜向量可感知方向扰动同样的幅度 0.50：")
    print(f"  潜向量 c 最大差 = {np.abs(c3[0, 0] - c0[0, 0]).max():.3f}，"
          f"K/V 相应变化 = {np.abs(Kn3[0, :, 0] - Kn0[0, :, 0]).max():.3f}")
    print(f"  → 结论：MLA 的 K/V 表达能力 ≈ 潜向量维度 Dc={Dc}（低秩瓶颈），")
    print("    能留几种关系就有上限；这是省显存与可能损质量之间的天平。")

    # ============================================================
    print()
    print("=" * 70)
    print("实验 B · 缓存账本：MHA / GQA / MLA 每 token 每层")
    print("=" * 70)
    # 关键公式（03 篇：MHA/GQA 缓存 = 2·N_KV·dh；MLA = 潜向量 + 位置段）
    n_kv = 4
    elems = {"MHA": 2 * H * dh, "GQA(4组)": 2 * n_kv * dh, "MLA": Dc + dr}
    print(f"迷你规格 H={H}·dh={dh}·kv 组 {n_kv}·潜向量 {Dc}·位置段 {dr}：")
    for k_, v_ in elems.items():
        print(f"  {k_:9s}  {v_:4d} 元素/token/层 → {v_*2:5d} 字节 (bf16)")
    mha_b, mla_b = elems["MHA"] * 2, elems["MLA"] * 2
    print(f"\n  MLA vs MHA = {mha_b/mla_b:.1f}×；vs GQA(4组) = {elems['GQA(4组)']*2/mla_b:.1f}×")
    print("  一句话：MHA/GQA 的账正比于「KV 头数 × 头维」，MLA 正比于「潜向量维」——头再多也不涨账")

    # ============================================================
    print()
    print("=" * 70)
    print("实验 C · DeepSeek-V2 规格账算（按公开配置折算，账算非实测）")
    print("=" * 70)
    # DeepSeek-V2：kv_lora_rank=512、n_head=128、head_dim=128、qk_rope=64
    H2, dh2, dr2, Dc2 = 128, 128, 64, 512
    n_kv2 = 8                            # 若同规格走 GQA-8 的对照组
    L2, ctx = 61, 131072                 # 61 层、128k 上下文（V2-236B 配置）
    e = {"MHA(128头)": 2 * H2 * dh2, "GQA(8组)": 2 * n_kv2 * dh2,
         "MLA": Dc2 + dr2}
    print(f"每 token 每层（元素 → bf16 字节）：")
    for k_, v_ in e.items():
        print(f"  {k_:10s} {v_:6d} 元素 → {v_*2:6d} B")
    print(f"128k 上下文 × 61 层：")
    for k_, v_ in e.items():
        giB = v_ * 2 * ctx * L2 / 1024**3
        print(f"  {k_:10s} → {giB:8.1f} GiB")
    mla_g, gqa_g = e["MLA"] * 2 * ctx * L2 / 1024**3, e["GQA(8组)"] * 2 * ctx * L2 / 1024**3
    mha_g = e["MHA(128头)"] * 2 * ctx * L2 / 1024**3
    print(f"\n  128k 长上下文：MLA {mla_g:.1f} GiB vs GQA-8 {gqa_g:.1f} GiB vs MHA {mha_g:.1f} GiB")
    print(f"  → 同规格下 MLA 比 MHA 省 {mha_g/mla_g:.0f}×、比 GQA-8 省 {gqa_g/mla_g:.1f}×；")
    print("    且 KV 账不再依赖头数——换更大的模型、更多头，MLA 这格基本不动")
    print("  * 若实现把位置段逐头缓存（dr×H 而非 dr），账会变大到 ", end="")
    alt = (Dc2 + dr2 * H2) * 2 * ctx * L2 / 1024**3
    print(f"{alt:.1f} GiB —— 这就是「同一个机制、账差 15×」：读模型卡时要看清实现口径")

    print()
    print("done · 一键复现：python code/scripts/mla_demo.py")


if __name__ == "__main__":
    main()
