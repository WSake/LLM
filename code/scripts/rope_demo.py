# -*- coding: utf-8 -*-
"""RoPE 旋转位置编码演示 + 2× 外推实验。

配合《02-核心原理/05-位置编码-Sinusoidal-RoPE与窗口扩展.md》使用。四段实验：

  A. 旋转·保范数：RoPE 只旋转不缩放，|R_m x| ≡ |x|（正交性）
  B. 旋转·平移不变：QK 点积只依赖相对距离 (i,j) 与 (i+δ,j+δ) 打分相同；
     对照"绝对正弦相加"编码，同样平移会改变打分——这就是 RoPE 叫"相对"的原因
  C & D. 长上下文之谜：在窗口 [1,32] 内学就近核 s(Δ)≈exp(-Δ/8)（真实 q、k 实现，
     打分全程有界）；直接外推 2× 时窗外冒出 4.6× 的伪峰抢走就近注意；
     位置插值 PI（位置÷2 再打分）把窗外距离缩回训练段，伪峰压到 0.14×

本机纯 CPU 可跑；seed 固定，全部数字一键复现。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.RandomState(0)


def theta_terms(d, base=10000.0):
    """RoPE 每维旋转频率：θ_k = base^(-2k/d)。"""
    return base ** (-2.0 * np.arange(0, d // 2) / d)


def rot_angle(pos, d, base=10000.0):
    """位置 pos 每维的旋转角：angles = pos·θ_k（逐 2 维一个角度）。"""
    return np.outer(pos, theta_terms(d, base))          # (num_pos, d//2)


def rot_cos_sin(pos, d, base=10000.0):
    a = rot_angle(np.atleast_1d(pos), d, base)
    return np.cos(a), np.sin(a)                          # 各 (num_pos, d//2)


def apply_rotary(x, cos, sin):
    """把 x 最后一维两两一组旋转：x'.(2i,2i+1) = R(角_i)·x.(2i,2i+1)。"""
    x = np.asarray(x)
    xe = x[..., 0::2]                                    # 偶数维
    xo = x[..., 1::2]                                    # 奇数维
    out = np.empty_like(x)
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xe * sin + xo * cos
    return out


def qk_dot(q, k):
    """(B,Ta,D) 与 (B,Tb,D) 逐行点积 → (B,Ta,Tb)，再广播回 2D 版。"""
    return q @ k.transpose(0, 2, 1)


def fit_recency_kernel(d=16, L=32, peak=16, sigma=4.0):
    """把目标"只看 2 步/Δ*=peak 的相对距离核"拟合成旋转频率的线性组合。

    s(Δ) = Σ_k [A_k·cos(Δθ_k) + B_k·sin(Δθ_k)]，
    在训练窗口 Δ∈[1,L] 上最小二乘拟合 → 得到可带任意 Δ 的延拓函数。
    返回 (coef, s, ok) —— coef 是 (2, d//2) 的 A/B 组合。
    """
    d2 = d // 2
    th = theta_terms(d)
    delta = np.arange(1, L + 1)
    target = np.exp(-((delta - peak) ** 2) / (2 * sigma**2))   # 高斯"回来找 Δ*"核
    Phi = np.column_stack([np.cos(np.outer(delta, th)),
                           np.sin(np.outer(delta, th))])       # (L, 2d2)
    coef, *_ = np.linalg.lstsq(Phi, target, rcond=None)

    def s(D):
        D = np.atleast_1d(D)
        Ph = np.column_stack([np.cos(np.outer(D, th)),
                              np.sin(np.outer(D, th))])
        return Ph @ coef

    return coef, s, peak, target, delta


def main():
    D = 16
    q = rng.randn(D)
    k = rng.randn(D)

    # ============================================================
    print("=" * 70)
    print("实验 A · RoPE 是纯旋转：保范数、可逆（正交变换）")
    print("=" * 70)
    m = 37
    c, s_ = rot_cos_sin(m, D)
    qr = apply_rotary(q[None, :], c, s_)[0]
    print(f"原向量 |q| = {np.linalg.norm(q):.6f}  旋转后 |q_r| = {np.linalg.norm(qr):.6f}")
    print(f"范数偏差 = {abs(np.linalg.norm(qr) - np.linalg.norm(q)):.2e}")
    # 逆旋转 = 旋转负角
    c_neg, s_neg = rot_cos_sin(-m, D)
    q_back = apply_rotary(qr[None, :], c_neg, s_neg)[0]
    print(f"再旋 -{m}° 回到原向量的最大偏差 = {np.abs(q_back - q).max():.2e}  → 旋转可逆")

    # ============================================================
    print()
    print("=" * 70)
    print("实验 B · 平移不变：QK 打分只依赖相对距离（RoPE 的灵魂）")
    print("=" * 70)
    i, j = 3, 10
    dag = 7                                       # 整体平移
    ci, si = rot_cos_sin(i, D); cj, sj = rot_cos_sin(j, D)
    cip, sip = rot_cos_sin(i + dag, D); cjp, sjp = rot_cos_sin(j + dag, D)
    qa, kb = apply_rotary(q[None], ci, si)[0], apply_rotary(k[None], cj, sj)[0]
    qa2 = apply_rotary(q[None], cip, sip)[0]
    kb2 = apply_rotary(k[None], cjp, sjp)[0]
    rel = np.dot(qa, kb)
    rel_shift = np.dot(qa2, kb2)
    print(f"位置对 ({i},{j}) 的 QK 打分: {rel:.6f}")
    print(f"两位置整体平移 {dag} → ({i}+{dag},{j}+{dag}) 打分: {rel_shift:.6f}")
    print(f"平移前后打分差 = {abs(rel - rel_shift):.2e}  → 只依赖相对距离 (j-i)={j-i}")

    # 对照组：绝对位置相加的 Sinusoidal 编码，平移后打分会变
    def sinusoidal_abs(pos, d):
        k_ = np.arange(d // 2)
        ang = pos / 10000.0 ** (2 * k_ / d)
        pe = np.empty(d)
        pe[0::2], pe[1::2] = np.sin(ang), np.cos(ang)
        return pe

    xa, xb = rng.randn(D), rng.randn(D)
    xa_pe = (xa + sinusoidal_abs(i, D)) @ (xb + sinusoidal_abs(j, D))
    xa_pe2 = (xa + sinusoidal_abs(i + dag, D)) @ (xb + sinusoidal_abs(j + dag, D))
    print(f"[对照·绝对Sinusoidal] 平移前打分 {xa_pe:.6f}，平移{dag}后 {xa_pe2:.6f}，"
          f"差 = {abs(xa_pe - xa_pe2):.4f}  → 能被平移改变 = 绝对位置感")

    # ============================================================
    print()
    print("=" * 70)
    print("实验 C & D · 长上下文之谜：RoPE 打分只在训练窗口内保真")
    print("（训练窗口 [1,32] 内学到'就近注意'核 s(Δ)≈exp(-Δ/8)）")
    print("=" * 70)
    D_, L, sigma = 64, 32, 8.0
    d2 = D_ // 2
    th = theta_terms(D_)

    def rot_matrix(Delta, d, th):
        """每个 Δ 一个分块旋转矩阵 R(Δ)：v 旋转 Δ 步 = R(Δ)·v。"""
        Delta = np.atleast_1d(Delta)
        a = np.outer(Delta, th)                        # (n, d//2)
        c, s_ = np.cos(a), np.sin(a)
        M = np.zeros((len(Delta), d, d))
        for i2 in range(d // 2):
            M[:, 2 * i2, 2 * i2] = c[:, i2]
            M[:, 2 * i2, 2 * i2 + 1] = -s_[:, i2]
            M[:, 2 * i2 + 1, 2 * i2] = s_[:, i2]
            M[:, 2 * i2 + 1, 2 * i2 + 1] = c[:, i2]
        return M

    # 1) 线拟合同核：在窗内把就近核表成 cos/sin 组合
    delta = np.arange(1, L + 1)
    target = np.exp(-delta / sigma) / np.exp(-1.0 / sigma)   # Δ=1 → 1
    Phi = np.column_stack([np.cos(np.outer(delta, th)),
                           np.sin(np.outer(delta, th))])
    coef = np.linalg.lstsq(Phi, target, rcond=None)[0]
    a_i, b_i = coef[:d2], coef[d2:]

    # 2) 用真实 q、k 精确实现这批系数：每个频率 i 令 q=(1,0)、k=(a_i,−b_i)，
    #    因为 qᵀR(Δ)k = Σ (q_e k_e+q_o k_o)cos(Δθ_i) + (q_o k_e−q_e k_o)sin(Δθ_i)，
    #    取 (q_e,q_o)=(1,0) 后 sin 系数就是 −k_o —— 打分真实且全程有界。
    q2 = np.zeros(D_); q2[0::2] = 1.0
    k2 = np.zeros(D_); k2[0::2] = a_i; k2[1::2] = -b_i
    bound = np.linalg.norm(q2) * np.linalg.norm(k2)          # |s| ≤ |q2||k2|

    def s(Deltas):                                           # 真实 RoPE 打分
        M2 = rot_matrix(Deltas, D_, th)
        return (M2 @ k2) @ q2

    s_in = s(delta)
    err_in = np.sqrt(np.mean((s_in - target) ** 2))
    print(f"窗内拟合 RMSE = {err_in:.4f}（真实 q,k 实现，打分 |s|≤{bound:.2f} 全程有界不爆炸）")
    print(f"窗内就近峰 s(Δ=1) = {s_in[0]:.3f}；窗尾 s(Δ=32) = {s_in[-1]:.3f}（应≈exp(-31/8) 贴近 0）")

    # 3) 外推 2×：把打分硬读到窗外
    D_full = np.arange(1, 2 * L + 1)
    s_full = s(D_full)
    far = s_full[L - 1:]
    far_max = far.max()
    far_d = D_full[L - 1 + int(far.argmax())]
    print(f"\n直接外推 2×：窗前峰 {s_full[0]:.3f}@Δ=1；窗外又冒一个峰 {far_max:.3f}@Δ={far_d}")
    print(f"  窗外峰/窗前峰 = {far_max / s_full[0]:.1f}  → 窗口外打分被三角函数形态接管，就近注意失效")

    # 4) 位置插值 PI：位置 ÷2 再打分 = s(Δ/2)；2× 时 Δ/2∈[0.5,32] 全塞回训练窗
    s_pi = s(D_full / 2.0)
    recent_pi = s_pi.max()                                   # 峰应回到 Δ≈2
    far_pi = s_pi[L - 1:].max()
    peak_pi = D_full[int(s_pi.argmax())]
    print(f"\n位置插值 PI：峰回到 Δ={peak_pi}（相邻位置依旧最高分）；窗外峰/窗前峰 = {far_pi / recent_pi:.2f}")
    print(f"  → 窗外/'就近'：直接外推 {far_max / s_full[0]:.1f} → PI {far_pi / recent_pi:.2f}，就近优先级保住")
    print("  原理一条线：RoPE 学的是'相对距离→打分'曲线，只对它见过的距离段保真；")
    print("  外推把打分读进窗外，峰落哪随三角函数形态；PI 把窗外距离缩回训练段再打分，曲线形态照旧成立。")

    print()
    print("done · 一键复现：python code/scripts/rope_demo.py")


if __name__ == "__main__":
    main()
