# -*- coding: utf-8 -*-
"""
peft_ft_demo.py
────────────────────────────────────────────────────────────────────────────
04-训练体系 06-全参微调与PEFT-LoRA-QLoRA 的复现脚本（04 目录第 5 个引擎脚本）。

显存不够时，PEFT（LoRA）偷走了哪几坨空间？本篇三段：
  A 账本（7B 账算，非本机实测）：全参微调每参数 12 字节
      （fp16 权重 2 + fp16 梯度 2 + AdamW 一阶矩 fp32 4 + 二阶矩 fp32 4）
      → 7B 约 78 GiB；LoRA 冻结主权重、只对可训练子集记梯度与优化器状态
      → 增量「~0.1GB 级」；QLoRA 再把主权重压 4-bit（NF4）省一大截
      （量化落地为业界做法，非本机实测）。
  B 低秩谱（本机实测）：合成任务里真值更新 ΔW* 是秩-3 目标（d=32）。
      Full 微调收敛后 ΔW 的前 3 个奇异值吃掉 ~99.9% 谱能量、其余 29 个
      几乎为零——「模型自己学出的更新本来就是低秩的」，LoRA 的低秩有
      物理依据。同时量 LoRA r=4 的 ΔW 与真值对齐度、可训练参数降幅。
  C 秩的刻度：r=2（低于自然秩 3）欠参 → 残差大；r=4/8（够）≈ Full——
      「秩踩到任务自然秩就够，不用满秩」。

纯 CPU、固定随机种子，一键可复现。
复现声明：run1==run2==run3 科学数字逐位一致，只有墙钟在跑次间浮动。
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

R = np.random.RandomState(7)
d = 32          # 输入/输出维度
r_true = 3      # 真值更新的自然秩
N = 256         # 样本数
STEPS = 600


# ──────────────────────────────────────────────────────────────────────────
# 0. 任务：把 W0 微调成 Wt = W0 + ΔW*，其中 ΔW* = 秩-3 目标
# ──────────────────────────────────────────────────────────────────────────

def build_task():
    U, _ = np.linalg.qr(R.normal(size=(d, r_true)))     # 列正交
    V, _ = np.linalg.qr(R.normal(size=(d, r_true)))
    W0 = np.eye(d)
    dW_star = U @ V.T                                   # 秩-3 更新
    Wt = W0 + dW_star
    X = R.normal(size=(N, d))
    Y = X @ Wt
    return W0, dW_star, Wt, X, Y


def mse_rel(W, Wt):
    """相对 MSE：||W−Wt||_F / ||Wt||_F"""
    return float(np.linalg.norm(W - Wt) / np.linalg.norm(Wt))


def top3_energy(dW):
    """前 3 个奇异值的谱能量占比（Σσ² 前 3 / 全）"""
    s = np.linalg.svd(dW, compute_uv=False)
    e3 = s[:3] @ s[:3] / (s @ s + 1e-30)
    return 100 * e3, s[3] / (s[0] + 1e-30)   # ppm: σ4/σ1


def cosinef(a, b):
    return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))


# ──────────────────────────────────────────────────────────────────────────
# 1. A 段：显存账本（7B 账算，非本机实测）
# ──────────────────────────────────────────────────────────────────────────

def account_book():
    P = 7e9
    per = {
        "模型权重 fp16": 2,
        "梯度 fp16": 2,
        "AdamW 一阶矩 fp32": 4,
        "AdamW 二阶矩 fp32": 4,
    }
    print("[A] 全参微调显存账本（7B 账算，本机不训 7B；GiB 为二进制、非本机实测）")
    total = 0
    for k, v in per.items():
        gb = P * v / 2**30
        total += gb
        print(f"    {k:20s} {v:2d} 字节/参数 → {gb:6.1f} GiB")
    print(f"    {'─' * 44}")
    print(f"    全参微调合计           12 字节/参数 → {total:6.1f} GiB（不含激活/中间量）")
    # LoRA：主权重冻结，只对可训练子集记梯度与优化器状态
    lr_d, lr_r = 4096, 8                     # 一层主线性 d=4096、LoRA r=8
    train_per_layer = lr_d * lr_r * 2        # A+B 两矩
    n_layers = 32
    total_lora = train_per_layer * n_layers * 4 / 2**30    # B+A 各 dp 各矩 fp32 8B/参数
    print(f"    LoRA（r=8、每层 {train_per_layer:,} 可训练参、{n_layers} 层）"
          f"：梯度+AdamW 状态 ≈ {total_lora:.2f} GiB，主权重冻结仍占 fp16 {P*2/2**30:.0f} GiB")
    print(f"        → 全参 ≈78 GiB vs LoRA ≈13+0.02 GiB；4-bit NF4 量化主权重再压到 "
          f"{(P*0.5)/2**30:.0f} GiB（QLoRA 落地，非本机实测）")
    print()


# ──────────────────────────────────────────────────────────────────────────
# 2. 训练器（全参 vs LoRA，同一损失同一 SGD 框架）
# ──────────────────────────────────────────────────────────────────────────

def train_full(W0, Wt, X, Y):
    """满秩线性微调的闭式解（最小二乘最小范数）＝全参最优基线。"""
    return np.linalg.pinv(X) @ Y


def train_lora(r, W0, Wt, X, Y, lr=0.20):
    A = np.zeros((r, d))                       # 惯例：A 从 0 起步，初始仍是 W0
    B = R.normal(0.0, 0.08, (d, r))
    for _ in range(STEPS):
        W = W0 + B @ A
        gW = X.T @ (X @ W - Y) / N
        gB = gW @ A.T                          # ∂L/∂B ∈ (d, r)
        gA = B.T @ gW                          # ∂L/∂A ∈ (r, d)
        B = B - lr * gB
        A = A - lr * gA
    return W0 + B @ A


# ──────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    W0, dW_star, Wt, X, Y = build_task()

    print("=" * 66)
    print("peft_ft_demo：全参账本 → 低秩谱 → 秩刻度（PEFT / LoRA 的显存账 + 更新谱）")
    print("=" * 66)

    # 段间 0：任务级信息
    print(f"[0] 任务：d=32 线性微调，真值更新 ΔW* 拥有自然秩 {r_true}（谱 = {[1.0]*r_true}），"
          f"样本 {N} 条；目标 Wt=W0+ΔW*")
    print(f"    Full 可训练参 1024 · LoRA r=4 可训练参 {4*d*2}（{4*d*2/1024:.1%}）"
          f" · LoRA r=8 可训练参 {8*d*2}")
    print()

    # A 段账本（7B 口径）
    account_book()

    # B 段：Full 学出的更新谱（低秩天然）
    Wf = train_full(W0, Wt, X, Y)
    dWf = Wf - W0
    e3f, s4f = top3_energy(dWf)
    print("[B] Full 微调收敛后 ΔW 的奇异值谱（它有权学满秩，实际只学到 3 维）")
    print(f"    前 3 奇异值谱能量占比 {e3f:.1f}% · σ₄/σ₁ = {s4f:.2e}")
    print(f"    → 模型自己学出的更新是低秩的（谱偏置）：LoRA 把更新限制在低秩子空间，"
          f"锁的就是这个谱")
    print()

    # LoRA r=4 对齐度
    Wl4 = train_lora(4, W0, Wt, X, Y)
    dWl = Wl4 - W0
    print(f"    LoRA r=4：相对 MSE {mse_rel(Wl4, Wt):.1e} · Full {mse_rel(Wf, Wt):.1e}"
          f"（同进机器精度）；ΔW 与真值对齐 cos(ΔW_lora, ΔW*) = {cosinef(dWl, dW_star):.4f}"
          f"（Full {cosinef(dWf, dW_star):.4f}）")
    print(f"    可训练参数 1024 → 256（-75.0%），且只记这些参数的梯度/优化器状态")
    print()

    # C 段：秩的刻度（r=2 欠参 < 自然秩；r=4/8 够）
    print("[C] 秩的刻度（自然秩 3；Full=最小二乘闭式解＝最优基线；LoRA 用 GD 600 步）")
    dWf_e = mse_rel(Wf, Wt)
    for r in (2, 4, 8):
        Wlr = train_lora(r, W0, Wt, X, Y)
        m = mse_rel(Wlr, Wt)
        note = "欠参（秩不够跨出三维，残差～1e-1）" if m > 1e-6 else "够用（≈Full，进机器精度）"
        print(f"    r={r:<3d} | 相对 MSE {m:12.2e} |  {note}")
    print(f"    Full  | 相对 MSE {dWf_e:12.2e} |  满秩（额外 1024 维参数零性价比）")
    print()
    print(f"墙钟 {1000*(time.time()-t0):.0f}ms（科学数字逐位一致，只有墙钟浮动）")


if __name__ == "__main__":
    main()
