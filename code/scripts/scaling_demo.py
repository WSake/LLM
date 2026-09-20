# -*- coding: utf-8 -*-
"""Scaling Law 与数据墙：把"loss 随参数/数据/算力幂律下降"在玩具里量一遍。

承接《12-预训练》同款 1-block 引擎，从一张谱上改尺寸与数据量（对标 Kaplan 2020
的 N/D/C 幂律 + Chinchilla 2022 的 compute-optimal 配比），四段实测：
  A  数据扫描：固定尺寸 C=96，唯一训练 token 数 D 从 57K 涨到 917K，
      eval loss（对真实转移分布的条件交叉熵）按 log-log 幂律下降 → 斜率 b≈?
      = 知识地图实践项的"画自己小模型的 loss vs 数据量曲线"
  B  参数扫描：固定 D，尺寸 C=32→128（参数 14K→205K），log-log 幂律下降 → 斜率 a≈?
  C  Chinchilla：5×5 网格 + early stopping，联合拟合 L(N,D)=E+A/N^α+B/D^β；
      等算力段用两条腿——同预算网格实测对照（预算往数据斜）+ 拟合面 argmin
     （玩具窗口里最优永远钉在 N 下界=最小参数、预算全给数据，因为数据未饱和；
       配比 D*/N* 的『常数』要等数据端饱和，即数据墙）
      并对照生成器熵 E_gen（1.5802）看拟合 E 的漂移坑（量程窄 → E 与幂次纠缠不可辨识）
  D  数据墙：有限语料池（P0=8K/16K token）循环 k 轮 vs 无限新样本，
      池世界 eval 弯向"池统计分辨率"决定的墙位（可解析预测、随池规模变动），
      越过墙后预算翻倍最佳 eval 一价不动——自然数据见底时"训练时间白给"

数据：阶 2 马尔可夫生成器（V=16 符号，转移 Dirichlet(0.2) 高峰态），生成器熵
可直接解析计算 ≈1.58 nats（< lnV=2.77）——E 地板有 oracle，不用猜。

评测口径：eval loss = 模型对【真实转移分布】的期望条件 CE（分布级、无采样噪声），
喂 <a,b> 读下一符号 logits 再按 π(a,b) 加权。模型拟合的是 P(c|a,b)。

科学数字逐位一致：seed 固定 + 单线程 BLAS + 确定性生成器/拟合（least_squares
无 RNG），run1==run2 逐位一致，只有墙钟浮动。
真实大模型数字（Kaplan/Chinchilla/Llama 等）在正文标 非本机实测，只做趋势参照。
一键复现：python code/scripts/scaling_demo.py
"""
import os
# 固定单线程 BLAS（无条件强锁）：多线程下 matmul 浮点低位逐进程浮动。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import time
import math
sys.path.insert(0, r"F:\00AI创业\01LLM\LLM\code\scripts")
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
import grpo_rlvr_demo as g

# ---------------------------------------------------------------------------
# 生成器：阶 2 马尔可夫，V=16 符号，转移 Dirichlet(0.2) 高峰态
# ---------------------------------------------------------------------------
V = 16
LEN = 28          # 每行真实 token 数（预测位 LEN-1）
T = 32
B = 32
PER = LEN - 1     # 每行被监督的预测位 = 27
gen = np.random.RandomState(77)
P = np.zeros((V, V, V))
for a in range(V):
    for b in range(V):
        P[a, b] = gen.dirichlet(0.2 * np.ones(V))

# 成对马尔可夫链 (a,b)->(b,c)，power iteration 求平稳分布 π 与过程熵 H
MM = np.zeros((V * V, V * V))
for a in range(V):
    for b in range(V):
        i = a * V + b
        for c in range(V):
            MM[b * V + c, i] = P[a, b, c]
pi = np.full(V * V, 1.0 / (V * V))
for _ in range(400):
    pi = MM @ pi
    pi /= pi.sum()
H = float(sum(pi[a * V + b] * (-(d * np.log(d + 1e-30)).sum())
              for a in range(V) for b in range(V) for d in [P[a, b]]))

# ---------------------------------------------------------------------------
# 引擎：1-block 仅解码器（12 章同款），符号 s_k -> id k，VOC=V
# ---------------------------------------------------------------------------
VOC = V


def init_c(Cc, seed):
    st = np.random.RandomState(seed)
    return {k: st.randn(r0, r1) * 0.06 for k, (r0, r1) in {
        "Wte": (VOC, Cc), "Wpos": (T, Cc), "Wq": (Cc, Cc), "Wk": (Cc, Cc),
        "Wv": (Cc, Cc), "Wo": (Cc, Cc), "Wf1": (Cc, 4 * Cc), "Wf2": (4 * Cc, Cc),
        "Wout": (Cc, VOC)}.items()}


def n_params(Cc):
    return 2 * VOC * Cc + T * Cc + 12 * Cc * Cc


g.B, g.T = B, T
g.NHEADS = 4


def set_c(Cc):
    g.C = Cc
    g.DH = Cc // 4
    g.mask = np.tril(np.ones((T, T)))


def draw(rng, pvec):
    return int(np.searchsorted(np.cumsum(pvec), rng.rand()))


def sample_walk(rng, n=LEN):
    i = int(np.searchsorted(np.cumsum(pi), rng.rand()))
    a, b = i // V, i % V
    seq = [a, b]
    for _ in range(n - 2):
        c = draw(rng, P[a, b])
        seq.append(c)
        a, b = b, c
    return np.array(seq, dtype=int)


lm = np.zeros((B, T), float)
lm[:, :PER] = 1.0


def make_batch(rng):
    tok_row = np.zeros((B, T), dtype=int)
    for j in range(B):
        r = sample_walk(rng)
        tok_row[j, :r.shape[0]] = r
    return tok_row


def eval_ce(p):
    """模型对真实转移分布的期望 CE：喂 (a,b) 读下一符号 logp，再按 π 加权。"""
    set_c(p["Wte"].shape[1])
    rows = np.full((V * V, T), 0, dtype=int)
    for a in range(V):
        for b in range(V):
            rows[a * V + b, 0] = a
            rows[a * V + b, 1] = b
    res = g.forward(rows, p)[1]["logp"][:, 1, :]
    tot = 0.0
    for a in range(V):
        for b in range(V):
            d = P[a, b]
            ce = -(np.log(np.maximum(np.exp(res[a * V + b]), 1e-30)) * d).sum()
            tot += pi[a * V + b] * ce
    return float(tot)


def train_cell(Cc, tokens, seed_tr=202, lr=3e-3):
    """训练 Cc 尺寸模型至 tokens 个监督位（early stopping 盯 eval CE 保最优）。
    返回 (best_e, tokens_used, curve) —— best_e 在 tokens_used 时刻取得。"""
    set_c(Cc)
    p = init_c(Cc, seed=1234)
    o = g.init_adam(p)
    tr = np.random.RandomState(seed_tr)
    steps = int(math.ceil(tokens / (B * PER)))
    interval = max(8, min(32, steps // 12))
    best_e = 1e9
    best_tok = 0
    curve = []
    for s in range(1, steps + 1):
        tok_row = make_batch(tr)
        tgt = np.concatenate([tok_row[:, 1:], np.zeros((B, 1), dtype=int)], axis=1)
        lr_now = lr * min(1.0, s / 200)
        loss, cache = g.forward(tok_row, p, tgt)
        gr = g.backward(p, cache, loss_mask=lm)
        g.adam_update(p, o, gr, lr_now)
        if s % interval == 0 or s == steps:
            e = eval_ce(p)
            curve.append((s, e))
            if e < best_e:
                best_e = e
                best_tok = s * B * PER
    return best_e, best_tok, curve


def main():
    t_start = time.perf_counter()
    print("=" * 78)
    print("Scaling Law 与数据墙（ch21）· 1-block 引擎 · 阶2马尔可夫数据")
    print("=" * 78)
    CS = [32, 48, 64, 96, 128]
    ND = [57 * 1024, 114 * 1024, 229 * 1024, 458 * 1024, 917 * 1024]
    print(f"符号 V={V} · 平稳熵 H={H:.4f} nats（lnV={math.log(V):.4f}）"
          f" · 每行 {PER} 个监督位 · 每步 {B*PER:,} tokens")
    print(f"尺寸 C∈{CS} → 参数 {n_params(CS[0]):,}~{n_params(CS[-1]):,}"
          f" · 数据 D∈{[int(x/1024) for x in ND]}K tokens（early stopping 护防过训练）")
    print("评测口径：eval loss = 模型对真实转移分布的期望条件 CE（分布级、无采样噪声）。")

    # ---- 5×5 网格
    print(); print("=" * 78)
    print("网格 · 5 尺寸 × 5 数据量（每格 early stopping 后的最优 eval CE）")
    print("=" * 78)
    grid = {}
    for Cc in CS:
        set_c(Cc)
        row = []
        for D in ND:
            t0 = time.perf_counter()
            best_e, tok_used, _ = train_cell(Cc, D)
            grid[(Cc, D)] = (best_e, tok_used)
            row.append(best_e)
            print(f"C={Cc:>3} N={n_params(Cc):>7,} D={D//1024:>4}K "
                  f"→ best CE {best_e:.4f} @ {tok_used/1024:.0f}K tokens "
                  f"· {time.perf_counter()-t0:5.1f}s")
        print()

    # 汇总矩阵（C 行 × D 列）
    hdrC = "        ".join(f"C{Cc}" for Cc in CS)
    print("汇总：evalCE(C,D)，行=D(列=尺寸)：")
    print(f"        D\\C  {hdrC}")
    for D in ND:
        print(f"{D//1024:>5}K   "
              + "   ".join(f"{grid[(Cc, D)][0]:.4f}" for Cc in CS))

    # ---- A 参数扫描（固定 D0=229K，Kaplan 第一支）：取网格里 C 切片，不重训
    D0 = 229 * 1024
    a_pairs = [(Cc, grid[(Cc, D0)][1], grid[(Cc, D0)][0]) for Cc in CS]
    for Cc, tok, e in a_pairs:
        print(f"  [A] C={Cc:>3} N={n_params(Cc):>7,} D≈{D0//1024}K "
              f"→ evalCE={e:.4f} @{tok//1024}K")
    A_N = np.array([n_params(Cc) for Cc in CS], dtype=float)
    A_E = np.array([e for _, _, e in a_pairs])
    ka_k, ka_b = np.polyfit(np.log(A_N), np.log(A_E), 1)
    print(f"  [A] N 幂律：log L = {ka_b:.3f} + ({ka_k:.3f})·log N  →  "
          f"L ∝ N^{ka_k:.3f}")

    # ---- B 数据扫描（固定 C0=96，Kaplan 第二支 + 知识地图实践项）：
    #      就是"画自己小模型的 loss vs 数据量曲线"，直接取网格 C0 那行
    C0 = 96
    b_pairs = [(D, grid[(C0, D)][1], grid[(C0, D)][0]) for D in ND]
    for D, tok, e in b_pairs:
        print(f"  [B] C=96 D={D//1024:>4}K(实际 {tok//1024:>4}K) → evalCE={e:.4f}")
    B_D = np.array([max(t, 1) for _, t, _ in b_pairs], dtype=float)
    B_E = np.array([e for _, _, e in b_pairs])
    kb_k, kb_b = np.polyfit(np.log(B_D), np.log(B_E), 1)
    print(f"  [B] D 幂律：log L = {kb_b:.3f} + ({kb_k:.3f})·log D →  "
          f"L ∝ D^{kb_k:.3f}")

    # ---- C Hoffmann 联合拟合 + 等算力（测量 + 拟合两条腿）
    print(); print("=" * 78)
    print("C · Chinchilla：联合拟合 L(N,D)=E+A/N^α+B/D^β；等算力=预算往哪放")
    print("=" * 78)
    all_N = np.array([n_params(Cc) for Cc in CS for D in ND], dtype=float)
    all_D = np.array([grid[(Cc, D)][1] for Cc in CS for D in ND], dtype=float)
    all_L = np.array([grid[(Cc, D)][0] for Cc in CS for D in ND], dtype=float)
    all_D = np.maximum(all_D, 1)

    def Lhat(theta, Ns, Ds):
        E, lA, la, lB, lb = theta
        return (E + np.exp(lA) * Ns ** (-np.exp(la))
                + np.exp(lB) * Ds ** (-np.exp(lb)))

    x0 = [H, math.log(1.0), math.log(0.5), math.log(1.0), math.log(0.5)]
    lo = [0.0, -10, -10, -10, -10]
    hi = [math.log(V), 5, 5, 5, 5]
    res = least_squares(lambda th: Lhat(th, all_N, all_D) - all_L,
                        x0, bounds=(lo, hi))
    E_, A_, alpha, B_, beta = (res.x[0], np.exp(res.x[1]), np.exp(res.x[2]),
                               np.exp(res.x[3]), np.exp(res.x[4]))
    pred = Lhat(res.x, all_N, all_D)
    rmse = float(np.sqrt(np.mean((pred - all_L) ** 2)))
    print(f"  拟合：E={E_:.4f} · A={A_:.3f} · α={alpha:.3f} · B={B_:.3f} · "
          f"β={beta:.3f} · 25 格 RMSE={rmse:.4f}")
    print(f"  ⚠ 拟合 E 漂到 {E_:.3f}，远低于数据过程真实熵 H={H:.4f}：")
    print("    量程太窄时 E 与幂指数互相补偿、不可辨识（Hoffmann 原论文跨 4000 格")
    print("    巨量程才钉住 E）。本章把 H 当『地板概念』做对照，拟合 E 不当精确解——")
    print("    它滑到 H 之下，正是『E 吸收了一部分可压缩项』的迹象，正文照实讲。")

    print()
    print("  等算力·实测腿：固定 N·D 预算档，模型越小 + 数据越多（Chinchilla 方向）：")
    for lab, cells in [
        ("预算≈4.8–5.2e10", [(64, 917), (96, 458), (128, 229)]),
        ("预算≈1.3–1.4e10", [(32, 917), (96, 114)]),
    ]:
        parts = []
        for c_, d_ in cells:
            ndt = n_params(c_) * grid[(c_, d_ * 1024)][1]
            parts.append(f"(C{c_},D{d_}K) L={grid[(c_, d_ * 1024)][0]:.4f}"
                         f" 预算={ndt:.2e}")
        print(f"  {lab:>16}  " + " · ".join(parts))
    print("  → 同一预算档里，把预算从参数挪给数据，loss 全部更小——"
          "玩具的『最优』指向数据侧。")

    print()
    print("  等算力·拟合腿：argmin_N L(N, c/N)（c=N·token = 算力代理）：")
    print("  " + "     ".join(["budget", "N*(最优参量)", "D*(tokens)",
                                "D*/N*", "位置"]))
    ratios = []
    for c in [1e9, 3e9, 1e10, 3e10, 1e11]:
        def obj(logN):
            Nn = math.exp(logN)
            Dd = c / Nn
            return Lhat(res.x, Nn, Dd)
        loN, hiN = min(all_N), max(all_N)
        lg = minimize_scalar(obj, bounds=(math.log(loN), math.log(hiN)),
                             method="bounded")
        Np = math.exp(lg.x)
        Dp = c / Np
        ratios.append(Dp / Np)
        print(f"  {c:9.0e}  →  N*={Np:8.0f}  D*={Dp:11.0f}  "
              f"D*/N*={Dp/Np:7.1f}  "
              f"{'下界Nmin' if abs(math.exp(lg.x) - loN) < 0.005 * loN else '内部'}")
    print("  → 最优一直钉在 N 下界（参数压紧、预算全给数据）：玩具的数据未饱和，")
    print("    配比 D*/N* 随预算涨成『数据无限便宜』的形状；Chinchilla 的常数")
    print("    （D*≈20 tokens/参数）要在数据端开始饱和之后才出现——数据墙那节再撞。")

    # ---- D 数据墙：有限池 vs 新样本。墙=池的统计分辨率，位置可解析预测
    print(); print("=" * 78)
    print("D · 数据墙：有限语料池循环训练 vs 无限新样本（C=96，early stopping）")
    print("=" * 78)

    def pool_floor(P0w, laplace=1.0, draws=6):
        """墙位的解析预测：模型把池子的经验分布记到极致（softmax 有 Laplace 平滑
        地板）后，对【真实过程】的期望 CE = H + 池子的统计分辨率噪声。"""
        acc = []
        for d in range(draws):
            pw = sample_walk(np.random.RandomState(300 + d), n=P0w)
            cnt = np.zeros((V, V, V), float)
            for i in range(len(pw) - 2):
                cnt[pw[i], pw[i + 1], pw[i + 2]] += 1
            ce = 0.0
            for a in range(V):
                for b in range(V):
                    n = cnt[a, b].sum() + laplace * V
                    if n < 1:
                        continue
                    phat = (cnt[a, b] + laplace) / n
                    ce += pi[a * V + b] * (-(np.log(np.maximum(phat, 1e-30))
                                             * P[a, b]).sum())
            acc.append(ce)
        return float(np.mean(acc))

    def train_pooled(pool, ktoks, rng_world):
        steps = int(math.ceil(ktoks / (B * PER)))
        p = init_c(C0, seed=1234)
        o = g.init_adam(p)
        interval = max(8, min(32, steps // 10))
        best_e = 1e9
        best_tok = 0
        for s in range(1, steps + 1):
            tok_row = np.zeros((B, T), dtype=int)
            for j in range(B):
                off = rng_world.randint(0, len(pool) - LEN - 1)
                tok_row[j, :LEN] = pool[off:off + LEN]
            tgt = np.concatenate([tok_row[:, 1:], np.zeros((B, 1), dtype=int)], axis=1)
            lr_now = 3e-3 * min(1.0, s / 200)
            loss, cache = g.forward(tok_row, p, tgt)
            gr = g.backward(p, cache, loss_mask=lm)
            g.adam_update(p, o, gr, lr_now)
            if s % interval == 0 or s == steps:
                e = eval_ce(p)
                if e < best_e:
                    best_e = e
                    best_tok = s * B * PER
        return best_e, best_tok

    fresh_cache = {}

    def fresh_best(ktoks):
        if ktoks not in fresh_cache:
            fresh_cache[ktoks] = train_cell(C0, ktoks, seed_tr=311)[:2]
        return fresh_cache[ktoks]

    set_c(C0)   # 网格末格 C=128 曾把引擎 DH 置为 32，池世界回到 C0 前必须复位

    for P0k, ks in [(8, (4, 16, 64, 128)), (16, (4, 16, 64))]:
        P0w = P0k * 1024
        floor_now = pool_floor(P0w)
        pool = sample_walk(np.random.RandomState(300), n=P0w)
        print(f"\n  P0={P0k}K 池：解析预测墙位 ≈{floor_now:.4f}（= H{floor_now - H:+.4f}）")
        for k in ks:
            bgt = P0w * k
            e_p, t_p = train_pooled(pool, bgt, np.random.RandomState(310))
            e_f, t_f = fresh_best(bgt)
            print(f"    {k:>3}×P0({bgt // 1024:>4}K tok)：池 best {e_p:.4f}"
                  f" @{t_p // 1024:>4}K · 新 best {e_f:.4f} @{t_f // 1024:>4}K"
                  f" → 池−新 {e_p - e_f:+.4f}")
    print("  读法：k 涨（预算翻倍）时新样本世界沿 D 幂律继续降；池世界弯向解析墙位，")
    print("  越过墙后预算翻倍最佳 eval 一价不动——『更多训练』当场白给。")

    print()
    print("done · 一键复现：python code/scripts/scaling_demo.py")
    print("复现自检键：H {:.6f} · A D0(229K) 各C CE [{}] · B C96 各D CE [{}]"
          " · 拟合E {:.6f} · α {:.6f} · β {:.6f}".format(
              H,
              " ".join(f"{grid[(Cc, 229*1024)][0]:.6f}" for Cc in CS),
              " ".join(f"{grid[(96, D)][0]:.6f}" for D in ND),
              E_, alpha, beta))
    print(f"done · 整脚本墙钟 {time.perf_counter() - t_start:.0f}s（本机纯 CPU 单线程）")


if __name__ == "__main__":
    main()
