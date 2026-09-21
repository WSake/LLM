# -*- coding: utf-8 -*-
"""Mistral / Mixtral 系列（复习 02-17 滑窗 + 02-18 MoE）：SWA 回望半径与第一代开源 MoE。

配合《03-模型家族/05-Mistral与Mixtral.md》使用。实验分四段：

  A. 滑窗注意力（SWA，复习 02-17）：① 成本账——full 因果与滑窗的打分对数，
     W=L/4 时比值恒定 2.3×、W 固定时序列翻倍比值翻倍（T·W ≈ L²/2 的代数）；
     ② 回望半径分诊——full 训练的 add-ICL 模型在推理时把掩码换成各窗宽，
     看 query 场上还剩几对完整演示对（结论：悬崖在 W=10→9——只把第 1 个
     完整演示对切出窗外 fresh 就 1.000→0.328 崩，窗内剩 2..1 对也救不回；
     回望半径是训练期雕进电路的，推理时改窗 = 换任务）；
     ③ 从零用滑窗训同一任务——窗内信息够就能学会（W=5:0.96·W=9:0.86），
     W=1 自注意力 only 无跨 token 信息 → 塌 0.10。
  B. 开源 MoE（复习 02-18）：400 步×bs8 对照——前 3 臂与 04 章 [B2] 逐位一致
     （经典 Top-2 / +共享专家 / +共享+无auxbias，同 rng 流位置重放），
     + 同参孪生对（第 5 次抽取、同种子、只差 aux）：控制 vs +aux（Mixtral/Switch
     方案，无共享、无 bias）。核心问题：共享专家是不是 MoE 必选项？——共享吸
     走通用语法（CE↘）、bias/aux 独立管均衡，两者正交；Mixtral 用 aux 把均衡写
     进优化目标（CE+aux 账），DeepSeek-V3 用 bias 写进参数（aux 恒 0）。
  C. （本家族无独立 C 段——SWA 的 KV 显存账并入 [A1]，参数/激活账并入 [D]）
  D. 派生账（公开配置换算，非本机实测）：打分对数 32k/4k→4.3×、128k/4k→16.3×；
     KV 行（SWA 不省 KV——rolling buffer 才省驻留）；Mixtral 8x7B 总参 46.7B /
     激活 12.9B≈27% vs DeepSeek-V3 5.5%——同为开 MoE，激活比例差 5 倍；家族
     演化减法账（SWA 的来去、θ=1e6 大 theta RoPE、Tekken 131k 词表、重回 MoE）。

纯 CPU 可跑（单线程 BLAS），seed 固定，stdout 字节确定（墙钟只在 stderr），
三遍逐位一致。总墙钟目标 ≈ 5-6 分钟。
"""
import os
import sys
import math
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r"F:\00AI创业\01LLM\LLM\code\scripts")
import numpy as np

np.set_printoptions(precision=4, suppress=True)
import llama_demo as L
import moe_demo as M
import deepseek_demo as D        # 复用 D.moe_fwd / D.moe_bwd / D.adam_safe（与 04 章同源）

_START = time.perf_counter()


def now(msg):
    sys.stderr.write(f"[{time.perf_counter()-_START:7.1f}s] {msg}\n")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def band_mask(T, W):
    """滑窗因果掩码：m[t,j]=1 iff 0 ≤ t−j < W；W≥T 即退化 full 因果。"""
    idx = np.arange(T)
    m = idx[:, None] - idx[None, :]
    return ((m >= 0) & (m < W)).astype(float)


def pair_visible(T, W):
    """t=T−1（预测位置）的滑窗里能看见的『完整 (x_i,y_i) 演示对』个数。
    add-ICL 序列：x1,y1,x2,y2,…,xK,yK,xq → 末位 xq 回望 [T−W, T) 区间。"""
    lo = max(0, T - W)
    n = 0
    for i in range(T):
        if (2 * i) >= lo and (2 * i + 1) < T - 1:      # x_i,y_i 都在窗内、且不是 xq 位
            n += 1
    return n


# ===========================================================================
# [A] 滑窗注意力（复习 02-17）
# ===========================================================================
def expA():
    print("=" * 66)
    print("[A] 滑窗注意力 SWA（复习 02-17：KV 缓存派生、02-03 GQA、02-05 RoPE）")
    print("    引擎 = 02-Llama 同款 2-block add-ICL（C=48 NH=8 NV=4 RoPE/SwiGLU/RMS）")
    print("=" * 66)

    # ---- A1 成本账：full 因果 vs 滑窗 的打分对数 ----
    print("  [A1] 成本账：打分对数全/窗对比（数数，不训练；self 含在对角线）")
    full_c = lambda T: T * (T + 1) // 2
    swa_c = lambda T, W: sum(min(min(t + 1, W), T) for t in range(T))
    print("        T/W       full对数        SWA对数     比值")
    for T, W in ((16, 4), (32, 4), (64, 4), (128, 4)):
        f, s = full_c(T), swa_c(T, W)
        label = "← 本实验 T=11 的量级" if T == 16 else ""
        print(f"        T={T:>3} W={W:<3} {f:>12,} {s:>12,}   {f / s:.1f}×  {label}")
    f, s = full_c(4096), swa_c(4096, 1024)
    print(f"        上行 W 固定、T 每翻倍 → 比值近似翻倍（极限 L/2W）；W=L/4 时比值恒定 {f / s:.1f}×")
    pf, ps = full_c(32768), swa_c(32768, 4096)
    print(f"        真实换算（非本机实测）：Mistral 7B → 32k 上下文、窗 4k："
          f"{pf:,} → {ps:,} = {pf / ps:.1f}× 打分省")
    qf, qs = full_c(131072), swa_c(131072, 4096)
    print(f"        128k 上下文、窗仍 4k：{qf:,} → {qs:,} = {qf / qs:.1f}×")
    print("        ※ 诚实注记：省的是『打分对数』（O(L²)→O(L·W)）；KV 缓存每 token 仍全量驻留，")
    print("          要省 KV 驻留得配 sliding-cache rolling buffer——SWA 本身不省 KV。")

    # ---- A2 回望半径分诊：full 训练 + 推理时换窗 ----
    print("  [A2] 回望半径分诊：full 训练 1200 步×bs96（同 Llama 章 Rx 预算，fresh(full)=1.000）")
    eng = L.Engine(V=13, T=11, C=48, NL=2, NH=8, NV=4)
    p = eng.init_params(7 + eng.C)
    o = L.init_adam(p)
    srng = np.random.RandomState(1234)
    mar = None
    for s in range(1, 1201):
        tok, tgt, lm = L.build_batch(srng, 13, 96, eng.T)
        loss, c = eng.mask_loss(p, tok, tgt, lm)
        g = eng.bwd(p, c, tgt, lm)
        L.adam_step(p, o, g, 3e-3 * min(1.0, s / 150))
    mar = (1200, float(loss))
    print(f"        满窗训练 1200 步末 CE={mar[1]:.3f} → 推理时把滑窗掩码逐格收窄评估（n=400/窗）")
    print(f"        {'窗宽 W':>6}  {'fresh':>6}    窗内完整演示对（末位 xq 能看到几对）")
    for W in (11, 10, 9, 7, 5, 3, 2, 1):
        eng.mask = band_mask(11, W)
        fr = L.eval_fresh(eng, p, np.random.RandomState(509), 13, n=400, S=0)
        eng.mask = band_mask(11, 11)
        print(f"        W={W:<5} {fr:6.3f}    {pair_visible(11, W):d} 对（满窗=5对）")
    print("        → 悬崖在 W=10→9：只把第 1 个完整演示对整对切出窗外，fresh 1.000→0.328 崩；")
    print("          W=7/5/3 窗内还剩 3..1 对也回不去 0.247-0.300——模型内部走的是『全场累积』通路，")
    print("          远距结构被切开 = 换任务，不是缩口径；W=2 反而弹到 0.463（只剩 y5 一个相邻")
    print("          token 时模型拿它当局部提示——统计捷径，不是回望半径回来了）；W=1 0.095 随机。")
    print("          SWA 因此是训练期决策：推理时改窗（服务端常见 OOD）会打断已雕好的电路。")
    # ---- A3 从零用滑窗训练：窗内信息够就能学会；窗内没有就塌 ----
    print("  [A3] 从零滑窗训练 600 步×bs96（窗宽 = 训练期就锁死的回望半径）")
    for W in (9, 5, 1):
        eng.mask = band_mask(11, W)
        p3 = eng.init_params(7 + eng.C)
        o3 = L.init_adam(p3)
        s3 = np.random.RandomState(1234)
        for s in range(1, 601):
            tok, tgt, lm = L.build_batch(s3, 13, 96, eng.T)
            loss, c = eng.mask_loss(p3, tok, tgt, lm)
            g = eng.bwd(p3, c, tgt, lm)
            L.adam_step(p3, o3, g, 3e-3 * min(1.0, s / 150))
        eng.mask = band_mask(11, W)
        fr_own = L.eval_fresh(eng, p3, np.random.RandomState(509), 13, n=400, S=0)
        eng.mask = band_mask(11, 11)
        fr_full = L.eval_fresh(eng, p3, np.random.RandomState(509), 13, n=400, S=0)
        print(f"        W={W:<4} 600步末CE={float(loss):.3f} · fresh@自窗 {fr_own:.3f}"
              f" · fresh@满窗 {fr_full:.3f} · 窗内完整演示对={pair_visible(11, W)}")
    print("        → A2/A3 合起来读：窗口把『哪些位置可信』在训练期锁死——满窗模型收窄")
    print("          必崩（1.000→0.328），自缚 W=5 的模型放成全窗也退化（0.955→0.562，")
    print("          训练时被压低到噪声级的远端在推理时突然可见=未校准干扰）；自缚 W=9")
    print("          （窗够宽）放全窗 0.978≈0.863 无害。改宽改窄都是 OOD，SWA 是训练期决策。")


# ===========================================================================
# [B] 开源 MoE：Mixtral 方案 vs DeepSeek 方案（复习 02-18）
# ===========================================================================
# ---- 单臂 400 步（DeepSeek 三件套各刀 + Mixtral aux 都走这一条路） ----
def run_arm(tag, p, use_sh, use_bias, alpha):
    """复用 deepseek_demo 的三件套循环；alpha!=None 时按 Mixtral/Switch 加 aux 梯度。"""
    if use_sh:
        p["Wsf1"] = np.random.RandomState(3).randn(M.C, M.M_FFN) * 0.06
        p["Wsf2"] = np.random.RandomState(4).randn(M.M_FFN, M.C) * 0.06
    if use_bias:
        p["bias"] = np.zeros(M.E_NUM)
    o_m = {k: np.zeros_like(v) for k, v in p.items() if k != "bias"}
    o_v = {k: np.zeros_like(v) for k, v in p.items() if k != "bias"}
    rd = np.random.RandomState(42)
    last = lastaux = None
    for step in range(1, 401):
        lr_now = 3e-3 * min(1.0, step / 200)
        tok, tgt = M.make_batch(rd)
        loss, ca = D.moe_fwd(tok, p, tgt, shared=use_sh)
        ca["tok"] = tok
        gg = D.moe_bwd(p, ca, shared=use_sh)
        if alpha is not None:                   # Mixtral/Switch：均衡写进辅助损失
            aux, f, Pbar, d_wg = M.aux_loss(ca["wg"], ca["gmask"], alpha=alpha)
            gg["Wg"] += (ca["n2"].transpose(0, 2, 1) @ d_wg).sum(axis=0)
        else:
            aux = 0.0
        if use_bias:                            # DeepSeek-V3：均衡写进参数（bias 标量步）
            f = ca["gmask"].mean(axis=(0, 1))
            p["bias"] = p["bias"] + 0.02 * (1.0 / M.E_NUM - f)
        D.adam_safe(p, gg, o_m, o_v, step, lr_now)
        if step in (200, 400):
            gates = ca["gates"]
            Pbar = gates.mean(axis=(0, 1))
            H = -float(np.sum(Pbar * np.log(Pbar + 1e-12)))
            load = ca["gmask"].mean(axis=(0, 1))
            last = (step, float(loss), H, load)
            lastaux = aux
    s, los, H, load = last
    note = f" (含aux {los+lastaux:.3f})" if alpha is not None else ""
    print(f"    {tag} step {s}: CE {los:.3f}{note} · 门控熵 H {H:.3f}（均匀界 "
          f"{math.log(M.E_NUM):.3f}）· 载荷 [{', '.join(f'{z:.2f}' for z in load)}]")
    now(f"[B] {tag.strip()} 400 步（stderr）")
    return last


def expB():
    print("=" * 66)
    print("[B] 开源 MoE（复习 02-18 双条腿 + aux）：共享专家是必选项吗？")
    print("    复用 02-18 语料/1-block C=96 · E=4 专家 × M=2C 细粒度 · Top-2 · 400 步×bs8")
    print("    前 3 臂与 04 章 [B2] 逐位一致（同 rng 流位置重放）；后 2 臂 = Mixtral 方案")
    print("    同参孪生对（同一参数、同一 42 种子，只差 aux 梯度一个变量）")
    print("=" * 66)
    # 复现 04 章 [B] 的模块 rng 消耗：expB 的 B1 对账画过一轮 init_params(moe=True)
    M.init_params(moe=True)                     # 第 1 次抽取（弃用，对齐流位置）
    ALPHA = 0.10                                # 演示强度：Switch 论文 α=0.01 面向万亿 token，
    #                                            # 400 步玩具需放大才可见（同 02-18 expD）
    print("    [B] 对照：前 3 臂 = DeepSeek 三件套（与 04 章 [B2] 逐位一致）；")
    print("        后 2 臂 = Mixtral 方案（无共享专家）同参孪生对：控制 vs +aux")
    arms = (("经典Top-2       ", False, False, None),
            ("+共享专家        ", True, False, None),
            ("+共享+无auxbias  ", True, True, None))
    rows = {}
    for tag, use_sh, use_bias, alpha in arms:
        p = M.init_params(moe=True)             # 2/3/4 次抽取：与 04 章逐位同参
        rows[tag] = run_arm(tag, p, use_sh, use_bias, alpha)
    # ---- 同参孪生对（第 5 次抽取）：控制 vs +aux，只差 aux 梯度一个变量 ----
    p5 = M.init_params(moe=True)                # 第 5 次抽取（全新、但两臂共用同一参数）
    pc = {k: v.copy() for k, v in p5.items()}   # 控制：深拷贝 → 两臂同起点出发
    rows["经典Top-2·同参  "] = run_arm("经典Top-2·同参  ", pc, False, False, None)
    rows["+aux(Mixtral)   "] = run_arm("+aux(Mixtral)   ", p5, False, False, ALPHA)
    l0 = rows["经典Top-2       "][3]; l1 = rows["+共享专家        "][3]; l2 = rows["+共享+无auxbias  "][3]
    lc = rows["经典Top-2·同参  "][3]; l3 = rows["+aux(Mixtral)   "][3]
    v0 = np.var(l0); v1 = np.var(l1); v2 = np.var(l2); vc = np.var(lc); v3 = np.var(l3)
    hc = rows["经典Top-2·同参  "][2]; h3 = rows["+aux(Mixtral)   "][2]
    print(f"    → 载荷方差：经典 {v0:.4f} → +共享 {v1:.4f} → +bias {v2:.4f}"
          f"（04 章 [B2] 逐位一致）")
    print(f"      同参孪生对：经典 {vc:.4f} → +aux(Mixtral) {v3:.4f}"
          f"（门控熵 {hc:.3f} → {h3:.3f}，均匀界 {math.log(M.E_NUM):.3f}）")
    print("    → 共享专家 vs 均衡机制正交：共享吸走通用语法、CE 落地但与载荷方差无关；")
    print("      均衡本身要单独买——Mixtral 用 aux 写进优化目标（总目标 CE+aux ≫ 纯 CE），")
    print("      DeepSeek-V3 用 bias 写进参数（aux 恒 0、目标只剩 CE）——同一结果，账本不同。")
    now("expB 完成")


# ===========================================================================
# [D] 派生账（公开配置换算，非本机实测）
# ===========================================================================
def expD():
    print("=" * 66)
    print("[D] 派生账（公开配置 × 公式，非本机实测）：SWA 不省 KV · 开 MoE 激活比 5 倍差")
    print("=" * 66)
    print("  ① 打分对数（同 [A1]）：Mistral 7B 32k/4k → 4.3×；128k/4k → 16.3×（=L/2W 极限）")
    print("  ② KV 行：SWA 每 token 的 KV 仍全量驻留；64k 上下文×L/层与滑窗无关——")
    print("     rolling buffer 才把驻留降到 O(W)。家族里 7B v0.1 / Mixtral 8x7B 用 SWA，")
    print("     v0.2/v0.3 改回全注意——但无论哪种布局，KV 都按全序列驻留记：")
    print("     SWA 省的是『参与打分的位置』，不是『缓存里躺着的位置』。")
    print("  ③ 参数账：Mixtral 8x7B 公开配置 46.7B 总参 / ~12.9B 激活（8 选 2 路由专家+注意）")
    act_r = 12.9 / 46.7
    print(f"     激活/总 ≈ {act_r:.1%}（≈27%，与 00 章/04 章口径一致）vs DeepSeek-V3 总/激活 671B/37B ≈ 5.5% →"
          " 同为开源 MoE,")
    print("     激活比差 5 倍；选 2/8（Top-2）与 V3 的 8+共享（细粒度）是算术差距来源。")
    print("  ④ 家族演化减法账（公开 release 口径，非本机实测）：Mistral 7B v0.1 全 SWA → v0.2 改回")
    print("     全注意（滑窗的『丢远距』量产不可用，官方 v0.2 直接砍 SWA）；→ v0.3 引用 θ=1e6")
    print("     大 theta RoPE（复习 02-05）；Nemotron 系 Tekken 131k 词表（中文/多语 token 压缩，")
    print("     解码提速）；Ministral 3B/8B 探端侧；Mistral Large/Large-2 走 123B 稠密；")
    print("     Mistral 3 8x6B 重回 MoE。")
    print("     —— 减法不是为了少参数，是把『买不起的 O(·)』逐项搬出推理热路径。")


def main():
    now("start（stderr）")
    expA()
    expB()
    expD()
    now(f"全脚本累计 {time.perf_counter()-_START:.1f}s")
    print()
    print("done · 一键复现：python code/scripts/mistral_demo.py")


if __name__ == "__main__":
    main()
