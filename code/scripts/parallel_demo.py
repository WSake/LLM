# -*- coding: utf-8 -*-
"""
parallel_demo.py
────────────────────────────────────────────────────────────────────────────
04-训练体系 07-训练并行-DP-ZeRO-FSDP-TP-PP-SP-EP 的复现脚本（04 目录第 6 个引擎脚本）。

单卡装不下 78 GiB，背景音是"把状态摊到多卡"（A）；但摊开就要运货——梯度、
权重在卡间搬的账决定选哪款并行（B）；流水线是"把时间切片"的并行，气泡到底
有多大我们用离散事件仿真真跑一遍（C，本机实测）；TP/SP/EP 各切"层内矩阵/
序列/专家"一维（D）。

四段性质：
  A 显存三档账   ——账算（7B 全参微调四件套 78.2 GiB 沿用 ch06，非本机实测）
  B 梯度货运账   ——账算（DDP ring vs ZeRO reduce-scatter；含假想平台假设）
  C 流水线气泡   ——离散事件仿真（本机真跑：fill-flush vs 1F1B 两种派活策略）
  D TP/SP/EP 账 ——账算（张量/序列/专家三把切刀各省什么、付什么通信）

纯 CPU，无随机性（全部确定性算术 + 确定性仿真）。
复现声明：run1==run2==run3 科学数字逐位一致，只有墙钟在跑次间浮动。
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

GIB = 2**30
P7 = 7e9          # 7B 参数（账算口径，同 ch06）
F_BYTES = 2       # fp16 梯度/权重每参数字节


# ──────────────────────────────────────────────────────────────────────────
# A 段：显存三档账 —— DP 不省、ZeRO 渐进、FSDP 同级
# ──────────────────────────────────────────────────────────────────────────

def a_ledger():
    w = P7 * 2 / GIB        # 权重 fp16 → 13.0 GiB
    g = P7 * 2 / GIB        # 梯度 fp16 → 13.0 GiB
    adam = P7 * 8 / GIB     # Adam 一阶+二阶 fp32 → 52.2 GiB
    total = w + g + adam    # 78.2 GiB
    print("[A] 显存三档账（7B 全参微调四件套 78.2 GiB 沿用 ch06；账算非本机实测；GiB=2³⁰）")
    print(f"    四件套：权重 {w:.1f} + 梯度 {g:.1f} + Adam 一/二阶各 {adam/2:.1f} = {total:.1f} GiB")
    print("    ────────────────────────────────────────────────────────────────")
    print("    单卡整份      : 78.2 GiB（40 GiB 卡坐不下，超桌 1.95×）")
    print("    DP（8 卡）    : 每卡仍是 78.2 GiB（整份拷贝，只摊数据，显存一分没省）")
    for cc in (8, 16, 32):
        print(f"    ZeRO-1（{cc} 卡）: 权重 13.0 + 梯度 13.0 + 优化器 52.2/{cc}={52.2/cc:.1f} "
              f"→ {26.0 + 52.2/cc:.1f} GiB/卡（只切优化器）")
    for cc in (8, 16, 32):
        print(f"    ZeRO-2（{cc} 卡）: 权重 13.0 + 梯度 13.0/{cc}={13.0/cc:.2f} + 优化器 "
              f"52.2/{cc}={52.2/cc:.1f} → {13.0 + 13.0/cc + 52.2/cc:.1f} GiB/卡（再切梯度）")
    for cc in (8, 16, 32):
        print(f"    ZeRO-3（{cc} 卡）: 78.2/{cc} = {78.2/cc:.1f} GiB/卡（三件全分片）")
    print(f"    FSDP（8 卡）  : 与 ZeRO-3 同语义、按层物化，稳态每卡同为 {78.2/8:.1f} GiB 级"
          f"（另含当前层 2× 单层权重临时物化的小头）")
    print()


# ──────────────────────────────────────────────────────────────────────────
# B 段：梯度货运账 —— DDP ring vs ZeRO（半分档少一半，全分档回到 DDP 量级）
# ──────────────────────────────────────────────────────────────────────────

def _comm_ms(G_bytes, P, eff_mult):
    # 每卡净传 = eff_mult × (P-1)/P × G_bytes；DDP ring =2；ZeRO reduce-scatter =1
    vol = eff_mult * (P - 1) / P * G_bytes
    return vol / 400e9 * 1000          # 互连单向 400 GB/s → 毫秒

def _compute_ms(t):
    return 6 * P7 * t / 100e12 * 1000  # 6·N·t FLOP，单卡有效 100 TFLOPS (bf16) → 毫秒

def b_comms():
    G_bytes = P7 * F_BYTES             # 14.0e9 B 全量梯度
    print("[B] 梯度货运账：每步 = 全量梯度 " f"{G_bytes/1e9:.0f} GB（7B×2B fp16）；"
          "DDP 全规约每卡净传 2(P-1)/P·G，ZeRO-1/2 用 reduce-scatter = (P-1)/P·G")
    print("    假设平台（账算非本机实测）：单卡有效算力 100 TFLOPS(bf16)、互连单向 400 GB/s；")
    print("    t = 每卡每步 token（全局 batch ÷ 卡数）")
    print("    ────────────────────────────────────────────────────────────────")
    print(f"    {'P':>4} {'t':>5} │ {'DDP 通信/步':>12} {'DDP 计算比':>10} │ "
          f"{'ZeRO1/2 通信/步':>13} {'ZeRO1/2 计算比':>11}")
    for P, t in ((8, 4096), (8, 512), (64, 128), (64, 16)):
        ddp = _comm_ms(G_bytes, P, 2)
        zer = _comm_ms(G_bytes, P, 1)
        cmpt = _compute_ms(t)
        print(f"    {P:>4} {t:>5} │ {ddp:>8.1f} ms {ddp/cmpt*100:>9.1f}% │ "
              f"{zer:>8.1f} ms {zer/cmpt*100:>10.1f}%")
    print("    → DP 只在 batch 充沛时划算（P=8·t=4096 只 3.6%）；t 掉到 512 就 28.5%；")
    print("      P=64·t=128 时 128%——通信主导。ZeRO-1/2 把整根曲线减半；ZeRO-3 通信回到")
    print("      DDP 量级（reduce-scatter + all-gather 两段），换的是显存自由，不是通信更省。")
    print()


# ──────────────────────────────────────────────────────────────────────────
# C 段：流水线气泡离散事件仿真（本机真跑）
# ──────────────────────────────────────────────────────────────────────────

def sim_pipeline(S, M, policy):
    """S 段、M 个微批；F=B=1 单位时间。

    policy:
      'naive' —— fill-flush：每卡先排完所有 F、再逆序 B（经典朴素流水线）。
      '1f1b'  —— 回传优先：B 一旦 ready 就做（梯度尽快回传），F 在无 B 时做。

    返回 (makespan, bubble_frac, stage0_resident_hw, global_inflight_hw)。
    确定性：无随机，唯一任务序列。
    """
    doneF = [[False] * M for _ in range(S)]
    doneB = [[False] * M for _ in range(S)]
    job = [None] * S               # (m, kind, remain)

    def ready_F(m, s):
        if doneF[s][m]:
            return False
        if s > 0 and not doneF[s - 1][m]:
            return False
        return True

    def ready_B(m, s):
        if doneB[s][m] or not doneF[s][m]:
            return False
        if s < S - 1 and not doneB[s + 1][m]:
            return False
        return True

    def res0():
        # stage0：F 已完成但 B 未完成 = 该卡缓存着的微批激活驻留数
        return sum(doneF[0][m] and not doneB[0][m] for m in range(M))

    def inflight():
        # 全局：至少还有一段任务未完的微批数（≈ 同时驻留的激活批次量）
        return sum(any(not doneB[s][m] for s in range(S)) for m in range(M))

    def pending():
        return any(not doneB[s][m] for s in range(S) for m in range(M))

    idle = 0
    hw0 = 0
    hwi = 0
    t = 0
    while pending():
        # 1) 收尾上一 tick 的任务
        for s in range(S):
            if job[s] is not None:
                m, k, rm = job[s]
                rm -= 1
                if rm <= 0:
                    (doneF if k == "F" else doneB)[s][m] = True
                    job[s] = None
                else:
                    job[s] = (m, k, rm)
        # 2) 记录本时刻驻留高水位
        hw0 = max(hw0, res0())
        hwi = max(hwi, inflight())
        # 3) 派活
        for s in range(S):
            if job[s] is not None:
                continue
            candF = [m for m in range(M) if ready_F(m, s)]
            candB = [m for m in range(M) if ready_B(m, s)]
            pick = None
            if policy == "naive":
                if candF:
                    pick = ("F", candF[0])
                elif candB:
                    pick = ("B", max(candB))     # 逆序回传（标准 flush）
            else:                                 # '1f1b'
                if candB:
                    pick = ("B", min(candB))      # 回传优先
                elif candF:
                    pick = ("F", candF[0])
            if pick is not None:
                kind, m = pick
                job[s] = (m, kind, 1)
            elif pending():
                idle += 1                          # 有活却没得派 = 气泡
        t += 1
    return t, idle / (S * t), hw0, hwi


def c_sweep():
    print("[C] 流水线气泡离散事件仿真（本机真跑；F=B=1 单位时间；fill-flush vs 1F1B）")
    print("    'naive' 行：气泡引入公式对照 (S-1)/(M+S-1)（经典结果）")
    print("    ────────────────────────────────────────────────────────────────")
    print(f"    {'S':>2} {'M':>3} {'policy':>6} │ {'makespan':>8} {'气泡实测':>7} "
          f"{'F公式':>7} {'F偏差':>7} │ {'s0驻留':>6} {'全局在飞':>7}")
    for S in (2, 4, 8):
        for M in (2, 4, 8, 16):
            for policy in ("naive", "1f1b"):
                ms, bubble, hw0, hwi = sim_pipeline(S, M, policy)
                formula = (S - 1) / (M + S - 1)
                if policy == "naive":
                    dev = bubble - formula
                    devs = " 0.0pp" if dev == 0 else f" {dev*100:+.1f}pp"
                    print(f"    {S:>2} {M:>3} {policy:>6} │ {ms:>8} {bubble*100:>6.1f}% "
                          f"{formula*100:>6.1f}% {devs:>7} │ {hw0:>6} {hwi:>7}")
                else:
                    print(f"    {S:>2} {M:>3} {policy:>6} │ {ms:>8} {bubble*100:>6.1f}% "
                          f"{'--':>6} {'--':>7} │ {hw0:>6} {hwi:>7}")
    print("    （1F1B 行气泡同级——1F1B 不减出米率、省的是驻留激活：s0驻留=stage0 缓存微批数，")
    print("      naive 可达 M、1F1B 封在 ~S；全局在飞=同时驻留的微批数。气泡补一档 M 就减半。）")
    print()


# ──────────────────────────────────────────────────────────────────────────
# D 段：TP / SP / EP 账 —— 切矩阵 / 切序列 / 切专家
# ──────────────────────────────────────────────────────────────────────────

def d_ledger():
    print("[D] TP / SP / EP 账（账算非本机实测）")
    # TP：每层 4 次 all-reduce（fwd 2 + bwd 2），每次传 B×seq×d
    act = 1 * 2048 * 4096 * F_BYTES / 1e6          # 16.8 MB
    print(f"    TP 层内：每层 fwd 2 次 + bwd 2 次 all-reduce；例 seq=2048·d=4096："
          f"激活 {act:.1f} MB/次 ×4 = {act*4:.1f} MB/层/步；32 层 ≈ {act*4*32/1024:.1f} GB/步")
    # SP：激活按序列维切
    full = 32768 * 4096 * F_BYTES / GIB * 1024     # B·seq·d·2B → MiB/层
    print(f"    SP 切序列：长上下文例 seq=32k（B=1·d=4096·fp16）：激活 {full:.0f} MB/层"
          f" → SP p=8 每卡 {full/8:.0f} MB/层（÷8）；32 层 {full*32/1024:.1f} → "
          f"{full*32/8/1024:.1f} GiB 级")
    # EP：all-to-all
    d = 1_000_000 * 2 * 4096 * F_BYTES / GIB * 2   # fwd+bwd 各一趟
    print(f"    EP 切专家：MoE 专家切 K 卡，token 去 top-k 专家所在卡 → all-to-all ≈ "
          f"D×topk×d×2B（fwd）再同量 bwd；例 D=1M token·topk=2·d=4096：每层每步 {d:.1f} GiB "
          f"（两趟合计）；专家驻留则按 K 摊")
    print()


# ──────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 66)
    print("parallel_demo：并行全家桶的显存账 / 通信账 / 气泡仿真（04 训练并行 §17.3.6）")
    print("=" * 66)
    print("[0] 口径：承接 ch06 的 7B 四件套账（12 B/参数 = 78.2 GiB，不含激活）。")
    print("    并行把这份账切散到多卡（A）；摊开就要运货（B）；流水线把时间切片（C 仿真）；")
    print("    TP/SP/EP 切层内矩阵/序列/专家（D）。A/B/D=账算非本机实测，C=本机离散事件仿真。")
    print()
    a_ledger()
    b_comms()
    c_sweep()
    d_ledger()
    print(f"墙钟 {1000*(time.time()-t0):.0f} ms（科学数字逐位一致，只有墙钟浮动）")


if __name__ == "__main__":
    main()
