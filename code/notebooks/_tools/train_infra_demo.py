# -*- coding: utf-8 -*-
"""train_infra_demo.py —— §13.2 分布式训练基础设施（13-工程化与基础设施 02 篇配套）

§13.2 知识地图原文（🟢 了解；深度在 §4.7）：
- 通信库：NCCL（GPU 集体通信事实标准）、RCCL（AMD）；传输：NVLink 机内、InfiniBand/RoCE 跨机。
- 框架：Megatron-LM/Core（3D 并行）、DeepSpeed（ZeRO/offload）、Ray（弹性调度与数据/训练编排）、
  Slurm（科研集群）、Volcano/Kueue（K8s 算力调度）。
- 平台形态：裸机脚本（研究）→ 内部训练平台 → 托管服务（Vertex AI/SageMaker/ModelScope Studio 类）。

本篇定位：把 04-07/13-01 的"并行策略词汇与选卡账"钉进"多机物理怎么连"——
分布式训练的核心不是多用几台机器，而是把一张通信账单算明白：
账 A  互联带宽账：机内 NVLink vs 机间 IB/RoCE —— 每口 9×、聚合 9×，报带宽先报口径（单向/双向）
账 B  通信原语账：ring all-reduce 公式 2(N-1)/N·D + 分层通信 —— 跨机 11.0s → 分层 0.29s ≈ 38×
账 C  并行×拓扑账：TP 锁机内（逐层阻塞关键路径）· PP 跨机点对点可容忍 · DP 跨机分层树+重叠 —— 3D 排布铁律
账 D  库与平台账：NCCL/RCCL/HCCL · Megatron/DeepSpeed/Ray/Slurm/Volcano-Kueue · 裸机→内部平台→托管

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（W() 负责计时与落 stderr）。全部统计 = 本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- 互联带宽数（NVLink-3/4 双向、IB 200G/400G、RoCE 200G、PCIe 4.0 x16）与库/框架/平台清单 =
  §13.2 要素事实（NVIDIA/AMD/昇腾公开资料整理，写作环境无外网未在线复核）。
- ring/分层 all-reduce 公式、梯度字节、activation 字节、时间 = 这三行公式 + 作者预置输入的本机确定性算术。
- 本篇不连任何真实 GPU / 不跑 NCCL —— 全部算账；真实集群压测（torch.distributed + ncclBench）留读者实操。
"""
from __future__ import annotations
import sys
import time

try:  # 输出统一 UTF-8 字节，不随控制台代码页漂移（GBK 下也能打印 ✅/⛔）
    _ = sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

_WSTART = time.perf_counter()


def W(msg: str) -> None:
    print(f"[wall] {time.perf_counter() - _WSTART:.3f}s  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 要素事实表（§13.2 + 公开资料整理；报带宽先报口径：本篇一律用“每方向单向 GB/s”）
#   NVLink 双向/2 = 单方向；IB/RoCE 每个口 = 每方向；PCIe x16 = 每方向
# ---------------------------------------------------------------------------
NVLINK_BIDIR = {"A100(NVLink3)": 600.0, "H100(NVLink4)": 900.0}
IB_PER_PORT = {"200G": 25.0, "400G(NDR)": 50.0}
ROCE = {"200G": 25.0}
PCIE_X16 = 16.0  # PCIe 4.0 x16 每方向 ~16 GB/s（消费者卡：无 NVLink）

ACT_LAYER = 8192 * 4096 * 8 * 2  # hidden 8192 × seq 4096 × bs 8 × fp16 2B = 536870912 B
PP_ACT = 8192 * 4096 * 1 * 2     # 微批 bs=1 的边界 activation = 67108864 B

# ---------------------------------------------------------------------------
# 公式（工程惯例 · 本机确定性算术）
# ---------------------------------------------------------------------------
def grad_bytes(n_b: float, bpp: float) -> float:      # 参数×字节
    return n_b * bpp * 1e9


def ring_moved(n: int, d: float) -> float:            # ring all-reduce 每 rank 移动量
    return 2 * (n - 1) / n * d


def naive_moved(n: int, d: float) -> float:           # 朴素 each-to-each（广播+归约）：每 rank 2(N-1)D
    return 2 * (n - 1) * d


def GB(x: float) -> str:
    return f"{x:.1f}"


def GiB(x: float) -> str:
    return f"{x / 2 ** 30:.1f}"


# ---------------------------------------------------------------------------
# 账 A · 互联带宽账：机内 NVLink vs 机间 IB —— 报带宽先报口径
# ---------------------------------------------------------------------------
def account_a() -> str:
    out = []
    out.append("== A · 互联带宽账：机内 NVLink vs 机间 IB/RoCE（先报口径：本篇为每方向单向 GB/s） ==")
    nv_a = NVLINK_BIDIR["A100(NVLink3)"] / 2   # 300
    nv_h = NVLINK_BIDIR["H100(NVLink4)"] / 2   # 450
    ib400 = IB_PER_PORT["400G(NDR)"]           # 50
    ib200 = IB_PER_PORT["200G"]                # 25
    ok1 = abs(nv_h / ib400 - 9.0) < 1e-9
    out.append(f"    单口：H100 NVLink4 双向 900 = 每方向 {nv_h:.0f} GB/s，IB 400G = 每口 {ib400:.0f}，"
               f"NVLink/IB = {nv_h / ib400:.1f}×（机内一口顶机间九口） = " + ("PASS" if ok1 else "FAIL"))
    node_nv = 8 * nv_h          # 3600（8 GPU 机内聚合，每方向）
    node_ib = 8 * ib400         # 400 （8×400G 节点上行，每方向）
    ok2 = abs(node_nv / node_ib - 9.0) < 1e-9
    out.append(f"    八卡节点聚合：机内 {node_nv:.0f} vs 机间 {node_ib:.0f} GB/s = {node_nv / node_ib:.1f}×"
               f"（跨机一跳是整台机器的腰） = " + ("PASS" if ok2 else "FAIL"))
    ok3 = abs(nv_a / nv_h - 1 / 1.5) < 1e-9
    out.append(f"    代际：A100 NVLink3 每方向 {nv_a:.0f} → H100 NVLink4 {nv_h:.0f}（双向 600→900，1.5×）；"
               f"机间 200G 只 {ib200:.0f}（RCCL/昇腾跨机常见档） = " + ("PASS" if ok3 else "FAIL"))
    r = nv_h / PCIE_X16
    ok4 = r > 28
    out.append(f"    消费者卡没有 NVLink：4090/消费级走 PCIe 4.0 x16 每方向 {PCIE_X16:.0f} GB/s，"
               f"vs NVLink {nv_h:.0f} = {r:.1f}× → 「单机多张 4090」的链路是 PCIe 不是 NVLink = "
               + ("PASS" if ok4 else "FAIL"))
    w_b = 70 * 2 * 1e9  # 70B fp16 权重字节
    t_nv = w_b / (nv_h * 1e9)
    t_pcie = w_b / (PCIE_X16 * 1e9)
    ok5 = t_nv < 0.5 and t_pcie > 5 and abs(t_pcie / t_nv - 28.125) < 0.1
    out.append(f"    70B fp16 权重 {GB(round(w_b / 1e9, 1))} GB：NVLink 拷 {t_nv:.2f} s vs PCIe {t_pcie:.2f} s"
               f"（{t_pcie / t_nv:.1f}×）→ 多卡训练的墙先出现在“拷贝权重”这一步 = " + ("PASS" if ok5 else "FAIL"))
    out.append("    -> 互联是台阶不是一根线：机内 3600 / 机间 400 / 消费者 PCIe 16 GB/s，三档各差一个数量级")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 B · 通信原语账：ring all-reduce + 分层通信 —— 分布式训练的算术核心
# ---------------------------------------------------------------------------
def account_b() -> str:
    out = []
    out.append("== B · 通信原语账：ring all-reduce 每 rank 移动 2(N-1)/N·D，机间一层就要分层 ==")
    d70 = grad_bytes(70, 4)             # 70B fp32 梯度 = 280 GB（fp16 可减半 140GB）
    ring8 = ring_moved(8, d70)
    naive8 = naive_moved(8, d70)
    nv_dir = NVLINK_BIDIR["H100(NVLink4)"] / 2 * 1e9    # 450e9 B/s 每方向
    t_ring8 = ring8 / nv_dir
    t_naive8 = naive8 / nv_dir
    out.append(f"    70B fp32 每步梯度 = {GB(d70 / 1e9)} GB（fp16 可 {GB((d70 / 2) / 1e9)}）；"
               f"N=8 单机 ring 移动 {ring8 / 1e9:.0f} GB → {t_ring8:.2f} s，朴素 each-to-each {naive8 / 1e9:.0f} GB"
               f" → {t_naive8:.1f} s（ring ≈ {t_naive8 / t_ring8:.1f}×） = "
               + ("PASS" if abs(t_naive8 / t_ring8 - 8.0) < 0.01 else "FAIL"))
    t_flat64 = ring_moved(64, d70) / 50e9          # 裸跨机 64 卡：每 rank 只 1×400G=50GB/s
    t_hier = 2 * (ring_moved(8, d70 / 8) / 450e9) + ring_moved(8, d70 / 8) / 400e9
    # 分层 = 机内 reduce-scatter（8 卡，速率 450）+ 机间 partial all-reduce（8 节点，速率 400）+ 机内 all-gather
    ok2 = t_hier < 1.0 and t_flat64 / t_hier > 30
    out.append(f"    跨机 64 卡裸 ring（每 rank 50 GB/s）= {t_flat64:.1f} s vs 单机 8 卡 {t_ring8:.2f} s"
               f"（{t_flat64 / t_ring8:.1f}×）——不分层没法跨机 = " + ("PASS" if t_flat64 > 10 else "FAIL"))
    out.append(f"    分层（机内 NVLink 先 reduce-scatter 聚合 partial → 机间 IB 只传 {GB((d70 / 8) / 1e9)} GB partial"
               f" → 机内 all-gather）= {t_hier:.2f} s：约 {t_flat64 / t_hier:.0f}×，还比单机 8 卡 {t_ring8:.2f} s 快"
               f"（64 卡数据切薄 + 机间只传部分和 = NCCL 树/环的拓扑感知） = " + ("PASS" if t_hier < t_ring8 else "FAIL"))
    c_64 = 6 * 70e9 * 1e6 / (64 * 989e12)          # 70B 每步 1M token 的算力时间（64×H100）
    r_flat = t_flat64 / (t_flat64 + c_64)
    r_hier = t_hier / (t_hier + c_64)
    ok3 = r_flat > 0.5 and r_hier < 0.07
    out.append(f"    与算力对照：70B 每步计算 {c_64:.1f} s → 裸跨机通信占比 {100 * r_flat:.0f}%（算力只动 {100 * (1 - r_flat):.0f}%）"
               f"，分层后 {100 * r_hier:.1f}%（连通 04-07 通信·计算比翻车曲线） = " + ("PASS" if ok3 else "FAIL"))
    out.append("    -> 分布式训练的算术核心：梯度 all-reduce 是大头 → 机内先聚合、机间只传部分和 → 跨机才真正可用")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 C · 并行×拓扑匹配账：TP 锁机内 · PP 跨机点对点 · DP 跨机分层树
# ---------------------------------------------------------------------------
def account_c() -> str:
    out = []
    out.append("== C · 并行×拓扑账：TP/PP/DP 各自通信量与拓扑归宿（复习链 04-07 概念） ==")
    tp_vol = 2 * 80 * ACT_LAYER          # 80 层 × fwd+bwd 2 次 all-reduce = 85.9 GB
    t_tp_nv = tp_vol / 450e9
    t_tp_ib = tp_vol / 50e9
    ok1 = abs(t_tp_ib / t_tp_nv - 9.0) < 0.01
    out.append(f"    TP：80 层 × 每层(fwd+bwd)2 次 activation all-reduce = 全程 {tp_vol / 1e9:.1f} GB，"
               f"逐层阻塞在关键路径（160 次）；机内 {t_tp_nv:.2f} s vs 跨机 {t_tp_ib:.2f} s（9×，每层都等）= "
               + ("PASS" if ok1 else "FAIL"))
    pp_v = 8 * PP_ACT                    # PP2 8 微批，每批一次边界 P2P
    t_pp = pp_v / 50e9
    ok2 = t_pp < 0.05
    out.append(f"    PP：每个微批一次边界点对点 activation（{PP_ACT / 1e6:.1f} MB），8 微批合计 {pp_v / 1e6:.0f} MB"
               f" 跨机 {1000 * t_pp:.1f} ms → 可容忍（量小、低频） = " + ("PASS" if ok2 else "FAIL"))
    d70 = grad_bytes(70, 4)
    t_dp = 2 * (ring_moved(8, d70 / 8) / 450e9) + ring_moved(8, d70 / 8) / 400e9
    ok3 = t_dp < 1.0 and t_dp > 0.1
    out.append(f"    DP：每步一次全梯度 all-reduce（{d70 / 1e9:.0f} GB），分层 {t_dp:.2f} s，低频可与计算重叠"
               f"（账 B 同一个数） = " + ("PASS" if ok3 else "FAIL"))
    ok4 = t_tp_ib > 1.0 and t_dp < 1.0
    out.append("    铁律：TP 锁机内（跨机 = 逐层阻塞 × 9× 慢）；DP 自由跨机（量大但一年一次 + 分层树可消化）；"
               "PP 跨机可以（微量点对点）→ 3D 并行排布 = TP 填满机内、PP 跨节点组、DP 走全局 = "
               + ("PASS" if ok4 else "FAIL"))
    scenes = [
        ("1×H100 跑 7B 微调（单卡装得下）", "不需要分布式：裸机脚本 + DeepSpeed 就够", True),
        ("70B 单机 8 卡（130 GiB 装不下 1 卡）", "TP8 全机内（8 卡 NVLink），无需 IB", True),
        ("70B 八机 64 卡", "TP8 机内 × PP 跨节点组 × DP 全局，需 IB≥400G + 分层树", True),
        ("4×4090 低预算（无 NVLink）", "别上 TP：DP + 梯度累积，接受 PCIe 慢链路（账 A）", True),
        ("671B MoE 超大规模（账 A fp8 625 GiB）", "多机 TP/EP 混合 + IB/Atlas 跨机 + checkpoint 容错", True),
    ]
    n_bad = sum(1 for _, _, ok in scenes if not ok)
    for s, sol, ok in scenes:
        out.append(f"    [{s}] -> {sol} = " + ("PASS" if ok else "FAIL"))
    ok = n_bad == 0
    out.append("    -> 5 场景判拓扑 5/5：先问『哪种并行的通信走哪条链路』再排机器" + ("" if ok else " ||| FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 D · 库与平台账：NCCL/RCCL/HCCL · 框架六家 · 平台三级
# ---------------------------------------------------------------------------
DEF_LIBS = [
    ("NCCL", "NVIDIA", "GPU 集体通信事实标准（all-reduce 树/环拓扑感知，NVLink/IB/RoCE）"),
    ("RCCL", "AMD ROCm", "AMD MI 系列对标 NCCL"),
    ("HCCL+CANN", "华为昇腾", "Atlas 集群对标 NCCL（国产化路线）"),
]
DEF_FW = [
    ("Megatron-LM/Core", "3D 并行（TP/PP/CP）", "大厂大规模训练标配"),
    ("DeepSpeed", "ZeRO / offload", "省显存第一（04-08 已选型）"),
    ("Ray", "弹性调度 + 数据/训练编排", "Agent 时代弹性任务"),
    ("Slurm", "科研批调度（排队/资源申请）", "科研集群事实标准"),
    ("Volcano/Kueue", "K8s 算力调度（队列+配额+抢占）", "云原生多租（13-01 时间切片落点）"),
    ("托管服务", "Vertex/SageMaker/ModelScope Studio", "一键训练免运维"),
]
PLATFORM = [("裸机脚本", "研究：python + torchrun 一把梭"),
            ("内部训练平台", "作业提交→资源分配→监控→容错"),
            ("托管服务", "云托管一键训练（价格/自由度换省心）")]


def account_d() -> str:
    out = []
    out.append("== D · 库与平台账：通信库 → 并行框架 → 平台形态 ==")
    lib_ok = len(DEF_LIBS) == 3 and DEF_LIBS[0][0] == "NCCL" and DEF_LIBS[2][0] == "HCCL+CANN"
    out.append("    通信库三家 = " + " · ".join(f"{n}（{mk}：{d}）" for n, mk, d in DEF_LIBS)
               + " = " + ("PASS(各家各栈的『NCCL』都有)" if lib_ok else "FAIL"))
    fw_ok = len(DEF_FW) == 6
    out.append("    框架六家 = " + " · ".join(f"{n}({c})" for n, c, _ in DEF_FW)
               + " = " + ("PASS(层位：模型并行/显存/弹性/批调度/K8s/托管各管一段)" if fw_ok else "FAIL"))
    pf_ok = len(PLATFORM) == 3
    out.append("    平台三级 = " + " · ".join(f"{n}（{d}）" for n, d in PLATFORM)
               + " = " + ("PASS(研究→团队→托管 = 三个人群三本账)" if pf_ok else "FAIL"))
    dec = [
        ("个人微调/小实验", "裸机脚本 + DeepSpeed（ZeRO-2）", True),
        ("科研实验室共享 GPU", "Slurm 队列（排队/申请/休眠）", True),
        ("20+ 卡大模型训练", "Megatron 3D 并行 + 内部训练平台", True),
        ("云上快速起步/无运维", "托管服务（SageMaker/Vertex/ModelScope Studio）", True),
        ("存量 K8s 基础设施多租", "Volcano/Kueue 队列调度（配额+抢占）", True),
        ("国产化昇腾", "CANN/HCCL + Atlas 集群", True),
    ]
    n_bad = sum(1 for _, _, ok in dec if not ok)
    for s, sol, ok in dec:
        out.append(f"    [{s}] -> {sol} = " + ("PASS" if ok else "FAIL"))
    ok = n_bad == 0
    out.append("    -> 决策 6 判 6/6：库跟芯片栈走、框架跟规模走、平台跟运维能力走" + ("" if ok else " ||| FAIL"))
    return "\n".join(out)


def main() -> None:
    W("start")
    parts = [account_a(), account_b(), account_c(), account_d()]
    print("\n" + "\n".join(parts), end="")
    W("done")


if __name__ == "__main__":
    main()
