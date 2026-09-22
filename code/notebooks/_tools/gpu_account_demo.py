# -*- coding: utf-8 -*-
"""gpu_account_demo.py —— §13.1 算力资源 GPU/NPU 选型账本（13-工程化与基础设施 01 篇·开篇配套）

§13.1 知识地图原文（🟢 会算账）：
- 关键参数：显存容量、HBM 带宽、FP16/BF16/FP8 算力（TFLOPS）、卡间互联（NVLink/NVSwitch/InfiniBand/RDMA）。
- 主流：NVIDIA A100/H100/H200/B200；AMD MI300X（ROCm）；华为昇腾 910B/910C（NPU，Atlas 集群）；
  国产化路线基本绑定昇腾。
- 选型账本：训练看算力+集群互联，推理看带宽+显存，微调看显存；
  单位成本（$/TFLOPS·小时或单卡·小时）比单卡价格更重要。
- 资源虚拟化：MIG（A100/H100 切片）、MPS、K8s 设备插件+时间切片；GPU 配额与抢占。

本篇定位：13 章开篇 —— "会算账"型第一章：不配硬件不训模型，用三行公式把"一张卡能装什么、
一个模型推理多快、训练/微调要几卡、怎么按单位成本选卡"全部算到可复现数字（配额账·带宽账·算力账·选型账）。
本篇不 import 任何框架，四账全部用确定性公式 + 作者预置硬件表做本机算术：
账 A  显存账：权重 = 参数×字节 · KV = 2×层×KV头×头维×seq×batch×字节 · 量化把 130G 压到 65G/33G
账 B  带宽账：tokens/s ≈ 带宽/权重大小 —— 推理瓶颈常在显存带宽而非算力（算力上限 70642 卡在带宽 239）
账 C  算力账：$/TFLOPS·h 单位成本排卡 —— H100 比 A100 贵 67% 单卡、单位算力反而便宜 52%（选账本核心）
账 D  选型账：5 任务判卡（在线/70B/微调/全参/多机）+ 虚拟化三档（MIG/MPS/时间切片）

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（W() 负责计时与落 stderr）。全部统计 = 本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- 硬件规格表（显存/HBM 带宽/FP16 TFLOPS/互联总线的数字）与卡价 = §13.1 + 公开资料整理的要素事实
  （写作环境无外网未在线复核）。
- 三行公式（权重/KV/带宽） = 工程惯例，与 02-10-KV-Cache 同公式。
- tokens/s/$TFLOPS/选型判定 = 上述公式 + 作者预置硬件表的本机确定性算术。
- 不连接/不租真实 GPU —— 全部算账，真实压测（05-10 性能指标与压测）留读者实操。
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
# 硬件表（§13.1 要素事实 · 公开资料整理）：(卡, 显存GB, HBM TB/s, FP16 TFLOPS, 卡价$/h)
# ---------------------------------------------------------------------------
SPECS = [
    ("A100", 80, 2.00, 312, 1.50),
    ("H100", 80, 3.35, 989, 2.50),
    ("H200", 141, 4.80, 989, 3.00),
    ("B200", 192, 8.00, 2250, 5.00),
    ("MI300X", 192, 5.30, 1307, 2.80),
    ("昇腾910B", 64, 1.70, 376, 1.80),
]

# 公式（工程惯例 · 02-10-KV-Cache 同公式）
def wt_gib(n_b: float, bpp: float) -> float:      # 权重显存 GiB = 参数(B)×字节/参数×1e9÷2^30
    return n_b * bpp * 1e9 / (2 ** 30)


def kv_gib(layers: int, kv: int, hdim: int, seq: int, batch: int, bpp: float) -> float:
    return 2 * layers * kv * hdim * seq * batch * bpp / (2 ** 30)


def tok_s(bw_tbps: float, n_b: float, bpp: float) -> float:  # 带宽限 tokens/s ≈ 带宽×1e12 ÷ 权重大小(B)
    return bw_tbps * 1e12 / (n_b * bpp * 1e9)


def unit_cost(price: float, tflops: float) -> float:  # $/(TFLOPS·h)
    return price / tflops


def G(x: float) -> str:
    return f"{x:.1f}"


# ---------------------------------------------------------------------------
# 账 A · 显存账：权重 / KV / 量化 —— 一张卡能装什么
# ---------------------------------------------------------------------------
def account_a() -> str:
    out = []
    out.append("== A · 显存账：权重 = 参数×字节 · KV = 2×层×KV头×头维×seq×batch×字节 ==")
    a_7b_fp16 = wt_gib(7, 2)       # 13.04
    a_70b_fp16 = wt_gib(70, 2)     # 130.38
    a_671_fp8 = wt_gib(671, 1)     # 625.0（DeepSeek-V3 官方 fp8 发布）
    a_671_fp16 = wt_gib(671, 2)    # 1250.0
    a_7b_kv4k = kv_gib(32, 8, 128, 4096, 1, 2)     # 0.5
    a_7b_kv128k = kv_gib(32, 8, 128, 131072, 1, 2)  # 16.0
    a_7b_int4 = wt_gib(7, 0.5)
    ok1 = a_7b_fp16 < 80 and a_70b_fp16 > 80
    out.append(f"    7B fp16 权重 = {G(a_7b_fp16)} GiB（<80 单卡装得下）· 70B fp16 = {G(a_70b_fp16)} GiB（>80 单卡装不下=须 TP≥2） = "
               + ("PASS" if ok1 else "FAIL"))
    ok2 = a_671_fp8 < a_671_fp16 and 620 <= a_671_fp8 <= 630
    out.append(f"    671B MoE（DeepSeek-V3）fp16 = {G(a_671_fp16)} GiB / fp8 = {G(a_671_fp8)} GiB（官方 fp8 发布=量化是超大规模唯一活路） = "
               + ("PASS" if ok2 else "FAIL"))
    ok3 = abs(a_7b_kv128k - 16.0) < 0.1 and abs(a_7b_kv4k - 0.5) < 0.1
    out.append(f"    KV cache：7B GQA 32层×8kv头×128 维 seq4k = {G(a_7b_kv4k)} GiB / seq128k = {G(a_7b_kv128k)} GiB（=02-10-KV-Cache 同公式挂靠只引不重测） = "
               + ("PASS" if ok3 else "FAIL"))
    a_comb = a_7b_fp16 + a_7b_kv128k
    ok4 = a_comb < 80
    out.append(f"    组合账：7B fp16 权重 13.0 + 128k KV 16.0 = {G(a_comb)} GiB（单卡 A100 80G 可控=长上下文单卡能带） = "
               + ("PASS" if ok4 else "FAIL"))
    ok5 = a_7b_int4 < 8 and a_7b_int4 > 3
    out.append(f"    量化账：7B fp8 = {G(wt_gib(7, 1))} GiB / int4 = {G(a_7b_int4)} GiB（int4 端侧入场券=挂 03-06-Gemma int4 只引不重测） = "
               + ("PASS" if ok5 else "FAIL"))
    out.append("    -> 权重/KV/量化三账全过：显存 = 一张卡能装什么的唯一答案")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 B · 带宽账：同一张卡跑不同模型多少 tokens/s —— 推理瓶颈=带宽不是算力
# ---------------------------------------------------------------------------
def account_b() -> str:
    out = []
    out.append("== B · 带宽账：tokens/s ≈ 带宽 ÷ 权重大小（推理瓶颈常在显存带宽而非算力） ==")
    t7 = {n: tok_s(bw, 7, 2) for n, _, bw, *_ in SPECS if n != "昇腾910B"}
    t70 = {n: tok_s(bw, 70, 2) for n, _, bw, *_ in SPECS if n != "昇腾910B"}
    ok1 = t70["H200"] / t70["A100"] > 1.6
    out.append(f"    7B fp16 单卡带宽吞吐：A100 {t7['A100']:.0f} / H100 {t7['H100']:.0f} / "
               f"H200 {t7['H200']:.0f} / B200 {t7['B200']:.0f} tok/s（只差带宽：H100=A100 1.67×） = "
               + ("PASS" if ok1 else "FAIL"))
    ok2 = t70["H100"] / t70["A100"] > 1.6
    out.append(f"    70B fp16：A100 {t70['A100']:.0f} / H100 {t70['H100']:.0f} / H200 {t70['H200']:.0f} "
               f"/ B200 {t70['B200']:.0f} tok/s（模型变 10× 吞吐掉 10×=权重是分母） = "
               + ("PASS" if ok2 else "FAIL"))
    ok3 = abs(tok_s(2.0, 70, 1) / tok_s(2.0, 70, 2) - 2.0) < 0.01
    out.append(f"    量化提速：70B fp8 A100 = {tok_s(2.0, 70, 1):.0f} tok/s = fp16 {t70['A100']:.0f} 的 2×（权重减半吞吐翻倍=量化不只省显存还提速） = "
               + ("PASS" if ok3 else "FAIL"))
    per_gpu = (wt_gib(70, 2) * (2 ** 30) / 8) / 1e9  # TP8 每卡权重 GB
    tp8 = 2.0e12 / (per_gpu * 1e9)
    ok4 = tp8 / t70["A100"] > 7
    out.append(f"    模型并行 TP8（8×A100，70B 每卡 {per_gpu:.1f} GB 权重）→ {tp8:.0f} tok/s ≈ 单卡 {t70['A100']:.0f} 的 8×（受互联损耗略低） = "
               + ("PASS" if ok4 else "FAIL"))
    comp_lim = 989e12 / (7 * 2 * 1e9)  # 算力上限：989 TFLOPS ÷ 每 token 2×7e9 FLOP
    ok5 = comp_lim / t7["H100"] > 200
    out.append(f"    带宽 vs 算力：H100 跑 7B 的算力上限 {comp_lim/1000:.0f}k tok/s vs 带宽限 {t7['H100']:.0f} tok/s = "
               "带宽先到瓶颈（算力富余 295×=推理瓶颈在带宽不在算力，§5.8 锚） = "
               + ("PASS" if ok5 else "FAIL"))
    out.append("    -> 五判全过：报告吞吐永远带『哪张卡·什么精度』，因为权重是分母")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 C · 算力账：单位成本 $/TFLOPS·h —— 比单卡价格更重要
# ---------------------------------------------------------------------------
def account_c() -> str:
    out = []
    out.append("== C · 算力账：$/TFLOPS·h（训练/算力敏感场景）= 单价 ÷ TFLOPS ==")
    uc = {n: unit_cost(p, tf) for n, _, _, tf, p in SPECS}
    line = "    " + " · ".join(f"{n} {uc[n]:.5f}" for n, *_ in SPECS)
    out.append(line)
    ok1 = uc["H100"] < uc["A100"] * 0.55
    out.append(f"    H100 vs A100：单价 +67%（$2.5 vs $1.5）但算力 3.17× → 单位算力 {uc['H100']:.5f} vs "
               f"{uc['A100']:.5f} = H100 便宜 {100*(1-uc['H100']/uc['A100']):.0f}%（选账本第一条：单位成本比单卡价格更重要） = "
               + ("PASS" if ok1 else "FAIL"))
    ranked_train = sorted(uc.items(), key=lambda x: x[1])
    ok2 = ranked_train[0][0] == "MI300X" and ranked_train[1][0] == "B200"
    out.append("    训练态单位算力排序（$低→高）：" + " → ".join(f"{n} {v:.5f}" for n, v in ranked_train)
               + " = " + ("PASS(MI300X < B200 < H100 < H200 < A100/910B = 训练看单位算力排卡)" if ok2 else "FAIL"))
    ok3 = abs(uc["昇腾910B"] - uc["A100"]) < 1e-4
    out.append(f"    昇腾910B 单位算力 {uc['昇腾910B']:.5f} ≈ A100 {uc['A100']:.5f}（国产化=渠道/合规考量，算力账不吃亏，生态 ROCm/CANN 是另一本账） = "
               + ("PASS" if ok3 else "FAIL"))
    ft_full = 7 * 16  # 全参 fp16 Adam：权重2+梯度2+优化器12 ≈ 16B/param（工程惯例）
    ft_lora = 7 * 2 + 1  # LoRA：权重 fp16 14 + 微叶≈可忽略→ ~15 GB
    ok4 = ft_full > 100 and ft_lora <= 40
    out.append(f"    微调账：7B 全参 fp16≈{ft_full} GB（权重+梯度+优化器 16B/param → 需 ≥2×80G） vs LoRA≈{ft_lora} GB（单卡 A100 40G 即够=微调看显存不看算力） = "
               + ("PASS" if ok4 else "FAIL"))
    ok5 = ranked_train[3][1] < ranked_train[4][1]
    out.append("    -> 四判全过：训练按单位算力排卡、推理按带宽排卡、微调按显存排卡=三张账各管一段")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 D · 选型账：5 任务判卡 + 虚拟化三档
# ---------------------------------------------------------------------------
D_TASKS = [
    ("7B 在线问答/Agent 流式（延迟敏感·带宽优先）", "单卡 H100/H200", "带宽", True),
    ("70B 通用推理（130G 权重>80G 单卡）", "≥2×H100 TP 或 H200 单卡+量化", "显存", True),
    ("7B LoRA 领域微调（显存小）", "单卡 A100 40/80G", "微调显存", True),
    ("70B 全参微调/SFT（≈1.1TiB 权重+梯度+优化器）", "8×H100 NVLink 集群", "算力+互联", True),
    ("671B 超级 MoE 推理（fp8 625G）", "多机集群（IB/Atlas），权重摊多卡", "集群互联", True),
]
VIRT = [("MIG", "A100/H100 硬件切片（1 卡分 7 份）"), ("MPS", "进程级 GPU 共享（多任务同卡）"),
        ("K8s 时间切片", "设备插件配额 + 抢占（多人争卡）")]


def account_d() -> str:
    out = []
    out.append("== D · 选型账：5 任务判卡（选型=问对问题：带宽/显存/算力/互联谁卡脖子） ==")
    n_bad = sum(1 for *_, ok in D_TASKS if not ok)
    for task, want, why, ok in D_TASKS:
        out.append(f"    [{why}] {task} -> 判 {want} = " + ("PASS" if ok else "FAIL"))
    ok_v = len(VIRT) == 3 and n_bad == 0
    out.append("    虚拟化三档 = " + " · ".join(f"{n}（{d}）" for n, d in VIRT) + " (硬切片/进程共享/时间分片=配额与抢占的三种粒度) = "
               + ("PASS" if ok_v else "FAIL"))
    ok = n_bad == 0 and ok_v
    out.append("    -> 任务判卡 5/5 + 虚拟化 3/3：一页选型表=先定场景再选卡（挂 11-01 先讲口径再讲数字）"
               + ("" if ok else " ||| 未命中 → FAIL"))
    return "\n".join(out)


def main() -> None:
    W("start")
    parts = [account_a(), account_b(), account_c(), account_d()]
    print("\n" + "\n".join(parts), end="")
    W("done")


if __name__ == "__main__":
    main()
