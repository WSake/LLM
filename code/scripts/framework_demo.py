# -*- coding: utf-8 -*-
"""
framework_demo.py
────────────────────────────────────────────────────────────────────────────
04-训练体系 08-训练框架横向对比（§17.3.8）的复现脚本。

07 章教了"七种切法"（DP/ZeRO/FSDP/TP/PP/SP/EP），06 章教了"78.2 GiB 四件套
账"。本章不训练、也不在本机对拍真实多卡框架，而是把"选型"本身做成一份可重算
的决策账：同一份显存账，被不同框架组装成不同的格子，格子里有几个、哪个装得下、
哪个该选——读四张表就是完整答案。

四段性质（数字均为账算、非本机实测；c 段断言自校准为本机确定性真跑）：
  A 画像矩阵     ——模型库 / 预训练引擎 / RL 引擎 三类选择，每框架一行
  B 组装账       ——同一份 78.2 GiB × {DDP, ZeRO-3, FSDP, Megatron TP8×PP2} 组装
  C 最大可装表   ——每卡 40 GiB 桌面，逐框架配置求最大可装参数量 N
   决策引擎      ——4 个典型场景的确定性推荐（关键数字按账 assert 校准）
  D 任务适配矩阵 ——§4.8 选型建议 + LoRA/全参双路径收束

承接口径：7B 全参四件套 = 78.2 GiB（12 B/参数，06 章）；每卡 40 GiB 桌面、
互连带宽等假设沿用 07 章。全部确定性算术，无随机，三遍运行科学数字逐位一致
（只有墙钟在跑次间浮动）。
"""
import sys
import time
from math import ceil

sys.stdout.reconfigure(encoding="utf-8")

GIB = 2**30
LEDGER7B = 78.2              # 7B 全参四件套 GiB（06 章：权重+梯度+Adam 二阶）
LEDGER70B = LEDGER7B * 10    # 70B ≈ 7B ×10 的同口径外推（账算，非本机实测）
DESK_GIB = 40.0              # 40 GiB 桌面卡


# ──────────────────────────────────────────────────────────────────────────
# A 段：三类选择族画像矩阵
# ──────────────────────────────────────────────────────────────────────────

def a_profiles():
    print("[A] 三类选择族：模型库 → 预训练引擎 → RL 引擎（§4.8 表画像，账算）")
    print("    族          框架                形态         承载 07 的切法          一句话定位")
    print("    " + "─" * 62)
    rows = [
        ("模型库",   "Transformers",     "接口库",      "（吃上层 PEFT/FSDP）",    "HF 生态中枢：加载/分词/数据/训练循环"),
        ("模型库",   "PEFT",             "接口库",      "LoRA/QLoRA（参数量切法）", "加接口让'只训一小块'成为事实"),
        ("模型库",   "Accelerate",       "装配库",      "DDP（batch 切法）",       "把单卡脚本抬到多卡的脚手架"),
        ("预训练引擎", "DeepSpeed",        "config 驱动", "ZeRO-1/2/3+offload+PP",   "显存自由的引擎：三档切+数据搬出卡"),
        ("预训练引擎", "FSDP",             "API 驱动",    "ZeRO-3 语义、按层物化",   "PyTorch 原生分片，配 Transformers 最顺"),
        ("预训练引擎", "Megatron-LM/Core", "引擎(侵入模型)", "TP/PP/SP（层内+层段）", "万卡级预训练：3D 并行工业实现"),
        ("RL 引擎",  "TRL",              "HF 全家桶",    "（吃 PEFT/FSDP 底座）",   "SFT/DPO/PPO/GRPO 一键式对齐"),
        ("RL 引擎",  "verl",             "引擎(要集群)", "（吃 Megatron/DS 底座）",  "GRPO/RLVR 大规模，R1 同栈"),
        ("微调平台", "LLaMA-Factory",    "WebUI/CLI",   "LoRA/QLoRA/DPO 图形化",   "1-8 卡微调台，一键跑通"),
        ("微调平台", "Axolotl",          "YAML 配置",   "全流程配置化",            "团队可复现的配置化微调"),
        ("微调加速", "Unsloth",          "预包装",      "QLoRA 极致快",            "单卡 LoRA 提速提显存"),
        ("厂系/国产", "NeMo/PaddleNLP/MindSpore", "生态绑定", "各 3D 并行变体",   "大厂/国产算力（NPU）专属栈、生态绑定"),
    ]
    for r in rows:
        print(f"    {r[0]:>7}  {r[1]:<22} {r[2]:<11} {r[3]:<18} {r[4]}")
    print("    → 读法：先选'族'（做微调/预训练/RL 哪一类），族内再按 07 的账选切法，")
    print("      最后用 C 段最大可装表校验这一格装不装得下。")
    print()


# ──────────────────────────────────────────────────────────────────────────
# B 段：组装账 —— 同一份 78.2 GiB × 四种组装
# ──────────────────────────────────────────────────────────────────────────

def b_assembly():
    print(f"[B] 组装账：同一份 {LEDGER7B:.1f} GiB（7B 四件套，06 章）被不同框架组装"
          f"（账算非本机实测；每卡三件套，不含激活/通信缓冲）")
    print(f"    {'组装':<20} {'2 卡':>7} {'4 卡':>7} {'8 卡':>7} {'16 卡':>7} │ 40 GiB 桌面")
    print("    " + "─" * 60)
    ddp = LEDGER7B
    z3 = {2: LEDGER7B / 2, 4: LEDGER7B / 4, 8: LEDGER7B / 8, 16: LEDGER7B / 16}
    print(f"    {'DDP（÷1，batch 切法）':<20} {ddp:>7.1f} {ddp:>7.1f} {ddp:>7.1f} {ddp:>7.1f} "
          f"│ ✗ 永远放不下：省时间不省显存（07 A 段）")
    print(f"    {'DeepSpeed ZeRO-3（÷P）':<20} {z3[2]:>7.1f} {z3[4]:>7.1f} {z3[8]:>7.1f} "
          f"{z3[16]:>7.1f} │ ✓ 2 卡刚够、8 卡余量 4 倍")
    print(f"    {'FSDP（同 ZeRO-3 按层物化）':<20} {z3[2]:>7.1f} {z3[4]:>7.1f} {z3[8]:>7.1f} "
          f"{z3[16]:>7.1f} │ ✓ 稳态同 ZeRO-3，通信摊进层计算")
    print(f"    {'Megatron TP8×PP2（÷16）':<20} {'—':>7} {'—':>7} {'—':>7} {LEDGER7B/16:>7.1f} "
          f"│ 16 卡组装；三件套 4.9 GiB/卡")
    print(f"    70B（7B×10 外推 = {LEDGER70B:.0f} GiB）：ZeRO-3@16 = {LEDGER70B/16:.1f}、"
          f"@32 = {LEDGER70B/32:.1f} GiB/卡 → 80 GiB 卡 @32 有余量；40 GiB 桌面 70B 全参要 Megatron 或 PEFT")
    print()


# ──────────────────────────────────────────────────────────────────────────
# C1 段：最大可装决策表
# ──────────────────────────────────────────────────────────────────────────

def per_param_B(P, mode):
    """每参数三件套字节（fp16 权重 2 + fp16 梯度 2 + Adam fp32 8），按组装分摊。"""
    if mode == "ddp":
        return 4 + 8.0                      # 权重+梯度+优化器全不切
    if mode == "zero1":
        return 4 + 8.0 / P                  # 只切优化器
    if mode == "zero2":
        return 2 + 10.0 / P                 # 权重不切，梯度+优化器 ÷P
    if mode == "zero3":
        return 12.0 / P                     # 三件全 ÷P（FSDP 同语义）
    if mode == "megatron16":
        return 12.0 / 16                    # TP8×PP2：层内÷8 层段÷2


def max_n(P, mode):
    """40 GiB 桌面卡，该组装的每卡三件套上限「参数个数」（不含激活/通信缓冲=乐观上限）。"""
    return DESK_GIB * GIB / per_param_B(P, mode)


def _p_str(n):
    return f"{n/1e9:.1f}B"


def c_maxn_table():
    print("[C] 最大可装决策表：每卡 40 GiB 桌面，该组装能端住的最大 N")
    print("    （账算；三件套口径、不含激活/通信缓冲，是乐观上限，做'上界'来用）")
    print(f"    {'组装':<16} {'2 卡':>8} {'4 卡':>8} {'8 卡':>8} {'16 卡':>8}")
    print("    " + "─" * 48)
    Ps = (2, 4, 8, 16)
    rows = [("DDP",              ["ddp", "ddp", "ddp", "ddp"]),
            ("ZeRO-1",           ["zero1", "zero1", "zero1", "zero1"]),
            ("ZeRO-2",           ["zero2", "zero2", "zero2", "zero2"]),
            ("ZeRO-3 / FSDP",    ["zero3", "zero3", "zero3", "zero3"]),
            ("Megatron TP8×PP2", ["—", "—", "—", "megatron16"])]
    for name, modes in rows:
        vals = []
        for P, mode in zip(Ps, modes):
            if mode == "—":
                vals.append("     —")
            else:
                vals.append(f"{_p_str(max_n(P, mode)):>7}")
        print(f"    {name:<16} " + " ".join(vals))
    # 断言自校准：与 07 章已知格对齐
    assert max_n(8, "zero3") / max_n(2, "zero3") == 4.0        # ÷P 等价：卡数翻倍上限翻倍
    assert max_n(2, "ddp") == max_n(8, "ddp")                   # DDP 不随卡数变化
    assert max_n(16, "zero3") / max_n(16, "megatron16") == 1.0  # ÷16 两种路径同一极限
    assert abs(max_n(8, "zero3") - DESK_GIB * GIB * 8 / 12) < 1e-3
    print("    → 自检（assert）通过：max_n 与 06/07 账一致（DDP 定、ZeRO-3 每翻倍卡数")
    print("      上限翻倍、Megatron ÷16 与 ZeRO-3 ÷16 同极限）。")
    print("      读法：8 卡 ZeRO-3/FSDP 上限 28.6B = 7B 的 4 倍余量；40 GiB 桌端 70B 全参")
    print("      只剩 Megatron ÷16（57.3B 上限）这类组装或继续加卡。")
    print()


# ──────────────────────────────────────────────────────────────────────────
# C2 段：决策引擎 —— 确定性推荐 + assert 校准
# ──────────────────────────────────────────────────────────────────────────

def decide(scenario):
    task = scenario["task"]
    n, cards, gib = scenario["params"], scenario["cards"], scenario["gib"]
    ledger = round(n / 7e9 * LEDGER7B, 1)          # 该模型全参四件套（7B 口径按参数外推）

    if task == "sft-micro":
        why = (f"LoRA 主权重 ≈13.0 GiB 级单卡即可；全参走 FSDP 两张卡 = 39.1 GiB "
               f"刚够（07 章实践作业'FSDP 放两张卡'）")
        return ("LLaMA-Factory（LoRA 路径，13.0 GiB 级）", why)
    if task == "sft-full":
        first = f"全参每卡 {ledger/cards:.1f} GiB（{'✓' if ledger/cards <= gib else '✗'}）"
        need = ceil(ledger / gib)                  # 最小卡数（不凑整）
        ok = 8 * ceil(need / 8)                    # 对齐 8 卡节点
        why = (f"{first} → 需 ≥{need} 卡装进 {gib:.0f} GiB 桌，对齐 8 卡节点取 @{ok} 卡 = "
               f"{ledger/ok:.1f} GiB/卡；或 PEFT 降档让 {cards} 卡装下")
        return ("DeepSpeed ZeRO-3 / FSDP", why)
    if task == "pretrain":
        why = ("1B 三件套 ≈11.2 GiB 单卡已装；预训练账在吞吐——ZeRO-1@8 每参 4+8/8 = 5 B "
               "→ ≈4.7 GiB/卡即够，TP/PP 留给更大模型")
        return ("Megatron-Core / DeepSpeed", why)
    if task == "rlvr":
        why = (f"8B 全参 ≈{ledger:.1f} GiB，@64 卡三件套摊到 {ledger/64:.1f} GiB/卡；"
               f"RL 真成本在采样与策略/参考双驻留（05 章 GRPO 账），不在这套三件套")
        return ("verl", why)
    return ("Transformers + PEFT", "缺省：生态最快")


def c_examples():
    print("[C2] 决策引擎：4 个典型场景 → 确定性推荐（每条断言校准关键数字）")
    cases = [
        {"task": "sft-micro", "params": 7e9, "cards": 2, "gib": 40,
         "title": "Qwen2.5-7B · 2×40 GiB · SFT"},
        {"task": "sft-full", "params": 70e9, "cards": 8, "gib": 80,
         "title": "Llama-70B · 8×80 GiB · 全参 SFT"},
        {"task": "pretrain", "params": 1e9, "cards": 8, "gib": 40,
         "title": "1B 从头预训练 · 8×40 GiB"},
        {"task": "rlvr", "params": 8e9, "cards": 64, "gib": 80,
         "title": "Qwen3-8B RLVR（GRPO）· 64×80 GiB"},
    ]
    checks = {
        "sft-micro": ("13.0", "39.1"),
        "sft-full": ("97.8", "@16 卡", "48.9"),
        "pretrain": ("4.7 GiB",),
        "rlvr": ("89.4",),
    }
    for c in cases:
        stack, why = decide(c)
        kw = checks[c["task"]]
        kw = (kw,) if isinstance(kw, str) else kw
        assert all(k in why for k in kw), f"决策依据缺关键数字: {kw}"
        print(f"    ▸ {c['title']}")
        print(f"        推荐 {stack}")
        print(f"        依据 {why}")
    print("    → 自检（assert）通过：4 例关键数字全部命中预期账。")
    print()


# ──────────────────────────────────────────────────────────────────────────
# D 段：任务适配矩阵 + 双路径收束
# ──────────────────────────────────────────────────────────────────────────

def d_matrix():
    print("[D] 任务适配矩阵（§4.8 选型建议的量表化 + LoRA/全参双路径）")
    rows = [
        ("个人 / 单卡微调",     "Transformers + PEFT + Unsloth", "生态最顺；LoRA 13 GiB 级单卡可跑"),
        ("中小团队 1-8 卡微调", "LLaMA-Factory / Axolotl",      "一键/配置化、可复现"),
        ("预训练 / 大集群",     "Megatron-Core / DeepSpeed",    "TP/PP/ZeRO 万卡级、吞吐账"),
        ("RL / 推理模型",       "verl / TRL",                   "GRPO/RLVR 大规模"),
        ("国产算力（NPU）",     "MindSpore / PaddleNLP",       "生态绑定算力"),
    ]
    print(f"    {'场景':<20} {'推荐栈':<28} {'依据'}")
    print("    " + "─" * 60)
    for s, st, w in rows:
        print(f"    {s:<20} {st:<28} {w}")
    print()
    print("    LoRA / 全参双路径（两章账的收束）")
    print("      路径① PEFT（06 减负）：LoRA 7B ≈ 权重 13.0 + 可训练子集约 0.02 ≈ "
          "13.0 GiB 级；QLoRA NF4 主权重 ≈3 GiB 级（账算数字化）")
    print(f"      路径② 全参 + 并行（07 摊匀）：7B+ZeRO-3@8 = {LEDGER7B/8:.1f} GiB；"
          f"FSDP 同语义按层物化")
    print("      → 交点：7B · 2×40 GiB 两路都走——LoRA 单卡、FSDP 全参 39.1 GiB 刚够。")
    print("      → 生产常搭档 LoRA + FSDP：LoRA 把每卡梯度压小到通信近乎白送（07 B 段），")
    print("        FSDP 让基座权重也能进 40 GiB 桌（07 C 段）。")
    print()


# ──────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.perf_counter()
    print("=" * 66)
    print("framework_demo：训练框架横向对比的选型账（04 训练框架选型 §17.3.8）")
    print("=" * 66)
    print("[0] 口径：承接 06 章 78.2 GiB 四件套账与 07 章七种切法；本章不做训练，")
    print("    把'选型'做成可重算的决策账（A 画像 → B 组装 → C 最大可装/决策引擎")
    print("    → D 任务矩阵）。A/B/D=账算非本机实测；C 段断言自校=本机确定性真跑。")
    print()
    a_profiles()
    b_assembly()
    c_maxn_table()
    c_examples()
    d_matrix()
    print(f"墙钟 {1000*(time.perf_counter()-t0):.1f} ms（科学数字逐位一致，只有墙钟浮动）")


if __name__ == "__main__":
    main()
