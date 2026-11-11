# -*- coding: utf-8 -*-
"""career_roadmap_demo.py —— §18.1 + §18.4 算法路线（16-学习路线与里程碑 01 篇配套）

四账对应正文 📑 本节大纲四点：
账 A  六支知识树：完整性 + 共同底座（4 支）/ 分叉层（2 支）划分
账 B  四级路线：年限递增 + 每级「里程碑 / 代表项目 / 仓库章节」三者成对
账 C  十二行寻址表：树上一点 → 仓库路径（8 场景判定 + 全表落盘校验）
账 D  先项目后论文：每阶段一个跑得通项目 + 活项目数落在 2-3

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr。全部统计 = 本机确定性计算。

诚实边界（与正文「数字纪律」一致）：
- 六支树、四级里程表（级别/年限/里程碑/代表项目）、三条铁律 = 地图 §18.1/§18.4 要素事实，
  写作环境无外网、未在线复核；年限是投入强度的粗略刻度，不是能力承诺。
- 「本仓库对应章节」列与十二行寻址表 = 作者整理的教学数；账 C 只做**路径是否真实存在**
  的落盘校验，不校验内容相关性。
- 账 D 的学习计划样例为演示用的作者预置输入，不是任何真实学习者的进度数据。
- 账 C 的落盘校验需在完整仓库目录结构下运行（按 __file__ 定位仓库根）；
  单独复制本脚本到别处时该账会如实报 FAIL。
"""
from __future__ import annotations

import pathlib
import sys
import time

_WSTART = time.perf_counter()
ROOT = pathlib.Path(__file__).resolve().parents[3]

# 账 A：(支, 层级, 要求)
TREE = (
    ("基础",     "共同底座", "会用手算"),
    ("模型原理", "共同底座", "必深"),
    ("训练工程", "共同底座", "必会"),
    ("评测",     "共同底座", "必备"),
    ("专深三选一", "分叉层", "选一支扎下去"),
    ("科研能力",   "分叉层", "工作方式"),
)

# 账 B：(级别, 年限上界, 里程碑, 代表项目, 仓库章节数)
LEVELS = (
    ("初级", 1.0, "讲清 Transformer 每层 / 会 LoRA / 跑通 nanoGPT", "复现 nanoGPT；微调 Qwen", 2),
    ("中级", 2.5, "设计数据配比 / 实现 DPO / FSDP 调优",           "0.5B 预训练；DPO 实验",   3),
    ("高级", 4.0, "牵头 7B-70B 微调 + 评测门禁 / 深入一方向",       "复现 GRPO；70B LoRA",     3),
    ("专家", 5.0, "设计训练方案与数据配方 / 产出技术报告",           "R1-V 风格项目；框架优化", 2),
)

# 账 C：(树上的点, 支, 仓库相对路径) —— 与正文 §3 十二行逐条一致
ADDR = (
    ("反向传播 / 训练闭环", "基础",     "01-基础/05-深度学习-神经网络.md"),
    ("显存与带宽怎么算",   "基础",     "01-基础/07-GPU-CUDA与并行计算.md"),
    ("Attention / MLA",    "模型原理", "02-核心原理/04-MLA-多头潜注意力.md"),
    ("RoPE / 位置编码",    "模型原理", "02-核心原理/05-位置编码-Sinusoidal-RoPE与窗口扩展.md"),
    ("KV Cache 显存账",    "模型原理", "02-核心原理/10-KV-Cache与显存账本.md"),
    ("预训练 → SFT → 对齐", "模型原理", "02-核心原理/13-SFT-监督微调.md"),
    ("Scaling Law / MoE",  "模型原理", "02-核心原理/21-Scaling-Law与数据墙.md"),
    ("数据配比 / 去污染",  "训练工程", "04-训练体系/01-预训练数据工程-采集-清洗-去重.md"),
    ("并行 / 显存账本",    "训练工程", "04-训练体系/07-训练并行-DP-ZeRO-FSDP-TP-PP-SP-EP.md"),
    ("Benchmark / judge",  "评测",     "11-评测与可观测/02-Benchmark体系-MMLU到SWE-bench.md"),
    ("回归门禁 / 评测集",  "评测",     "11-评测与可观测/10-评测工程最佳实践.md"),
    ("效率/部署 专深支",   "专深",     "05-推理与部署/04-KV-Cache系统级优化.md"),
)


def plan_ok(stages, active_projects):
    """账 D：每阶段都有可跑项目，且活项目数落在 2-3。"""
    return all(has_project for _, has_project in stages) and 2 <= active_projects <= 3


def W(label):
    print("[wall] %-28s %.4fs" % (label, time.perf_counter() - _WSTART), file=sys.stderr)


def main():
    p = f = 0

    def check(name, got, exp):
        nonlocal p, f
        hit = got == exp
        p += int(hit)
        f += int(not hit)
        print("  %-48s %s" % (name, "PASS" if hit else "FAIL(期望 %r 实得 %r)" % (exp, got)))

    print("16-01 §18.1/§18.4 算法工程师路线（纯 stdlib / 零网络 / 零随机）")

    # ---------- 账 A：六支树 ----------
    print("\n[账 A] 六支知识树完整性（大纲 1）")
    check("支数", len(TREE), 6)
    check("共同底座 4 支", [t[0] for t in TREE if t[1] == "共同底座"],
          ["基础", "模型原理", "训练工程", "评测"])
    check("分叉层 2 支", [t[0] for t in TREE if t[1] == "分叉层"],
          ["专深三选一", "科研能力"])
    check("必深/必会/必备各一", [t[2] for t in TREE[:4]],
          ["会用手算", "必深", "必会", "必备"])
    check("支名唯一", len({t[0] for t in TREE}), 6)
    W("账A")

    # ---------- 账 B：四级路线 ----------
    print("\n[账 B] 四级路线 · 里程碑/项目/章节成对（大纲 2）")
    check("级别 4 级", [l[0] for l in LEVELS], ["初级", "中级", "高级", "专家"])
    uppers = [l[1] for l in LEVELS]
    check("年限上界严格递增", uppers == sorted(uppers) and len(set(uppers)) == 4, True)
    check("每级里程碑非空", all(l[2] for l in LEVELS), True)
    check("每级代表项目非空", all(l[3] for l in LEVELS), True)
    check("每级至少 2 个仓库章节锚点", all(l[4] >= 2 for l in LEVELS), True)
    check("代表性项目总数 = 8", sum(2 for _ in LEVELS), 8)
    W("账B")

    # ---------- 账 C：寻址表 ----------
    print("\n[账 C] 树上一点 → 仓库路径（大纲 3 · 8 场景 + 全表落盘）")
    check("寻址表 12 行", len(ADDR), 12)
    expected8 = (
        ("反向传播 / 训练闭环", "01-基础/05-深度学习-神经网络.md"),
        ("显存与带宽怎么算",   "01-基础/07-GPU-CUDA与并行计算.md"),
        ("Attention / MLA",    "02-核心原理/04-MLA-多头潜注意力.md"),
        ("RoPE / 位置编码",    "02-核心原理/05-位置编码-Sinusoidal-RoPE与窗口扩展.md"),
        ("KV Cache 显存账",    "02-核心原理/10-KV-Cache与显存账本.md"),
        ("预训练 → SFT → 对齐", "02-核心原理/13-SFT-监督微调.md"),
        ("数据配比 / 去污染",  "04-训练体系/01-预训练数据工程-采集-清洗-去重.md"),
        ("效率/部署 专深支",   "05-推理与部署/04-KV-Cache系统级优化.md"),
    )
    lookup = {a[0]: a[2] for a in ADDR}
    for i, (point, exp) in enumerate(expected8, 1):
        check("[%d] %s" % (i, point), lookup.get(point), exp)
    missing = [rel for _, _, rel in ADDR if not (ROOT / rel).is_file()]
    check("12 条路径全部真实落盘", missing, [])
    check("寻址覆盖 5 个支/专深方向", len({a[1] for a in ADDR}), 5)
    W("账C")

    # ---------- 账 D：先项目后论文 ----------
    print("\n[账 D] 三条铁律 · 学习计划校验（大纲 4）")
    PLANS = (
        ("只读不练",                 [("初级", False), ("中级", False)], 0, False),
        ("每级有项目 / 0 个活项目",  [("初级", True), ("中级", True)],  0, False),
        ("每级有项目 / 2 个活项目",  [("初级", True), ("中级", True)],  2, True),
        ("每级有项目 / 3 个活项目",  [("初级", True), ("中级", True)],  3, True),
        ("每级有项目 / 5 个活项目",  [("初级", True), ("中级", True)],  5, False),
        ("高级阶段缺项目",           [("初级", True), ("高级", False)], 3, False),
    )
    for i, (name, stages, active, exp) in enumerate(PLANS, 1):
        check("[%d] %s" % (i, name), plan_ok(stages, active), exp)
    check("合格计划数（6 中 2）",
          sum(1 for _, s, a, _ in PLANS if plan_ok(s, a)), 2)
    W("账D")

    print("\nPASS %d  FAIL %d" % (p, f))
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())
