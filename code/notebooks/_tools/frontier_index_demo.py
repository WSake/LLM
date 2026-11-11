# -*- coding: utf-8 -*-
"""frontier_index_demo.py —— §15 前沿索引页（15-前沿技术 01 篇配套）

四账对应正文 📑 本节大纲四点：
账 A  成熟度矩阵：15 行完整性 + 三档分布（成熟 9 / 在途 4 / 探索 2）
账 B  定位 + 链接主干：8 行走主干索引、7 行进专题页，专题页 6 个（15-04 覆盖两行）
账 C  读法：成熟度 × 是否主干覆盖 × 必学标记 × 角色覆盖 → 学习动作（8 场景 8/8）
账 D  季度回炉：按本季成熟度变更判定哪些行需要改列，并校验「回炉不改方向编号」

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr。全部统计 = 本机确定性计算。

诚实边界（与正文「数字纪律」一致）：
- 15 个方向的名称与成熟度标注【成熟】/【在途】/【探索】= 地图 §15 要素事实，
  原文以 2025 年中期为基准；写作环境无外网、未在线复核，也未预测任何方向的胜出。
- 9/4/2 分布 = 对地图标注的本机确定性计数；「该学 / 该看 / 该知道 / 索引」四档动作、
  回炉四步、8 场景判定表 = 作者整理的**教学数**（规则全部写在本脚本 action() 与 CASES 中，可逐行核对）。
- 无真实发布数据、无论文抓取、无外部 API；本季变更样例为演示用的作者预置输入。
"""
from __future__ import annotations

import sys
import time

_WSTART = time.perf_counter()

# (编号, 方向, 成熟度, 覆盖类型, 必学标记)
MATRIX = (
    ("15.1",  "推理模型 / Test-time Scaling", "成熟", "主干", True),
    ("15.2",  "MoE 极致化",                   "成熟", "主干", False),
    ("15.3",  "超长上下文",                   "成熟", "主干", False),
    ("15.4",  "Agentic AI",                   "成熟", "主干", False),
    ("15.5",  "多模态原生",                   "在途", "主干", False),
    ("15.6",  "World Model",                  "探索", "专题", False),
    ("15.7",  "Diffusion Language Model",    "探索", "专题", False),
    ("15.8",  "SSM / Mamba",                  "在途", "专题", False),
    ("15.9",  "Hybrid Architecture",         "在途", "专题", False),
    ("15.10", "SLM / 端侧 AI",                "成熟", "专题", False),
    ("15.11", "AI Agent 全栈",                "成熟", "主干", False),
    ("15.12", "Computer Use / Coding Agent", "成熟", "主干", False),
    ("15.13", "MCP",                          "成熟", "主干", False),
    ("15.14", "合成数据与数据飞轮",           "成熟", "专题", False),
    ("15.15", "其他方向跟踪",                 "在途", "专题", False),
)

# 6 个专题滚动页 -> 覆盖的矩阵行（15-04 同时覆盖 15.8 与 15.9）
TOPIC_PAGES = {
    "15-02": ("15.6",),
    "15-03": ("15.7",),
    "15-04": ("15.8", "15.9"),
    "15-05": ("15.10",),
    "15-06": ("15.14",),
    "15-07": ("15.15",),
}

ACTIONS = {"成熟": "该学", "在途": "该看", "探索": "该知道"}


def action(maturity, covered, must_learn=False, role_override=None):
    """§3 判定规则（优先级从上到下，逐条可在正文 §3 找到）：
    1) 个人角色正好落在该方向 -> 强制该学
    2) 地图标「必学」-> 该学（15.1：学习发生在 02 章，但不降级为索引）
    3) 主干已讲透 -> 索引（去主干学，与成熟度无关）
    4) 其余按成熟度三档
    """
    if role_override:
        return "该学"
    if must_learn:
        return "该学"
    if covered == "主干":
        return "索引"
    return ACTIONS[maturity]


def rows_to_update(changes):
    """账 D：只有成熟度真的变了的行才需要改列；按矩阵固定顺序返回。"""
    current = {r[0]: r[2] for r in MATRIX}
    return [r[0] for r in MATRIX if r[0] in changes and changes[r[0]] != current[r[0]]]


def W(label):
    print("[wall] %-28s %.4fs" % (label, time.perf_counter() - _WSTART), file=sys.stderr)


def main():
    p = f = 0

    def check(name, got, exp):
        nonlocal p, f
        hit = got == exp
        p += int(hit)
        f += int(not hit)
        print("  %-46s %s" % (name, "PASS" if hit else "FAIL(期望 %r 实得 %r)" % (exp, got)))

    print("15-01 §15 前沿索引页 · 成熟度矩阵（纯 stdlib / 零网络 / 零随机）")

    # ---------- 账 A：矩阵完整性 + 三档分布 ----------
    print("\n[账 A] 15 行矩阵完整性（大纲 1）")
    check("方向总数", len(MATRIX), 15)
    check("编号唯一", len({r[0] for r in MATRIX}), 15)
    dist = {m: sum(1 for r in MATRIX if r[2] == m) for m in ("成熟", "在途", "探索")}
    check("三档分布 成熟/在途/探索", dist, {"成熟": 9, "在途": 4, "探索": 2})
    check("三档合计 = 15", sum(dist.values()), 15)
    check("必学标记仅 1 条", [r[0] for r in MATRIX if r[4]], ["15.1"])
    check("覆盖类型只有主干/专题", {r[3] for r in MATRIX}, {"主干", "专题"})
    W("账A")

    # ---------- 账 B：定位 + 链接主干 ----------
    print("\n[账 B] 主干索引 vs 专题页（大纲 2）")
    main_rows = [r[0] for r in MATRIX if r[3] == "主干"]
    topic_rows = [r[0] for r in MATRIX if r[3] == "专题"]
    check("主干索引 8 行", len(main_rows), 8)
    check("专题覆盖 7 行", len(topic_rows), 7)
    check("8 + 7 = 15", len(main_rows) + len(topic_rows), 15)
    check("专题页 6 个", len(TOPIC_PAGES), 6)
    covered = [x for rows in TOPIC_PAGES.values() for x in rows]
    check("专题页覆盖行去重 = 7", len(set(covered)), 7)
    check("专题页覆盖行与矩阵一致", sorted(covered), sorted(topic_rows))
    check("15-04 覆盖两行", TOPIC_PAGES["15-04"], ("15.8", "15.9"))
    W("账B")

    # ---------- 账 C：该学 vs 该看（8 场景） ----------
    print("\n[账 C] 成熟度 × 学习动作判定（大纲 3 · 8 场景）")
    CASES = (
        ("15.1 必学·主干",                "成熟", "主干", True,  None,     "该学"),
        ("15.2 成熟·主干",                "成熟", "主干", False, None,     "索引"),
        ("15.10 成熟·专题",               "成熟", "专题", False, None,     "该学"),
        ("15.5 在途·主干",                "在途", "主干", False, None,     "索引"),
        ("15.9 在途·专题",                "在途", "专题", False, None,     "该看"),
        ("15.6 探索·专题",                "探索", "专题", False, None,     "该知道"),
        ("探索方向但本人是研究者",         "探索", "专题", False, "研究者", "该学"),
        ("成熟·专题·标必学",              "成熟", "专题", True,  None,     "该学"),
    )
    for i, (name, m, c, must, role, exp) in enumerate(CASES, 1):
        check("[%d] %s" % (i, name), action(m, c, must, role), exp)
    check("全矩阵动作分布·索引", sum(1 for r in MATRIX if action(r[2], r[3], r[4]) == "索引"), 7)
    W("账C")

    # ---------- 账 D：季度回炉 ----------
    print("\n[账 D] 季度回炉变更判定（大纲 4）")
    changes = {"15.8": "探索", "15.10": "成熟", "15.6": "在途", "15.3": "成熟"}
    check("只有真变了的行需要改列", rows_to_update(changes), ["15.6", "15.8"])
    check("无变更季度不改任何行", rows_to_update({}), [])
    check("全档变更全部命中", len(rows_to_update(
        {r[0]: "探索" for r in MATRIX})), 13)
    check("回炉不改方向编号", [r[0] for r in MATRIX][:15],
          ["15.1", "15.2", "15.3", "15.4", "15.5", "15.6", "15.7",
           "15.8", "15.9", "15.10", "15.11", "15.12", "15.13", "15.14", "15.15"])
    W("账D")

    print("\nPASS %d  FAIL %d" % (p, f))
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())
