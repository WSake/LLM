# -*- coding: utf-8 -*-
"""applied_roadmap_demo.py —— §18.2 + §18.4 应用路线（16-学习路线与里程碑 02 篇配套）

四账对应正文 📑 本节大纲四点：
账 A  六支应用知识树：完整性 + 两支「核心」标记
账 B  四级路线：里程碑序列 + 每级「里程碑 / 代表项目 / 仓库章节」成对
账 C  四个代表项目 → 所需章节清单，含「每级必配评测/护栏」完整性校验
账 D  交叉地带预告：八项交叉 + 四块汇合 + 问题归属判定 6 例

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr。全部统计 = 本机确定性计算。

诚实边界（与正文「数字纪律」一致）：
- 六支树、四级里程表、八项交叉领域、四块汇合地带 = 地图 §18.2/§18.3/§18.4 要素事实，
  写作环境无外网、未在线复核；年限是投入刻度，不是能力承诺。
- 「本仓库对应章节」列、四项目章节清单、6 例问题归属 = 作者整理的教学数
  （规则与映射全部写在本脚本里，可逐行核对），不是任何团队的实际分工数据。
- 无真实项目数据、无性能测量、无外部 API。
"""
from __future__ import annotations

import sys
import time

_WSTART = time.perf_counter()

# 账 A：(支, 是否核心)
TREE = (
    ("基础",     False),
    ("知识工程", True),
    ("框架层",   False),
    ("Agent",    True),
    ("工程化",   False),
    ("业务",     False),
)

# 账 B：(级别, 年限上界, 里程碑, 代表项目, 仓库章节数)
LEVELS = (
    ("初级", 1.0, "API+Prompt+结构化输出 / 简单 RAG / 工具调用", "企业知识库 MVP",       3),
    ("中级", 2.5, "RAG 全栈优化 / 带状态 Workflow-Agent",        "生产级 RAG",           3),
    ("高级", 4.0, "多 Agent + 评测护栏 / MCP / 成本工程",         "客服 Agent",           3),
    ("专家", 5.0, "团队架构 + 数据飞轮 / AgentOps / 行业方案",     "行业数字员工平台",     3),
)

# 账 C：项目 -> (主章, 必配评测/护栏, 可延后)
PROJECTS = {
    "企业知识库 MVP":   (("06-应用开发", "08-RAG体系"), ("11-评测与可观测/05-RAG评测.md",), ("07-应用框架", "13-工程化")),
    "生产级 RAG":       (("08-RAG体系",), ("11-评测与可观测/10-评测工程最佳实践.md",), ("09-Agent体系", "14-垂直领域")),
    "客服 Agent":       (("09-Agent体系", "06-应用开发"), ("09-Agent体系/07-Agent安全.md", "11-评测与可观测/06-Agent评测.md"), ("13-工程化与基础设施",)),
    "行业数字员工平台": (("13-工程化与基础设施", "14-垂直领域"), ("11-评测与可观测/09-可观测工具链-Langfuse-Phoenix-OpenTelemetry.md",), ("05-推理与部署",)),
}

# 账 D：八项交叉 + 四块汇合 + 问题归属
CROSS = (
    "推理与 Serving", "评测体系", "数据工程", "Context / 长上下文",
    "成本模型", "安全基础", "微调基本盘", "MCP / 工具协议",
)
MERGE = ("推理成本", "评测", "数据", "安全")
OWNER = {
    "延迟降不下来该量化到几 bit": "共同",
    "RM 奖励被格式捷径劫持":       "算法",
    "检索没命中要不要拒答":         "应用",
    "换模型后要不要重跑评测":       "共同",
    "要不要为客服场景微调模型":     "共同",
    "MCP 工具的 schema 怎么设计":   "应用",
}


def owner(question):
    """账 D：问题 -> 该由谁答（算法 / 应用 / 共同）。"""
    return OWNER.get(question, "未知")


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

    print("16-02 §18.2/§18.4 应用工程师路线（纯 stdlib / 零网络 / 零随机）")

    # ---------- 账 A ----------
    print("\n[账 A] 六支应用知识树（大纲 1）")
    check("支数", len(TREE), 6)
    check("支序", [t[0] for t in TREE],
          ["基础", "知识工程", "框架层", "Agent", "工程化", "业务"])
    check("核心支恰两支", [t[0] for t in TREE if t[1]], ["知识工程", "Agent"])
    check("框架层不是核心支", dict(TREE)["框架层"], False)
    check("支名唯一", len({t[0] for t in TREE}), 6)
    W("账A")

    # ---------- 账 B ----------
    print("\n[账 B] 四级路线 · 项目形态逐级升级（大纲 2）")
    check("级别序列", [l[0] for l in LEVELS], ["初级", "中级", "高级", "专家"])
    uppers = [l[1] for l in LEVELS]
    check("年限上界严格递增", uppers == sorted(uppers) and len(set(uppers)) == 4, True)
    check("里程碑序列（MVP→RAG→Agent→平台）", [l[3] for l in LEVELS],
          ["企业知识库 MVP", "生产级 RAG", "客服 Agent", "行业数字员工平台"])
    check("每级里程碑非空", all(l[2] for l in LEVELS), True)
    check("每级至少 3 个仓库章节锚点", all(l[4] >= 3 for l in LEVELS), True)
    check("项目名与账 C 对齐", [l[3] for l in LEVELS], list(PROJECTS.keys()))
    W("账B")

    # ---------- 账 C ----------
    print("\n[账 C] 项目 → 所需章节清单（大纲 3）")
    check("项目 4 个", len(PROJECTS), 4)
    check("每项目都有主章", all(v[0] for v in PROJECTS.values()), True)
    check("每项目必配评测/护栏", all(v[1] for v in PROJECTS.values()), True)
    check("每项目都有可延后项", all(v[2] for v in PROJECTS.values()), True)
    check("客服 Agent 必配 2 项护栏/评测", len(PROJECTS["客服 Agent"][1]), 2)
    check("生产级 RAG 主章只 08", PROJECTS["生产级 RAG"][0], ("08-RAG体系",))
    check("MVP 不要求 13 工程化", "13-工程化与基础设施" in PROJECTS["企业知识库 MVP"][0], False)
    check("平台级必配可观测", "11-评测与可观测/09-可观测工具链-Langfuse-Phoenix-OpenTelemetry.md"
          in PROJECTS["行业数字员工平台"][1], True)
    W("账C")

    # ---------- 账 D ----------
    print("\n[账 D] 交叉地带预告（大纲 4 · 6 例归属判定）")
    check("八项交叉", len(CROSS), 8)
    check("四块汇合地带", MERGE, ("推理成本", "评测", "数据", "安全"))
    exp6 = (
        ("延迟降不下来该量化到几 bit", "共同"),
        ("RM 奖励被格式捷径劫持",       "算法"),
        ("检索没命中要不要拒答",         "应用"),
        ("换模型后要不要重跑评测",       "共同"),
        ("要不要为客服场景微调模型",     "共同"),
        ("MCP 工具的 schema 怎么设计",   "应用"),
    )
    for i, (q, exp) in enumerate(exp6, 1):
        check("[%d] %s" % (i, q), owner(q), exp)
    check("共同项 3 例", sum(1 for _, e in exp6 if e == "共同"), 3)
    check("未知问题不臆断", owner("一个没登记过的问题"), "未知")
    W("账D")

    print("\nPASS %d  FAIL %d" % (p, f))
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())
