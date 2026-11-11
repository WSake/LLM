# -*- coding: utf-8 -*-
"""16-03 交叉知识地带 —— 八项交叉表 + 6 例「问题→谁答」判定 + 四块汇合地带。

账 A  八项交叉领域表齐 + 每项深度刻度（共同底线）
账 B  6 例问题归属判定（主力手算法/应用/共同）
账 C  四块汇合地带 = 问题被归类的标签大写集合
"""
import sys
import time

_WSTART = time.perf_counter()

# (领域, 算法用途, 应用用途, 共同底线, 本仓库落点)
CROSS = (
    ("推理与 Serving", "推断真实训练成本", "选型压测容量规划", "会读显存/吞吐账", "05-推理与部署"),
    ("评测体系",       "用评测迭代模型",   "用评测迭代系统",   "评测集先行回归门禁", "11-评测与可观测"),
    ("数据工程",       "训练数据清洗合成", "知识库解析分块",   "数据KPI与模型KPI分开", "04-训练体系"),
    ("Context/长上下文", "长上下文训练评估", "上下文工程塞砍引", "知context是钱也是毒", "02-核心原理"),
    ("成本模型",       "训练成本预算",     "推理+标注成本",   "Token换算成钱三份账", "11-评测与可观测"),
    ("安全基础",       "模型责任偏越狱护栏", "产品责任注入权限", "先读威胁模型总览", "12-安全"),
    ("微调基本盘",     "主业何时全参PEFT", "判断要不要微调",   "微调决策树不开微调", "04-训练体系"),
    ("MCP/工具协议",   "评估工具调用能力", "直接使用选型护栏", "会读工具failure原因", "07-应用框架"),
)

# 四块汇合地带（must queue 照顾顺序）
MELT_ZONES = ("推理成本", "评测", "数据", "安全")

# 6 例归属：(例id, 问题, 主力手, 判据)
CASES = (
    ("A", "推理延时不达标，要不要换量化/小模型？", "算法", "卡点在模型侧与部署取舍"),
    ("B", "RAG 检索召回差，硬伤是分块还是 Embedding？", "应用", "卡点在知识工程"),
    ("C", "训练集涨了 4 倍，为什么评测还持平？", "算法", "卡点在数据配比与训练"),
    ("D", "上线后幻觉被反复投诉：是评测先动还是护栏先动？两侧都动", "共同", "评测门禁+产品护栏两线都动"),
    ("E", "长文档总结丢信息：context 塞太多还是索引 bug？两侧都查", "共同", "加载分块+压缩策略两侧查"),
    ("F", "微调后 A 涨 B 崩，怎么办？", "算法", "过拟合/配比，应用回传证据"),
)


def owner_of(problem):
    """把问题归类：算法/应用/共同。共同信号词优先，再按主题词。"""
    if any(k in problem for k in ("两侧", "同时", "都动", "都查", "双线", "两线")):
        return "共同"
    if any(k in problem for k in ("量化", "小模型", "部署", "推理延时", "选型")):
        return "算法"
    if any(k in problem for k in ("分块", "索引", "知识库", "检索", "上下文工程")):
        return "应用"
    if any(k in problem for k in ("训练集", "预训练", "微调", "过拟合", "数据配比", "评测", "幻觉", "护栏")):
        return "算法"
    return "共同"  # 兜底


def depth_note(domain):
    for d, _, _, note, _ in CROSS:
        if d == domain:
            return note
    return None


def run():
    ok, total = 0, 0

    def check(name, got, exp):
        nonlocal ok, total
        total += 1
        if got == exp:
            ok += 1
            print("  %-46s PASS" % name)
        else:
            print("  %-46s FAIL(期望 %r 实得 %r)" % (name, exp, got))

    print("\n[账 A] 八项交叉表齐")
    check("八项齐全", len(CROSS), 8)
    check("相互不重复", len(set(d for d, *_ in CROSS)) == 8, True)
    check("含推理与Serving", "推理与 Serving" in [d for d, *_ in CROSS], True)
    check("含MCP/工具协议", "MCP/工具协议" in [d for d, *_ in CROSS], True)
    check("每项有共同底线", all(note for _, _, _, note, _ in CROSS), True)

    print("\n[账 B] 6 例归属判定")
    for cid, prob, exp_owner, judge in CASES:
        got = owner_of(prob)
        check("例%s(%s)" % (cid, prob[:14] + "…"), got, exp_owner)

    print("\n[账 C] 四块汇合地带")
    owned = {owner_of(p) for _, p, *_ in CASES}
    check("三类主力手全覆盖", owned == {"算法", "应用", "共同"}, True)
    check("判定无死区", owner_of("摸不着头脑的问题"), "共同", )  # 兜底
    return ok, total


def W(label):
    print("[wall] %-24s %.4fs" % (label, time.perf_counter() - _WSTART),
          file=sys.stderr)


def main():
    W("cross_domain_zone_demo")
    ok, total = run()
    return 1 if total - ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
