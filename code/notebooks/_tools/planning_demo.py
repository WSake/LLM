#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""09-Agent体系 · 03-Planning与任务分解（知识地图 §9.3 + §9.2 计划节点 + §9.12 终结/评测）探针：
『规划』——把 02 账 D 埋下的「计划调用 1<3<4」展开成完整一课：
三种计划方式 × 任务分解树 × 计划质量评测 × 重规划成本。
量四本账：
  A 计划形态执行账 —— 静态 vs 动态两个场景 × 一次性计划/逐步计划：每种形态报
              计划调用数/执行动作数/白做步/显式重规划（静态省思考、动态少浪费）；
              外加『一次性+条件分支』=确定性任务对冲变化的低级外挂（白做 0 重规划 0）
  B 任务分解粒度账 —— 平铺(flat) vs 分层(tree) 分解同一 8 单元任务：分层用
              3 次子级编排税换『故障隔离』——flat 在中间某叶阻塞波及后面 2 叶白排，
              tree 只重拆该叶子分支（波及 0）
  C 计划质量评测账 —— §9.3 规划质量评测（PlanBench 思路）：四指标（可执行性/
              前提满足/覆盖/冗余）评价 1 个好计划 + 3 个坏计划（漏前提/幻想动作/
              冗余步），各恰被抓出 1 个缺陷——计划幻觉在执行前抓比执行后便宜
  D 规划成本与重规划账 —— 承接账A：重规划预算 cap=2，第 3 次重规划被拒→转人工
              （挂靠 02 账B 步骤上限 bean 思想）；工具账：Plan-and-Execute(2023)/
              LLM Compiler(2024)/代码即计划(天然形态)/PlanBench(2024)=要素事实
口径：任务/分解/计划/缺陷注入 = 作者按地图 §9.3 预置（仿真=非真实 LLM 运行的 Agent）；
     A-D 全部统计 = 本机确定性计算（纯 stdlib · 零随机 · 零网络 · stdout 逐字节可复现；
     wall-clock 只进 stderr）；规划工具/年份 = 要素事实（非本机实测）。
"""
import sys
from time import perf_counter

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ================= 账 A：三种计划形态 × 两场景 =================
SUBTASKS3 = [("s1", "查可行日期"), ("s2", "订酒店"), ("s3", "租车")]
BLOCKED = "s2"  # 动态场景：s2 处条件变化（酒店满房→换城市）


def top_down(subtasks, blocked=None):
    """一次性计划（Plan-and-Execute）：1 次计划列全部子任务，顺序执行；
    遇 blocked 则显式重规划一次，重规划后重新执行剩余步骤。
    返回 (计划调用, 执行动作数, 白做步, 显式重规划)。"""
    plan_calls, exec_actions, wasted, replans = 1, 0, 0, 0
    for i, (sid, _) in enumerate(subtasks, 1):
        exec_actions += 1
        if blocked and sid == blocked:
            wasted += 1          # 这一步做了但条件变了 = 白做
            plan_calls += 1      # 显式重规划
            replans += 1
            exec_actions += (len(subtasks) - i)  # 重规划后再执行剩余
            return plan_calls, exec_actions, wasted, replans
    return plan_calls, exec_actions, wasted, replans


def stepwise(subtasks, blocked=None):
    """逐步计划（ReAct 式）：每步行动前一次思考（=计划调用）；阻塞发生时下一步
    思考自然改口径，无需显式重规划。返回同 top_down。"""
    plan_calls = 0
    exec_actions = 0
    for _sid, _ in subtasks:
        plan_calls += 1
        exec_actions += 1
    return plan_calls, exec_actions, 0, 0  # blocked 时也是局部思考，不记显式重规划


def top_down_branch(subtasks, blocked=None):
    """一次性计划 + 条件分支：把不确定性（若 s2 满房→换城市）预先写进计划，
    遇 blocked 走分支，不再重规划。返回同 top_down。"""
    plan_calls, exec_actions, wasted, replans = 1, 0, 0, 0
    for _sid, _ in subtasks:
        exec_actions += 1
    return plan_calls, exec_actions, wasted, replans  # 分支吞掉了风险


def expA():
    print("=" * 72)
    print("[账 A] 计划形态执行账：静态 vs 动态两场景 × 一次性/逐步/一次性+分支")
    print("=" * 72)
    rows = [
        ("一次性计划(P&E)", "静态", *top_down(SUBTASKS3)),
        ("一次性计划(P&E)", "动态", *top_down(SUBTASKS3, BLOCKED)),
        ("逐步计划(ReAct)", "静态", *stepwise(SUBTASKS3)),
        ("逐步计划(ReAct)", "动态", *stepwise(SUBTASKS3, BLOCKED)),
        ("一次性+条件分支", "动态", *top_down_branch(SUBTASKS3, BLOCKED)),
    ]
    print(f'  {"形态":<16}{"场景":<6}{"计划调用":>8}{"执行动作":>8}{"白做步":>8}{"显式重规划":>10}')
    for form, scene, pc, ea, wa, rp in rows:
        print(f"  {form:<16}{scene:<6}{pc:>8}{ea:>8}{wa:>8}{rp:>10}")
    pe_s, pe_d, re_s, re_d, peb_d = rows
    assert (pe_s[2:]) == (1, 3, 0, 0)
    assert (re_s[2:]) == (3, 3, 0, 0)
    assert (pe_d[2:]) == (2, 3, 1, 1)
    assert (re_d[2:]) == (3, 3, 0, 0)
    assert (peb_d[2:]) == (1, 3, 0, 0)
    print("  → 静态：一次性 1 次计划 vs 逐步 3 次思考——确定性任务省 2 次思考（02 账D 的 1<3）")
    print("  → 动态：一次性 白做 1 步 + 显式重规划 1 次；逐步 0 白做 0 显式重规划（重规划摊进每步思考）")
    print("  → 一次性+条件分支：预先把『若 s2 满房→换城市』写进计划 → 白做 0 重规划 0 = 确定性任务对冲变化的低级外挂")
    print("  → 断言：静态 1<3 定理 · 动态 P&E(白做1/重规划1) vs ReAct(0/0) · 分支计划吞掉风险")


# ================= 账 B：任务分解粒度 × 故障隔离 =================
def flat_plan(units, fail_at):
    """平铺：depth=1 一次 8 步计划，顺序执行。fail_at 处阻塞 → 其后全部白排。
    返回 (计划事件数, 编排税, 阻塞波及叶子数)。"""
    events = 1 + units            # 1 计划 + 8 执行
    spread = units - fail_at      # 失败位置之后的叶子全白排
    return events, 0, spread


def tree_plan(root=1, subs=3, leaves=8, fail_at=6):
    """分层：root → 3 子 → 8 叶。失败叶子只重拆该分支，不波及兄弟。
    返回 (计划事件数, 编排税, 阻塞波及叶子数)。"""
    events = root + subs + leaves  # 根计划 1 + 子级编排 3 + 叶执行 8
    spread = 0                     # 故障隔离在叶子分支内
    return events, subs, spread


def expB():
    print()
    print("=" * 72)
    print("[账 B] 任务分解粒度账：同一 8 单元任务 · 平铺 vs 分层 · 中间某叶(第6)阻塞")
    print("=" * 72)
    f_e, f_tax, f_s = flat_plan(8, 6)
    t_e, t_tax, t_s = tree_plan()
    print(f"  平铺 flat ：计划事件 {f_e}（1 计划+8 执行）· 编排税 {f_tax} · 第 6 叶阻塞 → 波及后面 2 叶白排")
    print(f"  分层 tree ：计划事件 {t_e}（根计划 1+子级编排 3+叶执行 8）· 编排税 {t_tax} · 第 6 叶阻塞 → 波及 {t_s}（只重拆该分支）")
    assert (f_e, f_tax, f_s) == (9, 0, 2)
    assert (t_e, t_tax, t_s) == (12, 3, 0)
    print(f"  → 用 {t_tax} 次子级编排换『故障隔离 {f_s - t_s} 个叶子』——分层的代价是编排税，收益是局部重规划（§9.3 分层计划）")
    print("  → 断言：flat 波及 2 vs tree 波及 0 · 编排税 0 vs 3 —— 深度买隔离")


# ================= 账 C：计划质量评测（PlanBench 思路） =================
# 每个动作: (名字, 是否可执行, 产出集合, 前提集合)
ACTIONS = {
    "下载数据":      ("下载数据",        True,  {"原始文件"},     set()),
    "解析数据":      ("解析数据",        True,  {"结构化表格"},   {"原始文件"}),
    "按列统计":      ("按列统计",        True,  {"统计结果"},     {"结构化表格"}),
    "生成报告":      ("生成报告",        True,  {"报告段"},       {"统计结果"}),
    "HyperDB查询":   ("调用 HyperDB 幻方索引 API 查询", False, {"索引结果"}, {"原始文件"}),
    "打印日志":      ("打印日志到控制台", True,  {"日志"},         set()),
}
GOAL = {"报告段"}
PLAN_GOOD = ["下载数据", "解析数据", "按列统计", "生成报告"]
PLAN_SKIP = ["下载数据", "按列统计", "生成报告"]            # 漏了解析（跳过前提）
PLAN_FANTASY = ["下载数据", "HyperDB查询", "生成报告"]     # 幻想动作（库里没有）
PLAN_REDUNDANT = ["下载数据", "解析数据", "打印日志", "按列统计", "生成报告"]  # 冗余步


def eval_plan(plan):
    """四指标：可执行性 / 前提满足 / 覆盖 / 冗余。返回 (是否通过, 缺陷列表)。"""
    defects = []
    produced = set()
    for name in plan:
        if name not in ACTIONS:
            defects.append("不可执行")
            return False, defects
        _act_name, executable, produces, requires = ACTIONS[name]
        if not executable:
            defects.append("不可执行")
            return False, defects
        if not requires.issubset(produced):
            defects.append("前提不满足")
            return False, defects
        produced |= produces
    if not GOAL.issubset(produced):
        defects.append("覆盖不足")
        return False, defects
    # 冗余：产出与目标任务及后续需求都无关
    downstream_reqs = set()
    for name in plan:
        downstream_reqs |= ACTIONS[name][3]
    redundant = []
    for name in plan:
        produces = ACTIONS[name][2]
        if produces and not (produces & (GOAL | downstream_reqs)):
            redundant.append(name)
    if redundant:
        defects.append(f"冗余步({len(redundant)})")
    return not defects, defects


def expC():
    print()
    print("=" * 72)
    print("[账 C] 计划质量评测账（§9.3 PlanBench 思路）：可执行性·前提·覆盖·冗余 → 四计划")
    print("=" * 72)
    cases = [
        ("好计划（全步骤齐全）", PLAN_GOOD),
        ("坏计划①漏前提（跳过解析）", PLAN_SKIP),
        ("坏计划②幻想动作（HyperDB）", PLAN_FANTASY),
        ("坏计划③冗余步（打印日志）", PLAN_REDUNDANT),
    ]
    results = []
    for label, plan in cases:
        ok, defects = eval_plan(plan)
        print(f"  {label}: {'✅ 全指标通过' if ok else '❌ 缺陷=' + str(defects)}")
        results.append((ok, defects))
    assert results[0] == (True, [])
    assert results[1][1] == ["前提不满足"]
    assert results[2][1] == ["不可执行"]
    assert results[3][1] == ["冗余步(1)"]
    print("  → 四指标各抓一类缺陷：漏前提→前提不满足 · 幻想动作→不可执行 · 无用步→冗余(1)")
    print("  → 好计划 0 缺陷：可执行+前提链闭合+覆盖目标+0 冗余=『看起来像计划』不等于『能执行的计划』")
    print("  → 断言：3 个坏计划各被恰抓出 1 类缺陷 · 好计划全过（计划幻觉执行前抓比执行后便宜 = PlanBench 价值）")


# ================= 账 D：重规划预算 × 工具台账 =================
REPLAN_CAP = 2                       # 挂靠 02 账B『步骤上限』：重规划次数也封顶
TOOLS = [
    # (工具/方法, 年份, 一句话机制, 定位)
    ("Plan-and-Execute", 2023, "先列全步骤再执行", "三种方式之一（一次性）"),
    ("LLM Compiler", 2024, "把计划编译成可执行步骤图", "图上计划·显式控制流"),
    ("代码即计划", None, "代码本身就是可执行计划的天然形态", "Coding Agent 首选"),
    ("PlanBench", 2024, "规划质量评测基准（可执行/前提/覆盖）", "规划能力测量"),
]


def replan_budget(need_replans, cap):
    """重规划预算：need_replans 次阻塞需重规划，超过 cap 的第 cap+1 次被拒→转人工。"""
    accepted = min(need_replans, cap)
    if need_replans > cap:
        return accepted, "转人工", f"第 {need_replans - cap} 次重规划被拒（预算 cap={cap}）"
    return accepted, "完成", f"{need_replans} 次重规划都在预算内"


def expD():
    print()
    print("=" * 72)
    print("[账 D] 规划成本与重规划账：重规划预算 cap=2 · 规划工具台账（要素事实）")
    print("=" * 72)
    a1, r1, w1 = replan_budget(2, REPLAN_CAP)
    a3, r3, w3 = replan_budget(3, REPLAN_CAP)
    print(f"  重规划预算 cap={REPLAN_CAP}：需 2 次 → 接受 {a1} 次 · 终局【{r1}】（{w1}）")
    print(f"                  需 3 次 → 接受 {a3} 次 · 终局【{r3}】（{w3}）——重规划也受步骤上限约束（接 02 账B 护栏①）")
    assert (a1, r1) == (2, "完成")
    assert (a3, r3) == (2, "转人工")
    print(f"  工具台账（要素事实=非本机实测）——{' | '.join(f'{t}({y})={d}' for t, y, _, d in TOOLS)}")
    assert sum(1 for _t, y, _m, _d in TOOLS if y in (2023, 2024)) == 3
    print("  → 断言：重规划预算 2 兜底（超 1 次→转人工）· 工具台账 2023/2024 共 3 项 + 代码即计划")
    print("  → 四条路线一句话：确定性任务→一次性省思考 · 开放/会变→逐步少浪费 · 复杂大任务→分层买隔离 · 要可靠→计划先过评测闸（账C）")


def main():
    print("=" * 72)
    print("planning_demo：09-Agent体系 · 03-Planning与任务分解（想几次才做·拆多细才稳·计划可信吗）")
    print("    关联：地图 §9.3 ｜ 前提 02-Agent-Loop（护栏/停止判据）→ 计划质量=终局可靠的地基")
    print("=" * 72)
    print("[0] 口径：任务/分解/计划/缺陷注入 = 作者按地图 §9.3 预置（仿真=非真实 LLM 运行的 Agent）；")
    print("    A-D 全部统计 = 本机确定性计算（纯 stdlib·零随机·零网络）")
    expA()
    expB()
    expC()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 计划形态×场景 / B 分解粒度×隔离 / C 计划质量评测 / D 重规划×工具）")
    print("  A 静态 一次性 1 计划 < 逐步 3 思考（省 2 次）· 动态 一次性白做1/重规划1 vs 逐步 0/0 · 分支计划吞掉风险")
    print("  B 分层用编排税 3 换故障隔离 2 个叶子（flat 波及 2 vs tree 0）——深度买隔离")
    print("  C 四指标评测：好计划 0 缺陷 · 漏前提/幻想动作/冗余步各被抓出——计划幻觉执行前抓便宜")
    print("  D 重规划预算 cap=2（超 1 次→转人工）· 工具 2023 P&E / 2024 LLM Compiler·PlanBench / 代码即计划")
    print("一句话：Plan 有三种形态（一次性/逐步/分层）、两种粒度（深/浅）、一把尺（先过评测闸）——")
    print("         计划不是『列步骤』，是『把不确定性编进结构 + 用四指标证明它能被执行』")
    print("done · 一键复现：python code/notebooks/_tools/planning_demo.py")


if __name__ == "__main__":
    main()
    W(f"[planning_demo] wall-clock {perf_counter() - _t0:.3f} s")
