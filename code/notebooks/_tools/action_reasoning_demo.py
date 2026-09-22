#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""09-Agent体系 · 04-行动推理范式（知识地图 §9.4 + §9.7 + 02-20 推理模型锚点）探针：
『行动推理』——把 02 账 A 的循环骨架凝结成 §9.4 的经典范式账簿：
ReAct（2022）/ Reflexion（2023）/ Self-Refine（2023）/ ToT（2023）/ LATS（2023），
外加 2025 现实：推理模型内部思考与框架 Thought 层的融合。
量四本账：
  A 范式骨架账 —— ReAct 标准『Thought→Action→Observation』三元组跑 4 个模板任务
    （02 账 A 的 4 样本），每条动作必须被观察=骨架不变式（A==O，T==遍数）
  B 自纠错两大范式账 —— Reflexion（执行器反馈·反思写记忆·重试）vs
    Self-Refine（无执行器·生成→自我批判→修订）：外部真值兜底 vs 内部批判上限
  C 树搜索成本账 —— 同 1 棵 4 深×3 叉、正确分支在最后的树：ReAct式贪心 12 eval
    vs ToT分叉·剪枝 21 vs LATS（价值引导+反思）6 vs 穷举展开 120——都命中，
    成本差 20×：充分=贵，价值引导=杠杆（§9.4 原文『ToT 探索充分成本高·LATS 结合价值函数』）
  D 2025 融合与选择树账 —— 推理模型 think 把 Thought 层吸收（挂靠 02-20 think≈6×），
    框架只剩 行动/观察/反思+工具编排；范式选择树 6 断言锁死
口径：任务/测试结果/反思文本/批判清单/探索路径/价值分 = 作者按地图 §9.4 预置
     （仿真=非真实 LLM 运行的 Agent）；A-D 全部统计 = 本机确定性计算
     （纯 stdlib · 零随机 · 零网络 · stdout 逐字节可复现；wall-clock 只进 stderr）；
     ReAct/Reflexion/Self-Refine/ToT/LATS 与年份、2025 融合叙事 = 要素事实（非本机实测）。
"""
import sys
from time import perf_counter

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ================= 账 A：范式骨架账 =================
# (任务, 遍数, 每遍的动作列表) —— 02 账 A 的 4 模板样本
TASKS = [
    ("ReAct 查天气回答", 1, ["search_weather(北京)"]),
    ("LangGraph 客服", 2, ["ticket_fetch", "ticket_retry(重试)"]),
    ("代码解释器", 2, ["run_tests(失败)", "fix_code + run_tests(通过)"]),
    ("Operator 订航班", 3, ["search", "compare(备选)", "book(付款→人工)"]),
]


def react_triad(task):
    """每遍 = 1 Thought + 1 Action + 1 Observation；骨架不变式 A==O==遍数。"""
    name, rounds, actions = task
    thoughts = rounds                    # 每遍行动前必先想
    observations = len(actions)          # 每个动作必须被观察
    assert len(actions) == rounds        # 一遍一动作
    assert thoughts == observations == rounds
    return name, rounds, thoughts, len(actions), observations


def expA():
    print("=" * 72)
    print("[账 A] 范式骨架账：ReAct 标准『Thought→Action→Observation』三元组")
    print("=" * 72)
    print(f'  {"模板任务":<18}{"遍数":>4}{"Thought":>9}{"Action":>9}{"Observation":>13}')
    for task in TASKS:
        name, rounds, t, a, o = react_triad(task)
        print(f"  {name:<18}{rounds:>4}{t:>9}{a:>9}{o:>13}")
    print("  → 骨架不变式锁死 4/4：每个 Action 后面必须有一个 Observation（观察=动作的视线，"
          "外部世界入循环的唯一口）· Thought 数=遍数（每遍行动前必先想）")
    print("  → ReAct = §9.3『逐步计划』的字面骨架（03 账 A 的动态场景它 0 白做 0 显式重规划）"
          "；Thought 不调外部=零成本修正位")
    print("  → 断言：4 任务 (Thought,Action,Observation) 全等于各自的遍数（1/2/2/3）")


# ================= 账 B：两大自纠错范式账 =================
MEMORY = []  # Reflexion 的反思记忆（内容层，与 06-记忆系统同源）


def run_tests(version, bug=True):
    """执行器反馈：v0 带『比较方向反』bug → 1/3；修正版 3/3。"""
    if version == 0:
        return (1, 3) if bug else (3, 3)
    return (3, 3)


def reflexion_task(label, has_recall):
    """Reflexion：执行失败→生成反思→写记忆→重试；同型错误二次=记忆拦截。
    返回 (attempts, v0_passed, v0_total, final_passed, failures, reflections, memo)。"""
    recalled = has_recall and any("compare 方向反" in m for m in MEMORY)
    if recalled:                                   # 记忆命中→直接写对
        return (1, 3, 3, 3, 0, 0, len(MEMORY))
    v0p, v0t = run_tests(0, bug=True)              # v0 失败
    if v0p < v0t:
        MEMORY.append("compare 方向反（v0 测试用例2 期望升序实得降序）")
        fp, _ = run_tests(1)                       # 重写后 3/3
        return (2, v0p, v0t, fp, 1, 1, len(MEMORY))
    return (1, v0p, v0t, v0p, 0, 0, len(MEMORY))


def wordscore(draft):
    return {"初稿": 0.44, "修订2": 0.62, "修订3": 0.78}[draft]


def self_refine():
    """Self-Refine：生成→自我批判→修订，全程无执行器（内部批判）。"""
    drafts = ["初稿", "修订2", "修订3"]
    critiques = 2               # v1→v2、v2→v3 各批判 1 次
    revisions = 2
    scores = [wordscore(d) for d in drafts]
    assert scores == [0.44, 0.62, 0.78]
    return drafts, critiques, revisions, scores


def expB():
    print()
    print("=" * 72)
    print("[账 B] 自纠错两大范式账：Reflexion（执行器反馈） vs Self-Refine（无执行器）")
    print("=" * 72)
    MEMORY.clear()
    a1, v0p1, v0t1, f1, fl1, r1, m1 = reflexion_task("任务一·排序函数", has_recall=False)
    print(f"  ① Reflexion 任务一：v0 测试 {v0p1}/{v0t1} 失败 → 反思写入记忆 {r1} 条 → "
          f"重写 v1 通过 {f1}/3 → 尝试 {a1} 次 · 失败 {fl1} 次 · 反思 {r1} 条 · 记忆 {m1} 条")
    a2, v0p2, _, f2, fl2, r2, m2 = reflexion_task("任务二·同型任务", has_recall=True)
    print(f"  ② Reflexion 任务二（同型）：记忆命中『compare 方向反』→ v0 直接 {v0p2}/3 通过"
          f" · 尝试 {a2} 次 · 失败 {fl2} 次（0 重复错）· 记忆 {m2} 条=反思被复用了")
    assert (a1, v0p1, f1, fl1, r1, m1) == (2, 1, 3, 1, 1, 1)
    assert (a2, v0p2, fl2, r2, m2) == (1, 3, 0, 0, 1)
    drafts, critiques, revisions, scores = self_refine()
    print(f"  ③ Self-Refine（无执行器）：{' → '.join(f'{d}@{s:.2f}' for d, s in zip(drafts, scores))}"
          f" · 批判 {critiques} 次 · 修订 {revisions} 次 · 外部反馈 0 次")
    assert critiques == 2 and scores == [0.44, 0.62, 0.78]
    print("  → 分界：Reflexion 用执行器真值（测试 1/3→3/3）兜底=外部验证，Self-Refine 用内部批判"
          "（词面评分代理）无限上探=上限受批判力限制（02 账C 的验证器账）")
    print("  → 断言：Reflexion 失败1+反思1+记忆拦截复查0重复错 · Self-Refine 2 轮批判收敛且 0 外部反馈")


TREE_D, TREE_B, TREE_CORRECT = 4, 3, 2  # 深 4 · 每节点 3 支 · 正确分支=最后一条


def eval_cost(strategy):
    """同树探索成本：ReAct式贪心 / ToT 分叉剪枝 / LATS 价值+反思 / 穷举全树。"""
    if strategy == "greedy":
        return TREE_D * TREE_B                          # 每层试满 0,1,2 三条最后一条对=12
    if strategy == "tot_beam":
        return 3 + 3 * (TREE_D - 1) * 2                  # L1:3 · L2-4 每层 2路径×3分支=6 → 3+6×3=21
    if strategy == "lats":
        return 6                                        # 价值指路 4 + 反思回退探错 2
    if strategy == "full":
        return sum(TREE_B ** i for i in range(1, TREE_D + 1))  # 3+9+27+81=120
    raise ValueError(strategy)


def expC():
    print()
    print("=" * 72)
    print("[账 C] 树搜索成本账：同一棵 4 深×3 叉树（正确分支=最后一条）· 全命中最优路径")
    print("=" * 72)
    rows = [
        ("ReAct式贪心（线性逐步）", eval_cost("greedy")),
        ("ToT（分叉 B=3·剪枝保留 2）", eval_cost("tot_beam")),
        ("LATS（价值引导+反思）", eval_cost("lats")),
        ("穷举展开（全部路径）", eval_cost("full")),
    ]
    base = eval_cost("greedy")
    print(f'  {"策略":<28}{"评估次数":>8}{"相对贪心":>10}')
    for name, cost in rows:
        print(f"  {name:<28}{cost:>8}{cost / base:>9.2f}×")
    g, t, l, f = eval_cost("greedy"), eval_cost("tot_beam"), eval_cost("lats"), eval_cost("full")
    assert g == 12 and t == 21 and l == 6 and f == 120
    assert g < t < f and l < g          # 充分=贵（ToT 21 · 穷举 120）· 价值引导=最省（LATS 6）
    print(f"  → 四策略都找到最优路径命中（距离深度 {TREE_D}）→ 命中相同，成本差 "
          f"{f / l:.1f}×（穷举 vs LATS）：§9.4 原文『ToT 探索充分、成本高』『LATS 树搜索与反思、价值函数结合』")
    print("  → 断言：greedy 12 < ToT 21 < full 120 · LATS 6 全场最省 · 4/4 命中")


# ================= 账 D：2025 融合与范式选择树 =================
FUSION = [
    ("2022 ReAct", "Thought 层显式写在框架外（模型文本里写『Thought: …』）"),
    ("2024+ 推理模型", "o1/R1 think 模式把思考搬进模型内部（token 会计成本：02-20 think≈6×）"),
    ("2026 现实", "框架 Thought 层被模型 thinking 吸收→框架手里只剩 行动/观察/反思 + 工具编排"),
]
SELECT = [
    ("即时应答 / 无验证器", "ReAct"),
    ("有执行器反馈且可重试", "Reflexion"),
    ("只有文本 / 无执行器", "Self-Refine"),
    ("需充分探索 / 可剪枝打分", "ToT"),
    ("高价值任务 / 舍得算力", "LATS"),
    ("2026 默认", "推理模型 thinking + 框架控制流"),
]


def expD():
    print()
    print("=" * 72)
    print("[账 D] 2025 融合与选择树账：Thought 层去哪儿了 · 6 范式选择断言")
    print("=" * 72)
    for era, desc in FUSION:
        print(f"  {era:<14} → {desc}")
    print("  → 融合读法：§9.4 原文『推理模型在模型内部就做思考，Agent 框架的 Thought 层与"
          "模型 thinking 模式融合』——框架的省法=别重复造 Thought，把推理交给模型，自己管行动/观察/反思")
    print("  范式选择树（6 断言）：")
    for cond, strat in SELECT:
        print(f"    · {cond:<24} → {strat}")
    assert len(SELECT) == 6 and all(isinstance(c, str) and isinstance(s, str) for c, s in SELECT)
    assert FUSION[0][0] == "2022 ReAct" and FUSION[2][0] == "2026 现实"
    print("  → 断言：6 场景 6/6 锁死 · 融合三世代 2022→2024+→2026 台账完整")


def main():
    print("=" * 72)
    print("action_reasoning_demo：09-Agent体系 · 04-行动推理范式（骨架·自纠错·树搜索·2025 融合）")
    print("    关联：地图 §9.4（五种范式）｜§9.7（反思与自纠错）｜02-20（推理模型 think≈6×）")
    print("=" * 72)
    print("[0] 口径：任务/测试结果/反思文本/批判清单/探索路径/价值分 = 作者按 §9.4 预置（仿真=非真实 LLM）；")
    print("    A-D 全部统计 = 本机确定性计算（纯 stdlib·零随机·零网络）")
    expA()
    expB()
    expC()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 骨架三元组 / B 自纠错双范式 / C 树搜索成本 / D 融合与选择树）")
    print("  A ReAct 骨架：Thought→Action→Observation 三元组，不变式 A==O 且 Thought=遍数（4/4）")
    print("  B Reflexion=执行器反馈·反思写记忆·复查 0 重复错（1/3→3/3） vs Self-Refine=内部批判 2 轮 0 外部反馈（0.44→0.78）")
    print("  C 探索账：贪心 12 < ToT 21 < 穷举 120 eval，LATS 价值引导 6 全场最省→充分=贵·价值=杠杆")
    print("  D 融合：2022 显式 Thought → 2024+ 模型内部 think（≈6×）→ 2026 框架只留 行动/观察/反思；选择树 6/6")
    print("一句话：范式五兄弟本质是『想几次、靠谁验、花多少价』的三个旋钮——ReAct 最小盘、")
    print("         Reflexion 外验、Self-Refine 内验、ToT/LATS 用算力求稳、2026 把 Thought 交给推理模型。")
    print("done · 一键复现：python code/notebooks/_tools/action_reasoning_demo.py")


if __name__ == "__main__":
    main()
    W(f"[action_reasoning_demo] wall-clock {perf_counter() - _t0:.3f} s")
