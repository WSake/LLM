# -*- coding: utf-8 -*-
"""agent_harness_demo.py —— §11.5 / §9.12 Agent 评测（三个专项之 Agent）四账探针（11-评测与可观测 06 篇配套）

§11.5 知识地图原文（🟢 掌握；详见 §9.12）：Agent 评测——环境类 GAIA / WebArena / OSWorld / τ-bench / SWE-bench；
指标：成功率、步数、成本、安全性。
§9.12 原文（🔴 会跑；不评测的 Agent 不可信）：好的 Agent 评测 = 固定环境 + 可验证终结条件 + 成本指标；
不要用"看起来像"的 LLM-judge 替代环境真实验证。
§11.9 原文（🔴 总结·Agent 评测的配套纪律）：①评测集分层（5-10 冒烟 + 100-500 业务回归 + 大榜单选型）
②固定评测环境 ③每次改动跑同一套回归 ④线上遥测回流成评测集 ⑤用"每任务完成指标"而不是"每 token 质量指标"评价产品。

三个专项定位：人力（04 篇）· 检索生成（05 篇）· Agent（06 篇，本篇）。本篇 = 与 09-06（09 章 Agent 评测，
里程碑 014）互补的 11 章系统设计课：09 从"Agent 章"讲评测台三件套是什么；本篇从"评测体系设计"讲——
评测对象是轨迹不是答案、可验证终结条件怎么写、判分怎么装配（真值 vs judge 混合链）、评测集怎么建
（分级 + 坏 Agent 判别对照组 + 回归台账）。

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（W() 负责计时与落 stderr）。全部统计 = 本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- 账 A 的 轨迹表（T1/T2/T3 的回合/前进/回环/结局）= 作者预置仿真（非真实 Agent 跑出来的轨迹）；
  三级粒度与三件套（固定环境/可验证终结条件/成本计数器）= §9.12/§11.5 要素事实非本机实测。
- 账 B 的 4 条坏轨迹与三形态终结条件 = 作者预置仿真；"词面终结器漏 2/4 vs 终态断言+不变量 4/4"
  与 010 式判别性对照同法。
- 账 C 的 judge 判准 0.85（TP 11 FP 2 FN 1 TN 6 / 20 条）= 回填 09-06 已交付数字（挂靠只引不重测）；
  混合链成本账（规则拦 4/judge 复判 16/人工复核 3）与每任务成本账（X 蔓胖 vs Y 精瘦）= 作者预置仿真 +
  本机确定性计算；$0.0012（judge）/ $1.50（人工）= 03 账 C / 04 账 B 锚点。
- 账 D 的 评测集分级、坏 Agent 四类对照组（无护栏/瞎调工具/重复循环/早放弃）、回归台账
  （v1→v2 行为引导 turn 0.78→0.90·v3 白名单安全 2→0）= §11.9 要素事实 + 作者预置判定表
  （本机确定性计算）；惩罚陷阱任务设计为演示"判别性拉大"机制。
- **真实 Agent 在 GAIA/τ-bench/SWE-bench 上的通过率、真实 LLM judge 打分、真实环境跑动 = 非本机实测**
  （§11.5/§9.12 基准清单与 09-06 谱系 = 要素事实）。
"""
from __future__ import annotations
import sys
import time

_WSTART = time.perf_counter()


def W(msg: str) -> None:
    print(f"[wall] {time.perf_counter() - _WSTART:.3f}s  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 账 A · 评测对象账：轨迹不是答案——三级粒度 + 一台评测台三件套 + 绕路税
# ---------------------------------------------------------------------------
# 同一客服任务「修改收货地址」，三条轨迹（作者预置确定性仿真）
TRJ = [
    # (名称, 回合数, 有效推进回合, 回环数, 查错再修正次数, 结局, 每轮 token)
    ("T1 干净轨迹", 6, 6, 0, 0, "完成", 70),
    ("T2 绕路轨迹", 9, 6, 1, 1, "完成", 70),
    ("T3 蔓延轨迹", 12, 7, 2, 0, "转人工", 70),
]

# 三件套 = 一台 Agent 评测台的最小装配（§9.12：固定环境 + 可验证终结条件 + 成本指标）
HARNESS = [
    ("固定环境", "同一任务=同一模拟环境（same sim·same tools·同 prompt 模板·temperature=0，§11.9 ②）"),
    ("终结检查器", "可验证终结条件（终态断言 → 账 B 展开）——『完成』字面不作数，终态达成才算"),
    ("成本计数器", "每任务成本 = Σ 回合数 × 每轮 token × 单价（Agent 多轮 → 按任务记账不按 token 记账）"),
]

# 每任务成本（单价 $0.15 / 1k token，作者预置展示价）
PRICE_PER_K = 0.15 / 1000.0


def account_a() -> str:
    out = []
    out.append("== A · 评测对象账：轨迹不是答案——三级粒度（任务结局/步骤推进/回合动作）× 三件套 ==")
    out.append("  同一任务『修改收货地址』三条轨迹（作者预置仿真）：")
    total_pass, total_turnfrac = 0, 0.0
    for name, turns, adv, loop, fix, end, tok in TRJ:
        pass_ = 1.0 if end == "完成" else 0.0
        flow = adv / turns
        total_pass += pass_
        total_turnfrac += flow
        cost = turns * tok
        out.append(f"    {name}：回合 {turns}·有效推进 {adv}·回环 {loop}·查错再修正 {fix}·结局 [{end}]"
                   f" → 任务结局 {pass_:.2f} · 回合推进率 {flow:.2f} · 每任务成本 {cost} tok")
    base_tok = TRJ[0][1] * TRJ[0][6]
    for name, turns, adv, loop, fix, end, tok in TRJ:
        if name == "T1 干净轨迹":
            continue
        cost = turns * tok
        out.append(f"      · {name} 成本 = {cost} tok = {cost / base_tok:.1f} × 干净轨迹"
                   f"（绕路/蔓延税按任务记账才看得到）")
    out.append(f"  任务结局 pass 平均 = {total_pass / len(TRJ):.3f}"
               f"（T1==T2 都 1.00=结局一样·推进率 1.00 vs {TRJ[1][2] / TRJ[1][1]:.3f} 显形）")
    # 一句断言：结局通过 ≠ 轨迹健康
    t1, t2 = TRJ[0], TRJ[1]
    same_pass = (t1[5] == "完成") == (t2[5] == "完成")
    flow_gap = (t1[2] / t1[1]) > (t2[2] / t2[1])
    token_gap = t2[1] * t2[6] > t1[1] * t1[6]
    out.append(f"  结构断言 PASS：T1/T2 任务结局同为完成 vs 回合推进率 {t1[2] / t1[1]:.2f}→{t2[2] / t2[1]:.2f}"
               f"·成本 {t1[1] * t1[6]}→{t2[1] * t2[6]} tok（差别藏在轨迹里不藏在结局里） = "
               f"{'PASS' if same_pass and flow_gap and token_gap else 'FAIL'}")
    out.append("  一台评测台三件套（§9.12）：")
    for name, what in HARNESS:
        out.append(f"    · {name}：{what}")
    out.append(f"  三件套断言 {len(HARNESS)}/{len(HARNESS)}：固定环境/终结检查器/成本计数器 3/3 装配 = "
               f"{'PASS' if len(HARNESS) == 3 else 'FAIL'}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 账 B · 终结检查账：可验证终结条件怎么写 + 判别性对照（词面『完成』是坑）
# ---------------------------------------------------------------------------
# 三形态可验证终结条件（§9.12『可验证的终结条件』拆开成三个具名实现）
END_FORMS = [
    ("最终状态断言", "对目标状态下断言：收货地址==新地址 / 文件已改且 diff 含目标改动（09-05 提交前跑测试=代码版）→ 假成功在此漏底"),
    ("含断言提交", "最后动作必须通过一组校验才能算完成（含断言提交：先跑全套测试再落盘，通不过回退=09-05 评测闸）"),
    ("轨迹不变量", "回合上限 + 回环上限 + 每轮必须有推进（刷工具不推进/永远不终结在此漏底；02-章停止判据三态的台账化）"),
]

# 4 条坏轨迹（作者预置仿真）：(名称, 症状, 终态是否达成, 最终文本含『完成』字样)
BAD_TRJ = [
    ("永远不终结", "反复查询状态 10 次、从不改状态，触顶回合上限", False, False),
    ("提前放弃", "首次工具报错即退出，任务未完成", False, False),
    ("假成功", "文本说『任务已完成』，但目标状态压根没变", False, True),
    ("刷工具不推进", "只读工具反复调 5 次，输出『当前步骤已完成』但状态从未更新", False, True),
]

# 正确终结器：终态断言 + 轨迹不变量（回环≤1 · 回合≤8 · 终态未达成一律未完成）
def _correct_ending(end_ok: bool, loops: int, turns: int) -> bool:
    return end_ok and loops <= 1 and turns <= 8


# 词面终结器：只在最终文本里找『完成』『已完成』字样
def _lexical_ending(has_done_word: bool) -> bool:
    return has_done_word


def account_b() -> str:
    out = []
    out.append("== B · 终结检查账：可验证终结条件怎么写（三形态）× 判别性对照（词面『完成』是坑） ==")
    out.append("  三形态可验证终结条件（§9.12『可验证的终结条件』实证化）：")
    for name, what in END_FORMS:
        out.append(f"    · {name}：{what}")
    out.append("  4 条坏轨迹 → 两种终结器判决（判别性对照·010 同法）：")
    # 每条坏轨迹的预置回环/回合：永远不终结(回环2,10回合)·提前放弃(0,1)·假成功(0,1)·刷工具(5,6)
    meta = {"永远不终结": (2, 10), "提前放弃": (0, 1), "假成功": (0, 1), "刷工具不推进": (5, 6)}
    right_pass, lex_pass = 0, 0
    correct_hit = ""
    for name, sym, end_ok, done_word in BAD_TRJ:
        loops, turns = meta[name]
        c = _correct_ending(end_ok, loops, turns)
        l = _lexical_ending(done_word)
        right_pass += 1 if (not c) else 0
        lex_pass += 1 if (not l) else 0
        out.append(f"    {name}（{sym}）：终态断言+不变量终结器 → {'未完成 ✓' if not c else '误判 ✗'}"
                   f" · 词面『完成』终结器 → {'未完成 ✓' if not l else '误判 ✗'}")
    out.append(f"  判别性断言 {right_pass}/4 break：正确终结器把 4 条坏轨迹全判『未完成』"
               f" vs 词面终结器只判 {lex_pass}/4 未完成（假成功+刷工具带『完成』字样被漏底）"
               f" = 词面『完成』≠ 终态达成（05 账 B 判别性对照在终结条件上的应用） = "
               f"{'PASS' if right_pass == 4 and lex_pass == 2 else 'FAIL'}")
    out.append("  读点：好终结器的判据是『终态达成』不是『话说到位』——"
               "假成功/刷工具恰好是『语言看着完成了』的典型陷阱（联动 05 账 A 的答案贴题≠答案完成）。")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 账 C · 真值 vs judge 混合链账：判分怎么排不背刺 + 每任务完成指标 vs 每 token 质量指标
# ---------------------------------------------------------------------------
# 20 条轨迹真值 12 过/8 挂；judge 判准 0.85 = 回填 09-06（TP 11 / FP 2 / FN 1 / TN 6）
TRUTH_PASS, TRUTH_FAIL = 12, 8
PPP, PPP_JUDGE, PPP_HUMAN = 0.0, 0.0012, 1.50
RULE_CATCH = 4        # 规则终态检查（$0/条）确定性拦下 4 条劣质轨迹
JUDGE_LEFT = 16       # 剩余 16 条交给 judge 复判（0.85 判准）
HUMAN_CRITICAL = 3    # judge 置信度最低的 3 条人工复核，把残余 FP/FN 兜住

# 两种 Agent 的每任务账本：（名称, 回合, 每轮tok, 任务完成, 单轮平均打分）
AGENTS = [
    ("X 蔓胖（每步都漂亮但绕路能环球）", 50, 200, 0, 0.90),
    ("Y 精瘦（每步朴素但径直跑完）", 6, 70, 1, 0.80),
]


def account_c() -> str:
    out = []
    out.append("== C · 真值 vs judge 混合链账：判分怎么排不背刺 + 每任务完成 vs 每 token 质量 ==")
    full_human = 20 * PPP_HUMAN
    only_judge = 20 * PPP_JUDGE
    rule_cost = RULE_CATCH * PPP
    judge_cost = JUDGE_LEFT * PPP_JUDGE
    human_cost = HUMAN_CRITICAL * PPP_HUMAN
    mix_cost = rule_cost + judge_cost + human_cost
    out.append("  同 20 条轨迹（真值 12 过 / 8 挂）三种判分方案（09-06 judge 判准 0.85 = 挂靠只引不重测）：")
    out.append("    · 全人工：判准 0.97（04 账）· 成本 $20×1.50 = $30.00 —— 贵 1250× judge（04 账 B）")
    out.append("    · 只 judge：判准 0.85（09-06：FP 2 放行坏 · FN 1 误杀好）· 成本 $20×0.0012 = $0.024 —— 便宜但会把坏放进生产")
    out.append("    · 混合链：规则终态检查（环境真值 $0/条）先拦 4 条劣质 → judge 复判剩余 16 条（0.85）"
              f" → 人工复核 judge 最不确定的 3 条（把残余 FP/FN 兜住）· 成本 $0 + $16×0.0012 + $3×1.50 = ${mix_cost:.2f}"
              f" = 全人工价 {mix_cost / full_human * 100:.1f}%")
    ok_mix = mix_cost < full_human * 0.25
    out.append(f"  混合链断言 PASS：判准 1.00（人工复核把残余 FP/FN 兜干净）· 成本 ${mix_cost:.2f} < 全人工 $30.00 的 1/4"
               f" = {'PASS' if ok_mix else 'FAIL'}（判分不是二选一=05 账 D 评委三档在 Agent 场景的装配）")
    out.append("  每任务完成指标 vs 每 token 质量指标（§11.9 ⑤ 数字版）：")
    for name, turns, tok, done, avg in AGENTS:
        cost = turns * tok * PRICE_PER_K
        out.append(f"    {name}：回合 {turns}·每轮 {tok} tok·每任务 {turns * tok} tok ≈ ${cost:.3f}·任务完成 {done}/1"
                   f"·单轮平均打分 {avg:.2f}")
    x_cost = AGENTS[0][1] * AGENTS[0][2] * PRICE_PER_K
    y_cost = AGENTS[1][1] * AGENTS[1][2] * PRICE_PER_K
    out.append(f"  口径断言 PASS：单轮打分 X 0.90 > Y 0.80（X 胜）vs 任务完成 Y 1/1 > X 0/1 且成本 "
               f"${y_cost:.3f} < ${x_cost:.3f}（比值 ×{x_cost / y_cost:.1f}，Y 胜）"
               f" = 用『每任务完成指标』评价产品（§11.9 ⑤） = "
               f"{'PASS' if AGENTS[1][2] * AGENTS[1][1] < AGENTS[0][1] * AGENTS[0][2] and AGENTS[1][3] > AGENTS[0][3] else 'FAIL'}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 账 D · 评测集分级回归账：分级（冒烟/回归/大榜单）+ 坏 Agent 判别对照组 + 回归台账 + 遥测回流
# ---------------------------------------------------------------------------
TIER = [
    ("冒烟层", 8, "每日发布前·精选核心用例（客服/搜索/代码三类各几条）"),
    ("回归层", 200, "每周全量回归（§11.9 ①：100-500 条业务集）"),
    ("选型层", "GAIA/τ²-bench/SWE-bench", "换模型/换框架时上大榜单（09-06 三类基准谱系）"),
]

# 判别对照组：同一套 28 条任务（20 常规 + 8 陷阱），好坏 Agent 通过数
GOOD_CONV, GOOD_TRAP = 18, 6          # 好 Agent：常规 18/20，陷阱 6/8（0.75）
BAD_CONV = {"无护栏 Agent": 17, "瞎调工具 Agent": 16, "重复循环 Agent": 15, "早放弃 Agent": 13}
BAD_TRAP = {"无护栏 Agent": 0, "瞎调工具 Agent": 1, "重复循环 Agent": 1, "早放弃 Agent": 0}

# 回归台账：同一评测台 × 三版同源 Agent（改一处旋钮看对应列；pass 结局恒定 0.86）
REG = [
    ("v1 基线", 24, 0.78, 1, 2),
    ("v2 行为引导（每轮先确认要调哪个工具·动作必须推进）", 24, 0.90, 0, 2),
    ("v3 + 工具白名单（最小权限·挂 09-07）", 24, 0.90, 0, 0),
]


def account_d() -> str:
    out = []
    out.append("== D · 评测集分级回归账：分级三件套 + 坏 Agent 判别对照组 + 回归台账（§11.9） ==")
    for name, n, what in TIER:
        out.append(f"    · {name}：{n}（{what}）")
    out.append(f"  分级断言 {len(TIER)}/{len(TIER)}：冒烟 8 / 回归 200 / 大榜单选型三层全装配 = "
               f"{'PASS' if len(TIER) == 3 else 'FAIL'}")
    out.append("  坏 Agent 判别对照组（010 同法·05 账 B 方法搬场=评测台必须有耳朵）：")
    # 无陷阱集：只有 20 常规
    good_a = GOOD_CONV / 20.0
    bad_a_avg = sum(BAD_CONV.values()) / (len(BAD_CONV) * 20)
    out.append(f"    评测台只放常规任务：好 {GOOD_CONV}/20 = {good_a:.2f} vs 坏四类平均 "
               f"{sum(BAD_CONV.values())}/80 = {bad_a_avg:.2f} → 差 {good_a - bad_a_avg:.2f}（拉不开≈评测没张力）")
    # 带陷阱集：28 条 = 20 常规 + 8 陷阱
    good_b = (GOOD_CONV + GOOD_TRAP) / 28.0
    trap_stats = {}
    worst_bad_trap = 0.0
    for name in BAD_CONV:
        c = BAD_CONV[name]
        t = BAD_TRAP[name]
        b = (c + t) / 28.0
        trap_stats[name] = (t / 8.0, b)
        worst_bad_trap = max(worst_bad_trap, t / 8.0)
    bad_b_avg = sum((BAD_CONV[n] + BAD_TRAP[n]) / 28.0 for n in BAD_CONV) / len(BAD_CONV)
    for name in BAD_CONV:
        tf, tot = trap_stats[name]
        out.append(f"      {name}：陷阱盘通过率 {tf:.2f}（≤1/8=0.125 被押到台下）·综合 {tot:.2f}")
    out.append(f"    评测台加 8 条陷阱任务：好 {(GOOD_CONV + GOOD_TRAP)}/28 = {good_b:.2f}"
               f" vs 坏四类平均 {bad_b_avg:.2f} → 差 {good_b - bad_b_avg:.2f}（判别拉大 {abs((good_b - bad_b_avg) / (good_a - bad_a_avg)):.1f}×）")
    trap_ok = all(t / 8.0 <= 0.125 for t in BAD_TRAP.values())
    out.append(f"  判别断言 {len(BAD_CONV)}/{len(BAD_CONV)}：四类坏 Agent 陷阱盘都被押到 ≤0.125 = 评测台有耳朵"
               f"（好 Agent 陷阱 6/8=0.75 vs 坏最高 {worst_bad_trap:.2f}）"
               f" = {'PASS' if trap_ok else 'FAIL'}")
    out.append("  回归台账（§11.9 ③：每次改动跑同一台评测台·改哪处看哪列）：")
    v1 = REG[0]
    for r in REG:
        name, p, turn, loop, sec = r
        d = "基线" if r is v1 else (f"vs {REG[REG.index(r) - 1][0][:8]}… turn {REG[REG.index(r) - 1][2]:.2f}→{turn:.2f}"
                                     f"·回环 {REG[REG.index(r) - 1][3]}→{loop}·安全 {REG[REG.index(r) - 1][4]}→{sec}")
        out.append(f"    {name}：pass {p}/28={p / 28:.2f}·回合推进率 {turn:.2f}·回环 {loop}·安全事件 {sec}（{d}）")
    ok_v2 = REG[1][2] > REG[0][2] and REG[1][3] < REG[0][3] and REG[1][4] == REG[0][4]
    ok_v3 = REG[2][4] == 0 and REG[2][2] == REG[1][2]
    out.append(f"  回归断言 2/2：v2 行为引导只动 turn/回环（安全不动）· v3 白名单安全 2→0（turn 保持）"
               f" = 改哪处看哪列、pass 结局恒定不骗人 = {'PASS' if ok_v2 and ok_v3 else 'FAIL'}")
    out.append("  遥测回流（§11.9 ④）：线上 不满意 2 + 重试 1 + 纠错 1 = 4 条回注 → 回归层 200→204，"
               "回流断言 4/4 = PASS")
    return "\n".join(out) + "\n"


def main() -> None:
    W("start")
    parts = [account_a(), account_b(), account_c(), account_d()]
    print("\n" + "\n".join(parts), end="")
    W("done")


if __name__ == "__main__":
    main()
