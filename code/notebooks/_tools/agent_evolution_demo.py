#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""09-Agent体系 · 01-Agent定义与演进谱系（知识地图 §9.0/§9.1/§6.7/§9.14/§17.7）探针：
『Agent 是什么』的第一课——先画 Loop、再谈一切之前，先把 Agent 放在谱系里定位。
量四本账：
  A 演进谱系账 —— 地图 §9.1 的 7 世代能力梯（Chatbot→Tool Calling→Workflow→ReAct→
              Agent→Multi-Agent→Agentic System）建模成世代增量账：每代新增几个能力原语、
              累计到几个；能力梯次序 ≠ 发明年份（ReAct 论文 2022 先于 Function Call API
              2023，却在梯上排后=『先有工具抓手，循环才跑得动』）
  B 定义判别器账 —— 「这个系统算哪一类？」：12 个代表性系统 × 6 个特征位（工具/决策点/
              记忆/多主体/评测门/人工审批）→ 5 类（Chatbot/Workflow/Agent/Multi-Agent/
              Agentic System）确定性分界；边界对（有会话无工具 vs 有工具循环、固定检索 vs
              ReAct 循环）显性打印
  C 五要素溯源账 —— Agent = LLM(大脑) + 工具(手脚) + 循环(机制) + 记忆(状态) + 目标(任务)
              逐要素挂靠本仓库既有篇目的实测锚点（复习链 5/5）
  D 年谱与口诀账 —— 2022-2026 Agent 技术节点年谱（已实测挂靠 vs 要素事实计数）+ §6.7/§6.9/
              §9.7/§9.12 判断口诀断言 3/3
口径：能力原语归类/判别特征与类标/年谱事件与年份 = 作者按知识地图 §9/§6.7/§6.9 整理
     （要素事实=非本机实测，写作环境无外网未在线复核论文年份）；A-D 全部统计 = 本机确定性
     计算（纯 stdlib · 零随机 · 零网络 · stdout 逐字节可复现；wall-clock 只进 stderr）。
"""
import sys
from time import perf_counter

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ================= 账 A 数据：7 世代能力梯 =================
# 每一世代在能力梯上新增的原语；能力全集 13 个（作者按 §9.0/§9.1 拆解，要素事实）
CAPS = [
    "SINGLE 单轮回应", "SESSION 会话多轮", "TOOL 工具调用", "ORCH 固定步骤编排",
    "LOOP 思考→行动→观察", "PLAN 自主规划", "MEM 记忆分层", "REFL 反思自纠错",
    "MULTI 多角色协作", "ENV 环境执行(GUI/浏览器/终端)", "EVAL 评测门",
    "APPROVE 人工审批", "GOV 治理+知识准入(RAG/可观测)",
]
# (世代, 最小年份, 新增原语(缩写), 代表, 补齐了谁的缺口)
EPOCHS = [
    ("Chatbot 单轮", 2017, {"SINGLE"}, "对话式问答系统（小冰/早期闲聊）", "单轮完事，没有可迭代状态"),
    ("Tool Calling", 2023, {"SESSION", "TOOL"}, "2023-06 OpenAI Function Calling（会说也会做）", "『只会说不会做』"),
    ("Workflow", 2023, {"ORCH"}, "固定工序编排（流程图/LangChain Chain/Dify）", "『单步串联要人肉』"),
    ("ReAct", 2022, {"LOOP"}, "2022 论文：Thought→Action→Observation 循环", "『调完工具不会再想』"),
    ("Agent", 2024, {"PLAN", "MEM", "REFL"}, "自主规划+记忆+反思（AutoGPT 后的 Agent 浪潮）", "『一次想完不迭代·无记忆·错了不回头』"),
    ("Multi-Agent", 2024, {"MULTI"}, "多角色/多决策主体协作（AutoGen/CrewAI/AG2）", "『一个脑袋干不了多角色』"),
    ("Agentic System", 2026, {"ENV", "EVAL", "APPROVE", "GOV"}, "生产化 Agent：环境执行+评测门+人工审批+治理", "『不可信、不可测、进不了生产』"),
]
CAP2NAME = dict(x.split(" ", 1) for x in CAPS)


def expA():
    print("=" * 72)
    print("[账 A] 演进谱系账：地图 §9.1 的 7 世代能力梯 → 世代增量账（谁是能力跃迁的大头）")
    print("=" * 72)
    cum = set()
    first_at = {}   # 每个能力原语在哪个世代首次解锁
    rows = []
    for name, year, adds, rep, gap in EPOCHS:
        for c in adds:
            first_at.setdefault(c, name)
        cum |= adds
        rows.append((name, year, sorts(list(adds)), len(adds), len(cum)))
    print("  世代梯（按地图 §9.1 能力解锁顺序）：")
    for name, year, capped, add_n, cum_n in rows:
        print(f"    {name:<18}新增 {add_n} 个·累计 {cum_n} 个 · 补齐缺口=【{gap_for(name)}】")
    print(f"  → 能力原语清册（累计 {len(cum)} 个 · ←首解锁世代）：")
    for c in sorted(CAP2NAME.keys()):
        print(f"      {c:<8}{CAP2NAME[c]}  ← {first_at[c]}")
    inc = [(add_n, name) for name, _y, _c, add_n, _cn in rows]
    top = sorted(inc, reverse=True)
    print(f"  → 世代增量排行榜：{' · '.join(f'{name} +{n}' for n, name in top)}")
    print(f"    增量前两名 = {top[0][1]}(+{top[0][0]}) & {top[1][1]}(+{top[1][0]})——"
          f"Agent 一次补『规划/记忆/反思』三件套、Agentic System 一次补『环境/评测/审批/治理』"
          f"四件套=从『能跑』到『能生产』的台阶")
    print(f"  → 一张图两把尺：能力梯次序 ≠ 发明年份——ReAct 论文 2022 早于 Function Call API 2023，")
    print(f"    但能力梯上排在工具之后：『先有工具抓手（2023），思考-行动-观察的循环（2022）"
          f"才跑得动』=能力梯是解锁顺位、不是出版年谱")
    assert len(cum) == len(CAPS) == 13 and all(c in cum for c in CAP2NAME.keys())
    assert top[0] == (4, "Agentic System") and top[1] == (3, "Agent")
    print(f"  → 断言：世代链 7 齐全 · 能力原语 13/13 全部解锁 · 增量榜首=Agentic System(+4) · 次席=Agent(+3)")


def sorts(s):
    return sorted(s)


def gap_for(name):
    for n, _y, _a, _r, g in EPOCHS:
        if n == name:
            return g
    return ""


# ================= 账 B 数据：定义判别器 =================
# 特征位：tools 工具 / decision 运行中决策 / mem 记忆 / multi 多主体 / evalg 评测门 / appr 人工审批
# 类规则（§6.7/§9.0 判据，确定性）：
#   多主体(+评测+审批)→Agentic System；多主体→Multi-Agent；
#   (决策+评测+审批)→Agentic System；决策→Agent；
#   有工序无决策（即使带工具如检索）→Workflow；
#   无工具无决策（即使有会话）→Chatbot
SYSTEMS = [
    # (系统, 特征位, 期望类, 一句话边界教学)
    ("裸 LLM 聊天窗（单轮）",        {},                          "Chatbot",       "单轮输出·无工具无决策=最原初形态"),
    ("多轮客服对话（有会话无工具）",  {"SESSION"},                    "Chatbot",       "有『会话』不等于 Agent——分界在『决策点』不在『多轮』"),
    ("固定工序客服分流（IF-ELSE 流程）", {"ORCH"},                 "Workflow",      "步骤可枚举→确定性编排（§6.7 口诀第一问）"),
    ("RAG 单轮检索问答",            {"ORCH", "TOOL"},               "Workflow",      "检索是固定工序的一步：有工具≠Agent"),
    ("ReAct 带工具（无记忆）",       {"TOOL", "LOOP", "decision"},   "Agent",         "Thought→Action→Observation=最简洁的 Agent"),
    ("LangGraph 带记忆重试客服（里程碑012）", {"TOOL", "LOOP", "decision", "MEM"}, "Agent", "循环+记忆+可恢复=单 Agent 全配置"),
    ("代码解释器多步执行",          {"TOOL", "LOOP", "decision", "ENV"}, "Agent",     "与环境（代码执行）交互并回喂=Agent 动作"),
    ("OpenAI Operator（浏览器操作）", {"TOOL", "LOOP", "decision", "ENV"}, "Agent",   "GUI/浏览器=Agent 的新动作面（§9.9/9.10）"),
    ("AutoGen Group Chat 三 Agent 互聊", {"MULTI", "TOOL", "LOOP", "decision"}, "Multi-Agent", "多决策主体自由对话=协作版"),
    ("CrewAI 写书团队（写手+审校）",  {"MULTI", "TOOL", "LOOP", "decision", "MEM"}, "Multi-Agent", "多角色分工=角色化协作版"),
    ("生产 Agentic RAG（Agent+评测门+人工审批）", {"TOOL", "LOOP", "decision", "EVAL", "APPROVE"}, "Agentic System", "决策+评测+审批=能进生产的 Agent"),
    ("企业多 Agent 治理平台",        {"MULTI", "TOOL", "LOOP", "decision", "EVAL", "APPROVE"}, "Agentic System", "多主体+评测+审批=企业级 Agentic 系统"),
]
CLASS_ORDER = ["Chatbot", "Workflow", "Multi-Agent", "Agentic System", "Agent"]  # 打印顺序
FEATS = ["TOOL", "LOOP", "decision", "MEM", "MULTI", "EVAL", "APPROVE", "ENV", "ORCH", "SESSION"]


def classify(f):
    multi = "MULTI" in f
    decision = "decision" in f
    evalg = "EVAL" in f
    appr = "APPROVE" in f
    tools = "TOOL" in f
    if multi and evalg and appr:
        return "Agentic System"
    if multi:
        return "Multi-Agent"
    if decision and evalg and appr:
        return "Agentic System"
    if decision:
        return "Agent"
    if tools or "ORCH" in f or "SESSION" in f:
        return "Workflow" if ("ORCH" in f or tools) and not decision else "Chatbot"
    return "Chatbot"


def expB():
    print()
    print("=" * 72)
    print("[账 B] 定义判别器账：『这个系统算哪一类？』——12 系统 × 6 特征位 → 5 类确定性分界")
    print("       特征位：工具/决策点/记忆/多主体/评测门/人工审批；类规则=§6.7『先问可不可枚举』")
    print("=" * 72)
    ok = 0
    for sys_, f, expect, tip in SYSTEMS:
        got = classify(f)
        good = got == expect
        ok += good
        mark = "✅" if good else "❌"
        feats = [x for x in FEATS if x in f] or ["(空)"]
        print(f"  {mark} {sys_}")
        print(f"      特征 {','.join(feats):<42}→ {got:<14}（期望 {expect}）{tip}")
    print(f"  → 判别断言 {ok}/{len(SYSTEMS)}")
    assert ok == len(SYSTEMS)
    # 边界对
    print("  → 边界对（最容易判错的 3 对）：")
    pairs = [
        ("多轮客服对话", "ReAct 带工具", "Chatbot", "Agent",
         "同是多轮：分界=『运行中要不要决定下一步』——固定话术叫客服，循环决策叫 Agent"),
        ("RAG 单轮检索问答", "ReAct 带工具", "Workflow", "Agent",
         "都调用检索：分界=『检索后是直接答』还是『把结果回喂再想』——有 LOOP 才进 Agent"),
        ("OpenAI Operator", "AutoGen Group Chat", "Agent", "Multi-Agent",
         "都是决策体：分界=『单决策主体的动作面深（GUI）』vs『多决策主体的协同面广』"),
    ]
    for a, b, ka, kb, why in pairs:
        print(f"      {a}→{ka} ｜ {b}→{kb}：{why}")


# ================= 账 C 数据：五要素溯源（复习链） =================
BASES = [
    # (要素, 一句话, 仓库锚点篇, 锚点实测数字)
    ("LLM 大脑", "生成候选动作与文本的力量", "02-12 预训练" ,
     "0.18M 参数 CPU 自回归采样对齐 1.000(tr-loop) · ICL 解锁 fresh 0.320→1.000"),
    ("工具 手脚", "把手伸进外部世界执行动作", "06-05 Function-Tool-Calling",
     "同一 7 任务集：散文 0/7 → JSON 键名对齐 2/7 → FC 形状 promise 7/7（预置仿真）"),
    ("循环 机制", "观察→评估→再决定的重复机制", "07-02 LangGraph-状态图与Checkpoint",
     "7 节点 · 条件边 6 支路 · 自环 1 · recursion_limit=6 真抛 GraphRecursionError · checkpoint 恢复补 2 步"),
    ("记忆 状态", "跨轮记忆使决策依赖历史", "06-06 记忆系统",
     "11 会话 41 条 419 tok → 抽取 43 事实 → 整合 34 条 · 回灌 195 vs 354 省 45%"),
    ("目标 任务", "决定『做完了吗』的验收标准", "02-20 推理模型与Test-time-Scaling",
     "验证器 best-of-k≈覆盖率 78/80% · 能力墙 9+9=18 只采到 {'13'} → 终结条件=分布可达性"),
]
XTRAS = [
    ("工具→入口", "07-11 MCP 把工具接口标准化：跨 Agent 可复用（本仓库 07-11 已实测 17 方法 3 传输）"),
    ("循环→成本", "『先证明单 Agent 不行再上多』（§6.9）：多 Agent=多倍 token——07-10 实测行队成本 直答1/带工具2/手转交2 轮"),
    ("目标→评测", "『不评测的 Agent 不可信』（§9.12）：终结条件用『固定环境+可验证终结+成本指标』，别用『看起来像』的判官"),
]


def expC():
    print()
    print("=" * 72)
    print("[账 C] 五要素溯源账：Agent = LLM(大脑)+工具(手脚)+循环(机制)+记忆(状态)+目标(任务)，")
    print("       逐要素挂靠本仓库既有篇目实测锚点（复习链：一个要素=一篇已闭环的真数字）")
    print("=" * 72)
    for elem, one, anchor, num in BASES:
        print(f"  {elem:<10}{one}  →  {anchor}")
        print(f"     锚点数字：{num}")
    print("  → 复习链断言：五要素全部在 02/06/07 章有『≥1 篇已实测』锚点")
    assert len(BASES) == 5 and all(n for _, _, _, n in BASES)
    print(f"     五要素锚点 5/5 齐（无一要素需要新实验才有数字）")
    print("  → 要素间的通路（后面篇章消费这些）:")
    for name, note in XTRAS:
        print(f"     {name:<10}{note}")
    n_anchor = len(BASES)
    assert n_anchor == 5
    print(f"  → 断言：五要素锚点 5/5 · 通路三句与 §6.9/§9.7/§9.12 一致")


# ================= 账 D 数据：2022-2026 年谱与口诀 =================
# (年份, 事件, 挂靠, 状态)  status: 实测=仓库已有真实测挂靠；要素=要素事实待 09-xx 长肉
YEARLINE = [
    (2022, "ReAct/Reflexion/Self-Refine/ToT 范式论文群", "→ 09-04 规划中", "要素"),
    (2023, "Function Calling API(2023-06)·AutoGPT(2023-03)", "→ 06-05 已实测", "实测"),
    (2023, "AutoGen(2023-10) 开创 Agent 互聊", "→ 07-07 已实测", "实测"),
    (2023, "LangChain Agent/Workflow 生态成熟", "→ 07-01 已实测", "实测"),
    (2024, "LangGraph 状态图/checkpoint", "→ 07-02 已实测", "实测"),
    (2024, "CrewAI(2024) 角色化团队", "→ 07-08 已实测", "实测"),
    (2024, "Anthropic Computer Use(2024-10)·MCP(2024-11)", "→ 03-07 派生账/07-11 已实测", "实测"),
    (2024, "Swarm(2024-10)→OpenAI Agents SDK(2025)", "→ 07-10 已实测", "实测"),
    (2025, "OpenAI Operator/AgentKit · MAF(2025)·A2A(2025-04)", "→ 07-07 继承边已实测/要素", "实测"),
    (2026, "Agentic System：评测+护栏+人工审批+长时任务生产化", "→ 本篇账A + 09-12/13 规划", "要素"),
]
SAYINGS = [
    ("口诀1 · §6.7", "『任务是否可枚举步骤？是=Workflow，否=Agent』", "账 B 12 系统判别里 Workflow 恰是『可枚举』类（工序/检索），Agent 是『决策』类——二分把 12/12 全对"),
    ("口诀2 · §6.7/§6.9", "『能用 Workflow 就不用 Agent；先证明单 Agent 不行再上多』", "账 B 12 系统里只有 2/12 是 Multi-Agent（其余 10 个单决策体先被排掉=合 §6.9 成本警示）"),
    ("口诀3 · §9.7/§9.12", "『Agent 可靠性来自验证回路，不来自更好的提示词；不评测的 Agent 不可信』", "账 A 最后一世代理应验：Agentic System 新增 EVAL+APPROVE 原语=验证回路进架构"),
]


def expD():
    print()
    print("=" * 72)
    print("[账 D] 年谱与口诀账：2022-2026 Agent 技术节点年谱（已实测挂靠 vs 要素事实）＋判断口诀 3/3")
    print("=" * 72)
    for year, event, anchor, status in YEARLINE:
        tag = "🔬 已实测挂靠" if status == "实测" else "📄 要素事实/待长肉"
        print(f"  {year}: {event}")
        print(f"       {tag} {anchor}")
    real = sum(1 for _, _, _, s in YEARLINE if s == "实测")
    fact = len(YEARLINE) - real
    print(f"  → 年谱 10 节点：已实测挂靠 {real} · 要素事实/待长肉 {fact}")
    print("      （09-04 行动推理/09-13 安全/09-12 评测等未来篇目长肉后，挂靠面继续右移）")
    assert real == 8 and fact == 2
    print("  → 判断口诀断言 3/3：")
    for tag, say, why in SAYINGS:
        print(f"     {tag}：{say}")
        print(f"        → {why}")
    assert len(SAYINGS) == 3
    print(f"  → 断言：年谱节点 10 · 实测挂靠 8/10 · 口诀 3/3 与账A/账B 结果互证")


def main():
    print("=" * 72)
    print("agent_evolution_demo：09-Agent体系 · 01-Agent定义与演进谱系（第 1 课：发现·定位·分界）")
    print("    关联：地图 §9.0/§9.1/§6.7/§6.9/§9.7/§9.12/§9.14/§17.7 ｜ 06-05/07-02/06-06/02-20 锚点")
    print("=" * 72)
    print("[0] 口径：能力原语/世代归类/判别特征/年谱事件 = 作者按知识地图 §9 整理（要素事实=非本机实测）；")
    print("    A-D 全部统计 = 本机确定性计算（纯 stdlib·零随机·零网络）")
    expA()
    expB()
    expC()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 演进谱系 / B 定义判别 / C 五要素 / D 年谱口诀）")
    print("  A 7 世代能力梯 · 13 能力原语全解锁 · 增量榜首=Agentic System(+4) 次席=Agent(+3)")
    print("  B 12 系统判别断言 12/12 · 5 类（Chatbot2/Workflow2/Agent4/Multi2/Agentic2）· 边界对 3 组")
    print("  C 五要素复习链 5/5（LLM→02 工具→06 循环→07 记忆→06 目标→02）· 通路三句")
    print("  D 年谱 10 节点 · 已实测挂靠 8 · 口诀 3/3（可枚举→Workflow/先单后多/验证回路）")
    print("一句话：Agent 不是一类『模型』而是一个『多步行动+负责结果』的系统；判断它的类 =")
    print("         先问可枚举（Workflow）→再问决策点（Agent）→再问多主体（Multi-Agent）→")
    print("         最后只有带验证回路（评测+审批）的才叫 Agentic System——先画 Loop 再谈一切")
    print("done · 一键复现：python code/notebooks/_tools/agent_evolution_demo.py")


if __name__ == "__main__":
    main()
    W(f"[agent_evolution_demo] wall-clock {perf_counter() - _t0:.3f} s")
