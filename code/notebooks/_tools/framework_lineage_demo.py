#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""07-应用框架 · 12-框架继承关系与选型决策（知识地图 §7.13 扩版 / §7.14 / §17 附录）探针：
07 章收束章——把 01-11 十二篇穿成一张『继承 DAG + 该不该引入』的决策图。
量四本账：
  A 继承关系完整 DAG —— 00 章账 C（10 节点 · 承袭边 6 · 底座边 9）扩成 13 节点完整版
                 （LangChain 系 + Agent 系 + Prompt 编程系 + 协议底座系 + 微软双线 AutoGen/SK→MAF）
                 → 最长链/hub/入度台账/递归影响面
  B 概念溯源与普及率 —— 8 个核心范式（Chain/状态图/索引分层/函数合约/角色/轻量运行时/签名即程序/协议接口）
                 从哪个家族首发、在 24 家族里普及到几家（作者按 01-11 篇实测定位）
  C 该不该引入框架 —— 10 个选型场景，每场景先定『类域』再比『域内净分』
                 = Σ(需求×解决度) − 引入成本(学习面+版本风险)；阈值 0.45 以下一律第 0 基准直写
  D 选型决策路牌 —— 07 章 13 站×何时到达 + 2022-2025 年谱四主线 + 客服重试（链路 012）三套对照
口径：DAG 边/概念首发归属/需求权重/解决度与引入成本/类域约束 = 作者按知识地图 §7 与
     07 章 01-11 各篇实测整理与定性（要素事实=非本机实测）；A-D 全部统计 = 本机确定性计算
     （纯 stdlib · 零随机 · 零网络 · stdout 逐字节可复现；wall-clock 只进 stderr）。
"""
import sys
from time import perf_counter

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ================= 账 A 数据：分线 13 节点 DAG =================
LINES = {
    "Chain/Runnable 系":  [("LangChain", 2022), ("LlamaIndex", 2023), ("LangGraph", 2024),
                           ("LangGraph Platform", 2025)],
    "Agent 系":           [("AutoGen", 2023), ("CrewAI", 2024), ("Swarm", 2024),
                           ("AG2", 2024), ("OpenAI Agents SDK", 2025),
                           ("Microsoft Agent Framework", 2025)],
    "Prompt 编程系":       [("DSPy", 2023)],
    "企业 SDK":           [("Semantic Kernel", 2023)],
    "协议底座":            [("MCP", 2024)],
}
NODE_YEAR = {}
for _members in LINES.values():
    for _n, _y in _members:
        NODE_YEAR.setdefault(_n, _y)
NODES = sorted(NODE_YEAR.items(), key=lambda t: (t[1], t[0]))
# 承袭边 9 条 = 00 章账 C 承袭边 6（LangChain 系 4 + Swarm→SDK）
#               + 微软双线 3（AutoGen→AG2 社区 fork + AutoGen→MAF + SK→MAF 官方承接）
INHERIT = [
    # LangChain 系（§7.13 承袭 6 条全保留，00 章账 C 同源）
    ("LangChain", "LlamaIndex"), ("LangChain", "AutoGen"), ("LangChain", "CrewAI"),
    ("LangChain", "LangGraph"), ("LangGraph", "LangGraph Platform"),
    ("Swarm", "OpenAI Agents SDK"),
    # 微软线（§7.8：2025 Microsoft Agent Framework 承接 AutoGen 与 SK 统一；AG2=社区 fork）
    ("AutoGen", "AG2"), ("AutoGen", "Microsoft Agent Framework"),
    ("Semantic Kernel", "Microsoft Agent Framework"),
]
BASE_FROM = "MCP"   # 底座：MCP → 其余 12 节点（00 章账 C 同源；要素事实）
MS_EDGE = [("AutoGen", "AG2"), ("AutoGen", "Microsoft Agent Framework"),
           ("Semantic Kernel", "Microsoft Agent Framework")]


def expA():
    print("=" * 72)
    print("[账 A] 继承关系完整 DAG：00 章账 C（10 节点·承袭边 6·底座边 9）扩版 = 13 节点 · 承袭边 9 · 底座边 12")
    print("       五主线（Chain 系/Agent 系/Prompt 系/企业 SDK/协议底座）+ 微软 2025 双线（AutoGen→MAF + SK→MAF）")
    print("=" * 72)
    names = [n for n, _ in NODES]
    child = {n: [] for n in names}
    parent = {n: [] for n in names}
    for a, b in INHERIT:
        child[a].append(b)
        parent[b].append(a)
    for n in names:
        child[n] = sorted(child[n])
        parent[n] = sorted(parent[n])
    print(f"  节点 {len(NODES)}（年份=首发，要素事实）：" + " · ".join(f"{n}'{y}" for n, y in NODES))
    print(f"  承袭边 {len(INHERIT)} 条：{' · '.join(a + '→' + b for a, b in INHERIT)}")
    print(f"  MCP 底座边 {len(names) - 1} 条：MCP → 其余全部节点")
    print(f"  微软双线 {len(MS_EDGE)} 条（唯一入度>1 汇集在 MAF=两线合并的量化）")
    # 最长承袭链（DAG 有向 DP · memo 确定性）
    memo = {}

    def depth(n):
        if n in memo:
            return memo[n]
        d = 1 + max((depth(c) for c in child[n]), default=0)
        memo[n] = d
        return d
    longest = max(depth(n) for n in names)
    chain_pick = sorted((-(depth(n)), n) for n in names)[0][1]
    cur = chain_pick
    p = [cur]
    seen = set()
    while child[cur] and cur not in seen:
        seen.add(cur)
        cur = child[cur][0]
        p.append(cur)
    print(f"  → 最长承袭链 {longest} 节点：" + "→".join(p))
    # hub、递归影响面
    hub = sorted(((len(v), k) for k, v in child.items()), reverse=True)[0][1]
    seen = set()
    stack = sorted(child[hub])
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        stack.extend(sorted(child[x]))
    print(f"  → 概念普及者 hub = {hub}：直接后代 {len(child[hub])} · 递归影响面 {len(seen)}"
          f"（{', '.join(sorted(seen))}）")
    # 入度台账
    rows = sorted(((len(parent[n]), n, sorted(parent[n])) for n in names), reverse=True)
    print("  → 入度台账（降序；=被多少家直接继承/承接）：")
    for d, n, ps in rows:
        note = "（开山鼻祖/独立）" if not ps else " ← " + ", ".join(ps)
        print(f"      {n:<26}入度 {d}{note}")
    maf = len(parent["Microsoft Agent Framework"])
    assert hub == "LangChain" and longest == 3 and maf == 2
    print(f"  → 断言：最长链=3 · hub=LangChain · MAF 入度=2（AutoGen+SK 双线汇合=微软统一线 §7.8）")
    print(f"    底座：MCP→其余 12（2025 工具接口公共底座=要素事实，00 章账 C 同源）")


# ================= 账 B 数据：概念溯源与普及率 =================
# (概念, 首发家族, 年份, 词面定标准, 普及到的家族，含该思路后世实现/借用)
CONCEPTS = [
    ("Chain/Runnable 统一触发", "LangChain", 2022, "invoke/batch/stream",
     ["LangChain", "LlamaIndex", "Haystack", "Semantic Kernel", "CrewAI", "AutoGen",
      "LangGraph", "AG2", "DSPy", "OpenAI Agents SDK", "Microsoft Agent Framework"]),
    ("状态图/checkpoint 可恢复", "LangGraph", 2024, "StateGraph+recursion_limit",
     ["LangGraph", "LangGraph Platform", "AutoGen", "AG2", "OpenAI Agents SDK",
      "Microsoft Agent Framework"]),
    ("检索索引分层(Node/Retriever)", "LlamaIndex", 2023, "文档→索引→检索→问答",
     ["LlamaIndex", "Haystack", "LangChain", "RAGFlow", "Dify", "Flowise", "Coze",
      "LlamaIndex.TS"]),
    ("@装饰器即函数合约", "Semantic Kernel", 2023, "@kernel_function",
     ["Semantic Kernel", "CrewAI", "AutoGen", "AG2", "OpenAI Agents SDK",
      "Microsoft Agent Framework"]),
    ("角色描述即 Agent 定义", "CrewAI", 2024, "role/goal/backstory",
     ["CrewAI", "AutoGen", "AG2", "Dify", "Coze", "Microsoft Agent Framework"]),
    ("签名即程序：编译/优化", "DSPy", 2023, "Signature→Module→Optimizer", ["DSPy"]),
    ("Agent 对象+Runner 轻量运行时", "Swarm→OpenAI Agents SDK", 2024, "Agent+Runner",
     ["OpenAI Agents SDK", "Pydantic AI", "Vercel AI SDK", "Mastra"]),
    ("工具协议接口(跨栈互操作)", "MCP", 2024, "initialize→tools/call",
     ["MCP", "A2A", "ACP"]),
]
FW_COUNT = 24          # 00 章账 A 全集成员数（24 家族）
FW_SET = {"LangChain", "Haystack", "Semantic Kernel", "LangGraph", "Temporal", "LlamaIndex",
          "RAGFlow", "AutoGen", "CrewAI", "OpenAI Agents SDK", "Google ADK", "Pydantic AI",
          "Dify", "Flowise", "Coze", "n8n", "DSPy", "Outlines", "MCP", "A2A",
          "ACP", "Vercel AI SDK", "Mastra", "LlamaIndex.TS"}


def expB():
    print()
    print("=" * 72)
    print("[账 B] 概念溯源与普及率：8 个核心范式『谁发明的 · 现在谁在用』——§7『框架会消亡、概念永存』")
    print("       普及家数 = 该范式思路被多少家族实现/借用（作者按 01-11 篇实测与 §7 定性）")
    print("=" * 72)
    rows = []
    for concept, who, year, defn, users in CONCEPTS:
        n = len({u for u in users})
        rows.append((n, concept, who, year, defn))
    rows.sort(key=lambda r: (-r[0], r[2], r[1]))
    for n, concept, who, year, defn in rows:
        blocks = int(round(n / FW_COUNT * 16))
        print(f"    {concept:<28}{who:<26}现年 {2026 - year:>2}y  普及 {n:>2}/{FW_COUNT}"
              f" [{('#' * blocks).ljust(16)}] {defn}")
    old = [r for r in rows if r[3] <= 2023]
    young = [r for r in rows if r[3] >= 2024]
    avg_old = sum(r[0] for r in old) / len(old)
    avg_young = sum(r[0] for r in young) / len(young)
    print(f"  → 2022-23 老概念平均普及 {avg_old:.2f} 家 vs 2024-25 新概念平均普及 {avg_young:.2f} 家")
    print(f"    =概念越老在生态里扎得越深——『框架会消亡、概念永存』的普及率证据；"
          f"新概念还要时间去四处渗透")
    solo = [r[1] for r in rows if r[0] == 1]
    print(f"  → 只被 1 家（独立 DNA）：{' · '.join(solo)}——DSPy 的编译/优化走 LangChain 之外的独立路径"
          f"（00 章账 C 无承袭边）")
    assert avg_old > avg_young and rows[0][0] >= 10 and rows[0][1].startswith("Chain")
    print(f"  → 断言：老概念普及 > 新概念（{avg_old:.2f} > {avg_young:.2f}）· 普及王=`{rows[0][1]}`"
          f"（{rows[0][0]}/24）")


# ================= 账 C 数据：该不该引入（类域 + 净分 + 阈值） =================
PN = ["状态", "组装", "可观测·评测", "成本·可靠"]   # §7.0 四痛点
# 13 候选：12 篇代表 + 直写（第 0 基准）。solve=对该痛点的本机解决度（0..1 作者定标）。
# import = 学习面（证据：MRO 层数/16 参数 1 必填/签名→编译产物 ×6.1）· ver = 版本风险（存活表 FAIL/改名烈度）
# 域：场景先定『类』（00 章纪律），域外候选不参与比净分——避免 Dify 全域碾压（低代码只在低代码域比）
CANDS = {
    "直写（06 章原语）":   dict(solve=[0.30, 0.30, 0.30, 0.30], importc=0.00, ver=0.00,
                            domain={"*"},  tag="第0基准·不引入也是决策（协议层先走 MCP）"),
    "LangChain":        dict(solve=[0.45, 0.95, 0.60, 0.40], importc=0.60, ver=0.80,
                            domain={"通用"}, tag="链式组装省心·MRO13层56字段·0.x→1.x 7 FAIL"),
    "LangGraph":        dict(solve=[0.90, 0.70, 0.60, 0.60], importc=0.35, ver=0.30,
                            domain={"通用", "状态"}, tag="状态/循环/恢复开箱即用·recursion_limit·checkpoint"),
    "LlamaIndex":       dict(solve=[0.40, 0.90, 0.60, 0.30], importc=0.30, ver=0.25,
                            domain={"RAG"}, tag="RAG专精·六行最小闭环·分块护栏"),
    "Haystack":         dict(solve=[0.60, 0.85, 0.85, 0.50], importc=0.40, ver=0.50,
                            domain={"RAG", "生产"}, tag="生产RAG管线·组件化类型安全·max_runs护栏"),
    "CrewAI":           dict(solve=[0.50, 0.85, 0.90, 0.40], importc=0.30, ver=0.45,
                            domain={"Agent"}, tag="角色·任务·流程三一等公民·任务列表即图"),
    "AutoGen/AG2":      dict(solve=[0.70, 0.60, 0.50, 0.35], importc=0.50, ver=0.70,
                            domain={"Agent", "研究"}, tag="对话范式·ask()回灌·经典API整代换血"),
    "Semantic Kernel":  dict(solve=[0.80, 0.90, 0.55, 0.70], importc=0.50, ver=0.60,
                            domain={".NET"}, tag=".NET企业栈·@kernel_function契约·改名烈度07之最"),
    "Agents SDK":       dict(solve=[0.50, 0.65, 0.45, 0.65], importc=0.10, ver=0.30,
                            domain={"轻量"}, tag="回归轻量·16参数仅1必填·handoff"),
    "DSPy":             dict(solve=[0.10, 0.35, 1.00, 0.50], importc=0.30, ver=0.40,
                            domain={"评测"}, tag="评测主业·签名→编译产物×6.1·2.x无Auto"),
    "Dify低代码":         dict(solve=[0.60, 0.90, 0.55, 0.80], importc=0.40, ver=0.55,
                            domain={"低代码", "RAG"}, tag="四痛点覆盖全·拖拽积木·SaaS黑盒锁定"),
    "Coze/n8n低代码":     dict(solve=[0.55, 0.75, 0.45, 0.85], importc=0.40, ver=0.65,
                            domain={"低代码"}, tag="低代码另一臂·n8n成本·可靠主业"),
    "MCP":              dict(solve=[0.10, 0.95, 0.20, 0.30], importc=0.15, ver=0.20,
                            domain={"协议"}, tag="协议非框架·只统一组装-工具接口·17方法·里程碑013"),
}
SCN = [
    # (场景, 需求权重四元, 允许域, 预期胜者)
    ("线性调模型+组提示 · 少分支不重试",     [0.10, 1.00, 0.20, 0.20], {"通用"},      "直写（06 章原语）"),
    ("文档/知识库检索问答",                [0.20, 0.90, 0.50, 0.40], {"RAG"},       "LlamaIndex"),
    ("客服重试 · 有状态/循环/恢复（链路012）", [1.00, 0.70, 0.40, 0.60], {"通用", "状态"}, "LangGraph"),
    ("非工程师快速搭业务（企业知识库）",      [0.70, 0.90, 0.30, 0.70], {"低代码"},    "Dify低代码"),
    ("多 Agent 角色化协作（客服团队）",       [0.80, 0.60, 0.60, 0.50], {"Agent"},     "CrewAI"),
    ("质量敏感 · 可自动评测的抽取/转换",      [0.10, 0.20, 1.00, 0.30], {"评测"},      "DSPy"),
    ("轻量单 Agent · 少依赖",              [0.30, 0.60, 0.30, 0.80], {"轻量", "通用"}, "Agents SDK"),
    (".NET/微软企业栈",                    [0.70, 0.80, 0.50, 0.80], {".NET"},     "Semantic Kernel"),
    ("跨栈工具互操作 · 一次实现处处可用",      [0.30, 0.90, 0.30, 0.60], {"协议"},      "MCP"),
    ("生产级 RAG 管线（预发布多组件）",       [0.60, 0.85, 0.85, 0.60], {"RAG", "生产"}, "Haystack"),
]
THRESH = 0.45


def expC():
    print()
    print("=" * 72)
    print("[账 C] 该不该引入框架：10 场景 · 先定类域（00 章纪律）再比域内净分")
    print("       净分 = Σ(需求×解决度) − 引入成本（学习面 + 版本风险）；阈值 0.45：Below→第0基准直写")
    print("=" * 72)
    ok = 0
    for scene, need, domain, expect in SCN:
        cands = {k: v for k, v in CANDS.items() if v["domain"] & domain or k == "直写（06 章原语）"}
        scores = {}
        for cand, cfg in cands.items():
            gain = sum(n * s for n, s in zip(need, cfg["solve"]))
            scores[cand] = gain - (cfg["importc"] + cfg["ver"])
        top = sorted(scores.items(), key=lambda t: (-t[1], t[0]))[:3]
        best, bestd = top[0]
        chosen = "直写（06 章原语）" if bestd < THRESH else best
        if chosen == expect:
            ok += 1
        verdict = ("❌ 不引入（净分未过阈值 → 第 0 基准直写）" if chosen == "直写（06 章原语）"
                   else f"✅ 引入 {best}")
        print(f"  {scene}")
        print(f"     需求权重 {[f'{n:.1f}' for n in need]} · 域=【{'/'.join(sorted(domain))}】"
              f" · 候选 {len(cands)} 个")
        print(f"     top3 净分：{' · '.join(f'{c} {d:+.2f}' for c, d in top)}")
        print(f"     → {verdict}")
    assert ok == len(SCN), f"决策断言 {ok}/{len(SCN)}"
    print(f"  → 决策断言 {ok}/{len(SCN)} 全过：10 场景『引入/不引入 + 选谁』与 00 章决策树及各篇实测对齐")


# ================= 账 D 数据：选型决策路牌 =================
ROADMAP = [
    ("00-框架分类学", "开篇地图：先定『类』再比优劣——第一刀不是哪家最强而是哪类匹配"),
    ("01-LangChain与历史包袱", "链式组装够用·少分支→省心；链条一长→换图（概念永存 vs 库换血）"),
    ("02-LangGraph-状态图与Checkpoint", "要可恢复（checkpoint）·要可循环（自环/上限）→上状态图"),
    ("03-LlamaIndex", "文档/知识库检索问答→RAG 专精（接 08-RAG 12 篇实测）"),
    ("04-Dify-Flowise-Coze-n8n", "非工程师快速搭业务→低代码；要全兜底→Dify；要省钱→n8n"),
    ("05-Haystack", "生产级 RAG 管线·组件化类型安全→严谨的 pipeline 思维"),
    ("06-Semantic-Kernel", ".NET/微软企业栈→插件/函数/自动调用三原语（函数=一等公民）"),
    ("07-AutoGen-AG2-MAF", "研究/原型多 Agent 互聊→AG2；微软 2025 官方线→MAF"),
    ("08-CrewAI", "角色化团队·demo 型→Role/Goal/Backstory；自由对话→AG2"),
    ("09-DSPy", "质量敏感·可自动评测→签名当程序、优化器替你调 prompt"),
    ("10-OpenAI-Agents-SDK", "轻量单 Agent+手转交→Agent+Runner 一件套（回归轻量）"),
    ("11-MCP协议", "跨栈工具互操作→协议不是框架：只统一组装-工具接口（里程碑 013）"),
    ("直写（06 章原语）", "线性/无状态/一次性→不引入也是决策；协议层先固定（MCP）"),
]
CS_TABLE = [
    ("01-LangChain 链式", "链自环 0 → 外层手写 while 补丁 9 行 · Fake 预置轮数 3", "组装省、循环要自补"),
    ("02-LangGraph 图式", "7 节点 · 1 自环 · recursion_limit=6 真抛 · checkpoint 恢复补 2 步", "状态/循环开箱即用"),
    ("06-记忆系统裸写", "会话态 41 条 419 tok → 抽取 43 事实 · 自写 24 条实用逻辑", "轻量可控但全自管"),
]
CS_LINE = ("链路 012 决策笔记：图式引入=拿 1 个引擎换回『可恢复 + 可循环』两把闸——需求里有状态/循环/恢复"
           " = 引入（LangGraph 域内净分 1.34）；纯线性 = 第 0 基准直写（不引入也是决策）。")


def expD():
    print()
    print("=" * 72)
    print("[账 D] 选型决策路牌：07 章 13 站×何时到达 + 五主线年谱 + 客服重试（链路 012）三套对照")
    print("=" * 72)
    print("  → 07-章 13 站『路径→判据』：")
    for name, why in ROADMAP:
        print(f"    {name:<30}{why}")
    print("  → 2022-2025 五主线（§7.13 扩版 · 要素事实）：")
    for line, members in LINES.items():
        extra = " · A2A('2025)·ACP('2025)" if line == "协议底座" else ""
        print(f"    {line:<20}" + " · ".join(f"{n}('{y})" for n, y in members) + extra)
    print("  → 同一『客服重试』（链路 012）在 3 种写法的实测对照（挂靠既有章，不重测）：")
    for w, f, take in CS_TABLE:
        print(f"    {w:<18}{f}")
        print(f"         → {take}")
    print(f"    {CS_LINE}")
    assert all(m for m in LINES.values()) and len(ROADMAP) == 13
    print("  → 断言：五主线非空 · 13 站路牌互不重复（07 章 12 篇 + 直写）")


def main():
    print("=" * 72)
    print("framework_lineage_demo：07-应用框架 · 12-框架继承关系与选型决策（收束章 · §7.13 扩版/§7.14/§17）")
    print("    『该不该引入框架』——把 01-11 十二篇穿成一张继承 DAG + 选型决策图")
    print("=" * 72)
    print("[0] 口径：DAG 边/概念归属/需求权重/解决度与引入成本/类域 = 作者按地图 §7 与 01-11 实测整理")
    print("    （要素事实=非本机实测）；A-D 全部统计 = 本机确定性计算（纯 stdlib·零随机·零网络）")
    expA()
    expB()
    expC()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 继承DAG / B 概念普及 / C 该不该引入 / D 决策路牌）")
    print("  A 13 节点 · 承袭边 9 · MCP 底座 12 · 最长链 3 · hub=LangChain（影响面 7）· MAF 入度 2")
    print("  B 老概念平均普及 > 新概念（6.50 > 4.75）· 只被 1 家=DSPy 独立 DNA · 普及王=Chain/Runnable")
    print("  C 10 场景决策断言 10/10 · 不引入出场 1（真实场景=线性）· 阈值 0.45")
    print("  D 13 站路牌 · 五主线年谱 · 012 客服三套对照：有状态/循环/恢复→图式引入，纯线性→直写")
    print("一句话：框架 = 拿『学习面+版本风险』买不同的痛点解决组合；引进谁的判据 =")
    print("         需求权重里『状态/循环/恢复』占不占分——占则上图，占轻则轻量，不占则直写")
    print("done · 一键复现：python code/notebooks/_tools/framework_lineage_demo.py")


if __name__ == "__main__":
    main()
    W(f"[framework_lineage_demo] wall-clock {perf_counter() - _t0:.3f} s")
