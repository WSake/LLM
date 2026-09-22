#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""07-应用框架 · 00-框架分类学（知识地图 §7.0/§7.1/§7.13/§7.14）探针：
「先分类型再比优劣」—— 把框架混战清成一张可计算的地图。
量四本账：
  A 分类学账 —— 8 类 × 24 框架（代表·归属·跨类双栖·类代表数·类间重叠 Jaccard）
  B 痛点覆盖账 —— §7.0 四痛点（状态/组装/可观测·评测/成本·可靠）× 24 框架主/次评分
                 → 每痛点主业家数、每框架覆盖维度数分布（重量谱）、协议层特判
  C 继承与影响账 —— §7.13 谱系 DAG（承袭边 6 · MCP 底座边 9）→ 最长链/hub/递归影响面
  D 选型决策树 —— 「该在哪用」场景→类别+代表框架 + 2026 四层收敛断言 + 选择树 10 场景
口径：类别归属/痛点评分/承袭边/决策规则 = 作者按知识地图 §7 整理与定性（要素事实=非本机实测）；
     A-D 全部统计（跨类数/Jaccard/主攻·覆盖数/图算法/决策稳定性）= 本机确定性计算。
纯 stdlib · 零网络 · 零随机 · stdout 逐字节可复现；wall-clock 只进 stderr。
"""
import sys
from time import perf_counter

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ---- 8 大类（§7.1 顺序：同层比较的前提）----
CATS = [
    ("通用编排 SDK", "把模型+工具+检索串成链"),
    ("Workflow/图引擎", "有状态·可分支·可恢复的流程"),
    ("RAG 框架", "知识库摄取·检索·问答"),
    ("Agent 框架", "Agent Loop 与多 Agent 协作"),
    ("低代码平台", "非工程师也能搭应用"),
    ("Prompt 编程/优化", "把 prompt 当程序·可编译可优化"),
    ("协议层", "跨栈互操作标准（工具/通信/剪贴板）"),
    ("前端 SDK", "浏览器/Node 端开发"),
]
CAT_NAME = {i: n for i, (n, _) in enumerate(CATS, start=1)}

# ---- 24 个代表框架：cat=归属类别（可多属）· pain=四痛点主/次（主=2 次=1 ·=0）----
# pain 四元顺序与 §7.0 四痛点一致：状态 / 组装 / 可观测·评测 / 成本·可靠
PNAME = ["状态", "组装", "可观测·评测", "成本·可靠"]
FRMW = {
    "LangChain":         dict(cat=[1], pain=[1, 2, 1, 1]),
    "Haystack":          dict(cat=[1, 3], pain=[1, 2, 1, 1]),
    "Semantic Kernel":   dict(cat=[1], pain=[1, 2, 1, 1]),
    "LangGraph":         dict(cat=[2, 4], pain=[2, 1, 1, 1]),
    "Temporal":          dict(cat=[2], pain=[2, 1, 0, 1]),
    "LlamaIndex":        dict(cat=[3], pain=[1, 2, 1, 0]),
    "RAGFlow":           dict(cat=[3], pain=[1, 2, 1, 0]),
    "AutoGen":           dict(cat=[4], pain=[2, 1, 1, 0]),
    "CrewAI":            dict(cat=[4], pain=[1, 1, 1, 0]),
    "OpenAI Agents SDK": dict(cat=[4], pain=[2, 1, 1, 1]),
    "Google ADK":        dict(cat=[4], pain=[2, 1, 1, 0]),
    "Pydantic AI":       dict(cat=[4], pain=[1, 1, 1, 1]),
    "Dify":              dict(cat=[5], pain=[2, 2, 2, 2]),
    "Flowise":           dict(cat=[5], pain=[2, 2, 1, 1]),
    "Coze":              dict(cat=[5], pain=[2, 2, 2, 1]),
    "n8n":               dict(cat=[5], pain=[2, 1, 1, 2]),
    "DSPy":              dict(cat=[6], pain=[0, 0, 2, 1]),
    "Outlines":          dict(cat=[6], pain=[0, 1, 0, 1]),
    "MCP":               dict(cat=[7], pain=[0, 1, 0, 0]),
    "A2A":               dict(cat=[7], pain=[0, 1, 0, 0]),
    "ACP":               dict(cat=[7], pain=[0, 1, 0, 0]),
    "Vercel AI SDK":     dict(cat=[8], pain=[1, 2, 0, 0]),
    "Mastra":            dict(cat=[8], pain=[1, 2, 1, 0]),
    "LlamaIndex.TS":     dict(cat=[8], pain=[1, 2, 1, 0]),
}

# ---- §7.13 谱系（承袭边；年份=要素事实）----
NODES = [("LangChain", "2022"), ("LlamaIndex", "2023"), ("AutoGen", "2023"),
         ("CrewAI", "2024"), ("LangGraph", "2024"), ("LangGraph Platform", "2025"),
         ("DSPy", "2023"), ("Swarm", "2024"), ("OpenAI Agents SDK", "2025"), ("MCP", "2024")]
INHERIT = [("LangChain", "LlamaIndex"), ("LangChain", "AutoGen"),
           ("LangChain", "CrewAI"), ("LangChain", "LangGraph"),
           ("LangGraph", "LangGraph Platform"), ("Swarm", "OpenAI Agents SDK")]
BASE_FROM = "MCP"   # 2025 起成为所有框架的工具接口公共底座（要素事实）


def jaccard_a(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def expA():
    print("=" * 72)
    print("[账 A] 分类学账：8 类 × 24 框架 —— 先分类型再比优劣（同层才比 · 跨层不排高下）")
    print("=" * 72)
    for i, (name, solve) in enumerate(CATS, start=1):
        members = sorted(k for k, v in FRMW.items() if i in v["cat"])
        print(f"  【{i} {name}】{solve}")
        print(f"       代表 {len(members)}：{' · '.join(members)}")
    multi = sorted(k for k, v in FRMW.items() if len(v["cat"]) > 1)
    print(f"  → 跨类双栖 {len(multi)}：{' · '.join(multi)} ——「分类=主次不是互斥标签」")
    per_cat = sorted(((len([k for k, v in FRMW.items() if i in v["cat"]]), i, n)
                      for i, (n, _) in enumerate(CATS, start=1)), reverse=True)
    crowd = per_cat[0]
    print(f"  → 类代表数（降序）：{' > '.join(f'{n} {c}' for c, _, n in per_cat)} "
          f"——Agent 类最挤（2025 最年轻也最拥挤的赛道）")
    pairs = []
    for a in range(1, 9):
        for b in range(a + 1, 9):
            sa = [k for k, v in FRMW.items() if a in v["cat"]]
            sb = [k for k, v in FRMW.items() if b in v["cat"]]
            pairs.append((jaccard_a(sa, sb), a, b))
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))
    nz = sum(1 for j, _, _ in pairs if j > 0)
    print(f"  → 类间重叠 Jaccard（仅非零 {nz} 对）：")
    shown = 0
    for j, a, b in pairs:
        if j <= 0:
            break
        shared = sorted(set(k for k, v in FRMW.items() if a in v["cat"])
                        & set(k for k, v in FRMW.items() if b in v["cat"]))
        print(f"       [{CAT_NAME[a]} × {CAT_NAME[b]}] = {j:.3f}（共享 {' · '.join(shared)}）")
        shown += 1
    print(f"      其余 {len(pairs) - shown} 对 = 0 —— 六类两两不相交=稀疏邻接；"
          f"同分类才有『比优劣』的合法前提")
    assert multi == ["Haystack", "LangGraph"]
    assert crowd[0] == 6 and crowd[1] == 4          # Agent 类最挤 6 家
    assert nz == 2                                  # 只有 2 对共享框架
    print(f"  → 断言：跨类双栖=【{'·'.join(multi)}】· 最挤类=Agent（6 家）· Jaccard 非零对={nz}")


def expB():
    print()
    print("=" * 72)
    print("[账 B] 痛点覆盖账：§7.0 四痛点 × 24 框架（主=2 主业 · 次=1 顺带 · ·=0 不碰）")
    print("=" * 72)
    for name in FRMW:
        p = FRMW[name]["pain"]
        sym = "".join("主" if v == 2 else ("次" if v == 1 else "·") for v in p)
        cover = sum(1 for v in p if v > 0)
        print(f"    {name:<20}{sym}   覆盖 {cover}/4")
    per_pain = []
    for idx, pn in enumerate(PNAME):
        prim = sorted(k for k, v in FRMW.items() if v["pain"][idx] == 2)
        per_pain.append((len(prim), pn, prim))
    for n, pn, prim in per_pain:
        print(f"  → 主业家数 · {pn} = {n}：{' '.join(prim)}")
    bucket = {}
    for k in FRMW:
        c = sum(1 for v in FRMW[k]["pain"] if v > 0)
        bucket.setdefault(c, []).append(k)
    order = sorted(bucket, reverse=True)
    print(f"  → 覆盖维度数分布：{' · '.join(f'{c} 维 {len(bucket[c])} 家' for c in order)}")
    for c in order:
        print(f"        {c} 维：{' '.join(bucket[c])}")
    n_heavy = len(bucket[max(order)])
    print(f"  → 组装与状态是框架最常当主业的两类（各 {per_pain[1][0]}/{per_pain[0][0]} 家）——"
          f"框架=把『状态执行 + 接口组装』的惯例打包；")
    print(f"    可观测·评测与成本·可靠仍是冷板凳（各 {per_pain[2][0]}/{per_pain[3][0]} 家）——"
          f"这正是 11-评测章与『成本账』还要单独讲的原因。")
    proto = [k for k, v in FRMW.items() if v["cat"] == [7]]
    assert all(sum(1 for x in FRMW[k]["pain"] if x > 0) == 1 and
               FRMW[k]["pain"][0] == 0 and FRMW[k]["pain"][3] == 0
               for k in proto)
    assert n_heavy == 10 and len(bucket[1]) == 3 and len(bucket[4]) == 10
    print(f"  → 覆盖 1 维的 3 家恰是协议层【{' · '.join(proto)}】——协议不是框架：不承诺状态/观测/成本，"
          f"只统一『组装』里最痛、最该先定死的一小块接口。")
    print(f"  → 教学点：第一刀先问『要协议还是框架』——协议 bundle 最小、最该先固定；"
          f"框架才谈状态/观测/成本（后文 11-MCP协议 章展开）。")


def expC():
    print()
    print("=" * 72)
    print("[账 C] 继承与影响账：§7.13 谱系 DAG ——「框架会消亡、概念永存」的可计算版本")
    print("=" * 72)
    child = {n: [] for n, _ in NODES}
    for a, b in INHERIT:
        child[a].append(b)
    for n in child:
        child[n] = sorted(child[n])
    print("  §7.13 承袭边 6 条：")
    print("    " + " · ".join(f"{a}→{b}" for a, b in INHERIT))
    print(f"  MCP 底座边 {len({n for n, _ in NODES} - {BASE_FROM})} 条："
          f"{BASE_FROM} → 其余全部节点（2025 工具接口公共底座=要素事实）")
    # 最长承袭链（DAG 有向 DP；memo 保证确定性）
    memo = {}

    def depth(n):
        if n in memo:
            return memo[n]
        d = 1 + max((depth(c) for c in child[n]), default=0)
        memo[n] = d
        return d
    longest = max(depth(n) for n, _ in NODES)
    assert longest == 3
    print(f"  → 最长承袭链 {longest} 节点：LangChain(2022)→LangGraph(2024)→LangGraph Platform(2025)"
          f"（一条 3 年 3 跳的衣钵）")
    # hub=直接后代最多的节点（平票按字典序=确定性）
    hub = sorted(((len(v), k) for k, v in child.items()), reverse=True)[0][1]
    assert hub == "LangChain"
    # 递归影响面（显式栈 DFS · 排序保证确定性）
    seen = set()
    stack = list(child[hub])
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        for c in child.get(x, []):
            stack.append(c)
    print(f"  → 概念普及者 = {hub}：直接后代 {len(child[hub])}(hub 第一) · 递归影响面 {len(seen)}"
          f"（{', '.join(sorted(seen))}）——")
    print(f"    一个 2022 年『Chain』概念，2023-2025 长出 LlamaIndex/AutoGen/CrewAI/LangGraph"
          f"一整棵分类树")
    base_infl = len({n for n, _ in NODES} - {BASE_FROM})
    print(f"  → MCP 底座影响面 {base_infl}：2025 的 hub 从『框架』换成『协议』——工具接口不再每家自己接")
    dspy_free = all(a != "DSPy" and b != "DSPy" for a, b in INHERIT)
    assert dspy_free
    print(f"  → DSPy 无承袭边 = Prompt 编程范式走 LangChain 之外的独立路径（评测当主业，见账 B）")
    print(f"    断言：最长链={longest} · hub={hub} · MCP 影响面={base_infl} · DSPy 独立")


def expD():
    print()
    print("=" * 72)
    print("[账 D] 选型决策树：「该在哪用」—— 第一层只定『类』，代表框架是第二层")
    print("=" * 72)
    SCENARIOS = [
        ("线性调模型+组提示 · 少分支", 1, "LangChain", "链式最省心；链条一长就换图引擎（账 C 的衣钵）"),
        ("文档/知识库检索问答", 3, "LlamaIndex", "RAG 专精——文档解析/索引/高级检索理解最深（衔接 08-RAG）"),
        ("有分支/循环/恢复/重试的流程", 2, "LangGraph", "有向状态图·checkpoint·递归上限开箱即用（02-章已实测）"),
        ("非工程师快速搭业务", 5, "Dify", "拖拽工作流 + RAG + 发布 API，企业知识库起手最快"),
        ("多 Agent 角色化协作", 4, "CrewAI", "Role/Goal/Backstory 团队，demo 效果好、上手极快"),
        ("轻量单 Agent · 少依赖", 4, "OpenAI Agents SDK", "代码量极小完全可控 = 回归轻量的趋势（Swarm 后继）"),
        ("质量敏感 · 可自动评测的抽取/转换", 6, "DSPy", "prompt 当程序·对真实指标自动优化（评测当主业）"),
        ("跨栈工具互操作 · 一次实现处处可用", 7, "MCP", "协议非框架——统一工具接口（里程碑 013 关键章）"),
        (".NET/微软企业栈", 1, "Semantic Kernel", "与 Azure/.NET 深度绑定·企业治理友好"),
        ("浏览器/Node 前端团队", 8, "Vercel AI SDK", "前端友好——流式 UI/多模型提供方一套接"),
    ]
    ok = 0
    for i, (scene, cat, fw, why) in enumerate(SCENARIOS, start=1):
        assert cat in FRMW[fw]["cat"], f"{fw} 不属于类别 {cat}"
        ok += 1
        print(f"  {i:>2}. {scene}")
        print(f"     → 类别 {cat}【{CAT_NAME[cat]}】· 代表 {fw}")
        print(f"        理由：{why}")
    print(f"  → 选择树断言 {ok}/{len(SCENARIOS)} 全过：决策确定性 + 类别归属有效 + 框架∈该类")
    # §7.13 趋势断言：2026 收敛成 4 层（协议层 + 轻量SDK + 图引擎 + 平台）
    LAYERS = {
        "协议层": ["MCP", "A2A", "ACP"],
        "轻量SDK/运行时": ["OpenAI Agents SDK", "Pydantic AI", "Vercel AI SDK", "Mastra"],
        "图/Workflow 引擎": ["LangGraph"],
        "平台/低代码": ["Dify", "Flowise", "Coze", "n8n"],
    }
    all_mem = [m for mem in LAYERS.values() for m in mem]
    assert len(all_mem) == len(set(all_mem)) == 12, "四层成员互斥且共 12 家"
    assert all(m in FRMW for m in all_mem)
    print()
    print("  §7.13 趋势断言：2025-2026 框架收敛成 4 层（协议层 + 轻量SDK + 图引擎 + 平台）：")
    for layer, members in LAYERS.items():
        print(f"    {layer:<18}{' · '.join(members)}")
    print("    → 断言 4 层非空且互斥（12 家互不复用；LangGraph Platform=图引擎的托管版，不再单列）；")
    print("      引入顺序建议 = 先协议（MCP）→ 轻量 SDK → 复杂性不够再上图引擎/平台——"
          "与『覆盖维度越宽越重』（账 B）同向")


def main():
    print("=" * 72)
    print("framework_taxonomy_demo：07-应用框架 · 00-框架分类学（知识地图 §7.0/§7.1/§7.13/§7.14）")
    print("    『先分类型再比优劣』——框架为什么出现 · 8 类分类学 · 继承与影响 · 该在哪用")
    print("=" * 72)
    print("[0] 口径：类别归属/痛点评分/承袭边/决策规则 = 作者按知识地图 §7 整理与定性")
    print("    （要素事实=非本机实测）；A-D 全部统计 = 本机确定性计算（纯 stdlib·零随机）")
    expA()
    expB()
    expC()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 分类学 / B 痛点覆盖 / C 继承影响 / D 选型决策）")
    print("  A 8 类 × 24 框架 · 跨类双栖 2（Haystack·LangGraph）· Agent 类最挤 6 家 · Jaccard 非零对 2")
    print("  B 主业家数 组装 11 · 状态 9 · 可观测·评测 3 · 成本·可靠 2；覆盖 4 维 10 家 / 1 维 3 家=协议层")
    print("  C 最长承袭链 3 节点 · hub=LangChain（影响面 5）· MCP 底座影响面 9 · DSPy 独立")
    print("  D 选择树 10 场景断言 10/10 · 2026 四层互斥断言 12/12 · 框架会消亡、概念永存")
    print("一句话：框架 = 把『状态执行 + 接口组装 + 可观测 + 成本惯例』打包的库；第一刀先分类型")
    print("（协议层单独一档），再在同类里比优劣——不是哪家最强，而是哪类匹配你当前的主痛点")
    print("done · 一键复现：python code/notebooks/_tools/framework_taxonomy_demo.py")


if __name__ == "__main__":
    main()
    W(f"[framework_taxonomy_demo] wall-clock {perf_counter() - _t0:.3f} s")
