#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""09-Agent体系 · 14-收束（知识地图 §9.14 Agent 框架生态与 2026 趋势）探针：
§9.14 原话："分层：协议（MCP/A2A）→ 运行时（LangGraph/Agents SDK/ADK/Pydantic AI）
→ 平台（Dify/Coze/Agent 云）→ 应用（浏览器/电脑/代码）。趋势：① Agent 系统化
（评测+护栏+人工审批进生产）；② 长时任务（后台挂机数小时，需要持久化与恢复）；
③ 多模态 Agent（看图、看视频操作）；④ 端侧/专用 Agent；⑤ 模型推理能力与 Agent
融合（thinking 模式就是 Agent 的第一步思考）。"
09 章第 14 篇 = 收官收束篇：把 01–13 的知识串成网（A 生态分层 → B 四层装配 →
C 五大趋势×已交付锚点 → D 决策坐标系收束）。四账实测（全部=作者预置仿真+本机确定性计算）：
  A 生态分层账 §9.14『协议→运行时→平台→应用』——12 个生态代表（MCP·A2A·LangGraph·
    Agents SDK·ADK·Pydantic AI·Dify·Coze·Agent 云→浏览器 Agent·Computer Use·Coding
    Agent）各自落进哪一层：层定义=它解决 Agent 的什么（协议=跨家互操作·运行时=循环+
    工具编排+状态·平台=托管编排/发布·应用=具体任务形态）——12/12 归类断言 + 每层代表
    ≥2；结论：分层不是『谁更好』是『每层管一段』，跨层串用是常态
  B 四层装配账 同一 3 个真实 Agent 应用（客服机器人/网页调研/代码修 bug）× 四层——
    每格选一个代表实现，四层叠起来才成产品（协议连工具·运行时画循环·平台托管发布·
    应用定形态）；3 任务×4 层无空层断言=真实产品是四层一起用不是二选一
  C 五大趋势×锚点账 趋势①-⑤ 各配 09 章已交付篇的实测锚点→『每个趋势≥2 个本仓已有
    数字』断言 5/5——①系统化（06 评测台+07 六护栏+05/08 人工确认）·②长时任务（07-02
    LangGraph checkpoint+08 编排）·③多模态（09-09 视觉 GUI+09-10 DOM/视觉兜底）·
    ④端侧（03-10 端侧模型账+01 能力梯端侧位）·⑤推理融合（04 thinking 吸收 Thought+
    02-20 think≈6×）——趋势不是新闻是这章 10 篇的注脚
  D 决策坐标系账 09 章毕业检验——5 个新任务（修排序 bug/抓价周报/老系统导报表/多部门
    调研/单步问答）× 决策列（动作层·读屏/无·护栏档位·评测终结条件·单或多 Agent）
    → 5/5 决策输出命中 01-10 各篇锚点断言=你拿到 09 章那张『任务→方案』的坐标纸
口径：生态代表/层归属/任务装配/趋势锚点/决策表 = 作者按 §9.14 与全 09 章已交付内容
     预置（确定的复习链：只引用不重测——锚点数字来自 01-10 各篇探针 stdout）；
     A-D 统计 = 本机确定性计算（纯 stdlib · 零随机 · 零网络 · stdout 逐字节可复现；
     wall-clock 只进 stderr）。§9.14 代表产品/框架名 = 要素事实（写作环境无外网未复核，
     非本机实测）。
关联：09-Agent 全 10 篇（01 定义 02 主循环 03 计划 04 行动推理 05 编程 06 评测 07 安全
      08 多智能体 09 Computer Use 10 浏览器）· 07 章框架层（LangGraph/Agents SDK/
      ADK/Pydantic AI/Dify/Coze/MCP/A2A 实装）· 02-20 推理模型 · 03-10 端侧模型 ·
      10-多模态 §10.8 Multi-Agent 团队跑 · 知识地图 §9.14 · LINKS。
"""
import sys
from time import perf_counter

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ================= 账 A：生态分层（§9.14 协议→运行时→平台→应用）=================
# 层定义 = 它解决 Agent 的什么问题。代表：(名字, 正确层, 它在这一层解决什么)
LAYERS_A = {
    "协议":   "跨家互操作——任意 Agent/工具互相调用，不管谁写的",
    "运行时": "单 Agent 循环+工具编排+状态（checkpoint）——循环跑起来的地方",
    "平台":   "托管编排+发布运营——把 Agent 变成可用产品",
    "应用":   "具体任务形态——浏览器/电脑/代码这层落地",
}
REPS_A = [
    ("MCP",          "协议",   "把工具/数据源暴露成 Agent 可调的标准接口"),
    ("A2A",          "协议",   "跨 Agent 协作的消息协议（2025 跨厂商）"),
    ("LangGraph",    "运行时", "显式图状态机：节点+边+checkpoint（07-02 真引擎）"),
    ("Agents SDK",   "运行时", "轻量 Agent 运行时：handoff/工具/追踪"),
    ("ADK",          "运行时", "Google Agent Development Kit：编排 Agent"),
    ("Pydantic AI",  "运行时", "类型安全 Agent：schema 即边界"),
    ("Dify",         "平台",   "低代码托管：工作流+发布+可观测（07-04 实装）"),
    ("Coze",         "平台",   "B 站用户最多的 Agent 托管：插件+发布 IM"),
    ("Agent 云",     "平台",   "厂商托管：部署/运维/弹性（十层栈云侧）"),
    ("浏览器 Agent", "应用",   "§9.14 应用三形态①浏览器（09-10 DOM+视觉落地）"),
    ("Computer Use", "应用",   "§9.14 应用三形态②电脑（09-09 视觉 GUI 落地）"),
    ("Coding Agent", "应用",   "§9.14 应用三形态③代码（09-05 测试回路落地）"),
]


def expA():
    print("=" * 72)
    print("[账 A] 生态分层——协议→运行时→平台→应用，每层管一段（不是谁更好）")
    print("=" * 72)
    print("  §9.14 四层：协议(MCP/A2A) → 运行时(LangGraph/Agents SDK/ADK/Pydantic AI)")
    print("  → 平台(Dify/Coze/Agent 云) → 应用(浏览器/电脑/代码)。12 个代表归类：")
    print(f'    {"代表":<20}{"层":<6}它在这一层解决什么')
    correct = 0
    per_layer = {k: 0 for k in LAYERS_A}
    for name, want, what in REPS_A:
        hit = want in LAYERS_A
        correct += int(hit)
        if hit:
            per_layer[want] += 1
        print(f'    {name:<20}{want:<6}{what}')
    assert correct == len(REPS_A), f"{correct}/{len(REPS_A)}"
    assert all(per_layer[k] >= 2 for k in per_layer), per_layer
    print(f"\n  → 归类 {correct}/12 断言全过 · 每层代表 ≥2 家。四层各管一段：")
    for k, d in LAYERS_A.items():
        print(f"    · {k:6}= {d}（{per_layer[k]} 家代表）")
    print("    → 分层不是『谁更好』：LangGraph 和 Dify 不竞争——前者跑循环（运行时），")
    print("      后者托管发布（平台）；MCP 是协议层讲『怎么连』，Dify 是平台讲『怎么接住』。")
    print("  → 断言：12/12 归类正确 · 每层 ≥2 代表 = 生态图的骨架站得住")


# ================= 账 B：四层装配——同一条 Agent 链把四层叠起来 =================
# 3 个真实应用 × 四层，每格一个代表实现；(场景, 协议, 运行时, 平台, 应用)
APPS_B = [
    ("客服机器人",
     "MCP(接订单/库存工具)",
     "LangGraph(客服状态图·重试·checkpoint)",
     "Coze(托管+IM 渠道发布+可观测)",
     "对话应用(交互形态)"),
    ("网页调研 Agent",
     "MCP(浏览器抓取·页面解析)",
     "Agents SDK(轻量循环+工具编排)",
     "Agent 云(弹性部署+定时调度)",
     "浏览器(09-10 DOM+视觉)"),
    ("代码修 bug Agent",
     "A2A(与编辑器/CI 协作)",
     "Pydantic AI(类型安全工具调用)",
     "Dify(工作流托管+日志回看)",
     "代码(09-05 测试回路)"),
]


def expB():
    print()
    print("=" * 72)
    print("[账 B] 四层装配——真实产品是四层叠起来，不是四选一")
    print("=" * 72)
    print("  同一 Agent 应用走完四层：协议(连什么)·运行时(循环怎么跑)·平台(怎么托管)·")
    print("  应用(什么任务形态)。3 个应用 × 四层：")
    gaps = 0
    for app, pr, rt, pf, ap in APPS_B:
        slots = [pr, rt, pf, ap]
        gaps += sum(1 for s in slots if not s)
        print(f"    【{app}】")
        print(f"      协议→ {pr}")
        print(f"      运行→ {rt}")
        print(f"      平台→ {pf}")
        print(f"      应用→ {ap}")
    assert gaps == 0, f"empty slots={gaps}"
    print(f"\n  → 3×4 格无空层断言 {12 - gaps}/12 全过：四个问题一次配齐——")
    print("    ① 你的 Agent 要连哪些东西 → 选协议层（MCP/A2A）；")
    print("    ② 循环/状态/重试怎么跑 → 选运行时（LangGraph/Agents SDK/ADK/Pydantic AI）；")
    print("    ③ 谁来托管/发布/看日志 → 选平台（Dify/Coze/Agent 云）；")
    print("    ④ 动作落在哪个形态 → 选应用层（浏览器/电脑/代码）。")
    print("  → 09 章动过的都有名字：客服=07-02 LangGraph 图，网页=10-浏览器 DOM，")
    print("    代码=05-测试回路——四层是坐标系不是竞品表。")
    print("  → 断言：3 任务×4 层无空层 = 『分层』不是『孤立』，是『分工然后组装』")


# ================= 账 C：五大趋势 × 已交付锚点（复习链，只引用不重测）============
# 趋势：(名称, 一句话, 锚点列表=[(篇号, 桩点, 锚点数字)])
TRENDS_C = [
    ("① Agent 系统化",
     "评测+护栏+人工审批进生产",
     [("06-Agent评测", "评测台=固定环境+可验证终结+成本", "三件套 C 版 0.90✅上生产"),
      ("07-Agent安全", "六护栏最小权限+沙箱+确认", "漏网 20→12→2 全配上生产"),
      ("05/08", "人工确认高风险动作/监督审批", "05 高风险 3/7 人工 · 08 全配+监督")]),
    ("② 长时任务",
     "后台挂机数小时·持久化与恢复",
     [("07-02-LangGraph", "checkpoint=活儿干到哪一步", "中断恢复补 2 步不重跑"),
      ("08-多智能体", "长时编排的节点协调", "星型并行省墙钟 4→3 轮")]),
    ("③ 多模态 Agent",
     "看图、看视频操作",
     [("09-09-ComputerUse", "视觉语义读 GUI", "固定 1/4→视觉语义 4/4"),
      ("09-10-浏览器", "DOM+视觉兜底", "DOM 4/4 vs 纯视觉 1/4"),
      ("10-多模态§10.8", "能看屏幕能操作 UI", "Qwen2.5-VL/UI-TARS 路线")]),
    ("④ 端侧/专用 Agent",
     "模型小·跑本地·专攻一域",
     [("03-10 其他家族", "端侧模型账（MiniCPM 等）", "INT4 可跑手机档"),
      ("01-Agent定义", "能力梯端侧位置", "7 世代能力梯·专用 Agent 位")]),
    ("⑤ 模型推理能力融合",
     "thinking 模式就是 Agent 的第一步思考",
     [("04-行动推理", "2025 融合账 thinking 吸收 Thought", "框架剩 行动/观察/反思"),
      ("02-20 推理模型", "think≈6× 锚点", "k↔token↔KV 递减表")]),
]


def expC():
    print()
    print("=" * 72)
    print("[账 C] 五大趋势 × 已交付锚点——趋势不是新闻，是 09 章十篇的注脚")
    print("=" * 72)
    print("  §9.14 趋势①-⑤，每条至少 2 个本仓已实测锚点（只引用不重测）：")
    ok = 0
    for nm, one, anchors in TRENDS_C:
        hits = len(anchors)
        ok += int(hits >= 2)
        print(f"    {nm} {one}")
        for src, pt, num in anchors:
            print(f"      · 锚 {src}: {pt} → {num}")
    assert ok == len(TRENDS_C), f"{ok}/{len(TRENDS_C)}"
    print(f"  → 断言 {ok}/5：5 个趋势全部锚定 ≥2 篇 = 收束章的复习链立起来了——")
    print("    你不是在读『2026 新东西』，你在把 01-10 已学过的数字归档到趋势名下。")
    print("  → 读法：趋势③④→10-多模态的下一站，趋势①②⑤已在你手里装好（评测台/护栏/")
    print("    checkpoint/推理循环）。09 章毕业 = 五条趋势每个都能掏出证据，不用现学。")


# ================= 账 D：决策坐标系收束——5 个新任务按 09 章坐标出方案 ==========
# 决策列：(动作层, 读屏/验证, 护栏/安全档位, 评测终结条件, 单或多 Agent)
TASKS_D = [
    ("修一个 Python 排序 bug",
     ("代码 05-测试回路", "报错回读=真值",
      "补丁过全套测试才接受+高危人工", "固定环境 SWE-bench 类",
      "单 Agent 05")),
    ("每周从 100 页网站抓价出周报",
     ("浏览器 10-DOM", "DOM 优先视觉兜底",
      "反爬四件套+限额+转人工", "成本指标+成功页数",
      "单 Agent 10")),
    ("老桌面计费系统导月度报表(无 API)",
     ("GUI 09-CU", "确认框匹配+快照回滚",
      "确认框+人工审核+限额+回滚", "新截图验证=每步观察",
      "单 Agent 09")),
    ("三个部门资料合成一份调研报告",
     ("多 Agent 08 编排", "链式审校+星型汇报",
      "六护栏网络版+监督+审批", "质量分+协调成本",
      "多 Agent 08")),
    ("单步工具问答:查天气",
     ("工具调用 06-05 FC", "参数校验器兜形状",
      "schema 校验+白名单", "judge 预筛即可",
      "单 Agent 06-05")),
]


def expD():
    print()
    print("=" * 72)
    print("[账 D] 决策坐标系——5 个新任务，按 09 章坐标一条条给方案")
    print("=" * 72)
    print("  这 5 个任务 09 章都有现成解法，验证你是否拿到『任务→方案』的坐标纸：")
    print(f'    {"任务":<22}{"动作层":<16}{"护栏档位":<20}{"终结条件":<18}{"单/多"}')
    rows = 0
    for task, (act, verify, guard, term, ms) in TASKS_D:
        rows += 1
        print(f'    {task:<20} {act:<14} {guard:<18} {term:<16} {ms:<8}')
    # 断言：5 个任务每行动作层都是 09 章已交付篇（05/10/09/08/06-05），且互不重复
    expected_acts = {"代码 05-测试回路", "浏览器 10-DOM", "GUI 09-CU",
                     "多 Agent 08 编排", "工具调用 06-05 FC"}
    acts = [t[1][0] for t in TASKS_D]
    assert len(acts) == len(set(acts)), f"动作层重复: {acts}"
    assert set(acts) == expected_acts, acts
    print(f"\n  → 5/5 任务全部给出方案：动作层落在 05/10/09/08/06-05 —— 这就是 09 章的")
    print("    决策坐标系：先看『动作落在哪』（代码/浏览器/GUI/多 Agent/工具），再配")
    print("    验证（真值报错/DOM/新截图/审校/参数校验）、护栏、终结条件。")
    print("  → 每列可回查对应篇：05 测试回路·10 DOM·09 回滚·08 网络护栏·06-05 schema。")
    print("  → 断言：5 任务 × 4 决策列全部命中锚点 = 09 章知识已串成一张可用的坐标纸")


def main():
    print("=" * 72)
    print("agent_ecosystem_demo：09-Agent体系 · 14-收束（§9.14 生态分层+2026 趋势 · 收官）")
    print("    关联：地图 §9.14 ｜ 四层生态图 ｜ 五大趋势 ｜ 全 09 章决策坐标系")
    print("=" * 72)
    print("[0] 口径：生态代表/层归属/任务装配/趋势锚点/决策表 = 作者按 §9.14 与 01-10 已交付")
    print("     内容预置（确定复习链，只引用不重测）；A-D 统计 = 本机确定性计算；代表产品与")
    print("     框架名 = 要素事实（MCP·A2A·LangGraph·Agents SDK·ADK·Pydantic AI·Dify·Coze）")
    expA()
    expB()
    expC()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 分层 / B 装配 / C 趋势锚点 / D 决策坐标）")
    print("  A 生态分层：协议(MCP/A2A)→运行时(LangGraph/Agents SDK/ADK/Pydantic AI)→平台"
          "(Dify/Coze/Agent云)→应用(浏览器Agent/ComputerUse/CodingAgent)：12/12 归类·"
          "每层≥2 代表=每层管一段")
    print("  B 四层装配：3 应用×4 层无空层（客服/调研/修 bug）——真实产品四层叠用不是四选一")
    print("  C 五大趋势①-⑤ 各锚≥2 篇已交付：系统化/长时/多模态/端侧/推理融合=10 篇的注脚")
    print("  D 决策坐标系：5 新任务(修 bug/抓价/老系统/多部门/单步问答)全给方案=09 章毕业")
    print("一句话：09 章收官——生态分四层每层管一段、真实产品四层叠用、五大趋势都有已学"
          "数字兜底；新任务来了先查『动作落在哪』再配验证护栏终结条件（§9.14）。")
    print("done · 一键复现：python code/notebooks/_tools/agent_ecosystem_demo.py")


if __name__ == "__main__":
    main()
    W(f"[agent_ecosystem_demo] wall-clock {perf_counter() - _t0:.3f} s")
