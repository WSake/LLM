# 🎯 OpenAI Agents SDK：轻量 Agent 运行时 —— 「回归轻量」的手转交范式：一个 Agent 对象 + 一个 Runner 函数，把工具/护栏/手转交全做成了一等公民（Agent 系三连收尾篇）

> 对应知识点：知识地图 §7.11（**OpenAI Agents SDK（2025；位置：轻量 Agent 运行时）**——核心=Agent（tools+instructions）→ Handoff（Agent 之间转移权）→ Guardrails（输入输出校验）→ Sessions/Tracing；代码量极小、完全可控；代表「回归轻量、靠原生功能」趋势，不再追求大而全框架；Swarm（2024 实验）为其前身；同类 Google ADK / Pydantic AI）、§7.1（类 **4 Agent 框架**：LangGraph / AutoGen / CrewAI / **OpenAI Agents SDK** / Google ADK / Pydantic AI，六家最挤）、§7.14（决策树：「多 Agent 协作 → CrewAI / AutoGen / **Agents SDK（handoff）**」；「想少依赖框架 → OpenAI Agents SDK / Pydantic AI + 自己写工具协议（套 MCP）」）、§17.5.11（**OpenAI Agents SDK / 轻量 Agent 运行时**——以最小代码实现 Agent（tools+handoff+guardrail）的官方轻量 SDK；框架瘦身趋势代表）、§6.9（多 Agent=多倍 token 与延迟，先证明单 Agent 不行）。
> 前置：[00-框架分类学](./00-框架分类学.md)（账 A 类 4 六家最挤、账 B 状态主业；§7.13 谱系：Swarm（实验）→ Agents SDK（轻量）；选择树把「Agent 复杂流程」指给 LangGraph、「多 Agent 协作」指给 CrewAI/AutoGen/Agents SDK(handoff)）；[07-AutoGen-AG2-Microsoft-Agent-Framework](./07-AutoGen-AG2-Microsoft-Agent-Framework.md)（§7.8 **多 Agent 对话范式**=本篇左手：AutoGen『Agent 互聊』→ AG2 1.0 函数式→MAF；AG2 的一次 run=一条 prompt→一次 LLM 调用=与 Runner.run_sync 同族的最简执行原语，但 AG2 的续谈要显式 ask() 喂历史、本篇手转交=引擎自动）;[08-CrewAI](./08-CrewAI.md)（§7.14 角色化分工=本篇右手：role/task/crew 三一等公民的声明式团队流水线，缺了必填 role 构造期即 ValidationError；本篇反过来——没有 role/backstory/task 配方，就是 name+instructions 一把梭）;更早：[05-Function-Tool-Calling](../06-应用开发/05-Function-Tool-Calling.md)（工具声明→入参约束→执行→回喂，本篇在同引擎内封装成 function_tool 一等公民）、[06-记忆系统](../06-应用开发/06-记忆系统.md)（Sessions/会话层=要素事实的跨会话抽象）。
> 动手：`python code/notebooks/_tools/openai_agents_demo.py`（**真实 openai-agents 0.17.0 引擎**本机执行；模型决策=预置 stub（继承真 `OpenAIChatCompletionsModel` 重写 `get_response`）保确定性；零网络 · 零随机 · stdout md5 `d0689dfd…` 三个独立进程逐字节恒一）。
> 一句话：**OpenAI Agents SDK = 「回归轻量」的 Agent 运行时——Agent 是一个对象（name/instructions/tools/handoffs/护栏全挂它身上，16 个构造参数只有 name 必填），Runner 是一个函数（`run_sync` 一次调用从头跑到尾）；函数工具是引擎真执行的、护栏是引擎先看的（tripwire 拦截后模型 0 次调用）、手转交是一行 `handoff()` 把权交给另一个 Agent（`transfer_to_*` 自动工具名、目标 agent 原查询随到、`last_agent` 切换）——Swarm 的『你转给他』被做成了正式原语；角色化的重活归 CrewAI、博弈归 AG2、状态图归 LangGraph，本篇收掉 Agent 系第三个范式：轻量运行时。**
>
> 📌 诚实边界：**引擎语义 = 本机真实实测**（openai-agents `0.17.0`/openai `2.44.0`：`Agent` 构造参数面板、`Runner.run_sync` 全流程、`function_tool` 自动 schema、同一帧并行双工具、`input_guardrail` tripwire 前拦截并抛 `InputGuardrailTripwireTriggered`、items 列表输入 vs 裸字符串输入的首轮闸空 `[]` 怪癖、`handoff()` 自动工具名/`last_agent` 切换/`HandoffOutputItem`/目标 agent 上下文接力、22 名 import 探测）；**模型决策=预置 stub**（继承真模型类只重写 `get_response`，查表决定『这轮调哪个工具/转给谁/答什么』——真实 LLM 非本机实测：本机无外网无 key）；零网络保障=所有 agent 显式挂 stub、默认 OpenAI provider 永不被触（若被触到=401=环境事实）；**护栏=词面规则作者预置**（生产护栏常是 LLM 判定=要素事实）；**手转交的『模型决定要转』=预置仿真**（真实验证时由真实 LLM 编制工具名）；**版本节奏/决策树/`020` 最小实现=要素事实**（2025 Swarm→SDK、0.1→0.17 迁移路径=无外网未逐版复核）；openai-agents 与本机 langchain-openai 共享依赖、0.17.0 要求 `openai<3,>=2.26`（实测 2.44 为全仓兼容点）；墙钟只进 stderr（≈6.3–7.7 s=引擎导入+真跑，三进程浮动但 stdout md5 恒一）。

---

## 📑 本章目录

1. [为什么 Agent 系收束篇是 OpenAI Agents SDK？](#0-为什么-agent-系收束篇是-openai-agents-sdk)
2. [先把问题拆开：从「最轻」到「第一等公民」](#1-先把问题拆开从最轻到第一等公民)
3. [把「轻量运行时」降维成本机测量（探针设计）](#2-把轻量运行时降维成本机测量探针设计)
4. [实验 A · 轻量原语账（一件套 vs 三件套）](#3-实验-a--轻量原语账一件套-vs-三件套)
5. [实验 B · 工具与护栏账（引擎执行的工具、引擎先看的闸）](#4-实验-b--工具与护栏账引擎执行的工具引擎先看的闸)
6. [实验 C · 手转交账（§7.11『回归轻量』的转移原语）](#5-实验-c--手转交账711回归轻量的转移原语)
7. [实验 D · 版本存续与选择树（顶层最小面 + Agent 系收束）](#6-实验-d--版本存续与选择树顶层最小面--agent-系收束)
8. [常见坑（本机实测踩过的 6 条）](#7-常见坑本机实测踩过的-6-条)
9. [诚实边界（AAA 自我审查）](#8-诚实边界aaa-自我审查)
10. [参考与衔接](#9-参考与衔接)

---

## 0. 为什么 Agent 系收束篇是 OpenAI Agents SDK？

07-应用框架走到 Agent 系三连的最后一家。前两篇给的范式：

- **07-AutoGen/AG2**＝多 Agent **对话范式**——agent「互聊」靠消息循环：谁的回合谁答，`reply.ask()` 把历史**显式**回灌。一次 run 是「一条 prompt → 一次 LLM 调用」的最小执行单元，但多 agent 的「谁接话、谁降级」留在引擎外。
- **08-CrewAI**＝多 Agent **角色化分工**——role/goal/task 三一等公民：`You are {role}…` 系统配方、任务列表=声明式流水线、缺 role 构造期 `ValidationError`。团队怎么排由**声明式编排**说了算。

本篇的 §7.11 给出了第三节拍：**OpenAI Agents SDK**——「位置：轻量 Agent 运行时」。它代表的不是又一种编排，而是**框架瘦身的趋势本身**（§7.11 原文：代表「回归轻量、靠原生功能」的趋势——不再追求大而全；§7.13 谱系：`2024 Swarm（实验）→ 2025 OpenAI Agents SDK（轻量）`）。三句话定位它：

1. **最小代码量**：`Agent` 一个对象 + `Runner.run_sync` 一个函数，没有 `Crew`、没有 `Task`、没有 group chat 类、没有显式图边——SDK 把「干活」的全部职责钉在**一个对象 + 一个函数**上。
2. **三件一等公民**：`function_tool`（引擎真执行）、`input_guardrail/output_guardrail`（引擎先看的闸）、`handoff()`（Agent 之间的转移权）——这三件事正是 §7.11「核心：Agent（tools+instructions）→ **Handoff** → **Guardrails** → Sessions/Tracing」展开的实测对象。
3. **给 Agent 系收束**：CrewAI 的角色化、AG2 的对话、SDK 的手转交——§7.14 决策树三家并排点名（「多 Agent 协作 → **CrewAI / AutoGen / Agents SDK（handoff）**」）说的就是：同为类 4、分工不同，谁也别想吞谁。本篇账 D 把这张抉择表落成 8 断言收掉整章选择树。

## 1. 先把问题拆开：从「最轻」到「第一等公民」

「轻量运行时」要回答的三层：

1. **多轻才算轻**？→ 账 A：把一个 agent 从「能跑」到「什么都带」的所有参数枚一遍——`Agent` 构造签名 16 个参数只有 `name` 必填；最小闭环 = `Agent(...)` + `Runner.run_sync(...)` 两行。对照 CrewAI（role+backstory+goal 的 `Agent`、`Task`、`Crew` 三件套）与 LangGraph（`StateGraph`+节点函数+条件边显式拓扑），本篇把「代码量极小」量化成一张构造面。
2. **工具/护栏凭什么是一等公民**？→ 账 B：`@function_tool` 装饰一个普通 Python 函数，引擎自动导出 name/description/参数 schema、**真执行**并把结果回喂（工具轮不额外算模型轮）；`@input_guardrail` 在模型被询问**之前**跑、tripwire 一升直接抛 `InputGuardrailTripwireTriggered`——「闸拦截后模型 0 调用」是引擎真行为。并行双工具（同一帧两个 function_call）一次回喂=工具并发调度的最省轮形态。
3. **手转交到底转了什么**？→ 账 C：`handoff(agentB)` 挂在 A 的 `handoffs` 上，引擎生成自动工具名 `transfer_to_{B.name}`；模型（此处置预置决策）一调它，引擎真转移：`last_agent` 从 A 变 B、`result.new_items` 落一条 `HandoffOutputItem`、**原查询原样带到 B**（不用手工搬运上下文）。§7.11 的「Agent 之间转移权」从概念落地为可测的两次模型轮行队账。

一句话分层：**CrewAI 把多 Agent 写成「团队怎么排」的声明、AG2 把多 Agent 写成「谁和谁聊」的循环、SDK 把多 Agent 写成「谁把球传给谁」的转移**——前两者在编排层做文章，本篇在**执行层**做最小实现。

## 2. 把「轻量运行时」降维成本机测量（探针设计）

- **引擎**：`openai-agents` **0.17.0**（伴随 `openai` 2.44.0——见账 D 兼容段）真实安装、真实执行。四本账全部走 `Runner.run_sync` 真 API。
- **模型决策 = 预置 stub**：`StubBrain` 继承真实 `OpenAIChatCompletionsModel`、只重写 `get_response`——**引擎的调度/工具执行/护栏/手转交全部真实发生**，唯一被替换的是「模型这轮怎么想」。stub 按「上下文里已有的 function_call 轮次」决定：这轮返回纯文本 / 一个或多个工具调用 / 一条 handoff 调用。零网络（所有 agent 显式挂 stub，默认 provider 永不被触）。
- **量程**：四本账——A 轻量原语（构造面 + sys 透传 + 最小闭环行数）、B 工具护栏（自动 schema + 真执行回喂 + 并行双工具 + tripwire 前拦截 + items/裸串怪癖）、C 手转交（自动名 + 目标切换 + HandoffOutputItem + 上下文接力 + 三档行队成本）、D 版本存续（22 名导入面 + model 收敛 + 门面 3 例 + 选择树 8 断言）。
- **确定性**：三个独立进程跑完整探针，stdout md5 恒一 `d0689dfd…`（5588 字节）；`warnings.filterwarnings("ignore")` 前置、`sys.stdout.reconfigure(encoding="utf-8")`、stdout 以二进制整块写出（Windows 下文本模式会 `\n→\r\n` 污染 md5）；墙钟只进 stderr（≈6.3–7.7 s，引擎导入+真跑；三进程浮动但 stdout 逐字节一致）。

复现实录（严格逐字节 = 探针 stdout 三遍恒一）：

<details>
<summary>📊 探针 stdout 全量实录（md5 <code>d0689dfd…</code> · 墙钟仅 stderr）</summary>

````text
========================================================================
前置：OpenAI Agents SDK 轻量运行时（知识地图 §7.11 / §7.1 分类 · Agent 系收束）
  一句话：Agent 是一个对象 · Runner 是一个函数 · 工具/护栏/手转交都是一等公民
环境：openai-agents 0.17.0（真实运行）· 模型决策=预置 stub（零网络·零随机·stdout 可复现）
========================================================================

[实验 A] 轻量原语账 —— 一件套 vs 三件套（Agent 一个对象装全部）
  Agent 构造参数 16 个 · 必填仅 1：name=['name']（其余全带默认=可空手起 Agent）
  sys 指令原样透传：True（模型收到的就是 instructions 原文，无『You are {{role}}』配方——CrewAI 那套在这里没有）
  最小闭环 = 1 个 Agent 对象 + Runner.run_sync 一次调用 → final_output='您好呀。' · last_agent=sA · 模型轮数=1
  ------------------------------------------------------------------------
[实验 B] 工具与护栏账 —— 工具是引擎执行的、护栏是引擎先看的
  function_tool 自动 schema：name=get_weather · description='查询城市实时天气。' · 参数必填=['city']
  并行双工具（同一帧两个 function_call）：引擎真跑两个 Python 函数 → 结果都进下一轮 → 终答='广州晴 22-30℃，茅台涨 3.2%。' · 回合=[tool+tool, text]
  回合推进=[('tool', ['get_weather', 'get_stock']), ('text', 2)] · 总模型轮数=2（工具不另外算轮，只算模型往返）
  护栏词面闸（items 列表输入）：『帮我结个账』放行→模型 1 轮 · 『查违法』trip 拦截→模型 0 轮 · 异常=InputGuardrailTripwireTriggered
  闸函数观测：命中序列（True=拦住）=[False, True]
  引擎怪癖：裸字符串输入时 0.17.0 把首轮闸输入传空 []（闸看不见请求）；改传 items 列表闸即可见（本机实测）
  → 护栏=tripwire 前拦截（引擎真跑、真阻断：闸拦截后模型根本不会被询问）
  ------------------------------------------------------------------------
[实验 C] 手转交账 —— §7.11『回归轻量』：一行 handoff 给 Agent 装第二个大脑
  handoff 自动工具名='transfer_to_after_sale' · 描述前缀=模型看到的：'Handoff to the after_sale agent to handle the request. 客户要退货退款就转我。'
  run 结果：final out=归售后 '售后已接单，退 129 元。' · last_agent=after_sale（客服→售后已切换）· HandoffOutputItem>0=True 目标=after_sale
  上下文接力：售后收到首条 user 内容='电商:我要退货'（原查询带过去，不用手工搬运）
  客服侧回合=[('handoff', 'after_sale')] · 售后侧回合=[('text', 1)]
  行队成本（同一电商查询，三档队伍总模型轮数）：直答 1 · 带工具 2 · 手转交 2（转交轮 + 售后轮）——第二个大脑只多 1 轮
  ------------------------------------------------------------------------
[实验 D] 版本存续与选择树 —— 顶层最小化的导入面 + Agent 系三连收束
  导入面 22 名逐项实测（0.17.0）：
    OK 17 / FAIL 5
    FAIL 明细：agents.run_sync→FAIL(AttributeError)  ·  agents.FunctionAgent→FAIL(AttributeError)  ·  agents.runcmd.Agent→FAIL(AttributeError)  ·  agents.panic→FAIL(AttributeError)  ·  agents.arch→FAIL(AttributeError)
    → 顶层故意最小化：run_sync 挂 Runner、FunctionAgent 不存在、runcmd 是早期路径——0.1→0.17 的迁移动作=要素事实
    __version__=0.17.0
  模型收敛：Agent.model 类型=str|Model|None（可直插自定义 Model 实例——本探针四个 agent 全部真插 stub 模型）
  门面 API：set_default_openai_client / set_default_openai_key / set_tracing_disabled 都在 agents.*
  选择树 8 场景（§7.1 Agent 系收束）：断言 8/8
    S1 单 Agent 轻量问答/工具/护栏/手转交 → SDK（本篇）
    S2 角色化团队流水线（role/goal/task 分层） → CrewAI
    S3 多 Agent 自由对话/博弈 → AG2
    S4 显式状态图/循环/checkpoint/恢复/HITL → LangGraph
    S5 企业 .NET/云编排 → Semantic Kernel
    S6 生产 RAG 管线/类型安全接线 → Haystack
    S7 低代码画布/非工程师 → Dify·Flowise
    S8 协议级的跨栈工具互操作 → MCP（非框架）

========================================================================
台账汇总（A 轻量原语 / B 工具护栏 / C 手转交 / D 版本选择树）
  A  构造参数 16 仅 1 必填 · sys 原样透传 · 最小闭环 1 对象+1 调用 · 1 轮出答
  B  工具自动 schema（name/description/必填参数）· 真执行回喂 · 并行双工具同帧一次回喂 · 护栏 trip 前拦截模型 0 次
  C  手转交 auto 名 transfer_to_* · last_agent 切换 · HandoffOutputItem · 原查询随到 · 行队成本 1/2/2 轮
  D  导入面 17/22 OK · model 收敛 str|Model|None · 门面 3 例 · 选择树 8/8
一句话：OpenAI Agents SDK = 『回归轻量』的 Agent 运行时——一个 Agent 对象装下全部、Runner 一个函数跑完、
      一行 handoff 给 Agent 升级第二个大脑；工具它执行、护栏它先看、对话它来进行；角色化重活归 CrewAI、
      博弈归 AG2，Agent 系三连就此收束（对话/角色化/轻量转移三范式各占一格）
done · 一键复现：python code/notebooks/_tools/openai_agents_demo.py
========================================================================
````

</details>

## 3. 实验 A · 轻量原语账（一件套 vs 三件套）

**目标**：量化「轻量」到底轻在哪——把 `Agent` 的构造面、指令透传、最小闭环行数量开，和 CrewAI/AG2 拼一把「起一个 agent 的仪式成本」。

**过程**：`inspect.signature(Agent.__init__)` 枚出真实构造参数（探针打印 `{len(params)} 个 · 必填仅 {len(required)}`）；用 `Agent(name="sA", instructions="你是出口转内销客服小二，只处理中文。", model=StubBrain(...))` 跑 `Runner.run_sync`；stub 记录「模型收到的 system_instructions」，与传入的 `instructions` 逐字节比对。

**实测**（stdout 关键行）：

```
  Agent 构造参数 16 个 · 必填仅 1：name=['name']（其余全带默认=可空手起 Agent）
  sys 指令原样透传：True（模型收到的就是 instructions 原文，无『You are {{role}}』配方——CrewAI 那套在这里没有）
  最小闭环 = 1 个 Agent 对象 + Runner.run_sync 一次调用 → final_output='您好呀。' · last_agent=sA · 模型轮数=1
```

**结论**：
1. **构造面：一个对象装全部**——`name/handoff_description/tools/mcp_servers/mcp_config/instructions/prompt/handoffs/model/model_settings/input_guardrails/output_guardrails/output_type/hooks/tool_use_behavior/reset_tool_choice` 16 个参数全挂 `Agent` 身上，**必填只有 `name`**。对照：CrewAI 起一个 agent 至少要 `Agent(role=…, goal=…, backstory=…)` + `Task` + `Crew`（08 篇实测 role 缺失构造期即炸）；AG2 要 `Agent(name, prompt, config)` + 异步 `async with a.run()` 块（07 篇实测）。SDK 是**「npm 一座庙」式的极简**：想裸跑一个 agent，两行够。
2. **指令透传：无配方**——CrewAI 会把 role/goal/backstory 真拼成 `You are {role}…` 系统提示（08 篇逐字节实测，同角色 sys 长度恒定 46）；SDK **不做任何模板化**，`instructions` 字符串原封不动传给模型（探针 `sys 指令原样透传: True`）。这既是轻量（少一层包装）也是留白（「给模型说什么」完全你自己定）。
3. **1 轮出答的最小闭环**：`run_sync` 的返回 `RunResult` 上直接拿 `final_output`——没有显式「收集 agent 输出」的步骤。和 AG2 的 `run().result`、CrewAI 的 `crew.kickoff().raw` 对偶，但 SDK 连「Task」这种中间概念都没有。

## 4. 实验 B · 工具与护栏账（引擎执行的工具、引擎先看的闸）

**目标**：验证「一等公民」不是口头——工具（自动 schema + 真执行 + 并行 + 回喂）与护栏（模型被问之前就拦）都是引擎级行为。

**过程一（工具）**：两个普通 Python 函数 `get_weather(city: str)`、`get_stock(code: str)` 各挂 `@function_tool`；模型（stub 决策）在第一帧发**两个 function_call**，观察引擎是否真跑两个函数、结果是否都进下一轮、stub 第二帧直接出终答。工具自动 schema 直接读 `FunctionTool` 的 name/description/`params_json_schema`。

**过程二（护栏）**：`@input_guardrail` 装饰的一个词面闸（作者预置规则：命中「违法」置 tripwire）；同一 agent 跑两问——放行问一次、拦截问一次；分别数「模型被调了几轮」。

**实测**（stdout 关键行）：

```
  function_tool 自动 schema：name=get_weather · description='查询城市实时天气。' · 参数必填=['city']
  并行双工具（同一帧两个 function_call）：引擎真跑两个 Python 函数 → 结果都进下一轮 → 终答='广州晴 22-30℃，茅台涨 3.2%。' · 回合=[tool+tool, text]
  护栏词面闸（items 列表输入）：『帮我结个账』放行→模型 1 轮 · 『查违法』trip 拦截→模型 0 轮 · 异常=InputGuardrailTripwireTriggered
  闸函数观测：命中序列（True=拦住）=[False, True]
  引擎怪癖：裸字符串输入时 0.17.0 把首轮闸输入传空 []（闸看不见请求）；改传 items 列表闸即可见（本机实测）
```

**结论**：
1. **工具是真执行、不是模型编**：`get_weather("广州")` 是我们的 Python 函数被引擎真实调用（返回值进入下一轮上下文），stub 只是「决定调谁」。自动 schema 从签名+docstring 导出：`description='查询城市实时天气。'` 就是 docstring 原文——和 05 篇手搓 Function Calling 的「声明工具」对齐，这里由装饰器代劳。
2. **并行双工具 = 同一帧两个 function_call**：一次模型往返能发两个工具调用，引擎同回合执行两个函数、下一帧一次拿全结果（`回合推进=[('tool', ['get_weather', 'get_stock']), ('text', 2)]`，**总模型轮数=2**）。对照 05 篇串行回喂、LangGraph 的并行节点；SDK 把这默认折叠进 run loop（能否并发执行=引擎调度细节，未逐个计时）。
3. **护栏在模型被问之前**：命中的问闸时 tripwire_triggered=True，引擎直接抛 `InputGuardrailTripwireTriggered`——**「模型 0 轮」是硬证据**：闸是「先看」的，不是「看完再评」。这对应 §7.11 Guardrails 环节与 08-10 篇 judge 的「省 token」逻辑同族：烂请求根本不值得模型看第二眼。
4. **实测怪癖（本机 0.17.0）**：**裸字符串**作为输入时，首轮 `input_guardrail` 收到的 input 是空 `[]`（闸看不见请求内容）——改用 **items 列表形式**（`[{"role": "user", "content": …}]`）传输入，闸才能读到内容并正常拦截。这是引擎行为、不是文档写法（诚实边界：版本相关，见账 D 兼容段）。

## 5. 实验 C · 手转交账（§7.11『回归轻量』的转移原语）

**目标**：把 §7.11 的 **Handoff（Agent 之间转移权）**量开——一行 `handoff()` 到底在引擎里做了什么：自动工具名、目标切换、上下文接力、以及「第二个大脑」的行队成本。

**过程**：售后 agent H = `Agent(name="after_sale", handoff_description="客户要退货退款就转我。", …)`；客服 C = `Agent(name="cs_agent", handoffs=[handoff(H)], …)`。stub 决策：客服第一次被问「我要退货」就发一条 handoff 调用（工具名 `transfer_to_after_sale`）；售后第一次被问即回终答。取 `RunResult` 的 `final_output / last_agent / new_items`，并在售后 stub 里记录它收到的首条 user 内容。

**实测**（stdout 关键行）：

```
  handoff 自动工具名='transfer_to_after_sale' · 描述前缀=模型看到的：'Handoff to the after_sale agent to handle the request. 客户要退货退款就转我。'
  run 结果：final out=归售后 '售后已接单，退 129 元。' · last_agent=after_sale（客服→售后已切换）· HandoffOutputItem>0=True 目标=after_sale
  上下文接力：售后收到首条 user 内容='电商:我要退货'（原查询带过去，不用手工搬运）
  客服侧回合=[('handoff', 'after_sale')] · 售后侧回合=[('text', 1)]
  行队成本（同一电商查询，三档队伍总模型轮数）：直答 1 · 带工具 2 · 手转交 2（转交轮 + 售后轮）——第二个大脑只多 1 轮
```

**结论**：
1. **手转交=一次「带权移交的工具调用」**：`handoff(H)` 生成一个自动工具 `transfer_to_after_sale`（`Handoff.default_tool_name` = `transfer_to_{agent.name}`），模型把「该转」当作一次工具调用发出即可。引擎收到后执行真实转移：`last_agent` 从 `cs_agent` 变 `after_sale`、`new_items` 里落一条 `HandoffOutputItem(target_agent=after_sale)`——**转移本身是有型的状态变更，不是两个 agent 悄悄接着聊**。
2. **上下文是引擎搬的**：售后首条 user 内容=『电商:我要退货』原样到达——目标 agent 从**同一个用户查询**重新起跑（历史映射=引擎行为，账贴上一局）。对比 AG2 要把历史显式 `reply.ask()` 喂回去（07 篇实测 0/1/2 条计数），SDK 的 transfer 自带「带原问题去新大脑」。
3. **行队成本（同一需求三档队伍）**：直答 1 轮 / 带工具 2 轮 / 手转交 2 轮。手转交（转移轮+目标轮）**只比带工具多赚一个「第二个大脑」**，成本几乎平价——这就是 §7.11「最小代码量 + 多体能力」的甜点：**想升级第二个大脑，老板要付的多算的只有 1 轮模型往返**。这与 §6.9「多 Agent 是多倍 token」的告诫并不冲突：账在这里（2 vs 1），SDK 只是把它**压到最低**；真要多跳（客服→售后→仓储→返程），轮数线性加（要素事实，未在本篇实测多跳链——诚实边界）。

## 6. 实验 D · 版本存续与选择树（顶层最小面 + Agent 系收束）

**目标**：收束整章——一把量出 0.17.0 的导入面「多大/多小」，再把这十章的选型决策合成一张 8 断言的选择树，替 Agent 系画句号。

**过程一（导入面）**：22 个名字逐项 `_probe`（最长可导入模块前缀 + 残余属性 getattr，真 import 判定）：
- OK 组（17）：`agents`、`Agent`、`Runner`、`RunResult`、`function_tool`、`input_guardrail`、`output_guardrail`、`handoff`、`handoffs.Handoff`、`ModelSettings`、`GuardrailFunctionOutput`、`RunContextWrapper`、`tracing.set_tracing_disabled`、`WebSearchTool`、`Session`、`voice.VoicePipeline`、`models.openai_responses`。
- FAIL 组（5，全 AttributeError）：`run_sync`（顶层没有——在 `Runner` 上）、`FunctionAgent`（没有这个类）、`runcmd.Agent`（早期路径）、`panic`/`arch`（不存在名）。

**过程二（模型收敛 + 门面）**：`Agent.model` 的注解类型；门面 API 三个 `agents.*` 顶层函数。

**过程三（选择树 8 场景）**：探针断言 8/8（要素事实）。

**实测**（stdout 关键行）：

```
  导入面 22 名逐项实测（0.17.0）：OK 17 / FAIL 5
  FAIL 明细：agents.run_sync→FAIL(AttributeError)  ·  agents.FunctionAgent→FAIL(AttributeError)  ·  …·  agents.panic→FAIL(AttributeError)  ·  agents.arch→FAIL(AttributeError)
  __version__=0.17.0
  模型收敛：Agent.model 类型=str|Model|None（可直插自定义 Model 实例——本探针四个 agent 全部真插 stub 模型）
  选择树 8 场景（§7.1 Agent 系收束）：断言 8/8
```

**结论**：
1. **顶层是最小面、方法挂宿主**：核心原语 `Agent/Runner/function_tool/handoff/input_guardrail/…` 全在顶层（17/22 OK），直觉上该有的 `agents.run_sync` 顶层函数**不存在**——它在 `Runner.run_sync` 上；`FunctionAgent` 从没成为顶层名字（类 4 顶层就一个 `Agent`）。「顶层只导出该导出的」=**故意的瘦身**，与 Agent 系兄弟（CrewAI 顶层不出 `CrewBase`、AG2 顶层垫片移除）同一纪律。
2. **模型接入=接口收敛**：`Agent.model: str | Model | None`——字符串走 provider、**`Model` 实例直插**（本探针四 agent 全插 `StubBrain`）、None 走默认 provider。这是「引擎可测」的接口学：**把模型替换点做进类型注解**，才能把语言模型换成任何实现（stub/本地/别家）而调度不动。
3. **兼容账**（诚实边界段展开）：`openai-agents 0.17.0` 要求 `openai<3,>=2.26`；本机为不伤 langchain-openai 锁定 `openai 2.44.0`（2.45 曾把 `InputTokensDetails` 加必填字段、与 0.17.0 的默认工厂冲突，实测报 `cache_write_tokens Field required`——环境事实，见常见坑）。
4. **选择树 8/8**（§7.1 分类学 + §7.14 决策收束）：① 单 Agent 轻量问答/工具/护栏/手转交 → **SDK（本篇）**；② 角色化团队流水线 → **CrewAI**；③ 多 Agent 自由对话/博弈 → **AG2**；④ 显式状态图/循环/checkpoint/恢复/HITL → **LangGraph**；⑤ 企业 .NET/云编排 → **Semantic Kernel**；⑥ 生产 RAG 管线/类型安全接线 → **Haystack**；⑦ 低代码画布/非工程师 → **Dify·Flowise**；⑧ 协议级跨栈工具互操作 → **MCP（非框架）**。开头那句话就此闭环：**对话=AG2、角色化=CrewAI、轻量转移=SDK，Agent 系三范式谁也没吞谁**。

## 7. 常见坑（本机实测踩过的 6 条）

1. **裸字符串输入会让首轮 `input_guardrail` 收到空 `[]`**（实测，0.17.0）：`Runner.run_sync(agent, "请求文本")` 时护栏函数第一轮 `input` 为空列表，闸根本看不见请求，拦截必然 miss。**改用 items 列表** `[{"role":"user","content":"…"}]` 传输入即可（账 B 已演示）。这是引擎行为、非文档写法（诚实边界）。
2. **`openai` 版本顶格翻车**（实测）：本机曾把 `openai` 升到 3.x，`openai-agents` 0.22 的 `openai>=3.0.0` 硬约束赢了 resolver，结果 `langchain-openai 1.3.4`（`openai<3`）被打爆；即便留在 2.x，`openai 2.45.0` 又让 `agents.usage` 的默认工厂炸 `cache_write_tokens Field required`。**定型：`openai-agents 0.17.0` + `openai 2.44.0`**（`>=2.26,<3` 区间内全仓兼容点）。
3. **裸 top-level 没有 `agents.run_sync`**（实测）：直觉写 `from agents import run_sync` 会 AttributeError——它是 `Runner.run_sync` 的方法（账 D）；`run_sync_protected`/历史顶层函数位=要素事实。
4. **`handoff()` 不是 `Handoff(agent=…)`**（实测踩的）：0.17.0 的构造是装饰器式 `handoff(target_agent)`（挂在 source agent 的 `handoffs=[…]`）；`Handoff(agent=H)` 直构会 `TypeError: unexpected keyword argument 'agent'`。手转交工具名自动= `transfer_to_{agent.name}`，别手工拼错。
5. **给每个可能被激活的 agent 都要配模型**（实测）：手转交后**目标 agent 会真跑**；只给 source 配 stub、目标不配，引擎会去默认 provider 连网（本机直接 401）。零网络探针的铁律=让**所有** agent（含 handoff 目标）都显式挂 stub/模型。
6. **`contextlib.redirect_stderr` 才接得住引擎的日志**：`openai-agents` 导入/首跑很吵（版本检查、trace 控制台框），探针用 `set_tracing_disabled(True)` + 环境变量 + 对 stdout 整块二进制写出，才能保住 `md5 三进程恒一`；墙钟只进 stderr。

## 8. 诚实边界（AAA 自我审查）

- **引擎语义 = 本机真实实测**：`openai-agents 0.17.0` + `openai 2.44.0` 真实安装运行（stdout md5 `d0689dfd…` 三个独立进程逐字节恒一，墙钟 ≈6.3–7.7 s 仅 stderr=引擎导入+真跑，进程间浮动但 stdout 不变）。`Agent` 构造面、`Runner.run_sync`、`function_tool` 自动 schema、并行双工具、`input_guardrail` tripwire + `InputGuardrailTripwireTriggered`、items/裸串闸怪癖、`handoff()` 自动名/`last_agent`/`HandoffOutputItem`/上下文接力、22 名导入探测全部为真 API 实测。
- **模型决策 = 预置 stub**：`StubBrain` 继承真实 `OpenAIChatCompletionsModel`、只重写 `get_response` 查表决定「调哪个工具/转给谁/答什么」——真实 LLM 的推理/选工具/选转交对象 = **非本机实测**（本机无外网无 key）。引擎层的调度/执行/转移/拦截是测量对象；「模型怎么想」是预置仿真。
- **零网络保证**：所有 agent（含 handoff 目标）均显式挂 stub；默认 OpenAI provider 永不被触（被触则 401=本机环境事实）。
- **护栏=词面规则作者预置**：生产 `input_guardrail` 常是 LLM 判定（要素事实）；本篇仅演示「引擎侧先看、真拦截」的机制，不度量 LLM 闸的质量。
- **手转交的决策=预置**：「客服决定转售后」由 stub 表给出；真实手转交时由真实 LLM 编制 `transfer_to_*` 工具调用。
- **版本节奏/决策树/`020` 最小实现 = 要素事实**：2025 Swarm→Agents SDK、0.1→0.17 迁移路径（顶层函数位变化、`runcmd` 早期路径）、§7.14/§7.1 选择树条目均为知识地图文本（无外网未逐版复核）；`output_guardrail`、tracing、sessions 等未逐一展开（账 D 仅验证其导入面）。
- **openai-agents 依赖账**：`0.17.0` 标注 `openai<3,>=2.26`；本机锁定 `2.44.0` 是全仓兼容点（2.45 的 schema 变更冲突=环境事实，已在常见坑记录）。

## 9. 参考与衔接

- **本课地图**：§7.11（OpenAI Agents SDK——核心 Agent→Handoff→Guardrails→Sessions/Tracing；回归轻量趋势；Swarm 前身，同类 ADK/Pydantic AI）+ §7.1（类 4 Agent 框架六家最挤）+ §7.14（多 Agent 协作→CrewAI/AutoGen/Agents SDK(handoff)；想少依赖→Agents SDK/Pydantic AI+自写工具协议）+ §7.13（谱系：2024 Swarm→2025 Agents SDK）+ §17.5.11（应用开发框架条目）+ §6.9（多 Agent=多倍 token）+ 里程碑 `020`（Agents SDK 最小实现）。
- **选择树 8 场景**（探针断言 8/8，要素事实）：见账 D 结论 4——一句话：轻量单 Agent/工具/护栏/手转交归 SDK，角色化归 CrewAI，对话博弈归 AG2，状态图归 LangGraph，企业栈归 SK，生产 RAG 归 Haystack，低代码归 Dify，协议归 MCP。
- **前承**：00-框架分类学（类 4 六家最挤、§7.14 决策）；02-LangGraph（`recursion_limit`/checkpoint=状态图一格；SDK 的 run loop 是无状态的无环回）；07-AutoGen（对话范式=左手）；08-CrewAI（角色化分工=右手；「role 缺失创建即 ValidationError」vs 本篇「name-only 即起」、「`You are {role}` 配方」vs 本篇「sys 原样透传」——同一范式族两极端对照）；05-Function-Tool-Calling（手搓工具协议 → 本篇 `function_tool` 装饰器）；06-应用开发 02/03（指令/上下文直传 → 本篇 instructions 原样透传 + handoff 上下文接力）。
- **后启**：`09-DSPy`（✅ 已交付 v0.36：类 5：Prompt 编程 + 自动化优化——本篇 stub 的「预置决策表」正是 DSPy 想用优化器自动逼近的目标；SDK 手写决策 vs DSPy 编译决策两端对照）；`11-MCP协议`（✅ 已交付 v0.37：关键章——Agent 的工具出口；SDK 的 `mcp_servers` 挂接点 + 里程碑 `013` 手写 MCP Server 已交付）；`12-框架继承关系与选型决策`（收束章：完整 DAG + 「该不该引入框架」量化——本篇「16 参数仅 1 必填」是「轻量」一侧的证据）。
- **对应里程碑**：`020` Agents SDK 最小实现（本篇账 A/C 对口：两行起 agent、一行加第二个大脑）；前后里程碑 `012`（02-章 LangGraph 带记忆重试客服）已交付、`013`（手写 MCP Server）已在 `11-MCP协议` 章交付（v0.37）。

> 本篇完工于 2026-09-22（v0.35 批次）；探针 代码 `code/notebooks/_tools/openai_agents_demo.py`；stdout md5 `d0689dfd…`。
