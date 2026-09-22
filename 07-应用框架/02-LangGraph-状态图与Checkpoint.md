# 🕸️ LangGraph：把「流程」编译成可执行的有向状态图（节点=函数 · 边=条件路由 · checkpoint 状态持久化 · 重试与递归上限）

> 对应知识点：知识地图 §7.3（LangGraph：2024 LangChain 官方；**有向状态图＝节点=函数·边=条件路由**；循环 / 状态持久化 checkpoint / human-in-the-loop / 流式；**DAG 满足不了的、循环恢复它都能做**；生态 LangSmith / LangGraph Platform 2025）、§7.1（框架分类学：编排 SDK LangChain/Haystack/Semantic Kernel · Workflow/图引擎 LangGraph/Temporal · RAG 框架 LlamaIndex/Haystack/RAGFlow · Agent 框架 LangGraph/AutoGen/CrewAI/Agents SDK/ADK/Pydantic AI · 低代码 Dify/Flowise/Coze/n8n · Prompt 编程 DSPy/Outlines · 协议 MCP/A2A/ACP · 前端 SDK——先分类型再比优劣）、§17.7.4（Memory 一句话中的「框架=LangGraph checkpoint」）、§2.7 里程碑 `012`（LangGraph 带记忆重试的客服 Agent）。
> 前置：[06-记忆系统](../06-应用开发/06-记忆系统.md)（画像/账本=**内容层**记忆：记住用户是什么样；本篇把「读取剂量/遗忘预算」的纪律固化进**引擎层**：活儿干到哪一步——两层的叠加=带记忆的 Agent）；[05-Function-Tool-Calling](../06-应用开发/05-Function-Tool-Calling.md)（工具入参/结果循环=回喂闭环的即时版；本篇重试自环=回喂循环的**图化**）；[03-Context-Engineering](../06-应用开发/03-Context-Engineering.md)（装什么/装多少/要不要带=记忆检索注入的上层概念；本篇条件路由=3-D 路由闸的图化）；[04-结构化输出](../06-应用开发/04-结构化输出.md)（Schema=状态 channel 的字段模型）。
> 动手：`python code/notebooks/_tools/langgraph_demo.py`（本页四账 + 选择树一键复现；**真实 langgraph 1.0.10 引擎运行**，非模拟——引擎语义=本机实测）。
> 一句话：**LangGraph = 把「流程」编译成可执行的有向状态图——节点=实时函数、边=条件路由、状态=TypedDict channel、checkpoint=引擎层状态持久化；循环/重试/恢复/递归上限开箱即用。06 的画像/账本是「记住用户是什么样」（内容层），checkpoint 是「活儿干到哪一步」（引擎层），两层叠加＝带记忆重试的客服（§17.7.4 一句话的框架落地版）。**
>
> 📌 本篇引擎语义为**本机真实运行**（`langgraph 1.0.10`，`pip show` 来源；`_tools/langgraph_demo.py`，stdout md5 `91c11c5b…` 三个独立进程逐字节恒一）；节点**内部逻辑**（意图识别/记忆检索/路由判定/工具执行/答复模板）=词面规则作者预置（真实 Agent 这些节点是 LLM=非本机实测）；MemorySaver=进程内内存检查点（生产 SQLiteSaver/Redis/Postgres/Platform=要素事实非本机实测）；thread 隔离/中断恢复/GraphRecursionError=真实 langgraph 行为本机实测；流式/LangSmith/human-in-the-loop 审批流=要素事实未在本机实测；Tarjan SCC=纯 stdlib 确定性实现；墙钟只进 stderr。

---

## 📑 本章目录

1. [为什么 07 章第二课是 LangGraph？](#0-为什么-07-章第二课是-langgraph)
2. [先把问题拆开：从「记忆纪律」到「框架原语」](#1-先把问题拆开从记忆纪律到框架原语)
3. [把「图引擎」降维成本机测量（探针设计）](#2-把图引擎降维成本机测量探针设计)
4. [实验 A · 状态图建模账（拓扑与执行轨迹）](#3-实验-a--状态图建模账拓扑与执行轨迹)
5. [实验 B · 条件路由账（判定＝边上的一等公民）](#4-实验-b--条件路由账判定＝边上的一等公民)
6. [实验 C · checkpoint 账（thread 隔离 + 中断恢复 + 两层记忆的分工）](#5-实验-c--checkpoint-账thread-隔离--中断恢复--两层记忆的分工)
7. [实验 D · 循环与预算账（自环收敛 · SCC 环检测 · 递归上限兜底）](#6-实验-d--循环与预算账自环收敛--scc-环检测--递归上限兜底)
8. [常见坑（本机实测踩过的 6 条）](#7-常见坑本机实测踩过的-6-条)
9. [诚实边界（AAA 自我审查）](#8-诚实边界aaa-自我审查)
10. [参考与衔接](#9-参考与衔接)

---

## 0. 为什么 07 章第二课是 LangGraph？

06 章走到最后一课，把「跨会话连续性」的纪律写全了：画像 upsert · 账本 append · top-k 回灌 · 遗忘预算。但那些纪律全部**活在你自己写的 if/else 里**——你在 Python 进程里管理状态、在代码里写重试、在内存里存会话存档。隔天换一台机器、服务崩一下、想人工接管一下，这套"手写状态机"就露出边界：

- **状态存哪**：session 变量挂了就没了——需要一个跨请求的「运行状态存档」（checkpoint）。
- **流程停在哪**：工具出错要重试、缺信息要问人、重试要转人工——这些分支写成一坨 `if` 会随流程拉长而腐烂（03-D 的路由闸散落进每个节点）。
- **循环怎么表达**：线性链（LangChain 式的 `chain | chain`）表达不了"失败→再调一次"的**自环**，你只能在外面套 `while`，而 `while` 不属于这张图、不参与任何可视化/检查点。

§7.3 的原话一针见血：**DAG 满足不了的、循环恢复它都能做**。LangGraph 把这三件事——状态（TypedDict channel）、分支（条件边）、循环（条件边自环）+ 持久化（checkpoint）——固化成**图引擎的第一等原语**。本课（§2.7 里程碑 `012`：带记忆重试的客服）用真实框架把这张图跑起来，把你已经在 05/06 里手写过的"回喂 + 记忆"**重新编译成一张可执行的有向状态图**，量化四个问题：图建模长什么样 / 条件路由分掉了多少无谓工具 / checkpoint 到底记住了什么 / 循环与递归上限是谁在兜底。

## 1. 先把问题拆开：从「记忆纪律」到「框架原语」

06 的四条纪律在 LangGraph 世界里的对应物：

| 06 应用层纪律（内容层） | LangGraph 引擎原语（引擎层） | 本篇实测切入点 |
|---|---|---|
| 画像 upsert / 账本 append | `state` 的 TypedDict channel（节点头返回状态增量，`Annotated[list, operator.add]` 累积器=账本） | A 图建模：ps 累积器=免费轨迹可观测 |
| 读取剂量（top-k 回灌） | 节点函数里自己取（引擎不管"该喂多少"，只保证**通道可达**） | A 轨迹：查记忆直达 answer 少走工具 |
| 内存中管理状态 | **checkpoint**：引擎把每 thread 的运行状态快照化（`get_state().next` 记录下一跳） | C：thread 隔离 + 中断恢复 |
| 代码里写重试 | **条件边自环**（`call_tool→call_tool`）+ `recursion_limit` 轮上限 | D：自环收敛 + GraphRecursionError |

一句话分层：**06 管"内容"（记忆里的数据），LangGraph 管"过程"（执行到哪一步、怎么流、怎么恢复）。** 这就是本课最需要抓住的一层区别——很多教程把 checkpoint 和记忆混为一谈，本课在 C 账里把它量开。

业务载体 = §2.7 里程碑 `012`：带记忆重试的客服 Agent。节点：

```
parse → recall → route ⇉ call_tool ⇉ answer / ask_user / escalate
                 （记忆检索）   （工具执行·可失败重试）
```

- `parse`：意图/槽位/工具选择（词面规则代理）
- `recall`：从 06 的画像+账本里检索相关记忆（词面关键词命中）
- `route`：判定该走工具 / 直答 / 问用户（`go=tool/answer/ask`，返回给条件边）
- `call_tool`：模拟工具执行，可失败（`fail_policy` 控制：none/once/always）
- `answer` / `ask_user` / `escalate`：三种出口（直答 / 缺信息问 / 重试耗尽转人工）

## 2. 把「图引擎」降维成本机测量（探针设计）

- **引擎**：`langgraph 1.0.10` 真实运行（`StateGraph` + `MemorySaver` + `interrupt_before` + `GraphRecursionError` 全用真 API）；节点内逻辑 = 词面规则（确定性、无随机无网络）。
- **量程**：四本账——A 图建模（拓扑与 7 次会话的逐条轨迹）、B 条件路由（判定命中 + 反事实省步）、C checkpoint（双 thread 隔离 + 中断恢复不重跑）、D 循环预算（自环收敛 · Tarjan 环检测 · recursion_limit 真抛错）+ 选择树 8 场景。
- **确定性**：三个独立进程跑完整探针，stdout md5 恒一 `91c11c5b…`（5799 字节）；`sys.stdout.reconfigure(encoding='utf-8')`；墙钟只进 stderr（实测 ≈0.10 s）。
- **本轮是本门第一次第三方框架真实运行**：之前 45 个复现脚本全是纯 stdlib 自建引擎；`langgraph 1.0.10` 是真实安装的框架——图执行/状态/channel 合并/条件边/checkpoint/递归上限全部由框架真实完成，只有节点内业务逻辑是预置规则。

复现实录（严格逐字节 = 探针 stdout 三遍恒一）：

<details>
<summary>📊 探针 stdout 全量实录（md5 <code>91c11c5b…</code> · 墙钟仅 stderr）</summary>

````text
========================================================================
前置：LangGraph 图引擎（知识地图 §7.3 正题 / §7.1 分类 · 衔接 06-记忆）· 本机真实执行
  有向状态图：节点=函数 · 边=条件路由 · 状态=TypedDict channel · checkpoint=引擎层状态持久化
环境：LangGraph 1.0.10 真实运行 · 零网络 · 零随机 · stdout 逐字节可复现
========================================================================

[实验 A] 状态图建模账 —— 拓扑与执行轨迹（节点=函数 · 边=条件路由）
  图对象：节点 7 = parse/recall/route/call_tool/answer/ask_user/escalate
  边：无条件 3（START→parse→recall→route）· 条件 2 组 6 支路（route:tool/answer/ask · call_tool:ok/retry/esc）· 自环 1（call_tool→call_tool）· 终边 3
  t1 帮我查一下上海明天天气 → 意图=查天气  路由=tool   轨迹 parse→recall→route→call_tool→answer（5 步） · 答复=已查好：上海 明天 晴 20-28℃（我记得你住在上海）
  t2 把产品评审加到周六日程 → 意图=添加日程 路由=tool   轨迹 parse→recall→route→call_tool→answer（5 步） · 答复=好的，已把「产品评审」加入周六日程（你已有 7 条事件）
  t3 咱们团队例会定在什么时间 → 意图=查记忆  路由=answer 轨迹 parse→recall→route→answer（4 步） · 答复=查到了：团队例会（s2）
  t4 你好，谢谢你的服务 → 意图=问候   路由=answer 轨迹 parse→recall→route→answer（4 步） · 答复=您好，很高兴为您服务～
  t5 请再查一次北京的天气 → 意图=查天气  路由=tool   轨迹 parse→recall→route→call_tool→call_tool→answer（6 步） · 答复=已查好：北京 明天 晴 20-28℃（出差在外记得保暖）
  t6 我的物流单 999 现在什么状态 → 意图=查物流  路由=tool   轨迹 parse→recall→route→call_tool→call_tool→call_tool→escalate（7 步） · 答复=抱歉，多次尝试失败，已为您转接人工（重试 3 次）
  t7 我这周五的牙医预约是几点 → 意图=查记忆  路由=ask    轨迹 parse→recall→route→ask_user（4 步） · 答复=请问您说的是哪个城市/哪场安排？我这边还没记到～
  → 图引擎把『if/循环』写作显式拓扑：7 次会话全部走 graph.invoke（编译执行），ps 累积器=免费可观测轨迹

[实验 B] 条件路由账 —— 边上的判定 = 图的一等公民（对照：全走工具的最坏路径）
  路由判定：7 次会话 期望 全对账 → 命中 7/7（route 条件边按返回 go 选支路）
  可直达 #t3#t4#t7 本不需要工具（问候 / 查记忆有上下文）——条件边省掉无谓 call_tool
  → 反事实『每条都先调工具再答』= 38 步 vs 条件路由 35 步：省 3 步（约 7%）
  → 教学点：判定被建模成『边上函数』（route 返回 go），散落的 if 变成显式拓扑；与本门 03-D 路由闸同族

[实验 C] checkpoint 账 —— 引擎层状态持久化（thread 隔离 + 中断恢复）
  线程隔离：iso_a（查记忆）state.context=['团队例会'] vs iso_b（问候）state.context=[] → 串扰 0（隔离 = 每 thread 一册运行状态）
  checkpoint 快照：graph.get_state(next=('call_tool',)) —— 引擎记录『下一跳待跑节点』= 活儿干到哪一步
  中断恢复：第一段 parse→recall→route（interrupt_before 停在 call_tool 前）→ 同线程 invoke(None) 续跑 parse→recall→route→call_tool→answer
  → 恢复只补 2 步（call_tool+answer），parse/recall/route 不重跑：中断=挂起而非回滚

[实验 D] 循环与预算账 —— 自环收敛 · SCC 环检测 · recursion_limit 轮上限兜底
  自环收敛：t5（失败→重试一次成功）call_tool×2 · t6（恒失败）call_tool×3 → escalate 转人工（MAX_RETRY=3）
  环检测（Tarjan）：含自环图 SCC=7 含环 1（重试边）vs 去掉自环边 SCC=7 含环 0 —— 图论文『环=能反复执行的流程』
  递归上限：recursion_limit=6 跑 t6（需 7 步）→ Recursion limit of 6 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
  → 教学点：循环/恢复是图引擎的一等能力（§7.3；DAG 满足不了的、循环/恢复它都能做）；recursion_limit=框架级轮上限兜底，对应 06/05 的重试/轮限制

  选择树 8 场景（§7.1 分类学：先分类型再比优劣——图引擎=有状态/可分支/可恢复；协议层≠框架）；断言 8/8

========================================================================
台账汇总（A 图建模 / B 条件路由 / C checkpoint / D 循环预算）
  A  节点 7 · 边 条件 6 支路 + 自环 1；7 次会话编译执行 · 轨迹可观测（ps 累积器免费埋点）
  B  路由判定 7/7；对照「全走工具最坏路径」省 3 步（约 7%）——判定=边上函数
  C  thread 隔离串扰 0 · checkpoint.next=('call_tool',) 记录续跑点 · 中断恢复只补 2 步不重跑（挂起≠回滚）
  D  自环重试 call_tool×2→成功 ×3→转人工 · 环检测含环 1 vs 0 · recursion_limit 真抛 GraphRecursionError
一句话：LangGraph = 把『流程』编译成可执行的有向状态图——节点=函数·边=条件路由·状态=TypedDict channel·checkpoint=引擎层状态持久化；循环/重试/递归上限开箱即用；06 的画像/账本是『记住用户是什么样』（内容层），checkpoint 是『活儿干到哪一步』（引擎层），两层叠加=带记忆重试的客服，可调试、可恢复、可续跑
done · 一键复现：python code/notebooks/_tools/langgraph_demo.py
````

</details>

## 3. 实验 A · 状态图建模账（拓扑与执行轨迹）

**目标**：把「if/循环」重新建模成显式拓扑，量出"一张图 = 节点 + 边 + 状态通道"的骨架和每条会话的真实执行轨迹。

**过程**：`StateGraph(CS)` 声明 7 个节点；无条件边 3 条（`START→parse→recall→route`）；条件边 2 组 6 支路（`route → tool/answer/ask`、`call_tool → ok/retry/esc`）；**自环 1 条（`call_tool→call_tool`，重试）**；终边 3 条（`answer/ask_user/escalate→END`）。每次会话一个 thread_id，7 次会话全部走 `graph.invoke`。

**实测**（stdout 关键行）：

```
  边：无条件 3（START→parse→recall→route）· 条件 2 组 6 支路（route:tool/answer/ask · call_tool:ok/retry/esc）· 自环 1（call_tool→call_tool）· 终边 3
  t1 帮我查一下上海明天天气 → 轨迹 parse→recall→route→call_tool→answer（5 步） · 答复=已查好：上海 明天 晴 20-28℃（我记得你住在上海）
  t3 咱们团队例会定在什么时间 → 轨迹 parse→recall→route→answer（4 步）        · 答复=查到了：团队例会（s2）
  t5 请再查一次北京的天气 → 轨迹 parse→recall→route→call_tool→call_tool→answer（6 步）
  t6 我的物流单 999 现在什么状态 → 轨迹 parse→recall→route→call_tool×3→escalate（7 步） · 答复=…已为您转接人工（重试 3 次）
  t7 我这周五的牙医预约是几点 → 轨迹 parse→recall→route→ask_user（4 步）
```

**结论**：
1. **节点=函数**：每个节点是一次"状态增量的纯函数"（入=当前 state，出=字段更新）；边说了算的是「谁是下一步」，不是函数的调用栈——» 流程的**控制流图化**了。
2. **免费可观测性**：状态里挂一个加了 `operator.add` 归约器的 `ps` 通道（节点各 append 自己名字），整张图就自带**逐条轨迹审计**（对应 LangSmith 的追踪概念=要素事实）；不引入任何框架外埋点。
3. **自环被显式表达**：`t5` 的 `call_tool→call_tool→answer` 与 `t6` 的 `call_tool×3→escalate` 不再是 `while`，而是**图里的一条边**——它参与状态快照、参与递归上限、可被 get_state 观测。
4. 答复里 `（我记得你住在上海）` 来自 06 画像检索——**内容层记忆（06）已经被 recall 节点注入进图**，这正是里程碑 `012`「带记忆的客服」的最小闭环。

## 4. 实验 B · 条件路由账（判定＝边上的一等公民）

**目标**：量条件路由的性价比——判定从「散落的 if」升级成「边上函数」后，省掉多少无谓工具调用。

**过程**：`add_conditional_edges("route", lambda s: s["go"], {"tool": "call_tool", "answer": "answer", "ask": "ask_user"})`——路由判定是一个**返回边名**的函数，引擎按返回值选支路。对 7 次会话的期望路由做全对账；再反事实对照"每条都先调工具再答"的最坏路径。

**实测**：

```
  路由判定：7 次会话 期望 全对账 → 命中 7/7（route 条件边按返回 go 选支路）
  可直达 #t3#t4#t7 本不需要工具（问候 / 查记忆有上下文）——条件边省掉无谓 call_tool
  → 反事实『每条都先调工具再答』= 38 步 vs 条件路由 35 步：省 3 步（约 7%）
```

**结论**：
1. **判定变成图的一部分**：`route` 返回 `tool/answer/ask` 三个**边标签**，分支行为显式挂在边表里——新增一个出口（比如加一个 `route→pay`）不碰任何节点函数。这就是「路由闸从代码（03-D 的 7/8 词面闸）抬到拓扑」：可审计、可单测、可恢复（中途断点续跑时引擎记得你正停在哪个分支前）。
2. **条件边省的是"无谓动作"不是"思路"**：t3（查记忆有上下文）、t4（问候）、t7（缺信息）三条本来就不用工具，若强迫全走工具，轨迹各多 1 步 → 反事实 38 vs 35，省 3 步（≈7%）。数据小但有方向：**路由判定放边上的价值不止 token，而是不让"不该发生的动作"发生**（动工具=副作用/钱/延迟，和 03-D「要不要带」同一族逻辑）。
3. 与上轮的对话：03-D 路由闸是"要不要把材料装进上下文"，B 账是"要不要把动作派给工具"——两层都在回答「这一步值不值得发生」。

## 5. 实验 C · checkpoint 账（thread 隔离 + 中断恢复 + 两层记忆的分工）

**目标**：把「引擎层状态持久化」量开——checkpoint 记住的不是用户数据，是**运行状态**（停在哪个节点、下一步跑谁），并与 06 内容层记忆划清界限。

**过程一（隔离）**：同一个 checkpointer（一个 MemorySaver），线程 `iso_a` 跑"查团队例会"（查记忆）、线程 `iso_b` 跑"你好谢谢"（问候）；结束后各自取 `state.context` 看有没有串扰。
**过程二（中断恢复）**：编译时挂 `interrupt_before=["call_tool"]`；线程 `ir` 第一段 `invoke` 跑"查北京天气"→ 在 `call_tool` 前挂起；`get_state(cfg).next` 读续跑点；同线程 `invoke(None)` 续跑完成。

**实测**：

```
  线程隔离：iso_a（查记忆）state.context=['团队例会'] vs iso_b（问候）state.context=[] → 串扰 0
  checkpoint 快照：graph.get_state(next=('call_tool',)) —— 引擎记录『下一跳待跑节点』= 活儿干到哪一步
  中断恢复：第一段 parse→recall→route（interrupt_before 停在 call_tool 前）→ 同线程 invoke(None) 续跑 parse→recall→route→call_tool→answer
  → 恢复只补 2 步（call_tool+answer），parse/recall/route 不重跑：中断=挂起而非回滚
```

**结论**：
1. **thread = 一册运行状态**：不同 `thread_id` 的 state 完全隔离（`context` 各归各），同一线程多次 `invoke` 共享进度——**多会话/多用户天然分册**（对应生产多用户并发=要素事实）。
2. **checkpoint 记的是「活儿干到哪一步」**：`get_state().next==('call_tool',)`——引擎把"下一跳待跑节点"快照化。中断是**挂起不是回滚**：续跑只补 2 步（call_tool+answer），`parse/recall/route` 三个已完成节点不重跑——这正是 human-in-the-loop 的引擎底座（人工审批/工具需要人确认时，任务挂在这条边上等 `invoke(None)` 放行）。
3. **与 06 的分工（本课最关键的一层）**：06 的画像/账本是**内容**（张三住上海、团队例会 s2），权限/隐私/更新策略在数据层；checkpoint 是**过程**（这单正在查天气、挂起在 call_tool 前），会话存档/断点续跑在引擎层。**记忆（设备/画像icon） ≠ checkpoint（进度）**——前者回答"用户是谁"，后者回答"活儿干到哪"。真实系统两者都上：checkpoint 负责可恢复，memory 负责个性化，谁也不替代谁。

## 6. 实验 D · 循环与预算账（自环收敛 · SCC 环检测 · 递归上限兜底）

**目标**：量化图引擎的两个"循环相关"能力——重试自环能不能收敛、停不下来时谁来兜底。

**过程一（自环收敛）**：`add_conditional_edges("call_tool", tool_switch, {"ok": "answer", "retry": "call_tool", "esc": "escalate"})`，`tool_switch` 按 `tool_ok` 与 `retry < MAX_RETRY(3)` 判返回 `ok/retry/esc`。t5 走 once 策略（首败二胜）、t6 走 always 策略（恒败）。
**过程二（SCC 环检测）**：把图的邻接表喂纯 stdlib Tarjan，对比"含自环边"与"去掉自环边"两张图的含环 SCC 数。
**过程三（递归上限）**：`recursion_limit=6` 跑需要 7 步的 t6 → 观察引擎抛什么。

**实测**：

```
  自环收敛：t5（失败→重试一次成功）call_tool×2 · t6（恒失败）call_tool×3 → escalate 转人工（MAX_RETRY=3）
  环检测（Tarjan）：含自环图 SCC=7 含环 1（重试边）vs 去掉自环边 SCC=7 含环 0
  递归上限：recursion_limit=6 跑 t6（需 7 步）→ Recursion limit of 6 reached without hitting a stop condition. You can increase the limit by setting the `recursion_limit` config key.
```

**结论**：
1. **重试 = 图上一条自环边**：t5 首败后 `retry` 支路指回 `call_tool` 自己（第二次成功 → `ok` → answer）；t6 三次耗尽走 `esc` → escalate。**失败策略从"外层 while"变成"图上一条边"**——和 05 回喂的报错自愈（重试 1·1·2）同族，但这里是引擎层原语，参与 checkpoint/观测/上限。
2. **环不是 bug，是不收敛才要兜底**：Tarjan 显示含自环图只有 1 个含环 SCC（重试边），去掉自环边之后 0 个——图引擎的两重保障：**建模期**能静态发现环在哪（可审），**运行期**有 `recursion_limit` 兜底。实测超限抛 `GraphRecursionError`（真实框架异常，不是静默死循环）——这对应 05/06 里"轮上限/重试上限"的**引擎级实现**，代码里不用再手写 `while i<3` + 计数器。
3. **阈值是显式参数**：`{"recursion_limit": 6}` 按会话/请求级配置（默认 25=要素事实），生产按业务步数设预算——把你记得住的"重试 3 次"翻译成配置，而不是遗忘在某个函数里。

## 7. 常见坑（本机实测踩过的 6 条）

1. **TypedDict 里没声明的键会被静默丢弃**（实测）：状态 schema 里有 `go` 字段但 TypedDict 没写它 → `invoke` 传入的 `go` 直接消失，条件边函数读到的是 falsy → 恒走默认支，排查半天看不到报错。**铁律：所有要读的状态槽位必须先写进 `class CS(TypedDict)`**。
2. **缺 `add_edge(START, ...)` 编译期直接炸**（实测）：`ValueError: Graph must have an entrypoint`——图没有入口边，`compile` 不通过（比手写流程更早暴露结构错误）。
3. **条件边 path_map 必须合法**：`{"tool": "call_tool", ...}` 的值必须是节点名或 `END`；返回了 path_map 里没有的键会 `KeyError`（设计期就要穷举全部分支，别留 `default` 近路）。
4. **版本别从 `langgraph.__version__` 拿**（实测）：它是 `None`；用 `importlib.metadata.version("langgraph")` 才是真实版本号（本文 `1.0.10` 即 `pip show`/metadata 来源）。
5. **`interrupt_before` 对每次节点调度生效**：同一节点被重试再次调度时**还会再中断一次**（本课 C 账因此选了单次工具调用的会话做中断演示）；要"只在第一次停"需自己维护哨兵 channel，别假设挂起只发生一次。
6. **import 期警告噪音**：langgraph 依赖链会静默插 deprecation 过滤器（实测 `LangChainPendingDeprecationWarning`）——对 stdout 无影响，但脚本输出要干净就先 `import langchain_core` 再 `filterwarnings("ignore")`（本探针做法）。

## 8. 诚实边界（AAA 自我审查）

- **引擎语义 = 本机真实实测**：`langgraph 1.0.10` 真实安装运行，图编译/条件边/自环/checkpoint/thread 隔离/`interrupt_before`+`invoke(None)`/`GraphRecursionError` 全部真 API 实测（stdout md5 `91c11c5b…` 三进程恒一，墙钟 ≈0.10 s 仅 stderr）。
- **节点内部逻辑 = 词面规则作者预置**：`parse`（意图/槽位）、`recall`（记忆检索）、`route`（路由判定）、`call_tool`（工具执行成败注入）、`answer/ask/escalate`（答复模板）都是确定性规则——真实客服 Agent 这些节点是 **LLM 调用（非本机实测：无外网无 key）**；LangGraph 负责的图执行/状态/路由/持久化/递归上限才是本课测量的对象。
- **checkpoint 介质**：`MemorySaver`=进程内内存快照；生产级 SQLiteSaver/RedisSaver/Postgres/Platform 持久化=要素事实非本机实测。
- **流式 / LangSmith / LangGraph Platform / human-in-the-loop 审批流**：§7.3 要素事实，未在本机实测；本课实测的是与之同族的「中断恢复」最小闭环。
- **Tarjan SCC**：纯 stdlib 确定性实现（非 langgraph 能力）；环检测口径=含环 SCC（长度>1 或自环），与业务图邻接表一致。
- **token/墙钟口径**：本课不量 token（无 LLM 节点）；墙钟仅 stderr 且只代表引擎执行开销（≈0.10 s），无业务意义。

## 9. 参考与衔接

- **本课地图**：§7.1 分类学（编排 SDK vs 图引擎 vs RAG 框架 vs Agent 框架 vs 低代码 vs 协议）/ §7.3 LangGraph（有向状态图·节点=函数·边=条件路由·循环/checkpoint/HITL/流式·LangSmith+Platform）/ §2.7 里程碑 `012`。
- **选择树 8 场景**（§7.1：先分类型再比优劣；探针断言 8/8）：① 线性"调模型+组提示"→ 直写或通用编排 SDK（LangChain 链）；② 有分支/循环/恢复/重试 → **图引擎 LangGraph（本篇）**；③ 文档/知识库检索问答 → RAG 框架（LlamaIndex/Haystack，衔接 08）；④ 工具结果即时执行一次 → 05 回喂循环（不引入框架）；⑤ 多 Agent 协作/角色化 → Agent 框架（CrewAI/AutoGen/Agents SDK）；⑥ 非工程师快速搭 Demo → 低代码（Dify/Flowise/n8n）；⑦ 轻量单 Agent·少依赖 → OpenAI Agents SDK/直写工具协议；⑧ 跨栈工具互操作 → **MCP 协议层**（不是框架，是指令接口：衔接本目录 `11-MCP协议` 关键章·里程碑 `013`）。
- **前承**：06-记忆（内容层纪律 → 本篇引擎层）；05-FC（回喂循环 → 本篇条件边自环）；03-D（路由闸 → 本篇边上判定）。
- **后启**：`00-框架分类学`（先把 8 类框架的"该在哪用"钉死，本篇只是图引擎一格的展开）；`01-LangChain与历史包袱`（同为 LangChain 系，链 vs 图的分界）；`03-LlamaIndex`（✅ 已交付 v0.29：RAG 框架一格 = 选择树场景③「知识库检索问答」的落点，真实引擎四账实测）；`04-低代码`（✅ 已交付 v0.30：同样这张图换一种入口——低代码画布把它拖成积木；三家画布客服重试均有回边，与本文 1 自环同结论）；`05-Haystack`（✅ 已交付 v0.31：管线世界的上限=max_runs_per_component（组件运行数），与本文 recursion_limit（轮数）对偶——同一条防死循环纪律、两种计数单位；其账 A 自环禁令=本文自环原语的反例）；`06-Semantic-Kernel`（✅ 已交付 v0.32：护栏三态——SK 的护栏=FunctionChoiceBehavior 配置项（连接器内生效），不是本文 recursion_limit 那样的内核级强制；自动调用=规划器 LLM 化），`07-AutoGen-AG2-Microsoft-Agent-Framework`（✅ 已交付 v0.33：多 Agent 对话范式的整代重写——AG2 1.0 的护栏是『缺模型配置当场 ConfigNotProvidedError』这种创建/运行即栅栏，与本文 recursion_limit 轮数上限/FB 配置项并成三种姿势；`_tools/autogen_demo.py` 真实 ag2 1.0.6 四账）·`08-CrewAI`（✅ 已交付 v0.34：多 Agent 角色化分工——角色/任务/流程三一等公民、任务列表顺序即拓扑=『列表即图』的极简写法，无显式 add_edge（本文显式边为其对照）；护栏=缺 role 构造期 ValidationError、Flows 自环定义期拒绝；`_tools/crewai_demo.py` 真实 crewai 1.15.22 四账）·`10-OpenAI-Agents-SDK与轻量运行时`（✅ 已交付 v0.35：Agent 系收束——轻量运行时=一个 Agent 对象 + 一个 Runner 函数，手转交 handoff() 一行给 Agent 加第二个大脑、Transfer 后 last_agent 真切换=引擎级转移，非声明式拓扑；真实 openai-agents 0.17.0 四账，stdout md5 `d0689dfd…`）；`12-框架继承关系与选型决策`（收束章"该不该引入框架"——本篇 C/D 账给出两条判据：要不要可恢复（checkpoint）、要不要可循环（自环/上限））。

> 本篇完工于 2026-09-22（v0.26 批次）；探针 `_tools/langgraph_demo.py`；stdout md5 `91c11c5b…`。
