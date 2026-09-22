# 🎯 CrewAI：多 Agent 角色化分工 —— 「角色·任务·流程」三个一等公民，把多 Agent 建成团队流水线（研究/原型向 · 与 AG2 对话范式左右手）

> 对应知识点：知识地图 §7.14（**多 Agent 协作 → CrewAI / AutoGen / Agents SDK（handoff）**——决策树给三家名字本身就在说：同为类 4，分工不同）、§7.1 / §7.11（类 **4 多 Agent 框架**：CrewAI / AutoGen / Agents SDK 等——把「Agent 之间互聊/协作」做成范式的类；六家最挤）、§6.9（成本警句：**多 Agent = 多倍 token 与延迟，先证明单 Agent 不行再上多 Agent**）、§7.8（上篇 07-AutoGen 的「多 Agent 对话范式」——本篇是它的左手：**事前声明的团队分工 vs 运行时涌现的群聊**）。
> 前置：[00-框架分类学](./00-框架分类学.md)（账 A 类 4 六家最挤、账 B 状态主业——多 Agent 类把「状态」压进 agent 间对话；选择树场景⑤）；[07-AutoGen-AG2-Microsoft-Agent-Framework](./07-AutoGen-AG2-Microsoft-Agent-Framework.md)（§7.8 对话范式=本篇左右手：AG2「谁先说话=涌现」vs CrewAI「谁做什么=事先声明」；账 A 无配置即 `ConfigNotProvidedError` 的创建即栅栏面）；[02-LangGraph-状态图与Checkpoint](./02-LangGraph-状态图与Checkpoint.md)（账 A 护栏对偶：`recursion_limit` 轮上限真抛 vs 本篇 Flow 定义期拒自环 vs 缺 role 创建即栅栏）；[05-Haystack](./05-Haystack.md)（选择树账的 RAG 管线一格）；[06-应用开发 05-Function-Tool-Calling](../06-应用开发/05-Function-Tool-Calling.md)（账 D 第 9 行「单 Agent + 工具循环」=§6.9 省 token 那条路的面）。
> 动手：`python code/notebooks/_tools/crewai_demo.py`（本机真实执行 crewai 1.15.22 引擎；LLM 回复文本=预置 stub（BaseLLM 子类）保确定性；零网络 · 零随机 · stdout md5 `fd837cd2…` 三个独立进程恒一）。
> 一句话：**CrewAI 把「多 Agent 协作」做成 角色/任务/流程 三个一等公民——`role/goal/backstory` 被真实拼成系统提示 `You are {role}. {backstory}\nYour personal goal is: {goal}`（role 是必填原语，缺了创建即 `ValidationError`）；任务列表=声明式顺序流水线（每任务恰好一轮 LLM 调用、上文自动回填进下一任务提示=隐式链，全程无显式 `add_edge`）；流程由 `Process` 枚举约束——`sequential` 直链、`hierarchical` 走 manager_llm 多一跳（实测输出归属 `Crew Manager`）；2025 新面 **Flows** 提供事件 DAG（`@start / @listen / @router / or_`，路由的『返回值』喂给下游监听，定义期直接拒绝自环）。它不与 AG2 冲突而是左右手：要事先声明的团队流水线 → CrewAI，要运行时涌现的群聊 → AG2（§7.14/§7.8）。**
>
> 📌 诚实边界：**引擎语义 = 本机真实实测**（crewai 1.15.22 的 `Agent(role/goal/backstory)` 创建、`Crew(tasks=[…], process=…)` 顺序驱动、系统/任务提示模板逐字节、上下文接力『任务 N 可见前 N-1 个输出』、`Process.hierarchical` 的 manager_llm 一跳、Flows 的 `@start/@listen/@router/or_` 路由与 kickoff 返回值、缺 role 的 `ValidationError` 原文、自环定义期拒绝原文、`max_method_calls` 默认值、25 个 import 目标 OK/FAIL）；**但 LLM 回复文本 = 非本机实测**——探针里每条回复都是作者预置的 stub（`crewai.llm.BaseLLM` 子类）词面规则桩（真实 LLM 经 CrewAI 的 LLM 连接器输出=非本机实测；hierarchical 的 manager 委托链要在真实 LLM 判定下才出现，stub 只返文本=只实测『它有这么一跳』，不量『manager 分得对不对』）；**CrewAI 年份/路线/企业版、Flows 2025 发布、选择树规则 = 要素事实**（写作环境无外网、未在线复核官方文档，以官方 release/docs 为准）；**企业版/crew1 未安装 = 非本机实测**——账 C 里 `crewai.production` / `crewai.main` 的 FAIL 是本机「没有这个包」，不证明它们不存在。第三节嵌的每一行「实测」均可 `python code/notebooks/_tools/crewai_demo.py` 复现，stdout 与正文逐字节一致；墙钟 ≈8.1 s 仅进 stderr。

---

## 📑 本章目录

1. [为什么 08-CrewAI 值得单独讲一课？](#0-为什么-08-crewai-值得单独讲一课)
2. [先钉死事实：知识地图给 CrewAI 的定位（§7.14 / §7.1 / §7.11 / §6.9 / §7.8）](#1-先钉死事实知识地图给-crewai-的定位-714--71--711--69--78)
3. [把「多 Agent 角色化分工」降维成本机测量（探针设计 + 全量实录）](#2-把多-agent-角色化分工降维成本机测量探针设计--全量实录)
4. [实验 A · 角色化原语账（role/goal/backstory → 系统配方 · 缺 role 创建即栅栏）](#3-实验-a--角色化原语账roegoalbackstory--系统配方--缺-role-创建即栅栏)
5. [实验 B · 流程编排账（sequential 上下文接力 · hierarchical manager 多一跳）](#4-实验-b--流程编排账sequential-上下文接力--hierarchical-manager-多一跳)
6. [实验 C · 版本存续（三张 API 面 + 事件式 Flows 路由 + 0.x 路径换血）](#5-实验-c--版本存续三张-api-面--事件式-flows-路由--0x-路径换血)
7. [实验 D · 选择树账（9 场景断言 9/9）](#6-实验-d--选择树账9-场景断言-99)
8. [拿这张表怎么读本目录（07 章 13 篇的路牌）](#7-拿这张表怎么读本目录07-章-13-篇的路牌)
9. [常见坑（5 个）](#8-常见坑5-个)
10. [诚实边界（AAA 自我审查）](#9-诚实边界aaa-自我审查)
11. [参考与衔接](#10-参考与衔接)

---

## 0. 为什么 08-CrewAI 值得单独讲一课？

07 章路牌走到这里，前八篇分别把「分类学 / LangChain / LangGraph / LlamaIndex / 低代码 / Haystack / SK / AutoGen·AG2·MAF」各展开一格。**这篇展开的是 00-篇账 A 里『六家最挤』的类 4 多 Agent 框架——第二家专门谈的一家，但谈的是一个完全不同的姿势**：

- **07-AutoGen 讲的是「对话」**：agent 们围着一个群聊室互发消息，谁接下句是运行时涌现的。那个范式的问题——§7.8 原话「生产需自己加状态管理」——在于**团队不分工，协作靠聊**。
- **CrewAI 讲的是「团队」**：你先把团队建成型——每个成员有 `role`（角色）、`goal`（目标）、`backstory`（人设），再把工作拆成 `Task` 列表，让 `Crew` 按 `Process` 编排把任务按顺序滚过去。**协作发生在声明期**：谁做什么、做几轮、顺序如何，都在跑之前就定死了。

这是类 4 内部一对漂亮的左右手（§7.14 决策树同时点名 CrewAI 和 AutoGen 的原因）：**要事先声明的团队分工 → CrewAI；要运行时涌现的群聊 → AG2/对话范式**。此外 CrewAI 还有一个 2025 年补上来的第二张面——**Flows（事件 DAG）**：`@start/@listen/@router` 把 agent 之间的消息流做成显式路由图，让这家「角色化分工」的库也能做复杂分支/合并。

本篇要用真实 crewai 1.15.22 引擎，把这几句落成四本可复排的账：**角色化原语账（role/goal/backstory 到底怎么变成系统提示、role 缺了会怎样）→ 流程编排账（sequential 怎么传上下文、hierarchical 的 manager 多一跳在哪）→ 版本存续账（运行时/声明式/事件式三张 API 面 + 0.x 时代的路径死没死）→ 选择树账（多 Agent 各范式该在哪儿用）**。沿一条从前八篇接力下来的护栏对照线：**02 章 `recursion_limit`=「跑飞了才兜」、05 章 `max_runs_per_component`=「循环超限才兜」、06 章 `FunctionChoiceBehavior`=「连接器内配置项」、07 章 AG2 的 `ConfigNotProvidedError`=「没配置别开跑」——本篇加两类：缺 `role` 创建即 `ValidationError`、Flows 自环**定义期**直接拒绝**（同一门「框架怎么拦住你跑飞」的课，六家人有两种姿势）。

与 07 篇的衔接正是 §7.14 给的路牌：同一节把多 Agent 协作指向 CrewAI 与 AutoGen 两家，上篇收了对话范式，本篇收角色化分工。

## 1. 先钉死事实：知识地图给 CrewAI 的定位（§7.14 / §7.1 / §7.11 / §6.9 / §7.8）

五处来源各给一句可复核的定位：

1. **§7.14 专节盖章（本篇第一句话那么重要）**：决策树把「多 Agent 协作」指向 **CrewAI / AutoGen / Agents SDK（handoff）**——三家并排点名，说明类 4 内部「多 Agent」这个词本身还分着几派。本篇账 D 就是把这张表按 9 场景展开；其中 **CrewAI 的『团队流水线』、AG2 的『对话机』、Agents SDK 的『轻量 handoff』**各占一格，互不替代。
2. **§7.1 / §7.11 盖章（类 4 的代表之二）**：多 Agent 框架=把「Agent 之间互聊/协作」做成范式的类，六家最挤；CrewAI 与 AutoGen 同榜。**可测量对应**：「协作」→ 本篇账 A/B 量『任务列表=声明式流水线、每任务一轮调用、上文自动接力』；「互聊」→ 上篇账 A/B 量 AG2 的 run/ask 事件流。同一个类里，「声明式分工」与「运行时对话」是并列的两种实现。
3. **§6.9 成本警句（读所有 Agent 章节前先读这句）**：**多 Agent = 多倍 token 与延迟，先证明单 Agent 不行再上多 Agent**——读 CrewAI 前这条尤其重要：**角色化分工是「把一份上下文拆给 k 个角色各读一遍一份新的」**。账 B 会量到 sequential 每加一个任务，下一任务的提示就再带上前面所有输出——多几个角色，token 就按份数滚。**先证明单 Agent 不行，再上 CrewAI；先证明一个角色不行，再上第二个角色。**
4. **§7.8 的对话范式同章对照（本篇是其左手）**：上篇收了「AutoGen 开创多 Agent 互聊范式、AG2 重写、MAF 2025 承接」。本篇与它同属类 4 但不同姿势：AG2 的拓扑是**运行时**（谁接下句由对话演进），CrewAI 的拓扑是**声明期**（任务列表顺序即结构）。§7.14 同节点把两家并排给出的原因，就是「你要预定的团队分工，还是涌现的群聊」——这是选型时最先要回答的问题。

再钉反向边界：CrewAI **不**承诺帮你做「生产级状态管理/图检索」（状态图归 02 章，账 D 第 3 行）、**不**是轻量单 Agent 的答案（那是 10-篇 Agents SDK / 直写，账 D 行 8）、**不**做 RAG 管线（05/03 章）、**不**是低代码平台（04 章）。**它给的是「把多角色团队建成声明式流水线」这一种范式，以及（在本篇实测的版本里）角色/任务/流程这三个一等公民的精确行为。**

## 2. 把「多 Agent 角色化分工」降维成本机测量（探针设计 + 全量实录）

CrewAI 的口水文好写：介绍 `role/goal/backstory`、画一张 RPG 团队图就完。但要把它**变成可复排的本机实验**，得先把三个模糊问题变成精确断言：role/goal/backstory 到底以什么**字节**拼进系统提示？sequential 的「上下文传递」是真的还是文档话术？hierarchical 的 manager 一跳到底谁在调 LLM？本篇探针真实安装了 **crewai 1.15.22** 引擎（网线可达的一次性安装；运行过程零网络零随机），量 4 本可复排的账：

- **账 A 角色化原语账**：`Agent(role, goal, backstory)` + `Task(description, expected_output, agent)` 列表 → `Crew(agents, tasks).kickoff()`。逐字节锁定引擎生成的系统提示模板与任务提示模板，量「每任务一轮 LLM 调用」的配比，再故意创建缺 `role` 的 Agent 抓引擎真抛的护栏；
- **账 B 流程编排账**：`Process.sequential` 里任务 N 的提示有没有带上任务 0..N-1 的输出（用 stub 的哨兵输出数出来）；`Process.hierarchical` 配 `manager_llm` 时谁在收口（实测输出归属 `Crew Manager`）；
- **账 C 版本存续账**：三张 API 面——运行时（顶层 `Crew`/`Agent`/`Task`/`Process`/`LLM`/`Flow`）、声明式（`crewai.project.CrewBase` + `@agent/@task/@crew`）、事件式（`crewai.flow` 的 `Flow/start/listen/router/or_` 路由语义）——加一张 0.x-era 老路径的生存表（`agent_builder` 等**真 import 探测**）；
- **账 D 选择树账**：9 场景决策表（§7.14 + §7.8 + 前八篇各自的落点），断言 9/9，要素事实。

确定性纪律：`warnings.filterwarnings("ignore")` 最先执行；`sys.stdout.reconfigure(encoding="utf-8")`；全程序零网络、零随机、不打印对象地址/随机 Flow ID/哈希；任何 set 类内省一律 `sorted()`；每次 `kickoff()` 包在 `redirect_stdout/redirect_stderr` 里收框架控制台噪音；`suppress_flow_events=True` 关掉 Flows 富文本状态盒；stub LLM=词面规则预置（回复可复现）。**stdout md5 `fd837cd2…` 三个独立进程逐字节恒一**，墙钟 ≈8.1 s 仅进 stderr。

复现实录（严格逐字节 = 探针 stdout 三遍恒一）：

<details>
<summary>📊 探针 stdout 全量实录（md5 <code>fd837cd2…</code> · 墙钟 ≈8.1 s 仅 stderr）</summary>

```text
========================================================================
crewai_demo：07-应用框架 · 08-CrewAI（多 Agent 角色化分工）（知识地图 §7.14/§7.8/§7.1 类 4/§6.9）
    角色·任务·流程 = 三个一等公民：role/goal/backstory → 系统配方，任务列表 = 声明式流水线
[0] 口径：引擎语义=本机真实实测 crewai 1.15.22；LLM 回复文本=预置 stub(BaseLLM 子类)（真实 LLM 非本机实测）；版本节奏 / 企业版 / Flows 2025=要素事实（无外网未复核）
========================================================================
========================================================================
[账 A] 角色化原语账：Agent(role/goal/backstory) + Task 列表 = 声明式『角色→流水线』（引擎真跑 · stub LLM）
========================================================================
  3 任务 2 角色交替，LLM 被调用 3 次 = 任务数×1（每任务恰好一轮回复）
  按任务序各轮触发的角色 = ['研究员', '评审员', '研究员']
  系统配方（引擎真实模板）：
    sys = 'You are {role}. {backstory}\nYour personal goal is: {goal}'   长度恒定 46 字符 = 模板与 length 一致=同角色同系统（断言 True）
  任务提示（引擎真实模板）含 'Current Task:' / 'expected criteria…' / 'Provide your complete response:' = True
  护栏：Agent 缺 role 创建即栅栏（引擎真抛）:
    ValidationError | 1 validation error for Agent
role
  Field required [type=mis
  → 结论1：role/goal/backstory 被真实拼接成系统提示（role 是必填原语=缺栅栏）
  → 结论2：任务列表=顺序执行、每个任务独立调度一次 LLM——『谁做』由 agent 指定、『做几轮』由任务数决定
========================================================================
[账 B] 流程编排账：sequential 上下文接力 & hierarchical 多一跳（引擎真跑 · stub LLM）
========================================================================
  顺序流程：每任务的用户提示里自动带上『前序输出』= 上下文接力（引擎真拼）
    任务2 可见前序 = [0]（含任务0输出·SENT0）
    任务3 可见前序 = [0, 1]（含任务0+1输出） 累计级联=True
  分层流程：manager_llm 掌控, 任务输出由 manager 收口
    本机实测：调用 1 次（stub manager 直接产出），触达角色 = ['Crew Manager']
    输出归属 agent = ['Crew Manager']
  → 结论1：sequential = 隐式链（任务列表顺序即拓扑，上文自动回填下一任务）, 无显式 add_edge
  → 结论2：hierarchical 走 manager_llm 多一跳；与『对话范式』（AG2 谁先说话=涌现）对比 = 拓扑事先声明 vs 运行时自组织
  → 结论3：stub manager 只返文本不发起委托 → 实测 1 次调用；真实 manager 的委托链=LLM 判定，非本机实测
========================================================================
[账 C] 版本存续账：运行时/声明式/事件式三张 API 面（引擎真实探测）
========================================================================
  事件式 Flows（事件 DAG）：start 入口/监听传参/路由分叉/或合并 —— 2025 新面
    ChainFlow 入口 = ['begin']  路由点 = ['begin', 'converse', 'end']  kickoff = '下游收到:上游101'
    OrFlow    入口 = ['begin']  路由点 = ['A', 'B', 'converse', 'end']  kickoff = '组合收到:A'
    监听节点烫到的是 router 的『路由返回值』而非上游原始输出 —— OrFlow kickoff = '组合收到:A'
  护栏：自环事件定义期直接拒绝（引擎真抛） -> ValidationError | 1 validation error for LoopFlow
  Value error, Invalid flow definition f
  兜底：Flow.max_method_calls 默认 = 100（引擎字段；定义期拒自环→运行时无此环）
  导入面 importlib 探测：
    crewai                                       OK <module:crewai>
    crewai.Crew                                  OK <Crew>
    crewai.Agent                                 OK <Agent>
    crewai.Task                                  OK <Task>
    crewai.Process                               OK <Process>
    crewai.LLM                                   OK <LLM>
    crewai.Flow                                  OK <Flow>
    crewai.CrewBase                              FAIL AttributeError: module 'crewai' has no attribute 'Cr
    crewai.crew.Crew                             OK <Crew>
    crewai.agent.Agent                           OK <Agent>
    crewai.task.Task                             OK <Task>
    声明式面：
    crewai.project.CrewBase                      OK <CrewBase>
    crewai.project.agent                         OK <function>
    crewai.project.task                          OK <function>
    crewai.project.crew                          OK <function>
    事件式面：
    crewai.flow.Flow                             OK <Flow>
    crewai.flow.start                            OK <function>
    crewai.flow.listen                           OK <function>
    crewai.flow.router                           OK <function>
    crewai.flow.or_                              OK <function>
    LLM 注入面 / 0.x-era 残留 / 企业版：
    crewai.llm.BaseLLM                           OK <BaseLLM>
    crewai.llm.LLM                               OK <LLM>
    crewai.agents.agent_builder.base_agent_builder.BaseAgentBuilder FAIL ModuleNotFoundError: No module named 'crewai.agents.agent
    crewai.production.CrewAIEnterprise           FAIL ModuleNotFoundError: No module named 'crewai.production'
    crewai.main.crew1                            FAIL ModuleNotFoundError: No module named 'crewai.main'
  → 结论1：运行时(Crew/Agent/Task/Process/LLM)与事件式(Flow/start/listen/router/or_)全在顶层；
    声明式 CrewBase + agent/task/crew 装饰器只挂在 crewai.project 子包、顶层不导出——三张面在『导入进口』上分层
  → 结论2：0.x-era 子模块路径（agent_builder 等）整包移除=版本存续的换血证据（与本篇 AG2 对照）
  → 结论3：企业版/crew1 未装=本机环境事实，非存在性证明
========================================================================
[账 D] 选择树账：多 Agent 各范式该在哪儿用（§7.14 多 Agent 协作 → CrewAI/AutoGen/Agents SDK；§6.9 先证明单 Agent 不行再上）
========================================================================
  1. 团队型多角色固定流水线（研究→起草→评审 顺序）
     → CrewAI 角色化分工（角色·任务·流程三一等公民，账 A/B 实测）
  2. 多 Agent 自由对话/群聊（谁先说话=涌现）
     → AutoGen / AG2 对话范式（§7.8 武侠）
  3. 复杂状态图/checkpoint/条件边多
     → LangGraph 图状态（02 章，recursion_limit 真兜）
  4. 事件驱动 扇出/扇入/路由，无需多角色
     → CrewAI Flows 事件 DAG（账 C 实测 listen/router/or_）
  5. 严谨生产级 RAG 管线
     → Haystack pipeline 思维（05 章 双栖点别二）
  6. 企业 .NET/微软云栈
     → Semantic Kernel 三原语（06 章 + as_mcp_server）
  7. 低代码/非工程师快速落地
     → Dify / Coze 积木平台（04 章）
  8. 轻量 Agent 运行时/工具回喂
     → OpenAI Agents SDK（§7.11 handoff）
  9. 只想重试/更省 token，先别上多 Agent
     → 单 Agent + 工具循环（§6.9 多 Agent=多倍 token 与延迟）
  断言[团队流水线归 CrewAI] = True
  断言[自由对话归 AG2] = True
  断言[图状态归 LangGraph] = True
  断言[事件 DAG 归 Flows] = True
  断言[生产 RAG 归 Haystack] = True
  断言[企业栈归 SK] = True
  断言[低代码归 Dify] = True
  断言[轻量归 Agents SDK] = True
  断言[省 token 先单 Agent] = True
  → 选择树 9 场景断言 9/9 = True

========================================================================
台账汇总：A 角色化原语（role/goal/backstory→系统配方·任务列表=流水线·缺 role 创建即栅栏）/ B 流程编排（sequential 上下文接力·hierarchical manager 多一跳）/ C 版本存续三面（运行时顶层 6 原语全在 · 声明式 project.CrewBase · 事件式 Flow 路由簇 · 0.x 路径整包移除）/ D 选择树 9 场景 9/9
一话总结：CrewAI 把『多 Agent 协作』做成 角色/任务/流程 三个一等公民——
role/goal/backstory 拼成系统提示、任务列表=声明式流水线（顺序执行·上文自动接力）、
manager 可加到 hierarchical 多一跳、Flows 提供事件 DAG（2025）；
与 AG2 的『对话范式』是左右手：要事先声明的团队分工 → CrewAI，要运行时涌现的群聊 → AG2（§7.14/§7.8）。
done · 一键复现：python code/notebooks/_tools/crewai_demo.py
```

</details>

## 3. 实验 A · 角色化原语账（role/goal/backstory → 系统配方 · 缺 role 创建即栅栏）

**目标**：验证「角色化分工」到底建立在什么原语上——不是一句「配置华丽的人设」，而是**引擎真的把 `role/goal/backstory` 三个字段拼进系统提示**，且 `Task` 列表真的决定「做几轮」。再把 §6.9 那句「多 Agent=多倍调用」落到数字。

**过程**：建 2 个角色（研究员 / 评审员），配 3 个任务交替给角色（研究→评审→定稿），`Crew(agents, tasks, process=sequential).kickoff()`；stub LLM 记录每次被调用的 messages 与 `from_agent`，返回哨兵 `SENT{i}`；另建一个缺 `role` 的 `Agent` 抓护栏。

**实测**（stdout 关键行）：

```
  3 任务 2 角色交替，LLM 被调用 3 次 = 任务数×1（每任务恰好一轮回复）
  按任务序各轮触发的角色 = ['研究员', '评审员', '研究员']
  系统配方（引擎真实模板）：
    sys = 'You are {role}. {backstory}\nYour personal goal is: {goal}'   长度恒定 46 字符
  任务提示（引擎真实模板）含 'Current Task:' / 'expected criteria…' / 'Provide your complete response:' = True
  护栏：Agent 缺 role 创建即栅栏（引擎真抛）:
    ValidationError | 1 validation error for Agent
role
  Field required
```

**结论**：

1. **系统提示 = 引擎拼的真实模板**（逐字节实测）：`You are {role}. {backstory}\nYour personal goal is: {goal}`——三个字段按固定配方进 `system` 消息。对「角色化分工」来说这意味着：**你给的角色不是标签，是被拼进每次模型调用的上下文**；2 个角色交替 3 个任务 = 3 次调用各带各自角色名的系统提示（同角色两轮 sys 长度恒定）。
2. **任务列表 = 声明式流水线**：每任务恰好一轮 LLM 调用，轮数由任务数决定、谁做由 `task.agent=` 决定。`Crew` 是按列表顺序干的活，**拓扑在声明期定死**——这是与 AG2「谁先说话=涌现」的第一次正面差异（账 B 把它焊死）。
3. **缺 `role` = 创建即栅栏（本系列护栏线第 5 个姿势）**：`Agent(goal=..., backstory=...)` 直接抛 `ValidationError | Field required for role`。这不是运行期才兜（02 recursion_limit / 05 max_runs）也不是配置项（06 FunctionChoiceBehavior），而是和 07 章 AG2 的 `ConfigNotProvidedError` 同类：**原语构造时就拦**——多 Agent 框架把「身份」当强制契约。
4. **§6.9 的落点（数字版）**：3 个任务 = 3 轮 LLM 调用，且（见账 B）后一轮的提示包含前一轮输出——**每加一个任务/角色，就是把一份增长的上下文再发一遍**。「先证明单 Agent 不行再上多 Agent」在读 CrewAI 时 =「先一个角色跑顺，再扩团队」。

## 4. 实验 B · 流程编排账（sequential 上下文接力 · hierarchical manager 多一跳）

**目标**：把「流程编排」量成两个精确行为：sequential 的**上下文传递是真是假**（任务 N 的提示里到底有没有任务 0..N-1 的输出）；hierarchical 的 **manager 一跳到底谁在调 LLM**。

**过程**：① 同一角色 3 个任务，stub 返回哨兵 `SENT0/1/2`，数每个任务用户提示里实际出现的哨兵；② `Process.hierarchical` + `manager_llm=StubLLM`，记录 manager 的 `from_agent.role` 与调用次数、看 `CrewOutput.tasks_output` 的输出归属。

**实测**（stdout 关键行）：

```
  顺序流程：每任务的用户提示里自动带上『前序输出』= 上下文接力（引擎真拼）
    任务2 可见前序 = [0]
    任务3 可见前序 = [0, 1]  累计级联=True
  分层流程：manager_llm 掌控, 任务输出由 manager 收口
    本机实测：调用 1 次（stub manager 直接产出），触达角色 = ['Crew Manager']
    输出归属 agent = ['Crew Manager']
```

**结论**：

1. **sequential 的上下文传递是引擎真行为**：任务 2 的提示里真的带着任务 1 的输出（哨兵 SENT0 出现）、任务 3 带着前两个（SENT0+SENT1）。**这就是隐式链**：没有 `add_edge`、没有边声明，任务列表顺序即拓扑，上文自动回填下一任务。对比 02 章 LangGraph 要把每条边画出来——CrewAI sequential 是「**列表即图**」的最省写法，代价是它没有显式分支/条件边（要分支请走账 C 的 Flows 或 02 章图引擎）。
2. **hierarchical = 多一跳**：配 `manager_llm` 后，LLM 调用者从「干活 agent」换成 `Crew Manager`——engine 在任务前先插一个 manager 角色，由它在真实场景中决定谁干这活。**stub 的诚实边界**：stub manager 只返文本不发起委托 → 实测仅 1 次调用、输出归属 `Crew Manager`；真实 manager 的「委托谁、分几步」是 LLM 判定=（要素事实：hierarchical 的手册语义），非本机实测。
3. **与对话范式的对照（本课最重要的一张对比）**：AG2 的拓扑=运行时（谁接下句由对话演进，账 A 的 reply.ask 每步都可能是新角色）；CrewAI sequential=声明期（任务列表写死顺序与角色）。**一样的库内「多 Agent」，一个靠聊、一个靠排**——§7.14 把两家并排给的原因就在这。
4. **token 视角重申 §6.9**：任务 3 的提示=任务 3 模板 + 前 2 个输出；再加大任务 4、5… 提示只增不减。**多角色流水线 = token 按角色数与深度滚雪球**——读这篇的人要先回读 06-应用开发 06-记忆系统那套「挑着喂」才有抓手。

## 5. 实验 C · 版本存续（三张 API 面 + 事件式 Flows 路由 + 0.x 路径换血）

**目标**：给 CrewAI 做一张**三面生存表**——运行时面（你 `from crewai import` 什么）、声明式面（`@agent/@task/@crew` 装饰器在哪）、事件式面（2025 的 Flows 路由语义）——再量 0.x-era 老路径死没死。

**过程**：① 六个顶层运行时原语 + 子模块路径逐一 importlib 探测；② `crewai.project` 的 `CrewBase`/`agent`/`task`/`crew` 与 `crewai.flow` 的 `start/listen/router/or_` 探测；③ 三个 Flow 样例（ChainFlow 直传 / OrFlow router 分叉 / LoopFlow 自环）跑路由语义 + 抓定义期护栏。

**实测**（stdout 关键行）：

```
    ChainFlow 入口 = ['begin']  路由点 = ['begin', 'converse', 'end']  kickoff = '下游收到:上游101'
    OrFlow    入口 = ['begin']  路由点 = ['A', 'B', 'converse', 'end']  kickoff = '组合收到:A'
  护栏：自环事件定义期直接拒绝（引擎真抛） -> ValidationError | Invalid flow definition for LoopFlow
  兜底：Flow.max_method_calls 默认 = 100
    crewai.Crew                                  OK <Crew>
    crewai.Agent                                 OK <Agent>
    ...
    crewai.CrewBase                              FAIL AttributeError: module 'crewai' has no attribute 'CrewBase'
    crewai.project.CrewBase                      OK <CrewBase>
    crewai.project.agent / task / crew           OK <function>
    crewai.flow.Flow / start / listen / router / or_  全部 OK
    crewai.agents.agent_builder.base_agent_builder.BaseAgentBuilder FAIL ModuleNotFoundError
    crewai.production.CrewAIEnterprise           FAIL ModuleNotFoundError（未装=环境事实）
```

**结论**：

1. **运行时面 = 顶层 6 原语全在**：`Crew / Agent / Task / Process / LLM / Flow` 顶层可 `from crewai import`，子模块路径（`crewai.crew.Crew` 等）也可达——CrewAI 走的是**稳定的顶层 API 线**（对比 07 章 AG2 顶层经典 API 整包消失）。
2. **声明式面 = 只挂在 `crewai.project` 子包**：`CrewBase` 与 `@agent/@task/@crew` 装饰器是子包导出，**顶层 `crewai.CrewBase` 不存在（AttributeError）**。这解释了「搜笔记本看到 `from crewai import CrewBase` 就报错」——它是 `crewai.project.CrewBase`。三张面在「导入进口」上就分层了（§7.1 的「分类不只在语义，还在入口」）。这个分层本身有讲究：声明式（轨道化工程 01-章 CrewBase 全家桶）与运行时原语分开，避免顶层命名空间打架。
3. **事件式面 = Flows 路由簇全活 + 护栏是真**：`@start/@listen/@router/or_` 探测 OK；ChainFlow kickoff=`下游收到:上游101`（直传上游返回值）；OrFlow kickoff=`组合收到:A`（**router 的路由『返回值』喂给下游监听，不是上游原始输出**）。**自环定义期直接拒绝**：`@listen("loop")` 指向 `loop` 自己时，构造 `LoopFlow` 当场 `ValidationError | Invalid flow definition`——比 02/05 的「跑飞了兜」兜得更早，是**构造期护栏**（对偶 07 章 AG2 无配置构造期 `ConfigNotProvidedError`）。`max_method_calls` 默认=100 仍在，但定义期已把自环拦住。
4. **0.x-era 路径整包移除**：`crewai.agents.agent_builder.base_agent_builder` 真 import **FAIL**（ModuleNotFoundError）——CrewAI 的 0.x→1.x 并非没有换血，只是比 AG2 含蓄：顶层原语存活、内部模块路径重建。**企业版/crew1 未装=环境事实**，不是存在性证明。
5. **护栏线收官**：至此 07 章把「框架怎么拦住你跑飞」收集齐五种姿势——02 轮上限、05 组件运行上限、06 连接器配置项、07 无配置别开跑、本篇构造期拒自环/缺 role。**同一条问题，六位库作者六种答案。**

## 6. 实验 D · 选择树账（9 场景断言 9/9）

**目标**：把「多 Agent 各范式该在哪儿用」落成一张断言表（§7.14 多 Agent 协作 → CrewAI/AutoGen/Agents SDK + §7.8 对话范式 + §6.9 先单 Agent + 前八篇各自的落点，要素事实）。探针里 9 场景 9 断言逐条为真。

**实测**（stdout 关键行）：

```
  1. 团队型多角色固定流水线（研究→起草→评审 顺序）
     → CrewAI 角色化分工（角色·任务·流程三一等公民，账 A/B 实测）
  2. 多 Agent 自由对话/群聊（谁先说话=涌现）
     → AutoGen / AG2 对话范式（§7.8 武侠）
  4. 事件驱动 扇出/扇入/路由，无需多角色
     → CrewAI Flows 事件 DAG（账 C 实测 listen/router/or_）
  9. 只想重试/更省 token，先别上多 Agent
     → 单 Agent + 工具循环（§6.9 多 Agent=多倍 token 与延迟）
  → 选择树 9 场景断言 9/9 = True
```

**结论**：

1. **要「团队型、分工明确、顺序固定」→ CrewAI**：这是本账给 CrewAI 的格子——研究→起草→评审这类「步骤谁来做写死」的流水线。**注意再叠 §6.9**：先一个角色跑顺再扩团队，账 B 的「任务 N 带前 N-1 输出」就是那份滚雪球账单。
2. **要「自由对话/群聊」→ AG2（不是 CrewAI）**：本体说法仍按 §7.14——谁先说话、谁接下句由对话演进，放 AG2；要预定就放 CrewAI。这是类 4 内部第一条分界线。
3. **要「事件分支/合并」且不想上整张状态图 → CrewAI Flows**：账 C 实测 `@router/or_` 能扇出/扇入——CrewAI 的「第二张面」补了 sequential 没有的显式路由；但真复杂状态/checkpoint 仍归 02 章 LangGraph（行 3）。
4. **其余六格=各取各位**：状态图→02、生产 RAG→05、RAG 专精→03、.NET→06、低代码→04、轻量→10；行 9 是给所有多 Agent 故事的预警格——**省 token 或重试时，单 Agent + 工具循环（06-应用开发 05 篇）不该被略过**。

## 7. 拿这张表怎么读本目录（07 章 13 篇的路牌）

本页是 07 章「第九站」：00 篇回答了「框架分几类」，01 篇「通用编排家长子」，02 篇「图引擎一格」，03 篇「RAG 专精一格」，04 篇「低代码一格」，05 篇「生产级 RAG 管线一格」，06 篇「企业 .NET 栈一格」，07 篇「多 Agent 对话范式一格」，**本页回答「多 Agent 角色化分工一格」**。后文各篇的路牌（按依赖次序）：

| 后文篇 | 读它的理由（对应本页哪一格） | 状态 |
|---|---|---|
| `00-框架分类学` | 账 A 类 4 六家最挤、账 B 状态主业：角色化分工=把『团队结构』做进声明期 | 🔥 已交付（v0.27） |
| `01-LangChain与历史包袱` | 账 C 版本存续的对照母题：顶层原语稳定 vs 内部模块换血 | 🔥 已交付（v0.28） |
| `02-LangGraph-状态图与Checkpoint` | 账 B/C 护栏对偶：显式边 vs 列表即图；recursion_limit vs 构造期拒自环 | 🔥 已交付（里程碑 012） |
| `03-LlamaIndex` | 账 D 决策表另一臂：RAG 数据管线专精住进 03 | 🔥 已交付（v0.29） |
| `04-Dify-Flowise-Coze-n8n低代码` | 账 D 决策表另一臂：非工程师落 04（视觉拖拽的『团队』） | 🔥 已交付（v0.30） |
| `05-Haystack` | 账 D 场景⑤：生产级 RAG 管线归它（严谨 pipeline 思维） | 🔥 已交付（v0.31） |
| `06-Semantic-Kernel` | 账 D 场景⑥：企业栈归它；护栏三态 → 插 06 的 FunctionChoiceBehavior | 🔥 已交付（v0.32） |
| `07-AutoGen·AG2·MAF` | §7.14 同节点左右手：对话范式 vs 角色化分工（本篇账 B 终极对照） | 🔥 已交付（v0.33） |
| `08-CrewAI`（本篇） | 多 Agent 团队流水线：角色·任务·流程三一等公民 + Flows 事件 DAG | 🔥 已交付（v0.34） |
| `09-DSPy` | 账 A 类 5：质量敏感评测（提示图谱大师面，与角色化互补） | 🔥 已交付（v0.36） |
| `10-OpenAI-Agents-SDK` | 账 D 场景⑧：轻量运行时「回归轻量」先例（handoff）——对话/角色化/轻量转移三范式收束 | 🔥 已交付（v0.35） |
| `11-MCP协议` | 账 C "2025 公共底座"、里程碑 `013`；Agent 系的工具出口 | 🔥 已交付（v0.37） |
| `12-继承关系与选型决策` | 收束章：完整 DAG + "该不该引入"量化，00 篇 C 账扩版 | ⬜ |

读法口诀（本页的一页带走）：**遇到多 Agent 需求，先问『分工要不要事先定死』——要 → CrewAI（角色/任务/流程三个一等公民，账 A/B 的量就读读）；不要、想要聊出来的 → AG2/对话范式（07 篇）；要复杂状态图→02 章；要事件分支→本页账 C 的 Flows；但所有之前先压住 §6.9：先证明单 Agent 不行、先一个角色跑顺，再扩团队——账 B 的『任务 N 带前 N-1 输出』就是那份会滚的 token 账单。**

## 8. 常见坑（5 个）

1. **从 `from crewai import CrewBase` 开始写**（本课第一个坑）：账 C 真 import 探测——顶层 `crewai.CrewBase` 是 **AttributeError**，声明式装饰器在 `crewai.project.CrewBase`（`@agent/@task/@crew` 也是 project 子包）。看老笔记本报 `from crewai import CrewBase 失败` 先查你导的是顶层还是子包。
2. **以为「角色化分工」里你会看到显式的边/图**：账 A/B 实测——sequential 没有 `add_edge`，**任务列表顺序即拓扑**、上文自动接力。想画分支/条件边？那是 Flows（账 C：`@router/or_` 真路由）或 02-章 LangGraph，不是 sequential 的活儿。
3. **把 hierarchical 当作「manager 会自动派人」**：账 B 实测 stub 下 manager 调用 1 次、输出归属 `Crew Manager`——**manager_llm 是一跳真实存在**，但「它决定委托谁」要靠真实 LLM 判定（非本机实测）。stub 只证明「有这么一跳」，不证明「manager 分得对」。
4. **让 Flow 自己引用自己以为 max_method_calls 会兜**：账 C 实测——`@listen("loop")` 指向 `loop` 自己会在**构造期**直接 `ValidationError（Invalid flow definition）`，根本走不到 `max_method_calls=100`。自环不是「跑飞被兜住」，是「写都写不出来」——要无限循环就别在 Flow 里表达。
5. **不看 §6.9 就扩角色**：账 B 的 snowball 是每一次任务都带上前序全部输出——**多角色/多任务 = 上下文按份滚**。想重试/省 token 先落单 Agent + 工具循环（06-应用开发 05 篇 / 账 D 行 9）；遇到长上下文想挑着喂，回 08-RAG 06-上下文压缩那套剂量账。

## 9. 诚实边界（AAA 自我审查）

- **引擎语义 = 本机真实实测（crewai 1.15.22）**：`Agent(role/goal/backstory)` 创建、`Crew(tasks=[…])` 顺序驱动、系统提示模板逐字节（`You are {role}…`）、任务提示模板逐字节（`Current Task:` 等）、上下文接力（任务 N 提示含前 N-1 哨兵输出）、`Process.hierarchical` + `manager_llm` 的调用归属（`Crew Manager`）、Flows 的 `@start/@listen/@router/or_` 路由语义（Chain 直传 / Or 组合 / kickoff 返回值）、缺 role 的 `ValidationError` 原文、自环构造期拒绝原文、`max_method_calls` 默认 100、25 个 import 目标 OK/FAIL——全部在本机真实引擎上跑出（有真实报错原文为证）。凡可复现行为都对真实输出。
- **LLM 回复文本 = 非本机实测（stub 预置）**：探针里每条 LLM 回复（`SENT{i}`、`托管输出`）都是作者预置的词面规则桩（`crewai.llm.BaseLLM` 子类）——这是保证零网络零随机、stdout 三进程 md5 恒一的前提。**请勿把 stub 的「回复内容」与真实 LLM 的团队协作质量混淆**：真实 LLM（GPT/DeepSeek/Qwen…）经 CrewAI 的 LLM 连接器跑出来的「多 Agent 分工效果」=非本机实测；hierarchical 的 manager 委托链同理——账 B 只量「这一跳存在 + 归属 manager」，不量「分得对不对」。
- **版本存续行 = 本机真 import 探测 + 版本节奏要素事实**：六个顶层原语 / 子模块 / 声明式 / 事件式 25 个目标名的 OK/FAIL 是真实 importlib 探测（含 `AttributeError`、`ModuleNotFoundError` 原文）。但 **「CrewAI 2023 发布」「Flows 2025 推出」「企业版存在但需授权」这些年份/路线/模块归属 = 要素事实**，写作环境无外网、未在线复核官方文档，以官方 release/docs 为准。**账 C 的核心发现（顶层原语稳定、声明式只在 project 子包、`agent_builder` 路径整包移除）是本机 FAIL/OK 撑住的；企业版 `production`/`main.crew1` 的 FAIL 只代表本机未装。**
- **账 D 选择树 = 作者按知识地图 §7.14 / §7.8 / §7.11 / §6.9 与前八篇落点整理（要素事实）**：9 场景决策规则为作者整理定性，非本机可运行断言；它与 00-章账 D 场景⑤、07-章账 D、02-章选择树场景⑤ 的数据一致是硬纪律（跨章数字不打架）。
- **确定性纪律**：stdout md5 `fd837cd2…` 三个独立进程逐字节恒一（每次 `python code/notebooks/_tools/crewai_demo.py` 复现）；墙钟 ≈8.1 s 仅进 stderr；不打印随机 Flow ID/对象地址/哈希；set 类内省一律 `sorted()`；`suppress_flow_events=True` + `redirect_stdout/stderr` 收框架控制台噪音（否则 Flow 富文本盒会污染输出哈希）。

## 10. 参考与衔接

- **本页地图**：§7.14（多 Agent 协作 → CrewAI/AutoGen/Agents SDK——类 4 三家分工）+ §7.1/§7.11（类 4 多 Agent 框架、六家最挤）+ §6.9（多 Agent=多倍 token 与延迟，先证明单 Agent 不行）+ §7.8（上篇对话范式，本篇左手）；07-应用框架篇内：上接 `07-AutoGen·AG2·MAF`（§7.14 同节点）。
- **选择树 9 场景**（探针断言 9/9，要素事实）：① 团队型固定流水线 → CrewAI；② 自由对话/群聊 → AG2；③ 复杂状态图/checkpoint → LangGraph；④ 事件分支/路由 → CrewAI Flows；⑤ 生产 RAG → Haystack；⑥ .NET 企业栈 → SK；⑦ 非工程师 → 低代码；⑧ 轻量回归 → Agents SDK；⑨ 省 token/重试 → 单 Agent + 工具循环。
- **前承**：00-框架分类学（类 4 六家最挤、状态主业、选择树场景⑤）；07-AutoGen（§7.8 对话范式、账 A 无配置 `ConfigNotProvidedError` 构造期护栏面）；02-LangGraph（`recursion_limit` 轮上限真抛、显式边 vs 列表即图）；05-Haystack（`max_runs` 上限）；06-Semantic-Kernel（`FunctionChoiceBehavior` 配置项护栏）；06-应用开发 06-记忆系统（上下文按份滚的挑着喂解药）。
- **后启**：`09-DSPy`（✅ 已交付 v0.36：类 5 质量敏感评测，角色化互补——真实 dspy 2.6.27 四账：签名=契约、编译=自动化优化、DSPy 无承袭边独立的兑现）；`10-OpenAI-Agents-SDK`（✅ 已交付 v0.35：轻量运行时/Agents SDK 的回归轻量对照——本篇 role/goal/backstory 三套装它的 name-only 最小面（16 构造参数仅 1 必填），sys 原样透传 vs 本篇 `You are {role}` 配方，handoff §7.11 实测；真实 openai-agents 0.17.0 四账）；`11-MCP协议`（✅ 已交付 v0.37：Agent 系的工具出口，里程碑 `013`）；`12-继承关系与选型决策`（收束章：完整 DAG + "该不该引入"量化）。
- **对应里程碑**：`012`（02-章交付）之后，多 Agent 两杆（对话 / 角色化分工）分别由 07/08 两篇收；里程碑 `013`（手写 MCP Server）已在 `11-MCP协议` 章交付（v0.37），Agent 系的工具出口就位。

> 本篇完工于 2026-09-22（v0.34 批次）；探针 `code/notebooks/_tools/crewai_demo.py`；stdout md5 `fd837cd2…`。
