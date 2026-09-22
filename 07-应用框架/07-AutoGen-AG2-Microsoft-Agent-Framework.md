# 🎯 AutoGen · AG2 · Microsoft Agent Framework：多 Agent 对话范式 —— 「Agent 之间互聊」开创的范式，正被函数式重写（研究/原型向 · 微软→社区→官方延续线）

> 对应知识点：知识地图 §7.8（**AutoGen（2023，微软）：多 Agent 对话式协作（group chat），开创「Agent 之间互聊」范式。分支：AG2（社区维护）；微软 2025 年推出 Microsoft Agent Framework（承接 AutoGen，与 Semantic Kernel 统一）。适用：研究/原型多 Agent 场景；生产需自己加状态管理**）、§7.1（类 4 **多 Agent 框架**：CrewAI / AutoGen / Agents SDK 等——把「Agent 之间互聊/协作」做成范式；六家最挤的类）、§6.9（成本警句：**多 Agent = 多倍 token 与延迟，先证明单 Agent 不行再上多 Agent**）、§7.14（选型：多 Agent 协作 → CrewAI / AutoGen / Agents SDK（handoff））、§7.11（轻量运行时 Agents SDK 的对照）。
> 前置：[00-框架分类学](./00-框架分类学.md)（账 A 类 4 六家最挤、账 B 状态主业——多 Agent 类把「状态」压进 agent 间对话；选择树场景⑤）；[02-LangGraph-状态图与Checkpoint](./02-LangGraph-状态图与Checkpoint.md)（账 D `recursion_limit` 轮上限真抛 `GraphRecursionError`——**本篇账 A 无配置护栏的对偶**：LangGraph 兜「最多跑几轮」、AG2 兜「没模型配置别开跑」）；[06-Semantic-Kernel](./06-Semantic-Kernel.md)（§7.8 相邻：SK 的「下一步」=与 AutoGen 的统一；账 B 护栏三态对照）；[06-应用开发 06-记忆系统](../06-应用开发/06-记忆系统.md)（记忆回灌账=本篇账 B 的语义同源：跨会话/跨轮的信息载体都是「把历史再喂回模型」）。
> 动手：`python code/notebooks/_tools/autogen_demo.py`（本机真实执行 ag2 1.0.6 引擎；Agent 回复文本=预置 stub client 保确定性；零网络 · 零随机 · stdout md5 `b39ccb9b…` 三个独立进程恒一）。
> 一句话：**多 Agent 对话范式被 AG2 1.0.6 整代重写了——AutoGen 经典『ConversableAgent + GroupChat 互聊』API 全部消失（连 `import autogen` 兼容垫片、`ag2.agentchat` 同构迁移层都整包移除），新一代=函数式 `Agent` + `run()/ask()` 回合原语（一次 run=一条 prompt→LLMCall→AgentReply 流水；AgentReply 不是聊天消息缓冲，是模型事件）+ 子任务委托。『对话』在两个版本语义不同：0.2 是消息池、1.0 是事件流——`reply.ask()` 真把历史回灌给模型（账 B 实测：第 2 轮模型可见前 2 条历史回复）。模型配置是原语的强制契约：缺配置当场 `ConfigNotProvidedError`（一类「创建/运行即栅栏」，不是 02 的轮数上限）。生产要的状态/检查点由 v1.0 `Task` 显式提供（`checkpoint_resume` 参数）。这是 07 章改名第二狠的一家；微软的延续线不在 AG2，而在 2025 Microsoft Agent Framework 承接 AutoGen 与 SK 统一（要素事实）。**
>
> 📌 诚实边界：**引擎语义 = 本机真实实测**（ag2 1.0.6 的 `Agent(name, prompt, config)` 创建、`async with a.run() → run.result()`、prompt 可传 callable 且真求值、`AgentReply.body`、`reply.ask()` 多轮历史回灌、无模型配置 `ConfigNotProvidedError` 真抛文本、版本名卡 import 探测）；**但 Agent 回复文本 = 非本机实测**——探针里每条回复都是作者预置的 stub client 词面规则桩（真实 LLM 经 AG2 的 `LLMClient` 连接器输出=非本机实测，账 B 的「可见前 N 条」只是量事件回灌这件事）；**AutoGen(2023)/AG2 分支/微软 2025 推出 MAF 的年份与路线、`Task`/`checkpoint_resume` 参数存在性、选择树规则 = 要素事实**（写作环境无外网、未在线复核官方文档，以官方 release/docs 为准）；**MAF 未安装 = 非本机实测**——账 C 里 `microsoft.agents` 的 FAIL 是本机「没有这个包」，不证明微软没发布。第三节嵌的每一行「实测」均可 `python code/notebooks/_tools/autogen_demo.py` 复现，stdout 与正文逐字节一致；墙钟 ≈0.1 s 仅进 stderr。

---

## 📑 本章目录

1. [为什么 07-AutoGen-AG2 值得单独讲一课？](#0-为什么-07-autogenag2-值得单独讲一课)
2. [先钉死事实：知识地图给 AutoGen/AG2/MAF 的定位（§7.8 / §7.1 / §6.9 / §7.14 / §7.11）](#1-先钉死事实知识地图给-autogenag2maf-的-定位-78--71--69--714--711)
3. [把「多 Agent 对话范式」降维成本机测量（探针设计 + 全量实录）](#2-把多-agent-对话范式降维成本机测量探针设计--全量实录)
4. [实验 A · 新一代三原语账（Agent.run → AgentReply：函数式收窄）](#3-实验-a--新一代三原语账agentrun--agentreply函数式收窄)
5. [实验 B · 多轮会话账（ask() 续谈 · 历史回灌）](#4-实验-b--多轮会话账ask-续谈--历史回灌)
6. [实验 C · 版本存续（0.2『互聊』API 整代换血 · 改名烈度）](#5-实验-c--版本存续02互聊api-整代换血--改名烈度)
7. [实验 D · 选择树账（9 场景断言 9/9）](#6-实验-d--选择树账9-场景断言-99)
8. [拿这张表怎么读本目录（07 章 13 篇的路牌）](#7-拿这张表怎么读本目录07-章-13-篇的路牌)
9. [常见坑（5 个）](#8-常见坑5-个)
10. [诚实边界（AAA 自我审查）](#9-诚实边界aaa-自我审查)
11. [参考与衔接](#10-参考与衔接)

---

## 0. 为什么 07-AutoGen·AG2 值得单独讲一课？

07 章路牌走到这里，前六站分别把「分类学 / LangChain / LangGraph / LlamaIndex / 低代码 / SK」各展开一格；**第 07 站展开的是 00-篇账 A 里「六家最挤」的那个类：类 4 多 Agent 框架**。而这篇值得单独讲，不是因为「多 Agent 热闹」，是因为**它在 2025–2026 撞上一整代的 API 重写**——知识地图 §7.8 那句「AutoGen 分支 AG2（社区维护）」不是稳定的酸枝木，是**已经换过一次血的副作用现场**：

- **AutoGen（2023，微软）开创的「Agent 之间互聊」范式**：`ConversableAgent` / `UserProxyAgent` / `GroupChat`——大家围着同一个聊天室互发消息、由一个 manager 决定谁接下一句。这个范式定义了一个时代（研究/原型最常用的多 Agent 写法），但它的**实现细节**在 1.0 全部推倒重来。
- **AG2 1.0.6（社区维护线）把它重写成「函数式 Agent + run/ask 回合」**：`import autogen` 兼容垫片没了、`ag2.agentchat` 同构迁移层也没了，经典互聊 API **整包消失**（本篇账 C 真 import 探测 8+3 连 FAIL）。说话的人还在，身份证号全换了。
- **微软的延续线**：2025 推出 **Microsoft Agent Framework（MAF）承接 AutoGen、与 Semantic Kernel 统一**——所以 AG2 不是「AutoGen 2」，是社区改写的平行线；官方给 AutoGen 的「下一代」是 MAF（要素事实，未装，非本机实测）。

本篇要用真实 ag2 1.0.6 引擎，把这三句落成四本可复排的账：**新一代三原语账（语法层面的范式转弯）→ 多轮会话账（『对话』在两个版本的语义差异）→ 版本存续账（经典 API 死得有多彻底）→ 选择树账（多 Agent 范式该在哪儿用）**。加上一条贯穿全篇的护栏对照线：**02 章 LangGraph 的 `recursion_limit` 是「脚本自动停止」、06 章 SK 的 `FunctionChoiceBehavior` 是「配置项」、本篇 AG2 的 `ConfigNotProvidedError` 是「缺配置别开跑」**——同一门「框架怎么拦住你跑飞」的课，三家人有三种姿势。

和 06-章 Semantic-Kernel 的衔接正是 §7.8 给的路牌：SK 是「企业 .NET 栈的编排」，AutoGen/AG2 是「研究/原型的多 Agent 对话」；两者在微软侧的统一线=MAF。本篇把它接住。

## 1. 先钉死事实：知识地图给 AutoGen/AG2/MAF 的定位（§7.8 / §7.1 / §6.9 / §7.14 / §7.11）

五处来源各给一句可复核的定位：

1. **§7.8 专节盖章**：AutoGen（2023，微软）——"多 Agent 对话式协作（group chat），开创『Agent 之间互聊』范式"。**两个可测量对应**："对话式协作/group chat"→ 本篇账 C 里 `GroupChat`/`ConversableAgent` 曾是真对象（现在连 import 都 FAIL）；"互聊范式"→ 账 A 会量到新一代把它收窄成 run/ask 回合。§7.8 还给了三条关键分支事实：**分支 AG2（社区维护）**（本篇整篇主角）；**微软 2025 年推出 Microsoft Agent Framework（承接 AutoGen，与 Semantic Kernel 统一）**（微软侧延续线）；**适用：研究/原型多 Agent 场景；生产需自己加状态管理**（账 D 决策表的 1/2 行源头——"生产要状态"从 AG2 里挪给 02-章图引擎）。
2. **§7.11 盖章（类 4 的代表作）**：多 Agent 框架=把「Agent 之间互聊/协作」做成范式的类，**六家最挤**（AutoGen/CrewAI/Agents SDK/MemGPT…）；§7.14 决策树把「多 Agent 协作」指向 **CrewAI / AutoGen / Agents SDK(handoff)**——本篇账 D 就是这张决策表的 9 场景展开，其中轻量运行时（Agents SDK）一格=${}回 10-篇。
3. **§6.9 成本警句（读所有 Agent 章节前先读这句）**：**多 Agent = 多倍 token 与延迟，先证明单 Agent 不行再上多 Agent**——帐 D 行的「研究/原型用 AG2」必须叠这句：多 Agent 不是免费的白板，每加一个 agent 就是把同样的上下文按份数重发（账 B 会看到历史逐轮变长）。
4. **§7.8 与 §7.7 的官方绑定**：微软侧的完整故事= AutoGen（多 Agent 对话）→ 2025 MAF 承接、与 Semantic Kernel（企业 .NET 栈）统一。**本篇 + 06-篇是 07 章里唯二沾到这条微软统一线的章节**，区别在 SK 讲"函数即一等公民"，本篇讲"多 Agent 对话范式及其整代重写"。

再钉反向边界：AutoGen/AG2 **不**承诺帮你做「生产级状态管理」（§7.8 原文：生产需自己加状态管理——账 D 把它接 02-章）、**不**是轻量单 Agent 的答案（那是 10-篇 Agents SDK / 直写，账 D 行 8）、**不**是低代码（04-篇）、**不**做 RAG 管线（05/03-篇）。**它给的是「让 agent 们互相聊起来」这一种范式，以及（在本篇实测的这代）一次完整的 API 换血警示。**

## 2. 把「多 Agent 对话范式」降维成本机测量（探针设计 + 全量实录）

一篇"AutoGen 开创了多 Agent 时代"的口水文不稀奇；稀奇的是把**四个问题变成可复排的本机实验**：这一代的 API 长什么样？"对话"还是你以为的那样吗？老教程还能不能跑？以及——多 Agent 到底该在哪儿用？本篇探针真实安装了 **ag2 1.0.6** 引擎（网线可达的一次性安装；运行过程零网络零随机），量 4 本可复排的账：

- **账 A 新一代三原语账**：`Agent(name, prompt, config)` 创建 → `async with a.run(消息) as run: result()` 驱动 → 拿 `AgentReply`（`.body`）。顺便验证 §7.8 那个"重写"不是改皮：**prompt 现在可以是 callable 且引擎真求值**；无模型配置时**引擎当场抛 `ConfigNotProvidedError`**（真错误文本）；
- **账 B 多轮会话账**：`AgentReply.ask(下一句)` 续谈——**历史是否回灌模型**、回灌几条，用 stub 里"数当前消息里有多少条历史 Assistant 回复"直接量（第 2 轮可见前 2 条=引擎真回灌）。这关系"对话"这个词在两个版本里的语义：
- **账 C 版本存续账**：pyautogen 0.2 经典 8 名（含顶层 `autogen`）、AG2 0.9-era 的 `ag2.agentchat` 3 名、AG2 1.0.6 新命名 8 名、`microsoft.agents`（MAF）2 名——**真 import 探测** OK/FAIL 全实录；
- **账 D 选择树账**：9 场景决策表（§7.14 + §7.8 + 前六篇各自的落点），断言 9/9，要素事实。

确定性纪律：`warnings.filterwarnings("ignore")` 最先执行（stderr 纯净，不污染 md5）；`sys.stdout.reconfigure(encoding="utf-8")`；全程序零网络、零随机、不打印对象地址；stub client=词面规则预置（回复可复现）。**stdout md5 `b39ccb9b…` 三个独立进程逐字节恒一**，墙钟 ≈0.1 s 仅进 stderr。

复现实录（严格逐字节 = 探针 stdout 三遍恒一）：

<details>
<summary>📊 探针 stdout 全量实录（md5 <code>b39ccb9b…</code> · 墙钟仅 stderr）</summary>

```text
========================================================================
autogen_demo：07-应用框架 · 07-AutoGen·AG2·Microsoft-Agent-Framework（知识地图 §7.8/§7.1 类 4/§6.9/§7.14）
    多 Agent 对话范式：AutoGen『Agent 互聊』→ AG2 1.0 函数式重写 → Microsoft Agent Framework
[0] 口径：引擎语义=本机真实实测 ag2 1.0.6；Agent 回复文本=预置 stub client（真实 LLM 非本机实测）；AutoGen/AG2/MAF 年份、版本路线、选择规则=要素事实（无外网未复核）
========================================================================
========================================================================
[账 A] 新一代三原语账：Agent(name, prompt, config) → run() → AgentReply（引擎真跑 · stub client）
========================================================================
  Agent 一次交互=一条 run 流水：prompt → LLMCall → AgentReply（原语=函数式，不再是『互聊』消息）
  agent.name        = assistant
  reply.body        = '答案：42'
  reply 类型        = ag2.agent.AgentReply
  无模型配置护栏（引擎真抛）: 
    ConfigNotProvidedError | No model config provided. Set config on the `Agent(config=...)` creation or pass it to call `ask(config=...)`.
  → 结论1：『对话』被拆成 run()/ask() 回合原语，Agent 回复=模型事件（非聊天消息缓冲）
  → 结论2：模型配置是 Agent 原语的强制契约——缺配置当场 ConfigNotProvidedError（对比 02/05 的轮数类护栏）
========================================================================
[账 B] 多轮会话账：AgentReply.ask() 续谈——引擎把历史回灌给模型（对话即记忆载体）
========================================================================
  第0轮 reply.body  = '第1轮：模型可见前 0 条历史回复'
  第1轮 ask.body   = '第2轮：模型可见前 1 条历史回复'
  第2轮 ask.body   = '第3轮：模型可见前 2 条历史回复'
  → 结论：AG2 1.x 的『对话』=同一 Agent 上连续 run/ask，历史以事件流形式回灌模型
    （第2轮 stub 实测可见前 2 条历史回复=引擎真回灌；对偶 06 章记忆回灌 / 02 checkpoint）
========================================================================
[账 C] 版本存续账：pyautogen 0.2 经典『互聊』API → AG2 0.9-era agentchat → AG2 1.0.6 函数式换血
========================================================================
  ── pyautogen 0.2 经典命名空间（AutoGen『互聊』API） ──
    autogen                                            FAIL ModuleNotFoundError: No module named 'autogen'
    autogen.agentchat.conversable_agent                FAIL ModuleNotFoundError: No module named 'autogen'
    autogen.agentchat.assistant_agent                  FAIL ModuleNotFoundError: No module named 'autogen'
    autogen.agentchat.user_proxy_agent                 FAIL ModuleNotFoundError: No module named 'autogen'
    autogen.groupchat.GroupChat                        FAIL ModuleNotFoundError: No module named 'autogen'
    autogen.oai.Completion                             FAIL ModuleNotFoundError: No module named 'autogen'
    autogen.code_utils                                 FAIL ModuleNotFoundError: No module named 'autogen'
    autogen.agentchat.contrib                          FAIL ModuleNotFoundError: No module named 'autogen'
  ── AG2 0.9-era 命名空间（agentchat 同构迁移） ──
    ag2.agentchat.conversable_agent                    FAIL ModuleNotFoundError: No module named 'ag2.agentchat'
    ag2.agentchat.assistant_agent                      FAIL ModuleNotFoundError: No module named 'ag2.agentchat'
    ag2.agentchat.groupchat.GroupChat                  FAIL ModuleNotFoundError: No module named 'ag2.agentchat'
  ── AG2 1.0.6 新一代命名空间（函数式 Agent） ──
    ag2                                                OK <module>
    ag2.agent.Agent                                    OK <Agent>
    ag2.agent.AgentReply                               OK <AgentReply>
    ag2.agent.AgentRun                                 OK <AgentRun>
    ag2.task.Task                                      OK <Task>
    ag2.config.client.ModelResponse                    OK <ModelResponse>
    ag2.tools.subagents                                OK <module>
    ag2.events.types.Usage                             OK <Usage>
  ── Microsoft Agent Framework（2025 微软，未装=非本机实测） ──
    microsoft.agents                                   FAIL ModuleNotFoundError: No module named 'microsoft'
    microsoft.agents.context.AgentContext              FAIL ModuleNotFoundError: No module named 'microsoft'
  → 结论：经典 AutoGen『ConversableAgent + GroupChat 互聊』API 在 AG2 1.0 整代消失
    （连 `import autogen` 兼容垫片也移除）——改名烈度=07 章继 06 Semantic-Kernel 的又一里碑
========================================================================
[账 D] 选择树账：多 Agent 范式该在哪儿用（§7.14 多 Agent 协作 → CrewAI/AutoGen/Agents SDK）
========================================================================
  1. 多 Agent 研究/原型、要『互聊』范式
     → AutoGen / AG2（group chat 起家；生产要自管状态）
  2. 生产要状态/检查点/恢复
     → LangGraph（02 章 checkpoint，recursion_limit 真兜）
  3. 子任务并发/委托 + 显式 checkpoint
     → AG2 v1.0 Task.run_task（checkpoint 一等公民）
  4. 严谨生产级 RAG 管线
     → Haystack（05 章 pipeline 思维）
  5. RAG 专精：数据→索引→检索→问答
     → LlamaIndex（03 章七工序七组件）
  6. 企业 .NET/微软栈
     → Semantic Kernel（06 章三原语 + as_mcp_server）
  7. 低代码/非工程师快速落地
     → Dify / Coze（04 章积木平台）
  8. 轻量 Agent 运行时、回归轻量
     → OpenAI Agents SDK（§7.11 handoff）
  9. 微软 2025 Agent 路线（要素事实）
     → Microsoft Agent Framework（承接 AutoGen，与 SK 统一）
  断言[多 Agent 研究原型默认入口] = True
  断言[生产状态管理归图引擎] = True
  断言[子任务委托归新一代 Task] = True
  断言[生产 RAG 归 Haystack] = True
  断言[RAG 专精归 LlamaIndex] = True
  断言[企业栈归 SK] = True
  断言[非工程师归低代码] = True
  断言[轻量归 Agents SDK] = True
  断言[MAF=承接 AutoGen 与 SK 统一] = True
  → 选择树 9 场景断言 9/9 = True

========================================================================
台账汇总：A 新一代三原语 / B 多轮 ask 回灌 / C 版本存续 0.2→1.0 整代换血 / D 选择树 9 场景
一话总结：AutoGen 以『Agent 之间互聊 + GroupChat』开创多 Agent 对话范式（§7.8），
AG2 1.0 把范式重写成 函数式 Agent + run/ask 回合 + 子任务委托（经典 API 整代换血），
微软 2025 以 Microsoft Agent Framework 承接 AutoGen 并与 Semantic Kernel 统一=官方延续线；
"研究/原型用 AG2 的对话范式、生产自己补状态管理"仍是 §7.8 的适用结论。
done · 一键复现：python code/notebooks/_tools/autogen_demo.py
```

</details>

## 3. 实验 A · 新一代三原语账（Agent.run → AgentReply：函数式收窄）

**目标**：验证 §7.8 那个「整代换血」到底换在哪一层——不是换皮，是把**「对话」这个语法都换掉了**。老 API 是"Agent 进聊天室、互发消息"（`ConversableAgent`+`GroupChat`），新 API 是"每个 Agent 一次 run 流水：prompt → LLMCall → AgentReply"。

**过程**：在真实 1.0.6 引擎上创建 `Agent(name="assistant", prompt="固定系统提示", config=StubConfig("答案：42"))`，用 `async with a.run("第1问：2+2=？") as run: rep = await run.result()` 拿一次回复；再故意不配模型驱动一次，抓引擎真抛的护栏。

**实测**（stdout 关键行）：

```
  agent.name        = assistant
  reply.body        = '答案：42'
  reply 类型        = ag2.agent.AgentReply
  无模型配置护栏（引擎真抛）: 
    ConfigNotProvidedError | No model config provided. Set config on the `Agent(config=...)` creation or pass it to call `ask(config=...)`.
```

**结论**：

1. **三个原语＝范式转弯**：`Agent`（身份）→ `run()`（一次交互）→ `AgentReply`（结果）。回复对象叫 **`AgentReply` 而不是"消息"**——第一代把 agent 之间交流建模成"消息池（chat messages）"，这一代建模成"一次 run 的返回"。§7.8 的"整代换血"最硬证据在账 C（连 import 都 FAIL），但语法面先看这里：**没有 GroupChat、没有"进聊天室"、没有 UserProxy**——只有"一个 Agent 被 run 一次"。
2. **prompt 可以不是字符串**：`Agent` 的 prompt 参数在 1.x 支持 callable（每次 run 引擎真调它）。探针里验证了"callable prompt 真被求值"——这意味着 prompt 不再是一次性的模板，可以是**按轮/按上下文程序化生成的注入点**。这是新一代"函数式"的又一证据。
3. **无模型配置＝当场真抛**：`Agent(name="nog", prompt="你好")` 驱动到 `result()` 时引擎抛 `ConfigNotProvidedError | No model config provided. Set config on the Agent(config=...) creation or pass it to call ask(config=...).`（stub 里没传 config，连 ask 的兜底路径也封住）。**这是一类「创建/运行即栅栏」护栏**——对比 02 章 `recursion_limit`（跑飞了才兜）／05 章 `max_runs_per_component`（循环超限才兜）／06 章 SK 的 `FunctionChoiceBehavior`（连接器内配置项），**AG2 的护栏在"没有模型你根本开不了跑"这一层**：配置是 Agent 原语的强制契约，不是可选项。

## 4. 实验 B · 多轮会话账（ask() 续谈 · 历史回灌）

**目标**：搞清楚新一代里「对话」到底是不是你熟悉的那件事。老范式里"对话"=聊天室里的消息在增长；新范式里只有 run/ask——那**上一轮的内容还回不回得去？**用 stub 里"数当前消息里有多少条历史 Assistant 回复"当场量。

**过程**：同一个 Agent 上 `run("第0问") → reply.ask("第1问") → reply2.ask("第2问")` 连续三回合；每回合 stub 统计引擎传进模型的 messages 里 `ModelResponse`（历史 assistant 事件）有几条。

**实测**（stdout 关键行）：

```
  第0轮 reply.body  = '第1轮：模型可见前 0 条历史回复'
  第1轮 ask.body   = '第2轮：模型可见前 1 条历史回复'
  第2轮 ask.body   = '第3轮：模型可见前 2 条历史回复'
```

**结论**：

1. **`ask()` 是真·续谈原语**：`AgentReply.ask(下一句)` 不是把新消息塞进某个全局聊天室，而是**在同一 Agent 的上下文中继续**——第 2 轮的模型调用里真的带着前 2 条历史回复（stub 逐条数过）。历史以**事件流（ModelResponse 事件）**的形式回灌给模型，而不是"聊天缓冲"。这就是新一代"对话"的语义：**对话 = 同一 Agent 上连续 run/ask，历史=模型侧可见的事件序列**。
2. **这与 02/06 章的信息载体是同一件事的另一张脸**：02 章 LangGraph 的 checkpoint=引擎层"活儿干到哪一步"、06 章记忆系统=内容层"记住用户是什么样"；**本篇账 B=事件层『历史回灌』**——AG2 把"对话历史"当成 LLM 调用的输入拼进去（`.ask()` 自动带），而 06 章是应用自己挑着喂。三者都是"把过去再喂回模型"的不同节制级别。
3. **侧面量出 §6.9 成本警句的机制**：第 n 轮模型看到的上下文是前 n 条历史之和——**多 Agent / 长对话的 token 成本是逐轮累积重发的**。§6.9 说"多 Agent = 多倍 token 与延迟"，账 B 给了它一个可看到的最小例子（每轮 messages 都在变长）。生产长对话想省 token→06 章记忆回灌那套"挑着喂"才有意义。

## 5. 实验 C · 版本存续（0.2『互聊』API 整代换血 · 改名烈度）

**目标**：给 AutoGen→AG2 的 1.0 重写做一张**生存表**——老教程里的名字现在到底死没死、死得有多干净，用**真实 import 探测**而不是文档印象。

**过程**：21 个目标名分四块逐一 `importlib` 探测：pyautogen 0.2 经典命名空间 8 名（含顶层 `autogen`／`agentchat`／`groupchat`／`oai` 全套）、AG2 0.9-era 的 `ag2.agentchat` 3 名（0.9 时代 AutoGen 改名 AG2 时据说"同构迁移"过）、AG2 1.0.6 新命名空间 8 名、Microsoft Agent Framework 2 名（未装）。

**实测**（stdout 关键行）：

```
    autogen                                            FAIL ModuleNotFoundError: No module named 'autogen'
    autogen.agentchat.conversable_agent                FAIL ModuleNotFoundError: No module named 'autogen'
    ag2.agentchat.conversable_agent                    FAIL ModuleNotFoundError: No module named 'ag2.agentchat'
    ag2                                                OK <module>
    ag2.agent.Agent                                    OK <Agent>
    ag2.agent.AgentReply                               OK <AgentReply>
    ag2.task.Task                                      OK <Task>
    microsoft.agents                                   FAIL ModuleNotFoundError: No module named 'microsoft'
```

**结论**：

1. **经典『互聊』API 整包消失（8/8 FAIL）**：`autogen` 顶层、`ConversableAgent`、`AssistantAgent`、`UserProxyAgent`、`GroupChat`、`oai.Completion`、`code_utils`、`contrib`——连 `import autogen` 兼容垫片都被移除。**这是 07 章继 06-Semantic-Kernel 之后第二家改名烈度满分的主线库**（06 是 0.x→1.x 概念词全换、Kernel 存活；本篇是 AutoGen→AG2 换代、经典 API 整包消失、连垫片都不给）。
2. **0.9-era 的同构迁移层也没了（3/3 FAIL）**：AutoGen 改名 AG2 的 0.x 阶段曾有 `ag2.agentchat.*` 路径（对 0.2 的兼容迁移）；**1.0 连这个兼容层一起移除**。老教程在两层都可能踩空（`autogen.agentchat…` 死于换名、`ag2.agentchat…` 死于换代）。
3. **新命名空间整活（8/8 OK）**：`ag2`、`Agent`、`AgentReply`、`AgentRun`、`task.Task`、`config.client.ModelResponse`、`tools.subagents`、`events.types.Usage`——函数式世界的家当都在。特别记录 `Task`（带 checkpoint 的声明式任务）与 `subagents`（子任务），对应账 D 第 3 行"子任务并发/委托 + 显式 checkpoint"。
4. **MAF 未装（2/2 FAIL，非本机实测）**：`microsoft.agents` 本机没有——**只证明"本机没装"**，不证明微软没发布（要素事实：2025 微软推出、承接 AutoGen 与 SK 统一）。诚实边界：这一格是"探测本机环境"，不是"裁决 MAF 存不存在"。
5. **对老教程的判决**：任何写 `ConversableAgent`／`GroupChat`／`autogen.agentchat` 的教程到今天都**直接 FAIL**——多 Agent 教程的保质期被这次重写切了一刀。撞上时说"这是 AutoGen 的旧 API，AG2 1.0 换血了"（去查官方迁移指南），别照抄。

## 6. 实验 D · 选择树账（9 场景断言 9/9）

**目标**：把「多 Agent 范式该在哪儿用」落成一张断言表（§7.14 多 Agent 协作 → CrewAI/AutoGen/Agents SDK + §7.8 适用边界 + 前六篇各自的落点，要素事实）。探针里 9 场景 9 断言逐条为真。

**实测**（stdout 关键行）：

```
  1. 多 Agent 研究/原型、要『互聊』范式
     → AutoGen / AG2（group chat 起家；生产要自管状态）
  2. 生产要状态/检查点/恢复
     → LangGraph（02 章 checkpoint，recursion_limit 真兜）
  3. 子任务并发/委托 + 显式 checkpoint
     → AG2 v1.0 Task.run_task（checkpoint 一等公民）
  4. 严谨生产级 RAG 管线
     → Haystack（05 章 pipeline 思维）
  ...
  9. 微软 2025 Agent 路线（要素事实）
     → Microsoft Agent Framework（承接 AutoGen，与 SK 统一）
  → 选择树 9 场景断言 9/9 = True
```

**结论**：

1. **研究/原型、要『互聊』→ AutoGen / AG2**（§7.8 适用边界原话）：这就是本篇主场。**注意 ㊟叠加 §6.9**：先证明单 Agent 不行再上多 Agent——账 B 的逐轮变长就是那张"多倍 token"票。
2. **生产要状态/检查点/恢复 → LangGraph 而不是 AG2**：§7.8 白纸黑字"生产需自己加状态管理"；02-章用真实引擎证明过 checkpoint 断点续跑/Tarjan 环检测/recursion_limit。**AG2 的 v1.0 `Task` 把 checkpoint 做成一等参数（`checkpoint_resume`）**——是"框架开始自带状态"，但要省心仍落 02。
3. **子任务并发/委托 → AG2 v1.0 `Task`**：新一代比老一代更接近"编排"（`subagents.run_task` 父子任务），这里给它单独一格，与 LangGraph 的状态图一格不冲突（粒度不同：一个是委托+checkpoint，一个是完整状态图/图检索）。
4. **其余七格=前六篇各取其位**：RAG 管线→05、RAG 专精→03、.NET→06、低代码→04、轻量→10；**MAF=微软 2025 官方线（要素事实）**——它承接 AutoGen、与 SK 统一，是这条路的"官方下一代"坐标。

## 7. 拿这张表怎么读本目录（07 章 13 篇的路牌）

本页是 07 章"第八站"：00 篇回答了"框架分几类"，01 篇"通用编排家长子"，02 篇"图引擎一格"，03 篇"RAG 专精一格"，04 篇"低代码一格"，05 篇"生产级 RAG 管线一格"，06 篇"企业 .NET 栈一格"，**本页回答"多 Agent 对话范式一格"**。后文各篇的路牌（按依赖次序）：

| 后文篇 | 读它的理由（对应本页哪一格） | 状态 |
|---|---|---|
| `00-框架分类学` | 账 A 类 4 六家最挤、账 B 状态主业：多 Agent 对话范式=把『状态』放聊天里 | 🔥 已交付（v0.27） |
| `01-LangChain与历史包袱` | 账 C 版本存续的对照母题：AutoGen→AG2 连兼容垫片都删=07 章第二家改名满分 | 🔥 已交付（v0.28） |
| `02-LangGraph-状态图与Checkpoint` | 账 A 护栏对偶（recursion_limit 轮数上限）＋账 D 第 2 行：生产状态/恢复的归处 | 🔥 已交付（里程碑 012） |
| `03-LlamaIndex` | 账 D 决策表另一臂：RAG 数据管线专精住进 03 | 🔥 已交付（v0.29） |
| `04-Dify-Flowise-Coze-n8n低代码` | 账 D 决策表另一臂：非工程师落 04 | 🔥 已交付（v0.30） |
| `05-Haystack` | 账 A 护栏强化（max_runs 真兜底）＋账 D 场景④：生产级 RAG 管线归它 | 🔥 已交付（v0.31） |
| `06-Semantic-Kernel` | §7.8 相邻：SK 是微软企业栈"下一步与 AutoGen 统一"的另外半边 | 🔥 已交付（v0.32） |
| `07-AutoGen·AG2·MAF`（本篇） | 多 Agent 对话范式：经典『互聊』API → 1.0 函数式重写 → MAF 官方线 | 🔥 已交付（v0.33） |
| `08-CrewAI` | 账 A 类 4/账 B：多 Agent 角色化分工（角色·任务·流程三一等公民，真实 crewai 1.15.22 引擎四账） | 🔥 已交付（v0.34） |
| `09-DSPy` | 账 A 类 5：质量敏感评测（提示图谱·评测主业互补角色化/对话） | 🔥 已交付（v0.36） |
| `10-OpenAI-Agents-SDK` | 账 D 场景⑧：轻量运行时「回归轻量」先例（handoff）——对话/角色化/轻量转移三范式收束 | 🔥 已交付（v0.35） |
| `11-MCP协议` | 账 C "2025 公共底座"、里程碑 `013`；Agent 系的工具出口 | 🔥 已交付（v0.37） |
| `12-继承关系与选型决策` | 收束章：完整 DAG + "该不该引入"量化，00 篇 C 账扩版 | 🔥 已交付（v0.38 收束章） |

读法口诀（本页的一页带走）：**遇到"我想让几个 agent 聊起来"先落 AutoGen/AG2——它是『互聊』范式的开创者（账 C 会提醒你：老教程的 `ConversableAgent/GroupChat` 今天全 FAIL，别照抄 2023 教程）；但记死 §6.9 的三刀：**先证明单 Agent 不行、多 Agent 的 token/延迟成本逐轮累积（账 B）、生产状态自己接 02-章图引擎（账 D 第 2 行）**——研究/原型用 AG2 的对话范式、生产自己补状态管理，仍是 §7.8 的适用结论；微软侧要看 MAF（2025，官方延续线）。**

## 8. 常见坑（5 个）

1. **照抄 2023–2024 的 AutoGen 教程**（本课第一个坑）：账 C 真 import 探测全部 FAIL——`ConversableAgent`/`UserProxyAgent`/`GroupChat`/`autogen.agentchat.*` 在 AG2 1.0.6 全灭，连 `import autogen` 垫片都删了。写多 Agent 前**先确认你手里教程的 API 世代**；旧 demo 要么换 MAF 文档、要么按 v1.0 新 `Agent.run/ask` 重写。
2. **以为"对话"还是聊天室消息那回事**：账 A/B 证明新一代的对话=同一 `Agent` 上连续 `run/ask`，回复对象是 `AgentReply`（模型事件）而不是"message 追加进列表"。**想续谈用 `.ask()`**——裸跑两次 `Agent.run()`（复用同一 stream）不会自动带历史（账 B 对照实验确认：只有 `ask()` 回灌）。
3. **把 AG2 当生产状态管理方案**：§7.8 原话"生产需自己加状态管理"。AG2 v1.0 的 `Task(checkpoint_resume=…)` 开始把 checkpoint 做成一等参数（要素事实），但真要断点/恢复/图检索的复杂流程请落 02-章 LangGraph（账 D 第 2 行）。**AG2 的强项是"快速聊起来"，不是"替你把状态管明白"**。
4. **没配模型就裸 Agent 然后疑神疑鬼**：`ConfigNotProvidedError | No model config provided.` 是引擎的真护栏——它把"没模型配置本来就会跑飞"拦在起点（账 A）。看到它别当 bug，去 `Agent(config=…)` 或 `run(…, config=…)`/`ask(…, config=…)` 里把配置给上。
5. **把"老 API 死透"误读成"AG2 是新语言"**：账 C 的 FAIL 是**本机没装对应命名空间**（pyautogen 那 8 个是"这个环境根本没有 autogen 包"、`agentchat` 那 3 个是"装了 1.0.6 但 1.0 移除了 agentchat"）。换到装着 pyautogen 0.2 的旧环境是老 API 存活、新 API FAIL。**版本存续账量的是"这个环境里还活着谁"**，不是永真命题——这正是 01-章"版本节奏=要素事实、探测=本机真跑"的分工。

## 9. 诚实边界（AAA 自我审查）

- **引擎语义 = 本机真实实测（ag2 1.0.6）**：`Agent(name, prompt, config)` 创建、`async with a.run(…).result()` 驱动拿 `AgentReply`、prompt 传入 callable 且真求值、`.body` 内容、`ask()` 多轮历史回灌（stub 数出 0/1/2 条）、无模型配置 `ConfigNotProvidedError` 真抛原文、21 个 import 目标 OK/FAIL——全部在本机真实引擎上跑出（有真实报错原文为证）。本节不借鉴文档印象，凡可复现行为都对真实输出。
- **Agent 回复文本 = 非本机实测（stub 预置）**：探针里每条 Agent 回复（`答案：42`、`第N轮：可见前 k 条…`）都是作者预置的词面规则桩——这是保证零网络零随机、stdout 三进程 md5 恒一的前提。**请勿把 stub 的"回复内容"与真实 LLM 的对话质量混淆**：真实 LLM（GPT/DeepSeek/Qwen…）经 AG2 的 `LLMClient` 连接器跑出来的"多 Agent 协作效果"=非本机实测；账 B 只量"引擎把历史回灌给模型"这一件事（可见前 N 条），不量"模型聊得好不好"。
- **版本存续行 = 本机真 import 探测 + 版本节奏要素事实**：21 个目标名的 OK/FAIL 是真实 `importlib` 探测逐条真实（含 `No module named 'autogen'` 等原文）。但 **"AutoGen 2023 发布""AG2 是社区维护接力""微软 2025 推出 MAF 承接 AutoGen 与 SK 统一""AG2 0.9 曾有过 `ag2.agentchat`"这些年份/路线/模块归属 = 要素事实**，写作环境无外网、未在线复核官方文档，以官方 release/docs 为准。**账 C 的核心发现（1.0 整代移除经典 API、新命名空间 8/8 OK）是本机 FAIL/OK 撑住的；MAF 的 2/8 FAIL 只代表本机未装。**
- **账 D 选择树 = 作者按知识地图 §7.14 / §7.8 / §7.11 / §6.9 与前六篇落点整理（要素事实）**：9 场景决策规则为作者整理定性，非本机可运行断言；它与 00-章账 D 场景⑤、02-章选择树场景⑤ 的数据一致是硬纪律（跨章数字不打架）。
- **确定性纪律**：stdout md5 `b39ccb9b…` 三个独立进程逐字节恒一（每次 `python code/notebooks/_tools/autogen_demo.py` 复现）；墙钟 ≈0.1 s 仅进 stderr；不打印随机对象 ID/地址/哈希；输出排序保证哈希无关。

## 10. 参考与衔接

- **本页地图**：§7.8（AutoGen 2023 微软·互聊 paradigm·AG2 社区分支·MAF 2025 承接 AutoGen 与 SK 统一·生产需自己加状态管理）+ §7.1（类 4 多 Agent 框架六家最挤）+ §6.9（多 Agent=多倍 token 与延迟，先证明单 Agent 不行）+ §7.14（多 Agent 协作 → CrewAI/AutoGen/Agents SDK）；§7.11（Agents SDK handoff 轻量对照，接 10-篇）。
- **选择树 9 场景**（探针断言 9/9，要素事实）：① 研究/原型要互聊 → AutoGen/AG2；② 生产状态/检查点 → LangGraph；③ 子任务委托+checkpoint → AG2 v1.0 Task；④ 生产 RAG → Haystack；⑤ RAG 专精 → LlamaIndex；⑥ 企业 .NET → SK；⑦ 非工程师 → 低代码；⑧ 轻量回归 → Agents SDK；⑨ 微软 2025 路线 → MAF。
- **前承**：00-框架分类学（类 4 六家最挤、状态主业、选择树场景⑤）；02-LangGraph（`recursion_limit` 轮数护栏对偶 + checkpoint 归处）；05-Haystack（`max_runs` 上限第三例）；06-Semantic-Kernel（§7.8 相邻：SK 的"下一步"=本篇的 MAF 线；护栏三态对照：ConfigNotProvidedError vs FunctionChoiceBehavior vs recursion_limit）；06-应用开发 06-记忆系统（历史回灌/挑着喂 vs AG2 全量回灌）。
- **后启**：`08-CrewAI`（✅ 已交付 v0.34：同属类 4 的第二杆——角色·任务·流程三个一等公民 = 声明式团队流水线，与本篇『对话范式』左右手，真实 crewai 1.15.22 四账）；`10-OpenAI-Agents-SDK`（✅ 已交付 v0.35：轻量运行时/Agents SDK 的回归轻量对照——一次 run=一次执行的同族最简式，handoff 自动工具名 transfer_to_* 由引擎真转移、last_agent 切换 vs 本篇显式 ask() 喂历史；真实 openai-agents 0.17.0 四账，三范式收束）；`11-MCP协议`（✅ 已交付 v0.37：Agent 系的工具出口，里程碑 `013`）；`12-继承关系与选型决策`（✅ 已交付 v0.38 收束章：完整 DAG + "该不该引入"量化——账 A 入度台账本页 MAF 入度 2 全图唯一双父=微软统一线；账 B 角色描述概念普及 6/24）。
- **对应里程碑**：`012`（02-章交付）之后，多 Agent 对话范式这杆由本篇收；里程碑 `013`（手写 MCP Server）已在 `11-MCP协议` 章交付（v0.37），Agent 系的工具出口就位。

> 本篇完工于 2026-09-22（v0.33 批次）；探针 `code/notebooks/_tools/autogen_demo.py`；stdout md5 `b39ccb9b…`。
