# 🏷️ LangChain 与历史包袱：Concept 永存、Library 换代（Chain/Runnable · 套娃 · 换血 · 单向链）

> 对应知识点：知识地图 §7.2（LangChain，2022，**概念普及者**——"第一个把 Chain 概念普及，生态最大、模板最多；问题：抽象多而杂、版本变动剧烈、过度封装、性能与调试成本高；2024-2025 转型：重心转向 LangGraph 与 LangSmith，LangChain 退化为'库集合'；学习建议：用它入门理解 Chain 概念即可，生产逐步看 LangGraph 或直写"）、§17.5.8（**LangChain 与历史包袱**——"最早普及 LLM 编排概念的库，现以生态组件价值存在"）。
> 前置：[07-应用框架 00-框架分类学](./00-框架分类学.md)（开篇章：账 A 类 1「通用编排 SDK」/ 账 C 谱系**最长承袭链起点** LangChain→LangGraph→LangGraph Platform）；[06-应用开发 全 6 篇](../06-应用开发/01-Model-API与流式.md)（直写原语）；本文的**对照臂**是 [02-LangGraph-状态图与Checkpoint](./02-LangGraph-状态图与Checkpoint.md)——"链 vs 图的分界"，同是 LangChain 系两兄弟。
> 动手：`python code/notebooks/_tools/langchain_demo.py`（**真实 langchain 1.2.10 / langchain-core 1.4.9 引擎**四账 + 选择树一键复现；零网络 · Fake 模型确定性，stdout md5 `900181e7…` 三独立进程恒一）。
> 一句话：**LangChain 贡献的是「Chain/Runnable」这个概念（概念永存）；LangChain 库本身背的是「套娃 / 换血 / 单向链」的包袱（库会消亡）——用概念入门，生产看 LangGraph 或直写（§7.2 学习建议）。**
>
> 📌 诚实边界：本篇 **A–D 四账全部来自本机真实 langchain 1.2.10 / langchain-core 1.4.9 引擎**（importlib.metadata 实测版本；LCEL/Runnable/get_graph/MRO/pydantic 字段/import 生存表都是引擎真实运行结果，**不是演示纸面**）。**模型回复 = FakeListLLM 预置**（LLM 非本机实测，换取零网络 + 逐字节可复现）；**版本节奏/年份 = 要素事实**（写作环境无外网，未在线复核各版本发布日期）；**重试判定 = 词面规则**（`startswith('订单状态：已发货')`，真实工具循环语义挂靠 05-章）；**图式对照数字 = 挂靠 02-章真实引擎实测**（不重测）。stdout md5 `900181e7…` 三个独立进程逐字节恒一；墙钟 ≈15.7 s 仅进 stderr；`get_graph().nodes` 的节点 ID 是随机哈希（跨进程不稳定），探针只打**计数/定性**、绝不打 ID。

---

## 📑 本章目录

1. [为什么 LangChain 值得单独讲一课？](#0-为什么-langchain-值得单独讲一课)
2. [先钉死事实：LangChain 贡献了什么、背了什么（§7.2 / §17.5.8）](#1-先钉死事实langchain-贡献了什么背了什么7271758)
3. [把「概念与包袱」降维成本机测量（探针设计）](#2-把概念与包袱降维成本机测量探针设计)
4. [实验 A · 链式组装账（概念永存面：Chain 是什么）](#3-实验-a--链式组装账概念永存面chain-是什么)
5. [实验 B · 封装厚度账（包袱面：套娃的数字）](#4-实验-b--封装厚度账包袱面套娃的数字)
6. [实验 C · 版本存续账（包袱面 II + 概念永存：0.x→1.x 换血）](#5-实验-c--版本存续账包袱面-ii--概念永存0x1x-换血)
7. [实验 D · 链 vs 图分界账（该在哪用：客服重试的两种范式）](#6-实验-d--链-vs-图分界账该在哪用客服重试的两种范式)
8. [拿这张表怎么读本目录（07 章 13 篇的路牌）](#7-拿这张表怎么读本目录07-章-13-篇的路牌)
9. [常见坑（6 个）](#8-常见坑6-个)
10. [诚实边界（AAA 自我审查）](#9-诚实边界aaa-自我审查)
11. [参考与衔接](#10-参考与衔接)

---

## 0. 为什么 LangChain 值得单独讲一课？

07 章的路牌上，00-框架分类学把它分在**账 A 类 1「通用编排 SDK」**，谱系账里它是**账 C 最长承袭链的起点**（LangChain→LangGraph→LangGraph Platform）。这两个坐标都指向同一个问题：**一个被全网骂"套娃"、2025 年退居"库集合"的库，为什么还必须学？**

答案在标题里：**要分清楚"LangChain 这个概念"和"LangChain 这个库"是两个生命周期。**

- **概念**（Chain / Runnable = 把模型+工具+检索统一成"可调用的一段"）**永存**——现在每个框架（连原生 SDK）都在讲 runnable / pipeline / chain，你只是在别处又见到它一遍。
- **库**（`langchain` 这个 Python 包）**会消亡**——0.x 时代你背诵的 `LLMChain`、`ConversationBufferMemory`、`langchain.chains` 整包，到了本文实测的 1.2.10 上一个都不剩（账 C 生存表 7 条全 FAIL，这是真实 import 的结果）。

所以本文不是教你某版 API——API 反正下周就换；本文教你**怎么一眼认出"概念"和"包袱"**：概念让你在任何框架里都秒懂它的跟脚，包袱让你知道生产环境为什么谨慎引入它。四本账各负责一刀：

| 账 | 切的是什么 | 落点 |
|---|---|---|
| A 链式组装账 | **概念永存面**：Chain/Runnable 到底解决了什么 | steps 3 步 · 同链三触发 invoke/batch/stream |
| B 封装厚度账 | **包袱面 I**：为什么社区骂"套娃" | MRO 13 层 · 字段 56 · 隐形节点 2 |
| C 版本存续账 | **包袱面 II + 概念永存**：0.x→1.x 换血 | 生存表 7 FAIL / 3 OK · 库集合现状 |
| D 链 vs 图分界账 | **该在哪用**：什么时候还轮得到链 | 自环 0 · 重试手写 · 挂靠 02-章图式 |

## 1. 先钉死事实：LangChain 贡献了什么、背了什么（§7.2 / §17.5.8）

§7.2 用四个字概括 LangChain 的历史地位：**概念普及者**。它做对的第一件事是"编排层的统一抽象"——在它之前，接 OpenAI 一个口头 API、接检索用一个风格、接工具又一家 SDK，每换一个就要重写一遍参数风格；LangChain 的贡献是**把"模型、模板、解析、检索、工具"统一成同一个 `Runnable` 协议**：`|` 一拼，`invoke/batch/stream` 三种节奏全通。这个思想今天在 LlamaIndex、Haystack、原生 SDK 里全量复现——**概念确实永存**。

同时 §7.2 也一行写完它的包袱：**抽象多而杂、版本变动剧烈、过度封装**（社区戏称"套娃"）、**性能与调试成本高**。本文的全部实验就是给这三条"骂名"找**一手数字**：

- "抽象多而杂" → 账 B：一个 `ChatOpenAI` 继承 13 层 MRO、背 56 个 pydantic 字段、离 `Runnable` 隔着 10 层；
- "版本变动剧烈" → 账 C：0.x 时代七个代表性导入路径在 1.2.10 下**全部** `ModuleNotFoundError`，而核心概念在 `langchain_core` 里连续存活；
- "过度封装/套娃" → 账 A/B：一条 3 步链编译成内部 5 节点图，"少写 2 行"换的是"隐埋 2 个中间节点、跨 13 层 MRO 找状态"；
- "搬去哪" → 账 D：同样的"带记忆重试客服"，链式要手写 9 行重试补丁且无状态恢复，图式（02-章）由引擎自带循环/上限/checkpoint——**这就是 §7.2 那句"生产逐步看 LangGraph 或直写"的实证**。

## 2. 把「概念与包袱」降维成本机测量（探针设计）

一张"LangChain 好/坏"的口水文章不稀奇；稀奇的是把"概念 vs 包袱"变成**可复排的本机实验**。探针用**真实 langchain 1.2.10 / langchain-core 1.4.9 引擎**做了 4 件确定性的测量（模型全部用 `FakeListLLM` 预置回复→零网络、零随机、逐字节可复现）：

- **A 链式组装账**：`PromptTemplate | 模型 | StrOutputParser` 真实组装，数 `chain.steps`（概念面）/ `get_graph()`（实现面）/ `invoke·batch·stream` 同链三触发输出与预置逐字节一致；
- **B 封装厚度账**：真实 `ChatOpenAI.__mro__` 继承深度、`model_fields` 字段数、`FakeListLLM` 的 MRO、以及"3 步链→5 节点内部图"的隐形节点差；
- **C 版本存续账**：同机 6 包版本（importlib.metadata）、`langchain` 包本体现有子模块清单、十条 0.x 路径 iimport 生存表（真实 `importlib.import_module` 探测）；
- **D 链 vs 图分界账**：链内部图自环数、一次"客服答错重试"需要的外层手写补丁行数与轮数，对照 02-章图式数字。

确定性纪律：`warnings.filterwarnings("ignore")` 最先执行（拦 requests/pydantic 告警，**stderr 纯净**，不污染 md5；这是 02-章"导入期警告优先级"教训的延续）；`get_graph()` 节点 ID 随机→只打计数；全排序保证集合输出稳定。**stdout md5 `900181e7…` 三个独立进程逐字节恒一**，墙钟 ≈15.7 s（langchain 全家导入的固有开销）仅进 stderr。

复现实录（严格逐字节 = 探针 stdout 三遍恒一）：

<details>
<summary>📊 探针 stdout 全量实录（md5 <code>900181e7…</code> · 墙钟仅 stderr）</summary>

````text
========================================================================
langchain_demo：07-应用框架 · 01-LangChain与历史包袱（知识地图 §7.2 / §17.5.8）
    『Chain 概念永存 · LangChain 库有历史包袱』——统一抽象 vs 套娃/换血/单向链
========================================================================
[0] 口径：真实 langchain 1.2.10 / langchain-core 1.4.9 引擎（本机实测）；模型=FakeListLLM 预置（LLM 非本机实测）；版本节奏=要素事实（无外网未在线复核）；不打印随机图节点 ID
========================================================================
[账 A] 链式组装账：Runnable 统一抽象的最小闭环（概念永存面——Chain 是什么）
========================================================================
  steps 类型序列（确定性）：['PromptTemplate', 'FakeListLLM', 'StrOutputParser']
    -> 概念面 steps = 3 步 · 2 条 `|` 边 = 组装 2 行
    -> 实现面 get_graph 内部节点 = 5（多出 2 个隐形节点=封装的入场费，账 B 展开）
    -> 链 = langchain_core.runnables.RunnableSequence（instanceof 校验 True）——Runnable 协议统一了每一步
  → 同一条链 · 三种触发（Fake 预置回复 → 逐字节恒一）：
      invoke : 订单状态：已发货（编号 7281）
      batch  : ['回复一', '回复二', '回复三']   <- batch(3 条) 依次消费 3 个预置回复
      stream : ['流', '式', '回', '复', '甲']   <- 逐块流出（FakeStreaming 逐字）
  → 概念：invoke/batch/stream = 同一个 Runnable 协议的三种触发，换触发不需改代码——『`|` 组装一次、三种节奏都通』
  → 断言：steps==3 · 内部节点==5 边==4 · invoke/batch/stream 全与预置一致

========================================================================
[账 B] 封装厚度账：继承深 · 字段多 · 隐形节点（包袱面——『套娃』的数字）
========================================================================
  → ChatOpenAI MRO 继承链 深度 = 13 层 —— 离 Runnable 还隔 9 层（Runnable 在第 10 层）
      链：ChatOpenAI → BaseChatOpenAI → BaseChatModel → BaseLanguageModel → RunnableSerializable → Serializable → … → Runnable
  → ChatOpenAI pydantic 字段 = 56（一个『模型客户端』要背 56 个配置项）
  → FakeListLLM MRO 深度 = 13 · Runnable 位次 10
  → 隐形节点：3 步链内部图 = 5 节点（对账 A）——概念 3 步 vs 实现面多 2 个壳；『封装』=用『少写 2 行』换『多爬 2 层』
  → 结论：抽象深 ≠ 功能重；重封装的真正代价 = 调试要跨 13 层 MRO、56 字段给你无限 `**kwargs` 的选项面——『抽象多而杂』（§7.2）在此有数字
  → 断言：ChatOpenAI MRO=13 · Runnable 位次=10 · 字段=56

========================================================================
[账 C] 版本存续账：0.x→1.x 换血清单（包袱面 II + 概念永存）
========================================================================
  同机共存（importlib.metadata 实测）：
    langchain 1.2.10 · langchain-core 1.4.9 · langchain-openai 1.3.4
    langchain-community 0.4.1 · langgraph 1.0.10 · langsmith 0.7.9
  langchain 包本体：顶层命名空间 0 个公开符号（空）· 子模块只剩 6 个：agents · chat_models · embeddings · messages · rate_limiters · tools
    -> 旧『全家桶』（chains/memory/llms/schema/vectorstores/document_loaders…）整包拆走=『库集合』
  → 0.x 时代路径生存表（真实 import 探测）：
      | LLMChain               | langchain.chains.llm           | FAIL
      | ConversationBufferMemory | langchain.memory               | FAIL
      | OpenAI                 | langchain.llms                 | FAIL
      | BaseMessage            | langchain.schema               | FAIL
      | PromptTemplate         | langchain.prompts              | FAIL
      | TextLoader             | langchain.document_loaders.text | FAIL
      | FAISS                  | langchain.vectorstores         | FAIL
      | RunnableSequence       | langchain_core.runnables.base  | OK
      | PromptTemplate         | langchain_core.prompts         | OK
      | tool                   | langchain_core.tools           | OK
    -> 生存表：旧壳 7 条全 FAIL（ModuleNotFoundError）· 核心概念 3 条 OK——Runnable/PromptTemplate/@tool 在 langchain-core 里 0.x→1.x 连续存活
  → 版本节奏（要素事实=非本机实测，无外网未在线复核）：2023 v0.1（langchain-core 独立 · LCEL 定型）→ 2024 v0.2/v0.3（工具调用/大清理）→ 2025 v1.0（移除 legacy chain API）
  → 断言：生存表 FAIL=7 / OK=3 · 库集合子模块=6

========================================================================
[账 D] 链 vs 图分界账：一次『客服重试』的两种范式（该在哪用）
========================================================================
  → 链（LCEL）= 单向数据流：RunnableSequence 内部图 自环数 = 0（能分支不能自循环——这是 DAG，不是会恢复的图）
  → 想要『答错重试』→ 唯一选择 = 链外层自写 while 包装：9 行手写补丁（_retry_loop）
      实测轨迹：第 1/2 轮输出『查询失败』→ 第 3 轮『已发货…』停（Fake 预置 2 连不合格 + 1 合规）
  → 对照·图式（LangGraph，02-章真实引擎已实测）：自环=一条条件边 · 轮上限=引擎 recursion_limit 兜底 · checkpoint=中断恢复补 2 步不重跑（7 节点 6 支路 1 自环——挂靠 02 章数字）
  → 结论：链省的是『组装』（2 行 `|`）；图省的是『状态/循环/恢复』那一整类工程。线性/教学/脚本 → 链（概念入门）；状态/恢复/重试（生产 Agent）→ 图；LangChain 本体生产谨慎引入
  → 断言：链自环=0 · 重试轮数=3（Fake+词面判定，LLM 非本机实测）· 补丁行=9

  → 选择树 8 场景断言 8/8：链=概念入门工具（概念永存）；生产=图/原生（链库退居库集合）
      1. 教学/入门·只想懂 Chain 概念
         -> 直接读 langchain-core 的 Runnable 文档 + LCEL（概念），不装重量依赖
      2. 线性脚本：模板→调一次→拿字符串
         -> PromptTemplate | 模型 | StrOutputParser——2 行 `|` 就够（账 A）
      3. 批量/并发跑同一链
         -> Runnable.batch()——一个协议换触发，不重写代码（账 A）
      4. 需要状态/恢复/重试的客服
         -> 换 LangGraph 图引擎（优先于 chain）：checkpoint+recursion_limit（挂靠 02-章）
      5. 已经有 0.x 代码要迁移
         -> 先过生存表（账 C）：chains/memory/llms 已整包移除→迁 langchain-core 的 Runnable 组件
      6. 只想要某个工具/能力
         -> 取用 langchain-community / langchain 的『库集合』小件（账 C）而非整套框架
      7. 生产可观测链路
         -> LangSmith（同生态）或自接 Langfuse（账 D 之外，05-章后的横切章）
      8. 嫌封装太厚的团队
         -> 直写原生 SDK（06-章原语）或上轻量运行时（账 B 的 13 层 MRO 就是劝退理由）

========================================================================
台账汇总（A 链式组装 / B 封装厚度 / C 版本存续 / D 链 vs 图）
  A steps 3 步 · 内部图 5 节点（隐形 2）· invoke/batch/stream 同链三触发逐字节一致
  B ChatOpenAI MRO 13 层 · Runnable 位次 10 · pydantic 字段 56 · 封装=少写 2 行换多爬 2 层
  C 0.x 生存表 7 FAIL / 3 OK · langchain 本体 6 子模块库集合 · 同机 6 包版本实测
  D 链自环 0（重试手写 while·轮数 3）· 图式挂靠 02-章（自环 1·recursion_limit·checkpoint）
一句话：LangChain 贡献的是『Chain/Runnable』这个概念（概念永存）；LangChain 库本身背的是
套娃/换血/单向链的包袱（库会消亡）——用概念入门，生产看 LangGraph 或直写（§7.2 学习建议）
done · 一键复现：python code/notebooks/_tools/langchain_demo.py
````

</details>

## 3. 实验 A · 链式组装账（概念永存面：Chain 是什么）

**目标**：把"Chain 概念"降维成三条可复排的事实——① 它由几步拼成；② 它编译成几节点内部图（概念面 vs 实现面）；③ 同一条链换触发（invoke/batch/stream）代码零改动而输出逐字节一致。

**过程**：真实 langchain 1.2.10 里用三个 `|` 组装 `PromptTemplate | FakeListLLM | StrOutputParser`（订单客服问答，模型=Fake 预置回复）；读 `chain.steps` 拿概念面步骤类型；读 `chain.get_graph()` 拿实现面节点/边数；分别调 `invoke`（单问）、`batch`（3 问）、`stream`（流式）比对输出。

**实测**：

```
  steps 类型序列（确定性）：['PromptTemplate', 'FakeListLLM', 'StrOutputParser']
    -> 概念面 steps = 3 步 · 2 条 `|` 边 = 组装 2 行
    -> 实现面 get_graph 内部节点 = 5（多出 2 个隐形节点=封装的入场费，账 B 展开）
      invoke : 订单状态：已发货（编号 7281）
      batch  : ['回复一', '回复二', '回复三']
      stream : ['流', '式', '回', '复', '甲']
```

**结论**：

1. **"组装 2 行"就是 Chain 概念的整个卖点**：`PromptTemplate | 模型 | StrOutputParser`——模型、模板、解析器被统一成同一个 `Runnable` 协议，`|` 一拼即用。这个"把每一种角色都变成同一种可调用对象"的想法，就是 §7.2 历史贡献盖章的**概念普及**，今天的原生 SDK / LlamaIndex / LangGraph 全是它的后代（账 C 谱系）。
2. **同链三触发、输出逐字节一致**是"统一抽象"的实证：换触发方式不用改任何代码，`invoke→batch→stream` 只是同一个 Runnable 协议的不同入口。概念面"一次组装、三种节奏"；**这是学 LangChain 真正要学走的**——你在任何框架里认出 `invoke/batch/stream`，就认出了它。
3. **概念面 3 步 = 实现面 5 节点**（多出 2 个隐形节点）：框架偷偷插了模板预处理/后处理的袋子节点，"少写 2 行"换的是"隐埋 2 层"。这个差就是"封装"的计价单位——账 B 把它放大到继承链上。

## 4. 实验 B · 封装厚度账（包袱面：套娃的数字）

**目标**：把 §7.2 骂名之一"抽象多而杂 / 过度封装（套娃）"量成数字——一个真实模型客户端到底背多深的继承链、多少个字段，一条 3 步链实际埋几层。

**过程**：读真实 `ChatOpenAI.__mro__`（继承链）与 `model_fields`（pydantic 字段数），测 `Runnable` 在链中的位次；同测 `FakeListLLM` 的 MRO；并量化账 A 的"隐形节点"。

**实测**：

```
  → ChatOpenAI MRO 继承链 深度 = 13 层 —— 离 Runnable 还隔 9 层（Runnable 在第 10 层）
  → ChatOpenAI pydantic 字段 = 56（一个『模型客户端』要背 56 个配置项）
  → FakeListLLM MRO 深度 = 13 · Runnable 位次 10
  → 隐形节点：3 步链内部图 = 5 节点 —— 概念 3 步 vs 实现面多 2 个壳
```

**结论**：

1. **"套娃"是真实存在的层数问题**：`ChatOpenAI` 表面一个 `.invoke()`，底下是 13 层 MRO——`BaseChatOpenAI → BaseChatModel → BaseLanguageModel → RunnableSerializable → Serializable → BaseModel → Runnable`，你离 `Runnable` 隔着 10 层。**调试一个 `**kwargs` 传错了，要在 13 层里猜是哪一层收的**——这就是§7.2"性能与调试成本高"的物理来源。
2. **56 个 pydantic 字段 = 选择面爆炸**：一个"模型客户端"背 56 个配置项，每个都能通过 `**kwargs` 出现。抽象厚的反面不是"功能强"，而是**你的心智要背的选项面**。
3. **"封装"的计价单位**：概念 3 步 ↔ 实现 5 节点，"少写 2 行"换"多爬 2 层 + 隐状态 2 个"。这一点和账 D 连起来就是结论：**链适合线性/教学/脚本（概念面优势全开），越往状态/恢复/循环走，隐性层越碍事**。

## 5. 实验 C · 版本存续账（包袱面 II + 概念永存：0.x→1.x 换血）

**目标**：把 §7.2 骂名之二"版本变动剧烈"和§7.2"退化为库集合"钩到一手证据——本机一个 pip install 完成的 1.2.10 里，0.x 时代那些 API 到底还活不活；同时证明"概念永存"确实成立。

**过程**：importlib.metadata 实测同机 6 包版本；枚举 `langchain` 包本体现有子模块；对 10 条 0.x 时代著名导入路径逐一 `importlib.import_module` + `getattr` 探测（真实 import，不是纸面）。

**实测**：

```
  langchain 1.2.10 · langchain-core 1.4.9 · langchain-openai 1.3.4
  langchain-community 0.4.1 · langgraph 1.0.10 · langsmith 0.7.9
  langchain 包本体：顶层命名空间 0 个公开符号（空）· 子模块只剩 6 个：agents · chat_models · embeddings · messages · rate_limiters · tools
  → 0.x 时代路径生存表（真实 import 探测）：
      | LLMChain               | langchain.chains.llm           | FAIL
      | ConversationBufferMemory | langchain.memory              | FAIL
      | OpenAI                 | langchain.llms                 | FAIL
      | BaseMessage            | langchain.schema               | FAIL
      | PromptTemplate         | langchain.prompts              | FAIL
      | TextLoader             | langchain.document_loaders.text | FAIL
      | FAISS                  | langchain.vectorstores         | FAIL
      | RunnableSequence       | langchain_core.runnables.base  | OK
      | PromptTemplate         | langchain_core.prompts         | OK
      | tool                   | langchain_core.tools           | OK
```

**结论**：

1. **0.x 的一切都在今天失效——这是"版本变动剧烈"的最强证据**：`langchain.chains`、`langchain.memory`、`langchain.llms`、`langchain.schema`、`langchain.vectorstores`、`langchain.document_loaders` 六个旧命名空间**整包移除**，`LLMChain`、`ConversationBufferMemory` 这些 2023-2024 教程的绝对主角 import 即崩。**背 API 的教程保质期 ≈ 一个大版本**。
2. **概念在 `langchain_core` 里连续存活**：`RunnableSequence` / `PromptTemplate` / `@tool` 三条 core 路径全 OK——"框架会消亡、概念永存"在这张表上是一字不差的字面事实：**Library 换血，Concept 换皮在后代身上复活**。这也是§7.2"用概念入门"的量化依据：你学的应该是活的最久的那一层。
3. **`langchain` 包本体 = "库集合"**：顶层命名空间为空、只剩 6 个子模块——它不再是一个"框架"，而是框架整合目录。所以"我要不要用 LangChain"在新版里要先精确成："我要不要用它的**组件**（agents/tools/chat_models…）"。账 D 的场景 6 就是这条判断。

## 6. 实验 D · 链 vs 图分界账（该在哪用：客服重试的两种范式）

**目标**：把"链 vs 图的分界"落成一次双方都跑过的同一任务（里程碑 012 的**带记忆重试客服**）——本账跑链式臂，图式臂挂靠 02-章已实测的数字，别再重测。

**过程**：真实 langchain 组"订单客服"链（Fake 预置 3 个回复：失败、失败、成功）；数链 `get_graph()` **自环数**；用真实代码 `_retry_loop` 做"答错重试"，数手写补丁行数（`inspect.getsource`）和重试轮数。

**实测**：

```
  → 链（LCEL）= 单向数据流：RunnableSequence 内部图 自环数 = 0
  → 想要『答错重试』→ 唯一选择 = 链外层自写 while 包装：9 行手写补丁（_retry_loop）
    实测轨迹：第 1/2 轮输出『查询失败』→ 第 3 轮『已发货…』停
```

**结论**：

1. **链是 DAG，能分支、不能自循环**：`RunnableSequence` 自环数 = 0——LCEL 天生没有"循环原语"。你想要"答错再试一次"，唯一的语言级选项就是在链**外层**手写 `while`（实测 9 行补丁、本轮第 3 次命中）。这不是 bug，是定位：**链解决的是"顺一层"，没承诺"回来"**。
2. **同样这个任务，图式由引擎兜住**（挂靠 02-章真实引擎数字）：循环=一条**条件自环边**（Tarjan 含环 1）、轮上限=框架 `recursion_limit` 真抛 `GraphRecursionError`、checkpoint=中断恢复**补 2 步不重跑**、thread 隔离 0 串扰。**链条上要手写的东西，到了图上成了引擎字段**——这就是"链 vs 图"的分界：组装归链、状态/循环/恢复归图。
3. **第三条路：直写（判据与 00-章账 B 对齐）**：线性任务、无状态单跳 → 06 章原语（模板/一次调用/解析）已够，不必为"2 行组装"背 13 层 MRO 的调试成本（账 B）。**所以 §7.2 的"入门看 Chain、生产看 LangGraph 或直写"不是口号，是这张分界表的结论。**

## 7. 拿这张表怎么读本目录（07 章 13 篇的路牌）

本页是 07 章"地图页"（00 篇）之后的**第二站**：00 篇回答了"框架为什么出现、分几类、谁继承谁"，本页回答了"同一家长子 LangChain 的概念该学、库该怎么躲"。后文各篇的路牌（按依赖次序）：

| 后文篇 | 读它的理由（对应本页哪一格） | 状态 |
|---|---|---|
| `01-LangChain与历史包袱`（本篇） | 账 A 类 1 / 账 C 最长链起点：Chain 概念 vs 库的包袱 | 🔥 已交付（本文） |
| `02-LangGraph-状态图与Checkpoint` | 账 D 图式对照臂：链省组装、图省状态/循环/恢复 | 🔥 已交付（里程碑 012） |
| `03-LlamaIndex` | 账 A 类 3 / Haystack 双栖交集：RAG 专精（接 08 章） | 🔥 已交付（v0.29） |
| `04-低代码` | 账 B"四痛点全主"的平台（Dify/Coze/n8n） | 🔥 已交付（v0.30） |
| `05-Haystack` | 双栖点别二：生产级 RAG 管线的 pipeline 思维 | 🔥 已交付（v0.31） |
| `06-Semantic-Kernel` | 账 D 场景 .NET/微软企业栈：插件/函数/自动调用三原语（函数=一等公民） | 🔥 已交付（v0.32） |
| `07-AutoGen·AG2·MAF` | 账 A 类 4 代表：多 Agent 对话范式（互聊→函数式重写→MAF 官方线），§7.8 背书 | 🔥 已交付（v0.33） |
| `08-CrewAI` | 账 A 类 4、账 B：多 Agent 角色化分工（角色·任务·流程三一等公民） | 🔥 已交付（v0.34） |
| `09-DSPy` | "评测主业/状态不碰"、无承袭边独立路径 | 🔥 已交付（v0.36） |
| `10-OpenAI-Agents-SDK` | 账 D 场景 6：轻量运行时"回归轻量"先例（一个 Agent 对象 + 一个 Runner 函数） | 🔥 已交付（v0.35） |
| `11-MCP协议` | 账 B"覆盖 1 维"特判、账 C"2025 公共底座"、里程碑 `013` | 🔥 已交付（v0.37） |
| `12-继承关系与选型决策` | 收束章：完整 DAG + "该不该引入"量化，00 篇 C 账扩版 | ⬜ |

读法口诀（本页的一页带走）：**看到 LangChain 先分两层想——"我学的/用的这个东西是 Runnable 概念还是 `langchain` 包的 API"？概念背走（账 A），包的 API 每次发布验收一遍（账 C 生存表），需要状态/循环/恢复就下车换图（账 D）。**

## 8. 常见坑（6 个）

1. **背 API 当背概念**（本课第一个坑）：`LLMChain` 在 1.2.10 直接 `ModuleNotFoundError`（账 C）——教程保质期一个大版本；要背的是 `Runnable` 协议（invoke/batch/stream，账 A），它在 core 里连续存活。
2. **以为"用 LangChain = 用整个框架"**：现行 `langchain` 是 6 子模块"库集合"（账 C）——精准借用 `agents`/`tools`/`chat_models` 组件即可，不必整包引入再背它的世界观。
3. **拿链硬做循环/恢复**：LCEL 自环数 0（账 D），"答错重试"在链上就是手写 while——把状态装进循环再试图找回，就是"套娃"灾难的入口；该换 LangGraph。
4. **被"MRO 深度"吓退到非要用框架之外**：13 层继承、56 字段是"选择面"不是"能力"（账 B）——线性任务直写 06 章原语即可，别为 2 行组装背 10 层调试面。
5. **跨版本升级当小更新**：0.x→1.x 是整包位移（chains/memory/llms 全拆，账 C）；升级前先跑一遍生存表脚本，别信"pip install -U 就好"。
6. **2022 年的 LangChain 教程 + 2026 年的问题**：链/图分界在 2024 年起就以 LangGraph 为正统（账 D 挂靠 02 章）——写"LangChain agent"前先问：我要的是链的组装还是图的状态。

## 9. 诚实边界（AAA 自我审查）

- **引擎本体 = 本机真实运行**：langchain 1.2.10 / langchain-core 1.4.9 由 importlib.metadata 实测，A–D 账号的 **LCEL 组装、get_graph 拓扑、MRO、model_fields、import 生存表、自环数、重试轮数/补丁行**全部来自引擎真实结果，不是演示用的纸面数字。
- **模型 = FakeListLLM 预置（LLM 非本机实测）**：所有"模型输出"（订单回复/失败/成功/流式 chunk）都是预置字面量，真实 LLM 行为（含工具调用循环）未在本机运行——工具循环语义挂靠 05-章、客服任务挂靠里程碑 012（02 章里同样用词面规则预置节点逻辑）。**请勿据此推断真实 LLM 在 LangChain 上的品质，那需要真模型 + 真 API。**
- **版本节奏 / 年份 = 要素事实**：v0.1（2023）/ v0.2、v0.3（2024）/ v1.0（2025）为作者综述，写作环境无外网、未在线复核各release；若你看到的版本节奏有出入，以官方 changelog 为准——本账的关键结论（day-0 移除 vs day-0 存活）是本机可复排的硬事实，不依赖年份。
- **图式对照 = 挂靠 02-章**：账 D 的图式臂（7 节点/6 支路/1 自环/recursion_limit/checkpoint 数字）引自 02-LangGraph 章真实引擎实测，本页不做二次重测（主题是链式臂）。
- **确定性纪律**：stdout md5 `900181e7…` 三个独立进程逐字节恒一；墙钟 ≈15.7 s 仅 stderr；`get_graph()` 随机节点 ID 不打印（只打印计数）。

## 10. 参考与衔接

- **本页地图**：§7.2（LangChain 概念贡献 + 包袱清单 + 转型）+ §17.5.8（LangChain 与历史包袱）+ §7.13（谱系起点）。
- **选择树 8 场景**（探针断言 8/8）：① 学概念 → langchain-core 的 Runnable/LCEL，不装重量依赖；② 线性脚本 → `PromptTemplate | 模型 | StrOutputParser`（账 A）；③ 批量/并发 → `batch()` 换触发不换代码（账 A）；④ 需要状态/恢复/重试 → LangGraph 优先于 chain（账 D 挂靠 02 章）；⑤ 已有 0.x 代码 → 先过生存表迁移到 core 组件（账 C）；⑥ 只借组件 → 库集合小件（账 C）；⑦ 生产可观测 → LangSmith/自接 Langfuse；⑧ 嫌封装厚 → 直写原生 SDK 或轻量运行时（账 B）。
- **前承**：00-框架分类学（开篇章，类 1 与谱系定位）；06-应用开发全 6 篇（直写原语基准）；08-RAG（后文 LlamaIndex/Haystack 的挂靠面）。
- **后启**：02-LangGraph（✅ 已交付，账 D 图式对照臂）；`03-LlamaIndex`（✅ 已交付 v0.29：账 A 类 3：索引与检索把"数据-流"吃干净，LangChain 的检索组件变成其中一个子集，真实 llama-index-core 0.14.25 引擎四账实测，衔接 08-章 12 篇实测）；`04-低代码`（✅ 已交付 v0.30：账 A 类 5 低代码平台——把 06-07 原语尊成可视积木、同一客服重试三家画布 5-7 块带回边的确定性测量）；`05-Haystack`（✅ 已交付 v0.31：双栖点别二——生产级 RAG 管线「严谨的 pipeline 思维」，真实 haystack-ai 3.1.1 引擎四账实测，账 C 版本存续与本文同母题：顶层 Pipeline 存活、1.x nodes/pipelines 整包 FAIL）；`06-Semantic-Kernel`（✅ 已交付 v0.32：账 C 改名最狠的对照样本——0.x 概念词全换（Skill/Planner 消失，orchestration/skill_definition/planning/core_skills 四子包整包 FAIL）只剩 Kernel 身份存活，本文"概念永存 vs 库换血"再添一极）；`07-AutoGen-AG2-Microsoft-Agent-Framework`（✅ 已交付 v0.33：改名烈度第二家——AutoGen 经典『ConversableAgent+GroupChat 互聊』API 在 AG2 1.0 整包消失（连 `import autogen` 垫片都没了），真实 ag2 1.0.6 引擎四账）；`12-继承关系与选型决策`（收束章）。
- **对应里程碑**：`012`（带记忆重试客服）——本页与 02 章各跑一侧范式；里程碑 `013`（手写 MCP Server）已在 `11-MCP协议` 章交付（v0.37）。

> 本篇完工于 2026-09-22（v0.28 批次）；探针 `code/notebooks/_tools/langchain_demo.py`；stdout md5 `900181e7…`。
