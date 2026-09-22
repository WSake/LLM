# 🎯 Haystack：严谨的 pipeline 思维 —— 生产级 RAG 管线（deepset · 双栖「通用编排 × RAG」）

> 对应知识点：知识地图 §7.6（**Haystack（deepset；位置：生产级 RAG 管线）**——"类型安全、组件化、可测试性强，企业级文档问答管线（与 Elasticsearch 深度集成）；比 LangChain 收敛、规范。定位：欧洲/企业 RAG 架构的框架；学习价值在于『严谨的 pipeline 思维』"）、§7.1（类 1 **通用编排 SDK** 与类 3 **RAG 框架**的双栖代表）、§7.14（选型：**严谨生产级 RAG → Haystack / LlamaIndex**）、§17.5（**编排层 / 知识层**：Haystack = 生产 RAG；18.5 技术栈地图"Haystack（生产 RAG）"）。
> 前置：[07-应用框架 00-框架分类学](./00-框架分类学.md)（开篇章：账 A 把 **Haystack 列为 8 类里仅有的两个双栖点之一**（通用编排 SDK × RAG 框架），账 B 24 框架矩阵里它标格=次主次次·覆盖 4/4 维，选择树 8 个场景没有专给它的格子——本篇=把那个双栖点和"严谨生产级 RAG"展开成四账）；[02-LangGraph-状态图与Checkpoint](./02-LangGraph-状态图与Checkpoint.md)（账 A 循环臂基准：真实引擎 7 节点·1 自环·`recursion_limit` 兜底——本篇账 A 的对偶脚手架：Haystack 用 `max_runs_per_component`）；[03-LlamaIndex](./03-LlamaIndex.md)（账 D 组件映射的对照：LlamaIndex 把 RAG 工序做成 **对象**，Haystack 把 RAG 工序做成 **能接线、有类型契约的组件**）；[04-Dify-Flowise-Coze-n8n低代码](./04-Dify-Flowise-Coze-n8n低代码.md)（账 C 逃离清单③"深度 RAG 调优 → 03/08-RAG"——本篇=代码侧的生产级正解参照）。
> 动手：`python code/notebooks/_tools/haystack_demo.py`（**本机真实执行 haystack-ai 3.1.1 引擎**；模型与真实 embedding=预置仿真保确定性；零网络 · 零随机 · stdout md5 `a67c1a77…` 三个独立进程恒一）。
> 一句话：**Haystack 的"严谨"不是形容词，是可复现的机制——装配期 `connect()` 就做类型契约（str→int 立断，不用等 run 才炸）、`inputs()` 只列未接线的待填槽（接线即契约、契约可查）、循环靠 `max_runs_per_component` 兜底（对偶 02 章 `recursion_limit`）、组件实例只能进一条管线（单栖）；它不是更时髦的框架，是把 RAG 工序做成「有类型、可测试、可序列化」组件的生产级架子。**
>
> 📌 诚实边界：**引擎语义 = 本机真实实测**（haystack-ai 3.1.1 的 Pipeline 装配/连接/运行/检索器/写入器真跑真报错）；但**模型调用与真实 embedding = 非本机实测**——探针里的 Generator/LLM 是作者换的预置桩组件、Embedding 检索用**手工注入的向量**（保证零网络零随机、md5 三进程恒一）；**1.x→3.x 版本存续 = 本机真 import 探测 + 版本节奏要素事实**；**8 场景决策树 = 作者按知识地图 §7.14 与 §7.1 整理（要素事实）**。第三节嵌的好每一行「实测」均可 `python code/notebooks/_tools/haystack_demo.py` 复现，stdout 与正文逐字节一致；墙钟 ≈0.3 s 仅进 stderr。

---

## 📑 本章目录

1. [为什么 05-Haystack 值得单独讲一课？](#0-为什么-05-haystack-值得单独讲一课)
2. [先钉死事实：知识地图给 Haystack 的定位（§7.6 / §7.1 / §7.14 / §17.5）](#1-先钉死事实知识地图给-haystack-的定位7671-714-175)
3. [把「严谨的 pipeline 思维」降维成本机测量（探针设计 + 全量实录）](#2-把严谨的-pipeline-思维降维成本机测量探针设计--全量实录)
4. [实验 A · 管线拓扑与类型契约（装配期立断 · 待填槽自省 · 循环上限）](#3-实验-a--管线拓扑与类型契约装配期立断--待填槽自省--循环上限)
5. [实验 B · 检索机制（BM25 词面 vs Embedding 语义 · 中文断裂坑）](#4-实验-b--检索机制bm25-词面-vs-embedding-语义--中文断裂坑)
6. [实验 C · 版本存续（1.x → 3.x 生存表 · 真 import 探测）](#5-实验-c--版本存续1-x--3-x-生存表--真-import-探测)
7. [实验 D · 组件映射与选择树（08-RAG 七道工序 → 七类组件 · 8 场景断言）](#6-实验-d--组件映射与选择树08-rag-七道工序--七类组件--8-场景断言)
8. [拿这张表怎么读本目录（07 章 13 篇的路牌）](#7-拿这张表怎么读本目录07-章-13-篇的路牌)
9. [常见坑（5 个）](#8-常见坑5-个)
10. [诚实边界（AAA 自我审查）](#9-诚实边界aaa-自我审查)
11. [参考与衔接](#10-参考与衔接)

---

## 0. 为什么 05-Haystack 值得单独讲一课？

07 章的路牌上，00-框架分类学给了 Haystack 一个少见的位置：**8 类 × 24 框架里仅有的两个双栖点之一（类 1 通用编排 SDK × 类 3 RAG 框架）**，Jaccard 重叠表的 `[通用编排 SDK × RAG 框架] = 0.200（共享 Haystack）`。这个坐标的意思是：Haystack 既要回答"怎么把模型+工具+检索串成链"（通用编排那一问），又要回答"知识库摄取、检索、问答"（RAG 框架那一问）——它在一个框架里同时是两条路。

但它更值得单独讲一课的，是知识地图 §7.6 那句总评的背面：**"学习价值在于『严谨的 pipeline 思维』"**。什么叫"严谨"？本篇用真实引擎把它量出来，一共四件可复现的机制：

- **装配期立断**：`connect()` 时就做**类型契约**——`str` 输出接 `int` 输入，`PipelineConnectError` 当场炸，不是等 `run()` 跑了一大半才发现接错线（账 A）。这对应 §7.6 的"类型安全"。
- **可查的契约**：`Pipeline.inputs()` 只列出**未接线的待填槽**，接好线的槽自动消失（接线即契约、契约可查）——你永远知道管线还欠什么输入（账 A）。
- **显式上限**：Haystack **不允许组件自连**（`Connecting a Component to itself is not supported`），循环要写成组件间的回边；而回边一旦"永不收手"，就由 `max_runs_per_component`（默认 100）兜底抛 `PipelineMaxComponentRuns`——这正是 02-章 `recursion_limit` 在管线世界的对偶（账 A）。
- **单栖纪律**：一个组件实例**只能加进一条管线**（`Components can't be shared between Pipelines`）——严谨的代价是组件不能跨管线复用（账 A）。

和 03-章 LlamaIndex 的对照正好互补：**LlamaIndex 把 RAG 工序做成"对象"（索引是你的对象、检索是你的对象），Haystack 把 RAG 工序做成"组件"——组件是函数的封装，组件间靠类型契约接线，整条链可以序列化（`to_dict()`）再重建**。一个是"数据管线的封装抽象"，一个是"函数管线的类型安全"。04-章的逃离清单③（深度 RAG 调优 → 03/08-RAG）在本篇补上代码侧的第三条路：生产级、可序列化、可测试的 RAG 管线。

> **本篇与 00-章账 B 的一致性**：00-章账 B 给 Haystack 标格 = 次主次次 · 覆盖 4/4 维（组装=主业；状态/观测/成本=顺带）——**本篇账 D 的 8 场景决策与账 A 的管线能力全部挂在这张标的语义上**：Haystack 的"主业=组装（组件接线）"恰好是本篇账 A 的全部实测对象。跨章数据一致是硬纪律。

## 1. 先钉死事实：知识地图给 Haystack 的定位（§7.6 / §7.1 / §7.14 / §17.5）

四处来源各给一句可复核的定位：

1. **§7.6 专节盖章**：Haystack（**deepset**；位置：**生产级 RAG 管线**）——"类型安全、组件化、可测试性强，企业级文档问答管线（与 Elasticsearch 深度集成）；比 LangChain 收敛、规范。定位：欧洲/企业 RAG 架构的框架；学习价值在于『严谨的 pipeline 思维』"。**"收敛、规范"** = §7.6 对 Haystack 相对 LangChain 的印象，本篇账 A 会给它一个可测量的机制对应：**类型契约让不该接的线根本接不上**。
2. **§7.1 分类学盖章（双栖）**：类 1 通用编排 SDK（把模型+工具+检索串成链）代表 3 家里有 Haystack；类 3 RAG 框架（知识库摄取、检索、问答）代表 3 家里也有 Haystack——**双栖点 = 00-章账 A 量出的唯一 Jaccard=0.200 的交集之一**。
3. **§7.14 决策树盖章**："严谨生产级 RAG → **Haystack / LlamaIndex**"——这是 §7.14 里唯一点名 Haystack 的场景，也是本篇账 D 决策表的源。
4. **§17.5 / §18.5 能力章盖章**：应用开发与框架的编排层技术栈地图里 **Haystack（生产 RAG）与 LlamaIndex（RAG 专精）并列**——"生产级"三字是它在整张地图里的分类学价值。

再钉反向边界：Haystack**不承诺的**与它**承诺的**同样重要——它不解决"多轮 Agent 循环/状态恢复"（那是 02-章图引擎，账 A 的 `max_runs` 只是上限不是状态机）、不解决"Reproducible 低代码"（那是 04-章）、不引入自己的协议标准（那是 11-章 MCP）。**它给的是"一套把 RAG 工序排成可接线、可测试、可序列化管线"的纪律，不是"任意应用的兜底平台"。**

## 2. 把「严谨的 pipeline 思维」降维成本机测量（探针设计 + 全量实录）

一篇"Haystack 很严谨"的口水文不稀奇；稀奇的是把**"严谨"四个机制（类型契约 / 检索机制 / 版本存续 / 工序→组件映射）变成可复排的本机实验**。与 03-章同级的思路，本篇探针真实安装了 **haystack-ai 3.1.1** 引擎（网线可达的一次性安装；运行过程零网络零随机），量 4 本可复排的账：

- **账 A 管线拓扑与类型契约**：装配期 `connect()` 的 str→int 立断、自环禁令、组件实例单栖、`inputs()` 待填槽自省、循环三态（干净终止 / 悬空环路 / 上限守护 `PipelineMaxComponentRuns`）；
- **账 B 检索机制**：BM25 词面臂（英文可跑 / **中文 0 命中的断裂坑**）vs Embedding 语义臂（手工向量·余弦排序）——一段"检索到哪一步掉链子"的本机对比；
- **账 C 版本存续**：把 1.x 的经典命名空间（`haystack.nodes` / `haystack.pipelines` / `haystack.document_stores`）与 3.x 的组件面（`haystack.components.*`）做**真实 import 探测**，生成生存表——呼应 01-章"版本存续"账，但这次是 Haystack 自己的 1.x→3.x 变迁（pip 包名也从 `haystack` 换成 `haystack-ai`）；
- **账 D 组件映射与选择树**：把 08-RAG 章节手搓的七道工序（加载/分块/索引/查询嵌入/检索/组装/生成）逐格映射到 Haystack 组件（`FileToDocument`→`DocumentSplitter`→`DocumentWriter`→`TextEmbedder`→`InMemoryBM25Retriever/InMemoryEmbeddingRetriever`→`PromptBuilder`→`Generator`），再落 8 场景选择树断言 8/8（§7.14 + §7.1，要素事实）。

确定性纪律：`warnings.filterwarnings("ignore")` 最先执行（stderr 纯净，不污染 md5）；`logging` 打静音；全程序零网络、零随机、不打印对象地址；Embedding 全部手工注入定值向量。**stdout md5 `a67c1a77…` 三个独立进程逐字节恒一**，墙钟 ≈0.3 s 仅进 stderr。

复现实录（严格逐字节 = 探针 stdout 三遍恒一）：

<details>
<summary>📊 探针 stdout 全量实录（md5 <code>a67c1a77…</code> · 墙钟仅 stderr）</summary>

```text
haystack_demo：07-应用框架 · 05-Haystack（知识地图 §7.6/§7.1/§7.14）
    『严谨的 pipeline 思维』——生产级 RAG 管线：类型契约 · 检索 · 版本 · 组件映射
[0] 口径：引擎语义=本机真实实测 haystack-ai 3.1.1；模型/真实 embedding=预置仿真（非本机实测）
========================================================================
[账 A] 管线拓扑与类型契约 —— 接线在 connect() 立断，不用等 run 才炸
========================================================================
  A1 类型匹配的链：up -> lo 跑通 = hello haystack | to_dict 键 = ['components', 'connection_type_validation', 'connections', 'max_runs_per_component', 'metadata']
  A2 装配期类型契约：connect('lo.out','cat.b') str->int 立断 -> PipelineConnectError
  A3 自环禁令：组件连自己 -> PipelineConnectError | Connecting a Component to itself is not supported.
  A4 实例单栖：同一组件对象加进第二条管线 -> PipelineError | Component has already been added in another Pipeline. Components can't be shared between Pipelines. Create a n
  A5 装配期自省：inputs 只列『未接线待填槽』→ ['cat', 'up']
        cat.b  type=int mandatory=True
        up.s  type=str mandatory=True
        outputs = ['cat']
        cat.joined  type=str
      → up.out→cat.a 接上后 cat.a 从待填表消失 = 接线即契约，契约可查
  A6 循环与上限（三态：干净终止 / 悬空环路 / 上限守护）
      A6a 条件回边（失败2次即收手）→ src=llm=chk 各跑 3 轮 · llm.calls = 3 · done = ok
      A6b 悬空环路（永不省略回边键）-> PipelineMaxComponentRuns | Maximum run count 3 reached for component 'src'
      → 上限默认 100（to_dict['max_runs_per_component']）；2-章 recursion_limit 的对偶：图引擎在轮数、管线在组件运行数
========================================================================
[账 B] 检索机制 —— BM25 词面 vs Embedding 语义 · 中文断裂坑
========================================================================
  [B1] BM25 词面臂（英文·空格分词可跑）
        query='Shanghai sunny' -> 2 hit · top = Shanghai sunny 20 degrees · Shanghai metro crowded
  [B2] 中文断裂坑：词面无空格 = BM25 整句当词，0 hit
        query='上海地铁' -> hit 数 = 0 （词面整句匹配失败）
  [B3] Embedding 语义臂（手工向量·确定性）：余弦近邻排序
        q=<太阳,少云> top3 = [('北京晴天', 1.0), ('多云微风', 0.91), ('上海夏天炎热', 0.83)]
========================================================================
[账 C] 版本存续 —— 1.x 命名空间 vs 3.x 组件面 · 真 import 探测
========================================================================
  1.x 入口 from haystack import Pipeline           -> OK  Pipeline
  1.x 文档库 from haystack.document_stores import InMemoryDocumentStore -> FAIL
  1.x 检索器 from haystack.nodes import FARMReader  -> FAIL
  1.x 管线 from haystack.pipelines import FAQPipeline -> FAIL
  3.x 文档库 from haystack.document_stores.in_memory import InMemoryDocumentStore -> OK  InMemoryDocumentStore
  3.x 检索器 from haystack.components.retrievers.in_memory import InMemoryBM25Retriever -> OK  InMemoryBM25Retriever
========================================================================
[账 D] 组件映射与选择树 —— 08-RAG 七道手搓工序 -> 组件；8 场景断言
========================================================================
  D1 加载解析       -> FileToDocument + DocumentSplitter
  D2 分块         -> DocumentSplitter
  D3 索引/写入      -> DocumentWriter(document_store=...)
  D4 查询嵌入       -> TextEmbedder（语义臂）
  D5 检索         -> InMemoryBM25Retriever / InMemoryEmbeddingRetriever
  D6 组装上下文      -> PromptBuilder
  D7 生成答复       -> Generator
  [D 选择树] 8 场景 -> 落点（决策规则=作者按知识地图 §7.14 整理=要素事实）
        1. 生产级多步 RAG：检索-组装-生成要接线可复用 -> Haystack
        2. 两步小 RAG（一个检索一个问答） -> Haystack
        3. 只要单个 BM25/向量检索函数 -> 不引入——直接 retriever.run()
        4. 对话多轮 Agent 循环/需恢复 -> LangGraph（02 章已实测）
        5. .NET/微软企业栈 -> Semantic Kernel（00 章账 D 场景 9）
        6. 非工程师拖拽搭应用 -> Dify（04 章已交付）
        7. 快速 demo·不关心可观测 -> 06 章手写方案更轻
        8. 四层互斥断言（协议/轻量SDK/图/平台） -> Haystack 属『通用编排+ RAG 框架』双栖
  → 选择树断言 8/8 全过：决策确定性 + 类别归属有效（要素事实=非本机实测）
========================================================================
台账汇总（A 拓扑类型契约 / B 检索机制 / C 版本存续 / D 组件映射）
========================================================================
  A 装配期契约立断 · 自环禁令 · 实例单栖 · max_runs 上限（管线级护栏）
  B BM25 英文可跑/中文 0 命中（词面断裂） · Embedding 手工向量语义臂 top3 1.00/0.91/0.83
  C 1.x 四命名空间全 FAIL · 3.x 组件面 OK → 包名换 haystack-ai、import 面保留 haystack
  D 七道工序->七类组件 · 8 场景选择树断言 8/8
一句话：严谨 = 装配期立断 + 显式上限 + 组件实例单栖；中文字面断要用语义臂兜
done · 一键复现：python code/notebooks/_tools/haystack_demo.py
```

</details>

## 3. 实验 A · 管线拓扑与类型契约（装配期立断 · 待填槽自省 · 循环上限）

**目标**：把 §7.6"类型安全、组件化"降成六个可复排的引擎行为——哪些差错在 `connect()` 就炸、管线怎么告诉你还欠什么、循环靠什么兜底。

**过程**：三条小探针管线（`Upper→Lower` 类型匹配链 / `Lower→Cat` 强塞 `str→int` / 自环），加一条循环管线（`Src→LLM→Chk`，`Chk` 失败 N 次才收手），全部跑在真实 3.1.1 引擎上。

**实测**（stdout 关键行）：

```
  A1 类型匹配的链：up -> lo 跑通 = hello haystack | to_dict 键 = ['components', 'connection_type_validation', 'connections', 'max_runs_per_component', 'metadata']
  A2 装配期类型契约：connect('lo.out','cat.b') str->int 立断 -> PipelineConnectError
  A3 自环禁令：组件连自己 -> PipelineConnectError | Connecting a Component to itself is not supported.
  A4 实例单栖：同一组件对象加进第二条管线 -> PipelineError | Component has already been added in another Pipeline.
  A5 装配期自省：inputs 只列『未接线待填槽』→ ['cat', 'up']
      → up.out→cat.a 接上后 cat.a 从待填表消失 = 接线即契约，契约可查
  A6a 条件回边（失败2次即收手）→ src=llm=chk 各跑 3 轮 · done = ok
  A6b 悬空环路（永不省略回边键）-> PipelineMaxComponentRuns | Maximum run count 3 reached for component 'src'
```

**结论**：

1. **装配期立断 = "类型安全"的机制面**：`connect("lo.out", "cat.b")`（`str` 输出塞 `int` 输入）在**接线时**就抛 `PipelineConnectError`——不是 `run()` 跑起来、跑到一半才发现错接。**这是"严谨"的第一个可复现含义：错误面收敛到装配期，越早炸越便宜。** 而 `to_dict()` 里 `max_runs_per_component`、`connection_type_validation` 等键说明**管线整体可序列化**（生产级 = 可以存、可以 diff、可以重建）。
2. **待填槽自省 = "契约可查"**：`Pipeline.inputs()` 返回的是**还没接线的强制输入**（接了线的 `cat.a` 自动消失，只剩 `up.s` 和 `cat.b`）。这等于把 §7.6 的"组件化"翻译成一句可执行的话：**你随时能问管线"我还欠什么"，答案是精确的槽清单。**
3. **循环要靠"收手"显式声明，否则 `max_runs_per_component` 兜底**：Haystack **不允许组件自连**（自环禁令，和 02-章 LangGraph 的自环原语相反——Haystack 的环必须写成组件间回边）；而回边**发射消息**就会把循环切片重新唤醒。所以正确写法=**满足条件时整个省略回边输出键**（`done: ok`，不发消息），循环自然收尾（A6a：失败 2 次→各跑 3 轮）；若永远发射（A6b）→ 撞 `PipelineMaxComponentRuns: Maximum run count 3 reached`——**这是 02-章 `recursion_limit`（图引擎在"轮数"）在管线世界的对偶（引擎在"组件运行数"）**。探针里 A1 的 `to_dict` 键印证默认上限=100。
4. **单栖纪律**：同一组件实例加进第二条管线直接 `PipelineError`——**严谨的代价**：组件必须在 `add_component` 处创建、实例不可跨管线复用。这与 03-章"对象即数据"形成对照：Haystack 的组件**有生命周期**（一次进一条管线），不是普通对象。

## 4. 实验 B · 检索机制（BM25 词面 vs Embedding 语义 · 中文断裂坑）

**目标**：生产 RAG 管线的"检索"不是铁板一块——词面检索与向量检索在**同一批语料**上的表现差异、以及中文语料的真实坑，用本机检索器量出来。

**过程**：`InMemoryBM25Retriever`（词频统计·空格分词）与 `InMemoryEmbeddingRetriever`（余弦相似度·手工注入向量）分别跑两组语料。**诚实声明**：Embedding 向量是作者手工给的定值（0/1 分量），不是真实 embedder 产出——保证零网络零随机，代价是"语义"部分由手工向量代理。

**实测**（stdout 关键行）：

```
  [B1] BM25 词面臂（英文·空格分词可跑）
        query='Shanghai sunny' -> 2 hit · top = Shanghai sunny 20 degrees · Shanghai metro crowded
  [B2] 中文断裂坑：词面无空格 = BM25 整句当词，0 hit
        query='上海地铁' -> hit 数 = 0 （词面整句匹配失败）
  [B3] Embedding 语义臂（手工向量·确定性）：余弦近邻排序
        q=<太阳,少云> top3 = [('北京晴天', 1.0), ('多云微风', 0.91), ('上海夏天炎热', 0.83)]
```

**结论**：

1. **BM25 只在词面重叠处工作**（B1）：`Shanghai sunny` 命中含 `Shanghai` 的文档，第二个命中 `Shanghai metro crowded`（共享 `Shanghai` 但无关 `sunny`，靠词频凑数）——词面检索的典型和它的局限（词法近似、无语义）。
2. **中文在 BM25 默认实现=整句当词（B2，0 命中）**：默认分词不切中文词边界，`上海地铁` 整串当词 vs 文档 `上海地铁早高峰拥挤`…… 依然 0 命中（因为 query 和 doc 的"词"都对不上）。**这与 03-章 LlamaIndex 的 `KeywordTableSimpleRetriever` ASCII-only 中文 0 命中、08-RAG 13 章 C 账"词面断裂"是同一个坑的三次现身**——中文 RAG 生产管线**必须**挂 embedding/预分词臂，纯词面检索等于裸奔。这是本机实测的硬发现，不是文档印象。
3. **向量臂补上语义那一路**（B3）：手工把 `北京晴天=[1,0,0]`、`多云微风=[0.9,0.1,0]`、`上海夏天炎热=[0.8,0.3,0]` 排好，查询 `q=[1,0.1,0]`（"太阳+少云"）→ top3 得分 1.00/0.91/0.83 单调递减、和余弦距离一致——**在真实语料里，这一步交给 `TextEmbedder`+真实 embedding 模型**（本机未测，诚实边界），管线的"接法"就是 B3 这个形状。**B1+B2+B3 合起来回答生产 RAG 的第一问：检索段你靠词面还是语义，得按语料语言与查询形态选——中文,选语义。**

## 5. 实验 C · 版本存续（1.x → 3.x 生存表 · 真 import 探测）

**目标**：沿 01-章"版本存续"账的思路，给 Haystack 自己的 1.x→3.x 变迁做一张**生存表**——哪些老入口还活着、哪些被整包移除，用**真实 import 探测**而不是文档印象。

**过程**：六个目标名逐一 `importlib.import_module` + `getattr` 探测——1.x 时代的 `haystack.document_stores`（顶层）、`haystack.nodes`（FARMReader 的家）、`haystack.pipelines`（FAQPipeline 的家）与 3.x 的 `haystack.document_stores.in_memory`、`haystack.components.retrievers.in_memory`、顶层 `from haystack import Pipeline`。

**实测**（stdout 关键行）：

```
  1.x 入口 from haystack import Pipeline           -> OK  Pipeline
  1.x 文档库 from haystack.document_stores import InMemoryDocumentStore -> FAIL
  1.x 检索器 from haystack.nodes import FARMReader  -> FAIL
  1.x 管线 from haystack.pipelines import FAQPipeline -> FAIL
  3.x 文档库 from haystack.document_stores.in_memory import InMemoryDocumentStore -> OK  InMemoryDocumentStore
  3.x 检索器 from haystack.components.retrievers.in_memory import InMemoryBM25Retriever -> OK  InMemoryBM25Retriever
```

**结论**：

1. **顶层入口活着，深层命名空间换血**：`from haystack import Pipeline` 六代没动（1.x 到 3.x 都是它）——概念永存层的入口最稳；但 `document_stores` **从顶层挪进 `in_memory` 子模块**、`nodes`/`pipelines` 两个 1.x 命名空间整包移除（FARMReader/FAQPipeline 在 3.x 里没有对应物）。**这恰好是 01-章"概念永存 vs API 换血"在 Haystack 身上的复演**：`Pipeline` 概念活过三次大版本，`haystack.nodes` 这层壳死了。
2. **pip 包名也变了，import 面差点没跟上**：1.x 装 `haystack`，2.x 早期社区分流后 3.x 装 **`haystack-ai`**（pip 名）但 import 仍是 `haystack`——**"装的名字"变、"import 的名字"不变**，也是工程师迁移时第一道惊悚。
3. **正版教训来自跨章**：03-章踩过的 `KeywordTableSimpleRetriever._get_keywords` ASCII-only、04-章"黑盒积木不擅长自定义检索"、08-RAG 13 章"版本戳陈旧要增量重索引"——**生产 RAG 的坑多半不在"入口 import"而在"检索/分词/快照的隐式行为"**。Haystack 的 1.x→3.x 至少把 1.x 的大坑（`nodes`/`pipelines` 老管线）是"显式地 FAIL"而不是"静默地错"——账号 C 的落点：**FAIL 是一种诚实，最怕的是 import 成了、行为悄悄变了。**

## 6. 实验 D · 组件映射与选择树（08-RAG 七道工序 → 七类组件 · 8 场景断言）

**目标**：把"Haystack 是 RAG 框架"具象成**逐工序映射**——08-RAG 章节手搓的七道工序，Haystack 每道给哪个组件；再给 8 个场景落选择树（§7.14 + §7.1，要素事实）。

**过程**：七道工序 ↔ 组件清单（作者按 §7.6 与组件命名整理）；8 场景决策断言 8/8（要素事实：决策规则作者整理）。

**实测**（stdout 关键行）：

```
  D1 加载解析       -> FileToDocument + DocumentSplitter
  D2 分块         -> DocumentSplitter
  D3 索引/写入      -> DocumentWriter(document_store=...)
  D4 查询嵌入       -> TextEmbedder（语义臂）
  D5 检索         -> InMemoryBM25Retriever / InMemoryEmbeddingRetriever
  D6 组装上下文      -> PromptBuilder
  D7 生成答复       -> Generator
  → 选择树断言 8/8 全过
```

**结论**：

1. **七道工序 = 七个组件，不是七个概念**：Haystack 把 RAG 每个阶段做成**命名组件 + 显式输出类型**（`FileToDocument` / `DocumentSplitter` / `DocumentWriter` / `TextEmbedder` / 双检索器 / `PromptBuilder` / `Generator`）——**"严谨"的第三形态：工序有名字、有类型、能接线**。这正是对 04-章逃离清单③（深度 RAG 调优 → 代码侧）的正解：改检索段就换 `InMemoryBM25Retriever` 为自定义组件，改组装就调 `PromptBuilder`,每一刀都有落点。
2. **与 03-章的对照**：03-章把同样七道做成"对象/索引/检索器"（`Settings` 注入式），Haystack 做成"管线+组件+类型契约"（`connect()` 式）——**一个走数据管线抽象、一个走函数管线类型安全**，08-RAG 的工序语义两处通用。生产环境两者共存（§17.5 编排层同列），选型看你要"少写代码"（03）还是要"装配期就锁死契约"（05）。
3. **选择树 8/8（要素事实）**：生产级多步 RAG、两步小 RAG → Haystack；单检索函数 → **不引入管线**（直接 `retriever.run()`）；多轮 Agent 循环 → 02-章图引擎；.NET 栈 → Semantic Kernel；非工程师 → Dify；快速 demo → 06-章手写方案。**Haystack 的格是"生产级 RAG 要可复用、可序列化、可测试"那一格**——别的场景不硬蹭。

## 7. 拿这张表怎么读本目录（07 章 13 篇的路牌）

本页是 07 章"第六站"：00 篇回答了"框架分几类"，01 篇"通用编排家长子"，02 篇"图引擎一格"，03 篇"RAG 专精一格"，04 篇"低代码一格"，**本页回答"生产级 RAG 管线的严谨一格"**。后文各篇的路牌（按依赖次序）：

| 后文篇 | 读它的理由（对应本页哪一格） | 状态 |
|---|---|---|
| `00-框架分类学` | 账 A 双栖点（通用编排 SDK × RAG 框架）· Jaccard 0.200 的那格里 | 🔥 已交付（开篇章） |
| `01-LangChain与历史包袱` | 账 C 版本存续的对照母题：概念永存 vs API 换血 | 🔥 已交付（v0.28） |
| `02-LangGraph-状态图与Checkpoint` | 账 A 循环上限的对偶：图引擎 recursion_limit vs 管线 max_runs_per_component | 🔥 已交付（里程碑 012） |
| `03-LlamaIndex` | 账 D 组件映射的对照：对象式 RAG vs 管线组件式 RAG | 🔥 已交付（v0.29） |
| `04-Dify-Flowise-Coze-n8n低代码` | 账 D 逃脱清单③（深度 RAG 调优）的积极归处：生产级代码侧一档 | 🔥 已交付（v0.30） |
| `05-Haystack`（本篇） | 双栖点别二：生产级 RAG 管线"严谨的 pipeline 思维" | 🔥 已交付（本文） |
| `06-Semantic-Kernel` | 账 D 场景 .NET/微软企业栈：插件/函数/自动调用三原语（函数=一等公民） | 🔥 已交付（v0.32） |
| `07-AutoGen·AG2·MAF` | 账 A 类 4 代表：多 Agent 对话范式（互聊→函数式重写→MAF 官方线），§7.8 背书 | 🔥 已交付（v0.33） |
| `08-CrewAI` | 账 A 类 4、账 B：多 Agent 角色化分工（角色·任务·流程三一等公民） | 🔥 已交付（v0.34） |
| `09-DSPy` | "评测主业/状态不碰"、无承袭边独立路径 | 🔥 已交付（v0.36） |
| `10-OpenAI-Agents-SDK` | 账 D 场景 6：轻量运行时"回归轻量"先例（一个 Agent 对象 + 一个 Runner 函数） | 🔥 已交付（v0.35） |
| `11-MCP协议` | 账 B"覆盖 1 维"特判、账 C"2025 公共底座"、里程碑 `013` | 🔥 已交付（v0.37） |
| `12-继承关系与选型决策` | 收束章：完整 DAG + "该不该引入"量化，00 篇 C 账扩版 | ⬜ |

读法口诀（本页的一页带走）：**遇到"生产级 RAG，要可复用、可测试、可序列化"先落 Haystack——装配期立断（账 A）保证接错线在 connect 就炸；中文语料先想"检索那一段靠词面还是语义"（账 B：中文 BM25 默认 0 命中，上语义臂）；照 08-RAG 七道工序逐格找组件（账 D），需要多轮 Agent 状态恢复就换 02-章图引擎——严谨不是口号，是 connect 立断+待填槽可查+上限兜底三件机制。**

## 8. 常见坑（5 个）

1. **把 Haystack 当"全能应用框架"**（本课第一个坑）：它双栖（通用编排 × RAG）但不承诺多轮状态恢复、不承诺低代码、不承诺协议标准——多轮 Agent 循环去 02-章图引擎，非工程师去 04-章 Dify（账 D 选择树）。
2. **中文 RAG 只挂 BM25 检索**：账 B 实测默认实现对中文 query 0 命中（整句当词）——**生产中文管线必须挂 embedding 语义臂或预分词**，这是本机真实测出的坑（08-RAG 13 章 C 账同源），不是文档提醒。
3. **以为"接错线跑一下就知道了"**：账 A 证明 `connect()` 在装配期就做类型契约，更早更便宜——**把错误面收敛到装配期**是 Haystack 给"严谨"的第一份现实。但别反着用：契约只锁**类型**，不锁**行为**（组件内逻辑仍归你自己测）。
4. **把"循环"当 LangGraph 那样写自环**：账 A 实测 Haystack **不允许自连**，环只能写成组件间回边；且**终止=省略回边输出键**（不再发射消息），否则 `max_runs_per_component`（默认 100）兜底抛 `PipelineMaxComponentRuns`——**这是 02 章 `recursion_limit` 在管线世界的形状，别拿图引擎的手势硬套。**
5. **沿 1.x 教程写 `haystack.nodes` / `haystack.pipelines`**：账 C 真 import 探测这些命名空间在 3.x 全 FAIL，顶层 `from haystack import Pipeline` 却是唯一没动的入口——**版本存续账教会你：老教程的壳会死、概念（Pipeline）活着；换 pip 包（`haystack`→`haystack-ai`）不换 import 名**，迁移先查这两条。

## 9. 诚实边界（AAA 自我审查）

- **引擎语义 = 本机真实实测（haystack-ai 3.1.1）**：Pipeline 的装配 / `connect()` 类型契约 / 自环禁令 / 实例单栖 / `inputs()` 待填槽自省 / 循环与 `PipelineMaxComponentRuns` / BM25 与 Embedding 检索器 / `DocumentWriter` 输出键 / `to_dict()` 键集——全部在本机真实引擎上跑出（有真实报错原文为证）。本节不借鉴文档印象，凡可复现行为都对真实输出。
- **模型调用与真实 embedding = 非本机实测（预置仿真）**：探针里的 Generator/LLM 是作者写的桩组件（返回预定字符串、计数），Embedding 检索用**手工注入的定值向量**（0/1 分量）——这是为了保证零网络零随机、stdout 三进程 md5 恒一。**请勿把"手工向量的余弦排序"与时序上的真实 embedding 质量混淆**：真实嵌入需要真实模型（如 `sentence-transformers`），本机未跑，账 B 的"语义臂"只是接法示范（接法=真，向量=假）。
- **版本存续行 = 本机真 import 探测 + 版本节奏要素事实**：六个目标名是真实 `importlib` 探测（OK/FAIL 逐条真实）；但"1.x 曾长这样"、"pip 包名 haystack→haystack-ai" 的版本节奏 = 要素事实，写作环境无外网、未在线复核各版本文档，以官方 release 为准。**账户 C 的核心发现（顶层入口存活、nodes/pipelines 整包移除）是本机真实 FAIL 撑住的。**
- **账 D 选择树 = 作者按知识地图 §7.14 与 §7.1 整理（要素事实）**：8 场景决策规则为作者整理定性，非本机可运行断言；"08-RAG 七道工序→七组件"映射同为作者按组件命名与 §7.6 整理，组件本体部分在本机真实运行（A/B/C 账），映射关系本身是综述。
- **确定性纪律**：stdout md5 `a67c1a77…` 三个独立进程逐字节恒一（每次 `python code/notebooks/_tools/haystack_demo.py` 复现）；墙钟 ≈0.3 s 仅进 stderr；不打印随机对象 ID / 地址 / 哈希；排序保证哈希无关输出。

## 10. 参考与衔接

- **本页地图**：§7.6（Haystack：类型安全/组件化/可测试/企业级文档问答·比 LangChain 收敛规范·学习价值=严谨的 pipeline 思维）+ §7.1（类 1 通用编排 SDK + 类 3 RAG 框架双栖）+ §7.14（严谨生产级 RAG → Haystack / LlamaIndex）+ §17.5（编排层技术栈：Haystack（生产 RAG）与 LlamaIndex（RAG 专精）并列）。
- **选择树 8 场景**（探针断言 8/8，要素事实）：① 生产级多步 RAG → Haystack；② 两步小 RAG → Haystack；③ 单检索函数 → 不引入（直接 `retriever.run()`）；④ 多轮 Agent 循环/恢复 → LangGraph（02 章）；⑤ .NET 企业栈 → Semantic Kernel（00 章账 D 场景 9）；⑥ 非工程师 → Dify（04 章）；⑦ 快速 demo → 06 章手写；⑧ 四层互斥断言 → Haystack 属『通用编排+RAG 框架』双栖。
- **前承**：00-框架分类学（双栖点 + 账 B 标格次主次次·覆盖 4/4）；02-LangGraph（账 A 上限对偶：recursion_limit）；03-LlamaIndex（账 D 对象式 vs 管线组件式对照 + Keyword ASCII-only 中文坑）；04-低代码（逃离清单③ 的积极归处）；08-RAG 全 12 篇（工序语义源头，账 D 映射的母表）。
- **后启**：`06-Semantic-Kernel`（双栖外的企业 .NET 一格）；`07-AutoGen-AG2-Microsoft-Agent-Framework`（多 Agent 对话范式：AutoGen『互聊』→ AG2 1.0 函数式重写→MAF 官方线，真实 ag2 1.0.6 四账，账 D 场景①研究/原型默认入口）；`09-DSPy`（✅ 已交付 v0.36：质量敏感/评测主业对照——真实 dspy 2.6.27 四账，签名→模板编译产物 ×6.1 / Bootstrap 编译 4 调用 vs Labeled 0 调用 / 结构=签名定×内容=数据定）；`10-OpenAI-Agents-SDK`（✅ 已交付 v0.35：回归轻量的对照——Agent+Runner 一件套、手转交 §7.11 实测，真实 openai-agents 0.17.0 四账）；`12-继承关系与选型决策`（收束章：完整 DAG + "该不该引入"量化）。
- **对应里程碑**：`012`（02 章已交付）之后，双栖点 Fabrics 的这一格由本篇展开；里程碑 `013`（手写 MCP Server）已在 `11-MCP协议` 章交付（v0.37）。

> 本篇完工于 2026-09-22（v0.31 批次）；探针 `code/notebooks/_tools/haystack_demo.py`；stdout md5 `a67c1a77…`。
