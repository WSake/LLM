# 📇 LlamaIndex：把『数据→索引→检索→问答』标准化成对象（Document/Node/Index · 三种索引三种机制 · RAG 专精）

> 对应知识点：知识地图 §7.4（LlamaIndex，2022，**RAG 专精**——"任意数据源 → 可查询索引 → 检索 → 生成 标准化：Document/Node/Index/Retriever/QueryEngine；多种索引：vector / summary / keyword-table / tree 等；学习建议：如果你只做 RAG，LlamaIndex 的理解深度超过 LangChain"）、§17.5.7（**LlamaIndex**——"最成功的 RAG 框架，将数据摄取、索引、检索、问答全链路封装为对象"）。
> 前置：[07-应用框架 00-框架分类学](./00-框架分类学.md)（开篇章：账 A 类 3「RAG 框架」* / 账 A **双栖点** Haystack 之交 *：本篇=类 3 的代表展开）；[01-LangChain与历史包袱](./01-LangChain与历史包袱.md)（**同是数据侧买家**：LangChain 的检索只是其通用编排的组件之一，LlamaIndex 把「数据-流」整个吃干净——LangChain 检索组件变成它的子集）；[08-RAG体系](../08-RAG体系/02-核心流水线.md) 全部 12 篇实测（本篇账 D 的「手搓对照臂」：08 章自写 loader/切块/向量/索引/检索/生成/评测，LlamaIndex 的每一道工序都能找到 08-章那篇实测）。
> 动手：`python code/notebooks/_tools/llamaindex_demo.py`（**真实 llama-index-core 0.14.25 引擎**四账 + 选择树一键复现；零网络 · CountLLM 预置回复确定性，stdout md5 `cae6dffd…` 三独立进程恒一）。
> 一句话：**LlamaIndex 是 07 章里对 RAG 理解最深的框架（§7.4）——把『数据收取 → 分块 → 索引 → 检索 → 问答合成』每一步都做成了对象（Document/Node/Index/QueryEngine），你 08-章手搓的每一道工序，它都有一格封装；本文用真实引擎量四本账：数据管线、索引三种机制、合成 LLM 调用账、以及与 08-章工序的一一映射。**
>
> 📌 诚实边界：本篇 **A–D 四账全部来自本机真实 llama-index-core 0.14.25 引擎**（importlib.metadata 实测版本；Document→Node 切块、三索引构建/检索、RetrieverQueryEngine 合成、上下文护栏 ValueError，全部引擎真实运行结果，**不是演示纸面**）。**模型回复 = CountLLM 预置 canned 文本**（LLM 非本机实测，换取零网络 + 逐字节可复现；探针只量『调用了多少次 LLM、检索集是什么』，不量模型品质）；**embedding = TokenHashEmbedding 词袋哈希代理**（256 维桶哈希，余弦=词面近似的可复现代理，真实 embedding 的语义性=非本机实测）；**关键词 = 领域词表-最长匹配子类覆盖**（真实默认的 Llama/LLM 关键词抽取=非本机实测）；**2025 方向（Workflows/QueryPipeline/Agent/LlamaCloud）= 要素事实**（无外网，未在线复核）。stdout md5 `cae6dffd…` 三个独立进程逐字节恒一；墙钟 ≈20 s 仅进 stderr。

---

## 📑 本章目录

1. [为什么 LlamaIndex 值得单独讲一课？](#0-为什么-llamaindex-值得单独讲一课)
2. [先钉死事实：LlamaIndex 贡献了什么（§7.4 / §17.5.7）](#1-先钉死事实llamaindex-贡献了什么7471757)
3. [把「RAG 标准化」降维成本机测量（探针设计）](#2-把rag-标准化降维成本机测量探针设计)
4. [实验 A · 数据管线账（Document→Node 两级：分块粒度 knob · 元数据追溯）](#3-实验-a--数据管线账documentnode-两级分块粒度-knob--元数据追溯)
5. [实验 B · 索引语义账（同一语料三种索引 = 三种检索机制指纹）](#4-实验-b--索引语义账同一语料三种索引--三种检索机制指纹)
6. [实验 C · 检索-问答账（合成策略的 LLM 调用账 + 小上下文护栏）](#5-实验-c--检索-问答账合成策略的-llm-调用账--小上下文护栏)
7. [实验 D · 组件映射账（08-RAG 手搓工序 → LlamaIndex 组件 + 8 场景选择树）](#6-实验-d--组件映射账08-rag-手搓工序--llamaindex-组件--8-场景选择树)
8. [拿这张表怎么读本目录（07 章 13 篇的路牌）](#7-拿这张表怎么读本目录07-章-13-篇的路牌)
9. [常见坑（6 个）](#8-常见坑6-个)
10. [诚实边界（AAA 自我审查）](#9-诚实边界aaa-自我审查)
11. [参考与衔接](#10-参考与衔接)

---

## 0. 为什么 LlamaIndex 值得单独讲一课？

07 章的路牌上，00-框架分类学把它分在**账 A 类 3「RAG 框架」**，代表三家里与 Haystack/RAGFlow 并列的位置；谱系账里它是**账 C 受 LangChain '22 影响的四家之一**（LangChain→LlamaIndex 直接后代）。这两个坐标都指向一个独特之处：**LangChain 是「通用编排 + 顺手做检索」，LlamaIndex 是「只做 RAG、把它做深」**。

§7.4 的一句学习建议成了本课的引子：**"如果你只做 RAG，LlamaIndex 对 RAG 的理解深度超过 LangChain。"** 它为什么有资格这么说？因为 LlamaIndex 的选择是**把 RAG 的每一道工序都做成一等公民对象**：

- **Document**（文本源）→ **Node**（可检索块）→ **Index**（可查询结构）→ **Retriever**（取回）→ **QueryEngine**（合成问答），五样东西每个都有独立的类、可替换的实现、可单独拿出去用；
- **索引不止一种**：vector / summary / keyword-table / tree，同一个 `as_retriever()` 接口背后是完全不同的检索机制；
- **检索与生成之间**，还有 `response_mode`（compact/refine/accumulate）这一层的合成策略旋钮。

这跟 01-章 LangChain 的一句话对照正好互为镜像：LangChain 的贡献是「把每一种角色统一成 Runnable」，LlamaIndex 的贡献是「把 RAG 每一道工序做成对象」——前者横着铺开，后者竖着吃深。

而 08-章我们刚手搓过完整的 RAG 流水线（12 篇实测在案）：loader / 分块 / 向量化 / 索引 / 检索 / 压缩 / 生成 / 评测。**本篇就是"把 08 章手搓的骨架子，换到 LlamaIndex 引擎里再跑一遍"**——账 D 给出逐工序映射表，一栏 08-章自写实现、一栏 LlamaIndex 组件。

## 1. 先钉死事实：LlamaIndex 贡献了什么（§7.4 / §17.5.7）

§17.5.7 用一句话盖章它的地位：**"最成功的 RAG 框架，将数据摄取、索引、检索、问答全链路封装为对象。"** 这句话可以拆成三件确定的事实，本篇每件给一手数字：

1. **两级数据对象 = 追溯性**：`Document` 是整篇文本源（可带 `metadata`），`Node` 是它切出来的可检索块，每块带着 `ref_doc_id` 指回源文档。账 A 实测量：6 篇文档切成 12 块，`ref_doc_id` 12/12 非空——**"检索到哪一块、自动知道它来自哪一篇"**是框架白送的，不是你自己拼字符串拼出来的。
2. **多索引 = 机制是旋钮**：vector / summary / keyword-table 不是"三个 API"，是**三种检索哲学的落地**。账 B 用同一份语料、同一批查询同时跑三个索引，让「谁被召回、按什么序、带多少分」显出三套指纹：向量近似全检（打分排序）、摘要遍历全给（保真不减枝）、关键词词表精确（建表键之外一律漏）。
3. **检索与生成解耦 = LLM 调用账透明**：`RetrieverQueryEngine` 的 `response_mode` 决定检索集如何变成回答——compact 整包打包一次、refine 逐块精修 K 次、accumulate 逐块累积 K 次。账 C 用计数 LLM 实测：**同样一问，LLM 调用次数是 1 还是 K，一个旋钮说了算**。

再加上 §7.4 明确的两条边界（本文诚实边界也挂在这）：LlamaIndex **不解决概念引入（Prompt/上下文/工具）**——那归 06 章；**不当通用编排**——Agentic 那部分（Workflows/QueryPipeline/Agent 组件）是 2025 方向、要素事实，不是本篇引擎实测。

## 2. 把「RAG 标准化」降维成本机测量（探针设计）

一张"LlamaIndex 很好用"的口水文章不稀奇；稀奇的是把**"它到底把 RAG 哪几道工序做成了对象、三种索引机制差在哪、合成策略的 LLM 调用账"变成可复排的本机实验**。探针用**真实 llama-index-core 0.14.25 引擎**做了 4 件确定性的测量（模型全部用 `CountLLM` 预置回复→零网络、零随机、逐字节可复现）：

- **A 数据管线账**：6 篇客服手册文档 `Document`（带 metadata）→ `SimpleNodeParser(chunk_size=48, chunk_overlap=8)` → 数 Node 数、`ref_doc_id` 追溯、metadata 是否从 Document 注入 Node；再扫 chunk_size=24/48/96 看节点数怎么变；
- **B 索引语义账**：同一语料同时建 `VectorStoreIndex`（embedding=词袋哈希）、`SummaryIndex`、`KeywordTableIndex`，同一批 4 个查询各打一遍，比"返回指纹"；
- **C 检索-问答账**：`RetrieverQueryEngine` 的三种 `response_mode`（compact/refine/accumulate）各自触发几次 LLM 调用（`CountLLM` 计数器实测）；再主动把 `context_window` 调到装不下 prompt+检索集，接住引擎的护栏报错；
- **D 组件映射账**：把 08-章七道手搓工序逐条映射到 LlamaIndex 组件，数"从零到回答"的最少真码行数，再给出 8 场景选择树。

确定性纪律：`warnings.filterwarnings("ignore")` 最先执行（**stderr 纯净**，不污染 md5）；**引擎内置一个"护栏"会用裸 `print` 打到 stdout**（SentenceSplitter 遇 `chunk_size − metadata 后 < 50 token` 时打印 `Metadata length (5) is close to chunk size (48)…`）——探针用 `silent_parse()` 上下文管理器临时接管 stdout 捕获它、数清行数、收编为一句确定性的定台词（**告警字面=框架原样，未删改**，既保诚实又保复现）；三进程 set 输出全部排序。**stdout md5 `cae6dffd…` 三个独立进程逐字节恒一**，墙钟 ≈20 s（llama-index-core 导入 + 四账构建的固有开销）仅进 stderr。

复现实录（严格逐字节 = 探针 stdout 三遍恒一）：

<details>
<summary>📊 探针 stdout 全量实录（md5 <code>cae6dffd…</code> · 墙钟仅 stderr）</summary>

````text
========================================================================
llamaindex_demo：07-应用框架 · 03-LlamaIndex（知识地图 §7.4 / §17.5.7 · llama-index-core 0.14.25 引擎）
    『RAG 专精：把『数据→索引→检索→问答』标准化』——Document/Node/Index/QueryEngine
========================================================================
[0] 口径：真实 llama-index-core 引擎（importlib.metadata 实测）；模型=CountLLM 预置（LLM 非本机实测）；embedding=TokenHashEmbedding 词袋哈希（语义=非本机实测·余弦=词面近似代理）；关键词=领域词表-最长匹配子类覆盖（真实默认 LLM）；零网络·零 RNG
========================================================================
[账 A] 数据管线账：Document→Node 两级 + 分块粒度 knob（挂靠 08-04 分块与元数据）
========================================================================
  6 篇手册文档 → SimpleNodeParser(chunk_size=48, chunk_overlap=8) → 12 个 Node（每篇 2 块）
    | n0  | d1-阅读器    | 显示器与显示  | 产品 QX10 是一款支持夜间模…
    | n1  | d1-阅读器    | 显示器与显示  | 开启方式：设置-显示-夜间模式。…
    | n2  | d2-电池     | 电源与电池   | QX10 电池续航约八小时。快充…
    | n3  | d2-电池     | 电源与电池   | 更多信息见充电安全。…
    | n4  | d3-账户     | 账户与订单   | 账户充值支持支付宝与微信付款。退…
    | n5  | d3-账户     | 账户与订单   | 详细规则见退换货条款。…
    | n6  | d4-保修     | 服务与保修   | 产品保修期为一年。显示屏与电池享…
    | n7  | d4-保修     | 服务与保修   | 常见问题见故障排查。…
    | n8  | d5-应用     | 应用与同步   | 手机端 App 支持离线阅读。网…
    | n9  | d5-应用     | 应用与同步   | 更多说明见常见问题。…
    | n10 | d6-客服     | 服务与保障   | 客服工作时间为早九点到晚六点。联…
    | n11 | d6-客服     | 服务与保障   | 紧急问题优先处理。处理时效见服务…
  → 两级：Document=文本源（整篇可追溯）· Node=可检索块（切块后逐块可检索）
  → 追溯：ref_doc_id 非空 12/12（每块都能回到源文档）· metadata 携带 {doc,label} 由 Document 注入 Node
  → 框架护栏：SentenceSplitter 遇 chunk_size−metadata 后 <50 token 会裸 print 提示（本步捕获 6 行，字面=『Metadata length (5) is close to chunk size (48)…』——真实框架原样，未删改，收编此句保证复现输出逐字节恒定）
  → 分块粒度 knob（chunk_size=24/48/96 → 节点数 30/12/6）：粒度越细节点越多=召回粒度越细、上下文包越重（08-04 同一本账）；护栏再捕获 12 行
  → 断言：节点=12（每篇 2 块）· 追溯=12/12 · 粒度扫描=[30, 12, 6]

========================================================================
[账 B] 索引语义账：同一语料三种索引 = 三种检索机制指纹（§7.4 多种索引）
========================================================================
  查询集 Q1..Q4；每索引各打一遍，『返回指纹』=谁在、按什么序、带多少分
  ―― 向量索引 VectorStoreIndex（分数=词袋-余弦代理：查询向量 × 每块向量 打分全检）
    Q「QX10 夜间模式怎么开」→ d1-阅读器(0.4196) · d1-阅读器(0.2835)
    Q「退款的规则是什么」→ d3-账户(0.2887) · d5-应用(0.2227)
    Q「电池续航八小时」→ d2-电池(0.677) · d2-电池(0.4606)
    Q「客户几点上班」→ d2-电池(0.2843) · d6-客服(0.2641)
  → 撞桶示警：d2 尾块「更多信息见充电安全。」对 Q4 余弦=0.1361，但两者零词面重叠——256 维桶哈希把无关汉字撞进同坐标（本语料共享坐标 114/125），尘分 0.05~0.28 只配当提示、不配当语义；真命中档 0.42~0.68 与尘分有明确落差
  ―― 摘要索引 SummaryIndex（顺序遍历全给：无剪枝·保真）
    Q「QX10 夜间模式怎么开」→ 12/12 块全给（顺序首块=d1-阅读器·score=1.0）
    Q「退款的规则是什么」→ 12/12 块全给（顺序首块=d1-阅读器·score=1.0）
    Q「电池续航八小时」→ 12/12 块全给（顺序首块=d1-阅读器·score=1.0）
    Q「客户几点上班」→ 12/12 块全给（顺序首块=d1-阅读器·score=1.0）
  ―― 关键词索引 DeterministicKeywordTable（词表精确：查询词必须是建表键→再查表）
    Q「QX10 夜间模式怎么开」→ keys=[QX10,夜间模式] → d1-阅读器×2 · d2-电池×1
    Q「退款的规则是什么」→ keys=[规则] → d3-账户×1
    Q「电池续航八小时」→ keys=[八小时,电池,续航] → d1-阅读器×1 · d2-电池×2 · d4-保修×1
    Q「客户几点上班」→ keys=[空] → 无命中（查询词不在任何建表键=词面严格漏）
  → 结论：Vector=近似全检（默认给全库打分排序，但词袋代理带撞桶尘分）· Summary=遍历全给（保真不减枝）· Keyword=词表精确（建表键之外一律漏，词面一差就 0）
  → 断言：Q1 向量 top1=d1-阅读器 · Summary 全给 12/12 · Keyword Q1 有键命中、Q4 键空=0 命中

========================================================================
[账 C] 检索-问答账：RetrieverQueryEngine 合成策略的 LLM 调用账 + 小上下文护栏
========================================================================
  向量检索 top_k=2 → 三种合成策略（同一查询、同一检索集，只换 response_mode）
    mode=compact    llm_calls=1  sources=['d1-阅读器']  resp='根据检索到的资料，可以给出答案。'
    mode=refine     llm_calls=2  sources=['d1-阅读器']  resp='根据检索到的资料，可以给出答案。'
    mode=accumulate llm_calls=2  sources=['d1-阅读器']  resp='Response 1: 根据检索'
  摘要索引（检索集=全量 12 块）→ 同样三种策略（合成成本随检索集线性涨）
    mode=compact    llm_calls=1  sources=全量 6 组
    mode=refine     llm_calls=12  sources=全量 6 组
    mode=accumulate llm_calls=12  sources=全量 6 组
  → 合成成本账：compact=整包一次打包合成（1 次 LLM）· refine=逐块收发精修（K 次）· accumulate=逐块累积拼接（K 次）——换『忠实/省额』换调用数（挂靠 08-06 上下文压缩、06-03 组合策略）
  → 框架护栏：context_window 设小到装不下 prompt+检索集 → 内置报错（类比 02-章 recursion_limit）
      护栏实测：ValueError: Calculated available context size -64 was not non-ne…
  → 断言：compact=1 / refine=K / accumulate=K（K=检索块数）· 摘要全量 K=12 · 护栏=ValueError

========================================================================
[账 D] 组件映射账：LlamaIndex 把 08-RAG 手搓流水线封装成对象 + 8 场景选择树（该在哪儿用）
========================================================================
  08-RAG 章手搓工序 → LlamaIndex 组件（同一道工序，对象化）
    文档读取/解析     → SimpleDirectoryReader / Document       | 手搓≈08-章自写 loader / 纯文本读
    分块与元数据      → SimpleNodeParser + Document.metadata   | 手搓≈08-04 chunk_demo：字符串切块+标签
    embedding 化 → Settings.embed_model + VectorStoreIndex | 手搓≈08-01/08-02：自建 TF/哈希向量
    索引构建        → Vector/Summary/KeywordTableIndex       | 手搓≈08-02 手工索引矩阵
    检索          → Index.as_retriever(top_k / 模式)         | 手搓≈08-05 查询侧 top_k/混合
    生成合成        → RetrieverQueryEngine.response_mode     | 手搓≈08-06 压缩/重排后提示词拼装
    评测          → RAGAS 集成                               | 手搓≈08-07 RAGAS 手测
  → 最小闭环行数：6 行真码 = Settings 注入 2 + from_documents / as_retriever / as_query_engine / query 4（08-章手搓同名闭环需自写每道工序）
  → 选择树 8 场景断言 8/8：RAG 专项首选 LlamaIndex（§7.4 学习建议）
     1. 要完整 RAG 起步·文档多格式
       -> LlamaIndex（Document/Node/Index/Engine 全封装，本页全账）
     2. 只要分块/元数据管线
       -> SimpleNodeParser + Document.metadata（账 A；08-04 落点）
     3. 只要检索能力
       -> Index.as_retriever 组件（账 B；返回 NodeWithScore）
     4. 只要生成问答
       -> RetrieverQueryEngine（账 C；合成策略换调用账）
     5. 混合检索·关键词补充
       -> Vector + Keyword/BM25（账 B 关键词=词面精确臂；08-03 混合）
     6. Agentic RAG / 编排
       -> Workflows/QueryPipeline + Agent 组件（2025 方向=要素事实）
     7. 评测与可观测
       -> RAGAS / Langtrace（08-07 集成=要素事实·非本机实测）
     8. 私有/多样数据源
       -> LlamaHub / LlamaCloud（要素事实·非本机实测）

========================================================================
台账汇总（A 数据管线 / B 索引机制 / C 合成调用 / D 组件映射）
  A 6 文档 → 12 节点（每篇 2 块）· ref_doc_id 追溯 12/12 · 分块 knob 24/48/96 → 节点数递减梯队
  B Vector=近似全检（Q1 top1 d1-阅读器；撞桶尘分示警）· Summary=遍历全给 12/12 
    · Keyword=词表精确（Q1/Q3 命中纯正，Q4 键空=0 命中=词面严格漏）
  C compact=1 · refine/K=K（K=检索块数；摘要全量 K=12）· 小上下文触发 ValueError 护栏
  D 08-RAG 七道工序 → 七个组件 · 最小闭环 6 行真码 · 选择树 8 场景断言 8/8
一句话：LlamaIndex 把『数据→索引→检索→问答』标准化成对象——RAG 专项首选，
对 RAG 的理解深度超过 LangChain（§7.4）；手搓工序可对照 08-RAG 逐行搬家（账 D）
done · 一键复现：python code/notebooks/_tools/llamaindex_demo.py
````

</details>

## 3. 实验 A · 数据管线账（Document→Node 两级：分块粒度 knob · 元数据追溯）

**目标**：把"把数据摄取与分块做成对象"降成三条可复排的事实——① 6 篇文档切成几块、怎么排；② 每一块能不能追溯回源文档（`ref_doc_id`）；③ `chunk_size` 这个 knob 一动，节点数怎么变。

**过程**：真实 llama-index-core 里 6 篇客服手册 `Document`（每篇带 `{doc, label}` metadata）喂给 `SimpleNodeParser(chunk_size=48, chunk_overlap=8)`；逐个 Node 打印来源 doc/label/前 16 字；统计 `ref_doc_id`；再扫 `chunk_size=24/48/96` 三档看节点总数。框架的 SentenceSplitter 护栏（`chunk_size − metadata 后 < 50 token` 时裸 print）用 `silent_parse()` 捕获、数行。

**实测**：

```
  6 篇手册文档 → SimpleNodeParser(chunk_size=48, chunk_overlap=8) → 12 个 Node（每篇 2 块）
  → 追溯：ref_doc_id 非空 12/12（每块都能回到源文档）· metadata 携带 {doc,label} 由 Document 注入 Node
  → 框架护栏：SentenceSplitter 遇 chunk_size−metadata 后 <50 token 会裸 print 提示（本步捕获 6 行…）
  → 分块粒度 knob（chunk_size=24/48/96 → 节点数 30/12/6）…
```

**结论**：

1. **`Document`→`Node` 两级是 LlamaIndex 的根基**：文本源与可检索块是两个对象。源文档永远整篇可追溯（`Document`），切出来的每一块才是真正进索引的东西（`Node`）——这正好就是 08-04 分块章说的"分块是召回粒度与上下文包的权衡"在框架里的落地形态。
2. **`ref_doc_id` 追溯是框架白送的**：12/12 非空，不用你手动维护"这块来自哪篇"。RAG 生产上"引用来源"这个需求，在对象模型里是构造即自带的，不是事后凑的字符串。
3. **`chunk_size` 是第一个旋钮**：24→30 块、48→12 块、96→6 块。同一个 knob 在 08-04 里我们用字符串切片自己转，在 LlamaIndex 里是一个构造参数——**"分块粒度"从代码变成了配置**。这也是账 D 挂靠 08-04 的落点。
4. **框架护栏真实存在且会裸 print**：SentenceSplitter 在 `metadata 长度接近 chunk_size` 时打印 `Metadata length (5) is close to chunk size (48)…`，探针实测捕获 6 行（本账）+ 12 行（粒度扫描）——这是**引擎级放行但工程级的提示**（metadata 会占 chunk 预算），说明框架知道这是一个坑、但选择用 print 提醒而非报错。诚实收编，字面未删改。

## 4. 实验 B · 索引语义账（同一语料三种索引 = 三种检索机制指纹）

**目标**：把 §7.4 那句"多种索引"落成可复排的对照——同一份语料、同一批 4 个查询，vector / summary / keyword-table 三个索引各自返回什么、按什么序、带多少分。三种机制三套指纹，一目了然。

**过程**：同一 6 篇文档建三个索引（`VectorStoreIndex` 用 TokenHashEmbedding 词袋哈希、`SummaryIndex` 顺序遍历、`KeywordTableIndex` 子类用领域词表-最长匹配抽取）；4 个查询（夜间模式 / 退款规则 / 电池续航 / 客服时间）各打一遍。

**实测**：

```
  ―― 向量索引 VectorStoreIndex（分数=词袋-余弦代理：查询向量 × 每块向量 打分全检）
    Q「QX10 夜间模式怎么开」→ d1-阅读器(0.4196) · d1-阅读器(0.2835)
    Q「退款的规则是什么」→ d3-账户(0.2887) · d5-应用(0.2227)
    Q「电池续航八小时」→ d2-电池(0.677) · d2-电池(0.4606)
    Q「客户几点上班」→ d2-电池(0.2843) · d6-客服(0.2641)
  ―― 摘要索引 SummaryIndex（顺序遍历全给：无剪枝·保真）
    Q「…」→ 12/12 块全给（顺序首块=d1-阅读器·score=1.0）   ← 四个查询全部同款
  ―― 关键词索引 DeterministicKeywordTable（词表精确：查询词必须是建表键→再查表）
    Q「QX10 夜间模式怎么开」→ keys=[QX10,夜间模式] → d1-阅读器×2 · d2-电池×1
    Q「退款的规则是什么」→ keys=[规则] → d3-账户×1
    Q「电池续航八小时」→ keys=[八小时,电池,续航] → d1-阅读器×1 · d2-电池×2 · d4-保修×1
    Q「客户几点上班」→ keys=[空] → 无命中（查询词不在任何建表键=词面严格漏）
```

**结论**：

1. **Vector = 近似全检**：默认把全库每块都算分、排序取 top（top_k 默认 2 即打两行）。分数能分辨"谁更贴"：Q3 词面重叠最大（0.677/0.4606）、Q4 几乎无重叠（0.2843/0.2641）。**但它不是语义真值**：词袋代理下 Q4「客户几点上班」把 d2-电池顶到 top1——08-01/08-02 的 embedding 章说的"词面近似天花板"在这里亮了一次。
2. **撞桶尘分 = 词袋代理的诚实示警**：探针专门抽了 d2 的尾块「更多信息见充电安全。」与 Q4 比余弦 = **0.1361**，两者**零词面重叠**——256 维桶哈希把无关汉字撞进同坐标（本语料共享坐标 114/125）。这类尘分 0.05~0.28 只配当提示、不配当语义；真命中档 0.42~0.68 与它有明确落差。**这是嵌入代理的坑在引擎里的直观演示：真实 embedding 不会这样，但"余弦高≠语义近"的教训通用。**
3. **Summary = 遍历全给**：四个查询全返回 12/12 块、score 全 1.0、顺序首块恒为 d1——摘要索引不检索、它**把全语料原样打包交给生成**（保真不减枝）。账 C 会看到它合成时 LLM 调用数线性膨胀（12 次）。
4. **Keyword = 词表精确**：查询词必须是建表键→再查倒排表。词面一差就 0：Q4「客户几点上班」抽出的键为空（语料服务承诺里没有现成词）→ **无命中**。这是三机制里最"严格"的——精确是它的优点（Q1/Q3 命中纯正），词面刚性是它的代价（混合检索里它当补充臂正合适，08-03）。

## 5. 实验 C · 检索-问答账（合成策略的 LLM 调用账 + 小上下文护栏）

**目标**：把"检索与生成解耦"这个设计决定量成 LLM 调用账——同一个检索集，`response_mode` 换成三种，`CountLLM` 每次 `complete` 都记账，数调用次数；再把 `context_window` 调到装不下，看引擎怎么兜。

**过程**：同一个 `RetrieverQueryEngine`，`similarity_top_k=2`，分别以 compact / refine / accumulate 跑同一问；再对摘要索引（检索集=全量 12 块）跑同样的三种。护栏侧：`context_window=40, num_output=20` 触发引擎的上下文预算检查。

**实测**：

```
  向量检索 top_k=2：  compact=1 · refine=2 · accumulate=2
  摘要索引（12 块）：  compact=1 · refine=12 · accumulate=12
  护栏：context_window=40 → ValueError: Calculated available context size -64 was not non-ne…
```

**结论**：

1. **合成策略 = LLM 调用数旋钮**：compact 整包一次打包合成（1 次）；refine 逐块收发精修（K 次）；accumulate 逐块累积拼接（K 次）。**换策略不换检索结果，只换调用数与忠实/省额**——检索集 K 越大，refine/accumulate 的线性支出越明显（摘要全量 K=12 → 12 次）。
2. **成本随检索集线性涨 = 08-06 上下文压缩的同源问题**：摘要索引全给 12 块时，refine/accumulate 要 12 次 LLM——这正是 08-06"检索多用 vs 生成用精"张力的引擎版：**索引给全 → 生成侧替你分担成本**。compact 在 12 块下仍 1 次（整包打包）但 prompt 也会超长——于是引擎干脆加护栏。
3. **护栏 = 类比 02-章 recursion_limit 的框架底线**：`context_window=40` 装不下"prompt + 12 块检索集"，引擎抛 `ValueError: Calculated available context size -64 was not non-ne…`（剩余上下文 -64）。**框架宁可显式报错，不让你静默截断**——这是"显式失败优于隐性错误"的工程守则，和 LangGraph 的 `recursion_limit` 同一思路。
4. **LLM 调用数、非模型输出，是本账的量纲**：`resp='根据检索到的资料，可以给出答案。'` 是 CountLLM 的 canned reply（诚实边界）——本账要量的是"框架在模型前后替你做多少事"，不是模型本身。

## 6. 实验 D · 组件映射账（08-RAG 手搓工序 → LlamaIndex 组件 + 8 场景选择树）

**目标**：把本篇和 08-RAG 章焊死——LlamaIndex 的五个对象（Document/Node/Index/Retriever/QueryEngine）逐条对上 08 章手搓的七道工序；并数清"从零到第一次回答"最少几行真码；最后给出 8 场景选择树（该在哪儿用 §7.4 学习建议）。

**过程**：把 08-章工序清单（loader→分块→向量→索引→检索→生成→评测）与 LlamaIndex 组件逐条映射成表；数最小闭环样板码的行数（不算注释/import）；8 场景决策断言。

**实测**：

```
  08-RAG 章手搓工序 → LlamaIndex 组件（同一道工序，对象化）
    文档读取/解析     → SimpleDirectoryReader / Document
    分块与元数据      → SimpleNodeParser + Document.metadata
    embedding 化 → Settings.embed_model + VectorStoreIndex
    索引构建        → Vector/Summary/KeywordTableIndex
    检索          → Index.as_retriever(top_k / 模式)
    生成合成        → RetrieverQueryEngine.response_mode
    评测          → RAGAS 集成
  → 最小闭环行数：6 行真码 = Settings 注入 2 + from_documents / as_retriever / as_query_engine / query 4
```

**结论**：

1. **七道工序逐条可搬家**：08 章手搓的 loader（08 系列/核心流水线）、分块（08-04）、向量（08-01/02）、索引（08-02 手工矩阵）、检索（08-05）、生成拼装（08-06 压缩重排后提示词）、评测（08-07 RAGAS）——LlamaIndex 每个都有同名同职责的组件。**学 08-章的手搓，就是为了看穿这些封装**（账 A-D 正是拆封装的过程）。
2. **最小闭环只有 6 行真码**：`Settings` 注入节点解析器 + embedding 模型（2 行），`from_documents`（建索引）→ `as_retriever` → `as_query_engine` → `query`（4 行）。08-章同名手搓闭环要自写每一道工序——这就是"数据→索引→检索→问答"被标准化的行数差。
3. **8 场景选择树**：RAG 起步/只要分块/只要检索/只要生成/混合检索/Agentic RAG/评测观测/私有数据源——探针断言 8/8。核心判据：**RAG 专项先落 LlamaIndex（§7.4）**；当需求越过"知识库问答"进入流程编排/多 Agent，才考虑 Workflows/QueryPipeline 或把编排交给图引擎（02 章）。
4. **边界诚实交付**：场景 6/7/8 的 Workflows/QueryPipeline/Agent/LlamaHub/LlamaCloud/RAGAS 集成为**要素事实**（2025 方向，写作环境无外网未实测）——探针里引擎实测到的是账 A/B/C（Document/Node/Index/Retriever/QueryEngine + 三种索引 + 合成策略 + 护栏），决策层不在本机跑。

## 7. 拿这张表怎么读本目录（07 章 13 篇的路牌）

本页是 07 章"地图页"（00 篇）和"长子"（01 篇）之后的**第四站**：00 篇回答了"框架分几类"，01 篇回答了"通用编排家长子的概念与包袱"，本页回答了"RAG 专精这格为什么值得单独讲"。后文各篇的路牌（按依赖次序）：

| 后文篇 | 读它的理由（对应本页哪一格） | 状态 |
|---|---|---|
| `00-框架分类学` | 账 A 类 3：RAG 框架代表（LlamaIndex/Haystack/RAGFlow） | 🔥 已交付（开篇章） |
| `01-LangChain与历史包袱` | 账 C 承袭链父节点：LangChain 检索组件=LlamaIndex 子集 | 🔥 已交付（v0.28） |
| `02-LangGraph-状态图与Checkpoint` | 场景 6 的图引擎臂：RAG 之上的 Agentic 编排 | 🔥 已交付（里程碑 012） |
| `03-LlamaIndex`（本篇） | 账 A 类 3 / 双栖点交集：RAG 专精（接 08 章 12 篇实测） | 🔥 已交付（本文） |
| `04-低代码` | 账 B"四痛点全主"的平台（Dify/Coze/n8n） | 🔥 已交付（v0.30） |
| `05-Haystack` | 双栖点别二：生产级 RAG 管线的 pipeline 思维 | 🔥 已交付（v0.31） |
| `06-Semantic-Kernel` | 账 D 场景 .NET/微软企业栈 | ⬜ |
| `07-AutoGen·AG2` · `08-CrewAI` | 账 A 类 4、账 B 状态主业：多 Agent 对话 vs 角色化 | ⬜ |
| `09-DSPy` | "评测主业/状态不碰"、无承袭边独立路径 | ⬜ |
| `10-OpenAI-Agents-SDK` | 账 D 场景 6：轻量运行时"回归轻量"先例 | ⬜ |
| `11-MCP协议` | 账 B"覆盖 1 维"特判、账 C"2025 公共底座"、里程碑 `013` | ⬜ |
| `12-继承关系与选型决策` | 收束章：完整 DAG + "该不该引入"量化，00 篇 C 账扩版 | ⬜ |

读法口诀（本页的一页带走）：**遇到"知识库问答"先落 LlamaIndex——把 08 章手搓的工序名在脑子里过一遍，LlamaIndex 的每个对象都是那一道工序的封装（账 D）；要换机制（近似/保真/精确）就在索引层换（账 B）；要省 LLM 调用就在合成层换（账 C）；越过问答进入编排，才下车换图引擎（02 章）/低代码/Haystack。**

## 8. 常见坑（6 个）

1. **把"索引"当成一种东西**（本课第一个坑）：vector / summary / keyword-table 是三种检索哲学，召回机制完全不同（账 B）——"我建了索引"不等于"我建了向量索引"；小语料/强词面场景关键词索引更省（K 次 LLM vs 全库向量），别默认全上 vector。
2. **忽略 metadata 会占 chunk 预算**：SentenceSplitter 在 `chunk_size − metadata < 50 token` 时会裸 print 警告（账 A 实测捕获）——`metadata` 是注入 Node 的内容，会稀释每个 chunk 的正文预算；护栏提示真实存在，收编不等于它不需要处理。
3. **分块粒度当默认配置不调**：`chunk_size=48` 时 6 篇文档出 12 块、`96` 时出 6 块（账 A 扫表）——粒度决定召回粒度与上下文包轻重（08-04 同一本账），LlamaIndex 只是把 knob 变成了参数，账还得自己会算。
4. **让中文查询死在关键词索引上**：默认 `KeywordTableSimpleRetriever` 的 `_get_keywords` 走 `simple_extract_keywords`（**只认 ASCII 词，中文查询→空关键词=0 命中**，0.14.25 真实内置行为）——探针必须子类覆盖检索器+索引才让中文键跑通；真实生产换 LLM 抽取或 BM25 词表，别信默认。
5. **把合成策略当查询细节**：`response_mode` 直接决定 LLM 调用数与忠实/省额（账 C：compact=1 · refine/accumulate=K；摘要全量 K=12）——检索集大还开 refine 就是 12 次 LLM；先用 compact 跑通，按 08-06 压缩后才能放心放宽 K。
6. **撞桶尘分当语义信号**：词袋/哈希 embedding 下有 0.05~0.28 的"零词面重叠却非零余弦"（账 B 实测 0.1361）——余弦高≠语义近（08-01/08-02 的天花板教训）；用真实 embedding 时也记得设分数阈值过滤尘分，别把 top-k 无条件当真。

## 9. 诚实边界（AAA 自我审查）

- **引擎本体 = 本机真实运行**：`llama-index-core 0.14.25` 由 importlib.metadata 实测，A–D 四账的 **Document→Node 切块、ref_doc_id 追溯、分块扫描、三索引构建/检索返回指纹、RetrieverQueryEngine 合成 LLM 调用计数、ValueError 上下文护栏**全部来自引擎真实运行结果，不是演示纸面。
- **模型 = CountLLM 预置（LLM 非本机实测）**：所有"模型输出"（canned 回复字面量）都是预置的；探针只量"框架在模型前后替你做多少事"（调用次数/检索集/护栏），不量模型品质。**请勿据此推断任何真实 LLM 在 LlamaIndex 上的回答质量。**
- **embedding = TokenHashEmbedding 词袋哈希代理（语义=非本机实测）**：余弦=词面近似的可复现代理；真实 embedding 的语义性（同义/跨词面召回）未在本机运行。撞桶尘分（0.05~0.28 vs 真命中 0.42~0.68）正是"词袋代理"这一操作化边界的诚实演示。
- **关键词 = 领域词表-最长匹配子类覆盖**：真实默认的 Llama/LLM 关键词抽取=非本机实测（探针把两处默认都子类化成了确定性词表规则，建表/查询同源）；`simple_extract_keywords` 的 ASCII-only 行为=0.14.25 真实内置，作为坑 4 记录。
- **框架护栏收编 = 诚实保留**：SentenceSplitter 的裸 print（`Metadata length (5) is close to chunk size (48)…`）被临时重定向捕获、计数、收编为一句定台词——**字面=框架原样，未删改**；既有保证复现、又如实报告框架在边界上的真实行为。
- **2025 方向 = 要素事实**：Workflows/QueryPipeline/Agent 组件、LlamaHub/LlamaCloud、RAGAS/Langtrace 集成、§7.4 学习建议表述，写作环境无外网、未在线复核 → 要素事实非本机实测；探针实测止于账 A/B/C 的对象管线。
- **确定性纪律**：stdout md5 `cae6dffd…` 三个独立进程逐字节恒一；墙钟 ≈20 s 仅 stderr；不打印随机对象 ID/哈希。

## 10. 参考与衔接

- **本页地图**：§7.4（LlamaIndex：文档/节点/索引/检索/引擎 + 多种索引 + 学习建议"RAG 专精理解深度超过 LangChain"）+ §17.5.7（LlamaIndex：最成功的 RAG 框架，全链路封装为对象）+ §7.13（谱系：LangChain→LlamaIndex 直接后代）。
- **选择树 8 场景**（探针断言 8/8）：① 完整 RAG 起步/多格式 → LlamaIndex（本页全账）；② 只要分块/元数据 → SimpleNodeParser（账 A）；③ 只要检索 → as_retriever（账 B）；④ 只要生成 → RetrieverQueryEngine（账 C）；⑤ 混合检索补词面 → Vector + Keyword/BM25（账 B）；⑥ Agentic RAG/编排 → Workflows/QueryPipeline + Agent（要素事实）；⑦ 评测/可观测 → RAGAS/Langtrace；⑧ 私有/多样数据源 → LlamaHub/LlamaCloud。
- **前承**：00-框架分类学（账 A 类 3 定位）；01-LangChain（Runnable 检索组件=本页子集）；08-RAG 全 12 篇（账 D 手搓对照臂的挂靠面）。
- **后启**：`04-低代码`（✅ 已交付 v0.30：类 5 低代码平台——数据分析/无码不再是管线，是积木化+全兜底，账 C 四痛点专表铨实）；`05-Haystack`（✅ 已交付 v0.31：账 A 双栖点别二：生产级 RAG 管线的 pipeline 思维，与本文对照读——对象式 vs 管线组件式；真实 haystack-ai 3.1.1 引擎四账实测）；`09-DSPy`（Prompt 编程）；`12-继承关系与选型决策`（收束章）。
- **对应里程碑**：`012`（带记忆重试客服，02 章已交付）之后，RAG 对象管线落 LlamaIndex（本篇）；下一里程碑 `013`（手写 MCP Server）仍在 `11-MCP协议` 章。

> 本篇完工于 2026-09-22（v0.29 批次）；探针 `code/notebooks/_tools/llamaindex_demo.py`；stdout md5 `cae6dffd…`。
