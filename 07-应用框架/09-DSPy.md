# 🎯 DSPy：Prompt 编程 + 自动化优化——「思想 > 工具」：签名把 prompt 当程序（结构=声明式契约·内容=数据定），编译把『调 prompt』变成『跑优化器』——一个签名一句话、几行真实提示词就是 6.1× 的编译产物，Bootstrap 用自己的全对轨迹喂少样本、metric 把关弃用

> 对应知识点：知识地图 §7.10（**DSPy（2023 斯坦福；位置：Prompt 编程与优化**）——核心=把 LLM 应用写成语义程序（**signature → module → optimizer**），用编译与自动优化（针对真实指标）替代手写 prompt，并内置 **bootstrapping**（用示例自动构造 few-shot）；哲学=把玄学 prompt 变成可评测可优化的对象；适用=对质量敏感、可自动化评测的任务（分类/抽取/转换），与 RAG/Agent 结合（DSPy 2.0 支持 ReAct 等）；学习价值=以"编译器"思路看 LLM 应用）、§7.1（类 **5 Prompt 编程/优化**：DSPy / Outlines——把 prompt 当程序、可编译可优化；00-框架分类学账 C 已实测 **DSPy 无承袭边独立**）、§7.14（决策树：「质量敏感型任务 → **DSPy**」）、§17.5.9（一句话=把 prompt 流程写成程序并用优化器自动调优；前置=评测；衍生=自动 few-shot、多模块编译；关系=质量敏感任务的编译器）。
> 前置：[00-框架分类学](./00-框架分类学.md)（账 C 继承影响：**DSPy 无承袭边独立**=『框架会消亡、概念永存』的可计算版本——本篇兑现该断言：它的编译产物是自足的签名模块，不依赖任何相邻框架的继承边）；[06-应用开发/02-Prompt-Engineering](../06-应用开发/02-Prompt-Engineering.md)（从"咒语"到科学：模板账本·few-shot 示例预算·CoT 成本——本篇把"手写 few-shot 挑示例"升级成编译期自动收集）；[06-应用开发/04-结构化输出](../06-应用开发/04-结构化输出.md)（JSON 合法≠Schema 合规——本篇 JSONAdapter 是引擎内置的第三个『解析器兜格式』现场：Chat 适配器首试失败→回退 JSON 适配器）；[06-应用开发/05-Function-Tool-Calling](../06-应用开发/05-Function-Tool-Calling.md)（工具入参 Schema——本篇 ReAct 是引擎内建 Agent 模块的导入面）；[07-应用框架/10-OpenAI-Agents-SDK与轻量运行时](./10-OpenAI-Agents-SDK与轻量运行时.md)（后启语已预告：本篇 stub 的『预置决策表』正是 DSPy 想用优化器自动逼近的目标——SDK 手写决策、DSPy 编译决策）；更早：[08-RAG体系/13-失败模式与修复](../08-RAG体系/13-失败模式与修复.md)（§8.14 诊断手册里的『生成 vs 筛选』账：DSPy 的 bootstrap 同源——先让模型生成轨迹再按 metric 决定采纳与否）。
> 动手：`python code/notebooks/_tools/dspy_demo.py`（**真实 dspy 2.6.27 引擎**本机执行；模型决策=预置 stub（继承真 `dspy.clients.lm.LM` 重写 `__call__`，读提示词格式指令照吟+词面规则给答案）保确定性；零网络 · 零随机 · stdout md5 `c45b24fd…` 三个独立进程逐字节恒一）。
> 一句话：**DSPy = 「Prompt 编程 + 自动化优化」——思想大于工具**：你不写提示词、你写签名（一句话声明输入输出契约），引擎把它编译成一页真实的机器提示词（一句话 27 tok → 编译产物 166 tok，×6.1）；你不挑 few-shot 示例、编译期 optimizer 替你挑（Bootstrap=用自己的全对轨迹喂 demos、metric 把关弃用——4 个训练例跑 4 次学出 4 条 full-traces；Labeled=纯直抄 0 次调用）；你不拼长链、把模块当函数组合（ChainOfThought 自动多一个推理字段、双模块链=2 次调用、数据在模块间按字段流转）——结构粘死在签名上、内容活在数据里，换数据不换签名=只换 demo 不动骨架。质量敏感型任务的编译器。
>
> 📌 诚实边界：**引擎语义 = 本机真实实测**（dspy `2.6.27`：签名→真实 system/user 模板全文、ChatAdapter/JSONAdapter 双适配器与解析回退、`Predict`/`ChainOfThought`/`BootstrapFewShot`/`LabeledFewShot` 编译期调用账与演示对收集、metric 把关 full-traces 4 vs 0、`import` 面 18 OK/10 FAIL、`__version__`）；**模型决策 = 预置 stub**（`dspy.LM` 子类只重写 `__call__`，按提示词格式指令返回 JSON/标记并查词面关键词表决定 label——真实 LLM 非本机实测：本机无外网无 key）；零网络保障=所有程序挂 stub、任何真实 provider 永不被触；**『模型把提交还给了哪条轨迹』=预置仿真**（真实 bootstrapping 由真实 LLM 生成提交）；**Bootstrap 告示行文本来自引擎 print**（随版本可能变化，本批次 2.6.27 逐字节恒一）；**版本节奏/0.x→2.x 迁移方向/选择树=要素事实**（无外网未逐版复核）；dspy 2.6.27 依赖 `openai>=0.28.1` 无上界，本机仍锁定 openai 2.44.0 作为全仓兼容点（实测互不影响）；墙钟只进 stderr（≈1.4 s 三进程浮动但 stdout md5 恒一）。

---

## 📑 本章目录

1. [为什么 07 章现在讲 DSPy？](#0-为什么-07-章现在讲-dspy)
2. [先把问题拆开：从 prompt 玄学到语义程序](#1-先把问题拆开从-prompt-玄学到语义程序)
3. [把「编译器思路」降维成本机测量（探针设计）](#2-把编译器思路降维成本机测量探针设计)
4. [实验 A · 编译产物账：一句话签名 → 一页提示词](#3-实验-a--编译产物账一句话签名--一页提示词)
5. [实验 B · 编译账：数据 → 少样本，是一场生成 + 筛选的收费活动](#4-实验-b--编译账数据--少样本是一场生成--筛选的收费活动)
6. [实验 C · 程序账：模块直推、组件组合、签名复用](#5-实验-c--程序账模块直推组件组合签名复用)
7. [实验 D · 版本存续与选择树](#6-实验-d--版本存续与选择树)
8. [常见坑（本机实测踩过的 4 条）](#7-常见坑本机实测踩过的-4-条)
9. [诚实边界（AAA 自我审查）](#8-诚实边界aaa-自我审查)
10. [参考与衔接](#9-参考与衔接)

---

## 0. 为什么 07 章现在讲 DSPy？

07-应用框架之前十篇已经把「编排」的四种骨架走完了：

- 链（LangChain/LCEL）、图（LangGraph）、RAG 对象化（LlamaIndex/Haystack）、Agent 三范式（对话=AG2·角色化=CrewAI·轻量转移=Agents SDK）。

但所有这些框架都默认一件事：**prompt 是你手写的**——LangGraph 的节点里写什么提示、CrewAI 的 `You are {role}` 配方、Agents SDK 的 `instructions` 原样透传——框架只管「怎么组织、怎么跑、怎么接工具」，**怎么把一段提示词写对写稳，它们全交给你的手感**。这是 00-框架分类学账 C 里「DSPy 无承袭边独立」的另一面：DSPy 不是又一种"编排"，而是**把 prompt 本身当程序**——签名（声明式契约）→ 模块（Predict/ChainOfThought/ReAct）→ 优化器（BootstrapFewShot/MIPROv2/COPRO），用编译和自动化优化（针对真实指标）替代手写 prompt。

知识地图给它的位置写得很清楚（§7.10）：

> 核心：把 LLM 应用写成语义程序（signature → module → optimizer），用编译与自动优化（针对真实指标）替代手写 prompt，并内置 bootstrapping（用示例自动构造 few-shot）。哲学：把玄学 prompt 变成可评测可优化的对象。适用：对质量敏感、可自动化评测的任务（分类/抽取/转换）。

三句话定位本篇：

1. **思想 > 工具**：DSPy 想自动化的不是"再包一层链"，而是**你调 prompt 的那双手**——few-shot 挑哪些示例、CoT 要不要、提示词措辞怎么改。optimizer 把这三件事从手调变成编译。
2. **它比所有邻居都激进**：LangChain 说「链是程序」、LangGraph 说「图是程序」、DSPy 说**「整个调 prompt 的过程是程序」**。所以账 A 实测的是编译器最核心一步：一句话签名到底被编译成什么样、放大多少倍。
3. **给 07 章收口前的补位**：下一篇 `11-MCP协议` 是协议关键章、再后是 12 收束章；DSPy 作为类 5「Prompt 编程/优化」的唯一一课，正好把「框架 + 提示词都归场景」之前最后一块没盖的功能区补上——**质量敏感型任务的编译器**（§7.14 决策树点名）。

## 1. 先把问题拆开：从 prompt 玄学到语义程序

「把 prompt 当程序」这句话拆成三层落地：

1. **签名（Signature）**＝声明式契约：`class QA(dspy.Signature)` 里写 `text → answer` 两个字段和一句文档字符串，**不写一句提示词**。引擎要能自己把它编译成机器提示词——本篇账 A 抓的就是这份编译产物：逐字节拷出 system 模板和 user 模板，量『声明 27 tok → 产物 166 tok（×6.1）』的结构放大。签名是 DSPy 的"类型声明"：结构由它定、内容由数据定（账 C 用一个签名两批数据验证）。
2. **模块（Predict/ChainOfThought/ReAct）**＝可调用对象：`prog = dspy.Predict(QA); prog(text="…").answer`。模块是函数的函数式组合单元（账 C 把两个 ChainOfThought 串成管道、量调用数与跨模块字段），也是编译的对象（账 B 把数据喂进去让 optimizer 往模块里塞 demos）。
3. **优化器（BootstrapFewShot/LabeledFewShot/MIPROv2/COPRO）**＝编译的那一步：`compile(student=prog, trainset=…)`。编译做什么？账 B 用调用账回答——**Bootstrap 是真收费**（每个训练例跑 1 次模型、metric 把关收轨迹），**Labeled 是直抄**（0 次模型调用、纯把训练集搬进 demos）。「先用模型生成、再按指标筛选」= bootstrapping，这正是 08-RAG 13 章「生成 vs 筛选」账的体系内版本。

一句话分层：**LangChain 把链当程序、LangGraph 把图当程序、DSPy 把『写 prompt + 挑示例 + 调措辞』当程序**——前两者在运行时层面做程序化，本篇在**编辑期（编译期）层面**做程序化。你写的是思想（签名、指标、数据），工具负责把它变成提示词。

## 2. 把「编译器思路」降维成本机测量（探针设计）

- **引擎**：`dspy` **2.6.27**（Python 3.13.9）真实安装、真实执行。四本账全部走真 API：`dspy.Predict/ChainOfThought/BootstrapFewShot/LabeledFewShot/configure/Example/Signature`。
- **模型决策 = 预置 stub**：`Stub` 继承真实 `dspy.clients.lm.LM`、只重写 `__call__`——**引擎的模板编译/解析/回退/编译期收集全部真实发生**，唯一被替换的是「模型这轮怎么回」。stub 分两档：
  - **遵格式 stub**：读提示词里的格式指令照吟（JSON 指令→按字段序回 JSON；Chat 标记指令→回 `[[ ## field ## ]]` 标记行并以 `[[ ## completed ## ]]` 收尾），答案由词面关键词表决定（`点赞/很快/好评→正面`、`太慢/贵/差→负面`）。零网络（`model="stub-1"`，永不触 provider）。
  - **恒 JSON stub**：一律回 JSON 对象，用于量『适配器双试』——Chat 适配器解析标记失败→回退 JSONAdapter，一次推理两次调用。
- **量程**：四本账——A 编译产物（声明 tok vs 产物 tok、真实模板全文、适配器双试）、B 编译学费（Bootstrap 编译期调用数/demos/full-traces 行 对 Labeled 0 次调用、metric 全刷回退、few-shot 剂量价签 166→276 tok）、C 程序（CoT 推理字段、组合链 2 模块 2 调用、同签名两批数据的结构×内容指纹）、D 版本存续（28 名导入面 + `__version__` + 8 场景选择树）。
- **确定性**：三个独立进程跑完整探针，stdout md5 恒一 `c45b24fd…`（字节数约 8.2 KB）；`warnings.filterwarnings("ignore")` 前置、`sys.stdout.reconfigure(encoding="utf-8")`、stdout 以二进制整块写出（Windows 文本模式 `\n→\r\n` 会污染 md5）；Bootstrap 编译期的引擎告示用 `contextlib.redirect_stdout` 静音后按行回放，墙钟只进 stderr（≈1.4 s，三进程浮动但 stdout 逐字节一致）。

复现实录（严格逐字节 = 探针 stdout 三遍恒一）：

<details>
<summary>📊 探针 stdout 全量实录（md5 <code>c45b24fd…</code> · 墙钟仅 stderr）</summary>

````text

========================================================================
07-应用框架 · 09-DSPy「Prompt 编程 + 自动化优化——思想 > 工具」探针
环境 = dspy 2.6.27 · Python 3.13.9 · 零外网（stub LM 词面规则预置）
========================================================================
一句话：签名 = 声明式契约；编译 = 自动化优化（数据出来，prompt 进化）；程序 = 可组合可复用。
关键收敛行 → A 签名 27 tok 扩成编译产物 166 tok ×6.1 的程序性提示词
               → B 编译学费：Bootstrap(metric) 编译期 4 次调用 / 4 条演示对 / full-traces 4，Labeled 0 次调用 = 优化要不要模型眼
               → C 同签名两批数据：system 指纹恒同（结构=签名定）× demos 指纹相异（内容=数据定）

========================================================================
账 A｜从一句话到一页提示词——签名与编译产物
========================================================================
签名声明（类 docstring + 输入/输出字段描述）＝ 27 tok
Predict(QA)  1 次推理 = 1 次 LM 调用（遵格式 stub 首试即中），r.answer = 正面
编译产物（真实提示词）：system 111 tok + user 55 tok = 166 tok，约 6.1×
····························································
—— ChatAdapter 真实 system 模板（机器生成，照抄入文）——
Your input fields are:
1. `text` (str): 评论文本
Your output fields are:
1. `answer` (str): 正面/负面
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## text ## ]]
{text}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        判断电商评论文本的情感倾向。
····························································
—— 真实 user 模板 ——
[[ ## text ## ]]
物流很快

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.
····························································
适配器双试：不遵格式 stub（恒 JSON）Chat 适配器首试解析失败 → 回退 JSONAdapter 共 2 次调用；遵格式 stub 首试即中 1 次调用
→ 同一条链路，模型是否『照格式吟』决定走几次解析器；真实模型通常 1 次成功，兜底在解析器不在人。

========================================================================
账 B｜数据 → 少样本：编译是一次『生成 + 筛选』的收费活动，还是直抄
========================================================================
编译前 裸 Predict：推理消息演示对 0 条 · 上下文 166 tok
BootstrapFewShot(max=4, metric=…) 编译：引擎告示 →
  「Bootstrapped 4 full traces after 3 examples for up to 1 rounds, amounting to 4 attempts.」
  编译期 LM 调用 4 次（每个训练例子跑 1 次学生）· 演示对收进 4 条 · 编译只进 stdout 探针已静音
编译后 同输入推理：演示对 4 条 · 上下文 276 tok（few-shot 剂量价签 = 110 tok）
LabeledFewShot：编译期 0 次模型调用（纯直抄训练集）· 演示对 4 条
metric 全刷场景：编译期仍 4 次调用，但引擎告示 →
  「Bootstrapped 0 full traces after 3 examples for up to 1 rounds, amounting to 4 attempts.」
  演示对仍是 4 条 = 未达标的模型轨迹一条不采纳，退回『gold 直抄』兜底
→ 编译 = 生成 + 筛选双向收费：学费买的是『谁写出了得分轨迹』，metric 把关决定采纳还是弃用。

========================================================================
账 C｜程序：模块的直推、组件的组合、签名的复用
========================================================================
C1 ChainOfThought(QA)：模板自动多一个推理字段 reasoning · 1 次调用 · 输出 reasoning=规则命中 / answer=正面
····························································
C2 组合链 = CoT(QA) → CoT(Reply)：2 个模块 = 2 次 LM 调用 · 模块间数据 = 上一个模块的输出字段
   输入『价格贵，差评』→ 情感 负面 → 客服回复『抱歉给您添堵了。』
→ 组合不是拼提示词，是拼字段；数据在模块间按字段流转，调用数 = 模块数。
····························································
C3 同签名 QA、两批数据各自编译 1 次：
   system 指纹  ad606a55 vs ad606a55  恒同=True（模板结构 = 签名定）
   demos 指纹  0cb9c03b vs 668af40a  相异=True（演示内容 = 数据定）
   最后一条 user 指纹 0924006c == 0924006c（输入适配对调用方不变）
→ prompt 是编译产物：结构粘死在签名上，内容活在数据里。换数据不换签名 = 只换 demo，不动骨架。

========================================================================
账 D｜版本存续 + 8 场景选择树
========================================================================
  [OK    ] dspy.Signature
  [OK    ] dspy.InputField
  [OK    ] dspy.OutputField
  [OK    ] dspy.Predict
  [OK    ] dspy.ChainOfThought
  [OK    ] dspy.ReAct
  [OK    ] dspy.BootstrapFewShot
  [OK    ] dspy.LabeledFewShot
  [OK    ] dspy.MIPROv2
  [OK    ] dspy.COPRO
  [OK    ] dspy.Evaluate
  [OK    ] dspy.LM
  [OK    ] dspy.Retrieve
  [OK    ] dspy.Example
  [OK    ] dspy.Prediction
  [OK    ] dspy.Program
  -- 子模块面 --
  [OK    ] dspy.predict.react
  [OK    ] dspy.predict.retry
  [FAIL(AttributeError)] dspy.retry（顶层无子模块）
  -- 顶层缺 / 更名 --
  [FAIL(AttributeError)] dspy.Retry
  [FAIL(AttributeError)] dspy.OpenAI
  [FAIL(AttributeError)] dspy.OpenAIChat
  [FAIL(AttributeError)] dspy.MIPRO
  [FAIL(AttributeError)] dspy.GroundedFA
  [FAIL(AttributeError)] dspy.Generator
  [FAIL(AttributeError)] dspy.PythonTool
  [FAIL(AttributeError)] dspy.agent
  [FAIL(AttributeError)] dspy.Auto（2.x 无 Auto，动态入口让位给显式模块）
导入面：18 OK / 10 FAIL。迁移方向（要素事实）：OpenAI/OpenAIChat → dspy.LM 统一接入；Generator → Predict/ChainOfThought；MIPRO → MIPROv2；agent → ReAct；Retry → dspy.predict.retry 子模块。
····························································
8 场景选择树断言 8/8（要素事实：按 §7.1 八类 × §7.14 决策要点）。

========================================================================
诚实边界（要素事实 / 非本机实测）
========================================================================
· 引擎语义（签名→模板、JSON 解析、Bootstrap 收集、适配器回退、导入面）= dspy 2.6.27 本机真实实测；
· LLM 决策 = dspy.LM 子类 stub 词面规则预置（真实 LLM 非本机实测 · 零网络）；
  label 由『点赞/很快/好评/太慢/贵/差』关键词决定，推理的『对错』因此不代表模型能力；
· 版本节奏 / 0.x→2.x 迁移方向 / 选择树 = 要素事实（无外网未逐版复核）；
· Bootstrap 告示行文本来自引擎 print，随版本可能变化，本批次在 2.6.27 上逐字节恒一。

========================================================================
台账汇总（本批）
========================================================================
签名→产物    27 tok → 166 tok ≈×6.1 · 遵格式 1 调用 / 恒 JSON 双试 2 调用
Bootstrap 学费 编译期 4 调用 · 演示对收 4 条 · demo 对上屏 4 条 · full-traces 4/4 例
Labeled 学费  编译期 0 调用 · 演示对 4 条（直抄训练集，无模型参与）
metric 把关   metric 全刷：full-traces 0 条仍演示对 4 条 = 弃用回退 gold
程序组合     CoT 单模块 1 调用 → 双模块链 2 调用 · 跨模块字段 1 个（情感→回复）
结构×内容     system 指纹恒同 ad606a55 · demos 指纹相异 0cb9c03b vs 668af40a
版本面        导入面 18 OK / 10 FAIL · dspy 2.6.27 · dspy.retry 顶层缺 / dspy.predict.retry OK
选择树        8 场景断言 8/8
````

</details>

## 3. 实验 A · 编译产物账：一句话签名 → 一页提示词

DSPy 的第一承诺是「不用写提示词、写签名」。那么签名到底被怎么放大？探针逐字节抓取。

**声明端**（我写的全部「代码」）：`class QA(dspy.Signature)`，docstring = `判断电商评论文本的情感倾向。`，两个字段 `text: 评论文本`（InputField）、`answer: 正面/负面`（OutputField）。量成 token：**27 tok**。

**产物端**（引擎编译出的真实提示词，探针逐字节拷出）——ChatAdapter 的 system 模板：

````text
Your input fields are:
1. `text` (str): 评论文本
Your output fields are:
1. `answer` (str): 正面/负面
All interactions will be structured in the following way, with the appropriate values filled in.

[[ ## text ## ]]
{text}

[[ ## answer ## ]]
{answer}

[[ ## completed ## ]]
In adhering to this structure, your objective is: 
        判断电商评论文本的情感倾向。
````

user 模板：

````text
[[ ## text ## ]]
物流很快

Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.
````

量：system **111 tok** + user **55 tok** = **166 tok**，正好是声明的 **6.1×**。这就是「编译产物」的体感：你打一句话，引擎糊了一页格式条款（字段清单、标记符 `[[ ## … ## ]]`、收尾符 `[[ ## completed ## ]]`、目标声明）把它包住。产物里没有任何一句是你写的话术——**全部是结构**，这正是「签名定结构」的出处。

**适配器双试**（2.6 的解析架构）：同一条推理链路，DSPy 先试 ChatAdapter（标记模式），解析失败再回退到 JSONAdapter。探针用两档 stub 分隔变量：
- **写 JSON stub**（不遵格式，一律回 `{"answer":…}`）：Chat 适配器拿不到 `[[ ## answer ## ]]` 标记 → 解析失败 → 回退 JSON 适配器 re-format 后再调一次。**1 次推理 = 2 次 LM 调用**。
- **遵格式 stub**（照吟标记收尾）：Chat 适配器一次解析成功。**1 次推理 = 1 次调用**。

结论：兜底不在人、在解析器。真实模型通常句句循格、一次成功；stub 恒 JSON 的 2 次调用是**引擎替你兜格式**的可见成本。它也顺带回答了知识地图 §7.10「把玄学 prompt 变成对象」里最容易忽略的一半——**prompt 是解析对象，格式错了引擎会替你 re-format 重试**。

*（A 账还顺带验证了结构化输出的老话题：`r = prog(text="物流很快")` 返回 `Prediction` 对象，`r.answer` 直接取到 `正面`——输出字段与签名一一对应，不必正则去剥字符串。挂靠 06-应用开发 04-结构化输出。）*

## 4. 实验 B · 编译账：数据 → 少样本，是一场生成 + 筛选的收费活动

知识地图说 DSPy 内置 bootstrapping（用示例自动构造 few-shot）。编译到底花多少算力？账 B 把四笔账打平：

| 场景 | 编译期 LM 调用 | 演示对（demos） | 引擎告示（编译 stdout，静音后回放） |
|---|---|---|---|
| 编译前 裸 Predict | — | **0 条** | — |
| BootstrapFewShot(max=4, metric=宽松) | **4 次**（每训练例跑 1 次学生） | **4 条** | `Bootstrapped 4 full traces after 3 examples for up to 1 rounds, amounting to 4 attempts.` |
| LabeledFewShot | **0 次** | **4 条**（直抄训练集） | — |
| BootstrapFewShot + metric 全刷 | **4 次** | **4 条** | `Bootstrapped 0 full traces after 3 examples for up to 1 rounds, amounting to 4 attempts.` |

三个结论：

1. **加 few-shot 是真收费、不是白送**：Bootstrap 编译期花 4 次调用（每训练例跑一次学生）换来 4 条演示对；编译后再推理时这两条演示进上下文，剂量价签立现——同输入 `物流很快`，上下文 **166 → 276 tok（+110 tok 的 few-shot 剂量）**。这就是 06-应用开发 02 章「few-shot 示例预算」账在 DSPy 里的编译器版本：示例挑得好不好，由 optimizer 在编译期替你做。
2. **LabeledFewShot = 到嘴边就抄**：0 次模型调用、纯把训练集的 4 条直接搬进 demos。Bootstrap 贵在它**先跑一次生成轨迹**；Labeled 连跑都不跑。`require_trace_metrics` *（注：2.6 实测 compile 无需传 metric 也能编，但 metric 是准入筛选的必要条件）*。
3. **metric 是准入器，决定『谁被采纳』**：metric 全刷（`return False`）时，引擎仍然跑满 4 次（生成照旧），但告示从「Bootstrapped 4 full traces」掉到「**0 full traces**」——**每条刚生成的轨迹都因不达标被弃用**，demos 却仍是 4 条：这是 `gold` 直抄兜底，不是模型轨迹。两行的差异就是「生成了但没被采纳」与「生成了且 4 条全被采纳」的唯一隔断。

> 一句话：**编译 = 生成（花调用）+ 筛选（metric 把关）双向收费**；学费买的是『谁的轨迹能得分』，你怎么逼它也省不掉生成，但 Labeled 给了一条不花钱的近道——前提是你信训练集直抄够用。08-RAG 13 章「生成 vs 筛选」的同款账，在 DSPy 里变成了引擎内建的鼎形过程。

## 5. 实验 C · 程序账：模块直推、组件组合、签名复用

**C1 ChainOfThought 自动加推理字段**：`dspy.ChainOfThought(QA)` 编译出的模板里，输出字段从 `answer` 变成 `['reasoning', 'answer']` 两个——1 次调用返回 `reasoning=规则命中`、`answer=正面`。CoT 不是一个「模型行为开关」，是**签名层面多声明了一个输出字段**，结构里被编译进模板。这就是把"think step by step"从咒语变成了可观测的字段（对照 06-应用开发 02 章 CoT 成本账）。

**C2 模块组合 = 函数组合**：把 `CoT(QA)` 和 `CoT(Reply)` 串起来——

```python
def pipeline(text):
    a1 = c1(text=text).answer          # 模块1：情感
    r2 = c2(text=text, sentiment=a1).reply  # 模块2：客服话术，吃模块1的输出
    return a1, r2
```

输入 `价格贵，差评` → 情感 `负面` → 客服回复 `抱歉给您添堵了。`。两个模块 = **恰好 2 次 LM 调用**，模块间数据 = 上一个模块的**输出字段**（`sentiment=负面` 作为第二个签名的输入字段被灌进下一环节）。**组合不是拼提示词，是拼字段；数据在模块间按字段流转，调用数 = 模块数**。对照 AG2 的 `Agent` 三原语、CrewAI 的 `role/task`：DSPy 的"程序"是最轻的——就是函数。

**C3 同签名、两批数据 → 结构×内容指纹分离**：同一个 `QA` 签名，分别用「全正面」和「全负面」两批训练集编译两次，探针抓三次推理消息做 md5：
- `system` 指纹 **ad606a55 == ad606a55**：模板结构一个字节都没变 → **结构 = 签名定**；
- `demos` 指纹 **0cb9c03b ≠ 668af40a**：灌进上下文的演示内容随数据翻面 → **内容 = 数据定**；
- 最后一条 `user` 指纹 **0924006c == 0924006c**：对调用方而言输入适配一个字节未动。

> 这就是「prompt 是编译产物」的可证明版本：**换签名 = 换骨架，换数据 = 只换 demo**。你调 review 面试官给的一段提示词时改的永远是「结构」（签名）；你换数据集时改的是「内容」（demos）——两件事在编译器视角下被分得干干净净。00-框架分类学账 C 说 DSPy「无承袭边独立」，账 C 给了它一个更彻底的优势：**结构不依赖任何相邻框架，内容不依赖任何手写示例**。

## 6. 实验 D · 版本存续与选择树

**导入面（28 名逐项 `_probe`，dspy 2.6.27）**：

- **OK（18）**：`Signature`/`InputField`/`OutputField`/`Predict`/`ChainOfThought`/`ReAct`/`BootstrapFewShot`/`LabeledFewShot`/`MIPROv2`/`COPRO`/`Evaluate`/`LM`/`Retrieve`/`Example`/`Prediction`/`Program`（16 顶层）+ `dspy.predict.react`/`dspy.predict.retry`（2 子模块）。
- **FAIL（10，`AttributeError`）**：顶层缺 `Retry`/`OpenAI`/`OpenAIChat`/`MIPRO`/`GroundedFA`/`Generator`/`PythonTool`/`agent`/`Auto` + 顶层无子模块 `dspy.retry`。
- `__version__` = **2.6.27**。

迁移方向（要素事实）：0.x 时代的模型接入单体类 `OpenAI/OpenAIChat` → 2.x 统一收进 `dspy.LM`；`MIPRO` → `MIPROv2`；`agent` → `ReAct` 模块；`Generator` → `Predict/ChainOfThought`；`Retry` → `dspy.predict.retry` 子模块。2.6 的 `Auto`（动态编排入口）没了——**动态入口让位给显式模块**（要什么模块、写什么模块），与 07 章全家纪律一致（CrewAI 顶层不出 `CrewBase`、AG2 顶层垫片移除、Agents SDK 顶层不设 `run_sync`）。

**选择树（8 场景断言 8/8，要素事实）**：

| 场景 | 归属 |
|---|---|
| 提示词反复试错 · 特征式任务 · 要自动化优化（账 A 类 5） | **DSPy（本篇）** |
| 状态机 · 自环重试 · checkpoint · 图式路由 | LangGraph |
| 多 Agent 互聊 / 群聊式工作流 | AutoGen(AG2) |
| 角色 · 任务 · 流程的角色化流水线 | CrewAI |
| 文档→节点→索引→检索→问答标准化 | LlamaIndex |
| 拖拽积木搭整套应用（状态+组装+观测+成本） | 低代码平台（Dify/n8n） |
| 轻量运行时单 Agent + 工具护栏 + 转交 | OpenAI Agents SDK |
| 跨框架协议互操作 / 工具标准化 | MCP |

## 7. 常见坑（本机实测踩过的 4 条）

1. **裸 `Predict(QA)` 一次推理可能是 2 次 LM 调用**（实测，2.6.27）：默认适配器先试 Chat（标记模式），stub 若不循格式回 JSON，解析失败 → 回退 JSONAdapter 再调一次。不是引擎 bug，是**兜格式**；灾难只在你想数「每次推理 N 次调用」时计入预算。解法：模型配置/提示词让它循格式（真实模型基本如此），或量化时用遵格式 stub 测 1 次、恒 JSON stub 测 2 次两个口径（本篇账 A 就是这么打的）。
2. **Bootstrap 编译期的告示打到 stdout**（实测）：`compile()` 时引擎 `print` 「Bootstrapped 4 full traces after …」到 stdout——你如果像探针一样对 stdout 做 md5，必须 `contextlib.redirect_stdout` 静音再按行回放，否则 md5 被版本句尾数字污染。
3. **`dspy.Example` 必须 `.with_inputs("text")`**（实测）：裸 `Example(text=…, answer=…)` 进 trainset，bootstrap 会报「Inputs have not been set」。这是构造细节，漏了编译才炸、错误藏日志里。
4. **demos 在 `comp.demos`，不在 `comp.predict.demos`/`state_dict`**（实测）：编译结果是 **Predict 本身**（继承而非包装），demos 直接挂在编译对象上；`state_dict` 没有 `.demos` 键、旧 API 的 `comp.predict` 是空壳。取值口诀：**只要数 demos 就 `len(compiled.demos)`**。

另外两笔环境账（供读者参考，均已实测锁定）：
- **dspy 2.6.27 资源账**：依赖 `openai>=0.28.1`（无上界）、litellm、boto3/botocore、optuna、pandas、datasets、diskcache、cloudpickle 等十几个包——安装时 resolver 未动任何既有锁（openai-agents 0.17.0 / openai 2.44.0 / langchain-openai 1.3.4 / crewai 1.15.22 / ag2 1.0.6 / semantic-kernel 1.44.1 全部保活）。**dspy 2.6 与 openai 2.44.0 实测互不影响**（0.17.0 的 Usage 工厂要求 `openai 2.x`，dspy 并无上界钉——这点上它比 openai-agents 温和）。
- **Python 版本带**：dspy 2.6.27 要求 `>=3.9,<3.14`；本机 3.13.9 实测通过。

## 8. 诚实边界（AAA 自我审查）

- **引擎语义 = 本机真实实测**：dspy `2.6.27` 真实安装运行（stdout md5 `c45b24fd…` 三个独立进程逐字节恒一，墙钟 ≈1.4 s 仅 stderr）。签名→真实 system/user 模板全文、ChatAdapter/JSONAdapter 双适配器与解析回退（恒 JSON stub 2 次 vs 遵格式 1 次）、`Predict/ChainOfThought/BootstrapFewShot/LabeledFewShot` 编译期调用账与演示对收集、metric 把关 full-traces 4 vs 0、few-shot 剂量 166→276 tok、同签名两批数据的 md5 指纹、28 名 `import` 探测、`__version__`=2.6.27——全部为真 API 实测。
- **模型决策 = 预置 stub**：`Stub`/`JsonOnlyStub` 继承真实 `dspy.clients.lm.LM`、只重写 `__call__`——label 由词面关键词表决定（`点赞/很快/好评→正面`、`太慢/贵/差→负面`），`reasoning=规则命中`、`reply` 按情感正负回话。模型的推理/选轨迹/选示例 = **非本机实测**（本机无外网无 key）——**任何"编译后比分更高/更低"的字面都不成立**，本篇从不声明正确率提升，只量调用/次数/token/指纹等引擎可观测量。
- **零网络保证**：所有程序挂 `model="stub-1"` 的 stub，默认 provider 永不被触（碰则 401=本机环境事实）。
- **『模型把提交还给了哪条轨迹』=预置**：真实 bootstrap 由真实 LLM 生成提交、metric 由真实模型判——本探针只有词面真伪判定，轨迹质量不代表任何真实模型能力。
- **版本节奏 / 0.x→2.x 迁移方向 / 选择树 = 要素事实**：知识地图 §7.10/§7.14/§17.5.9 文本（无外网未逐版复核）；`Auto`/`MIPRO`/`OpenAI` 等顶层缺是 2.6.27 实测，迁移方向是要素事实。
- **Bootstrap 告示行文本来自引擎 print**：随版本措辞可能变；本批次 2.6.27 逐字节恒一（三个进程）。
- **依赖账**：dspy 2.6.27 要求 `openai>=0.28.1`（无上界）+ 十几个伴生包，安装后全仓既有 pin（含 openai-agents 0.17.0 的 `openai<3`）未动，`openai 2.44.0` 仍为全仓兼容点。

## 9. 参考与衔接

- **本课地图**：§7.10（DSPy——signature→module→optimizer、bootstrapping、把玄学 prompt 变成可评测可优化对象）+ §7.1（类 5 Prompt 编程/优化 DSPy/Outlines）+ §7.14（质量敏感型任务 → DSPy）+ §17.5.9（DSPy：自动 few-shot、多模块编译）。
- **选择树 8 场景**（探针断言 8/8，要素事实）：表见账 D——一句话：quality-sensitive/自动评测任务归 **DSPy**，状态图归 LangGraph，对话归 AG2，角色化归 CrewAI，RAG 归 LlamaIndex，低代码归 Dify，轻量不足归 Agents SDK，协议归 MCP。
- **前承**：00-框架分类学（账 C「DSPy 无承袭边独立」——本篇账 C 用 md5 指纹兑现「结构不依赖相邻框架、内容不依赖手写示例」）；06-应用开发 02（few-shot 预算账 → 本篇编译期自动挑示例）；06-应用开发 04（JSON 合法≠Schema 合规 → 本篇 JSONAdapter 是解析器兜格式第三现场）；06-应用开发 05（工具入参 Schema → 本篇 ReAct 导入面）；10-OpenAI-Agents SDK（stub 预置决策表 vs 本篇 optimizer 自动逼近——SDK 手写决策、DSPy 编译决策，两端对照）；08-RAG 13（生成 vs 筛选账 → 本篇 bootstrap 的 metric 把关同源）。
- **后启**：`11-MCP协议`（关键章：Agent 的工具出口——DSPy 的 ReAct 模块对接工具/检索的挂接点，链路 `013` 手写 MCP Server）；`12-框架继承关系与选型决策`（收束章：完整 DAG——本篇「Prompt 编程/优化」与 MCP 协议是两类独立的优化面）。
- **对应里程碑**：无独立编号；§17.5.9 练习「用 DSPy 优化一个分类任务的 prompt」= 本篇账 A/B 对口（分类签名→编译→few-shot）。

> 本篇完工于 2026-09-22（v0.36 批次）；探针 代码 `code/notebooks/_tools/dspy_demo.py`；stdout md5 `c45b24fd…`。
