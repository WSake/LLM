# 📡 MCP 协议（Model Context Protocol）：协议层的「握手→能力→调用→分层」，把「组装 + 工具接口」标准化成 2026 标配（07 章协议关键章 · 链路 013）

> 对应知识点：知识地图 §7.12（**MCP（Model Context Protocol，2024-11 由 Anthropic 提出，2025 成为开放标准）**——定位=**不是框架，是协议**：统一「模型/Agent 与外部工具/数据源」的接口（client-server：MCP Host 与 MCP Server）；为什么是分水岭=此前每个 Agent 框架都要自己接工具 SDK，MCP 后工具一次实现、处处可用；配套 A2A（Agent 间通信）/ACP；2025 年 OpenAI/谷歌宣布支持；实践=「会写一个 MCP Server（stdio/HTTP+SSE、工具/资源/提示三大原语）是 2026 年应用工程师的标配技能」）、§7.1（协议层==**覆盖 1 维的特判类型**：跨栈互操作标准——MCP/A2A/ACP，不是框架、不承诺状态/观测/成本）、§7.13（**MCP 底座**：2025 协议标准化成为所有框架的公共底座、「框架分层 = 协议层(MCP/A2A) + 轻量SDK + 图引擎 + 平台」）、§7.14（决策树）、§17.5.10（应用编程接口类）。
> 前置：[00-框架分类学](./00-框架分类学.md)（账 B 协议层特判「覆盖 1 维」、账 C 继承影响「MCP 底座影响面 9」、8 类×24 框架里协议层单独一档——本篇把这张选型表里的「协议」从格子升格成整章）；[05-Function-Tool-Calling](../06-应用开发/05-Function-Tool-Calling.md)（工具协议手搓版：声明工具→入参校验→执行→回喂——本篇回答「这套协议标准化之后长什么样」）；[06-Semantic-Kernel](./06-Semantic-Kernel.md)（`k.as_mcp_server()`→MCP `tools/list` 真实 1:1 注册——框架侧出场先例）；[10-OpenAI-Agents-SDK与轻量运行时](./10-OpenAI-Agents-SDK与轻量运行时.md)（`mcp_servers` 挂接点是 SDK 的「下一个工具出口」）；更早：[08-RAG 全系](../08-RAG体系/01-Embedding与检索基础.md)（检索→生成桩工具在位，MCP 是为「把检索/算力/数据库包成工具」准备的协议壳）。
> 动手：`python code/notebooks/_tools/mcp_demo.py`（**真实 mcp 1.28.1 引擎**本机执行；进程内 memory-stream 环回——零网络 · 零随机 · stdout md5 `5bf0d14d` 三个独立进程逐字节恒一；手工 JSON-RPC 帧级实录 + 版本协商边缘 + 工具/协议双层错误分层 + who-speaks-MCP 生态属性探测）。
> 一句话：**MCP = 把「组装 + 工具接口」标准化成协议的 2026 标配——Server 自报能力（`capabilities{tools/prompts/resources}`）→ Client 按能力用请求发起工具调用（17 个请求方法·三种传输 stdio/HTTP-SSE/streamable-HTTP）；握手靠 `initialize→result→notifications/initialized` 三帧定协议版本与能力（实测：客户端谎报 `1999-01-01`，服务端照样接受、协商到最新档 `2025-11-25`——协议是「双方谈」，不是「一方验」）；工具业务错误与协议错误分两层——missing/错型/工具不存在都落在 `CallToolResult` 带内 `isError=True` 的文本结果里（对 LLM 是「一段可读的错误」，不是异常），只有方法名这类协议层错误才出带外 JSON-RPC `-32602`（客户端 SDK 才抛异常）；工具 `@mcp.tool()` 即注册、`tools/list` 一张 Schema 契约面；生态侧 SK/Agents SDK/CrewAI 三家本机就有 MCP 接口，LangChain/DSPy/AG2/LlamaIndex/Haystack 没有——协议的赢面不在某个框架里，在跨框架的都接它。**
>
> 📌 诚实边界：**引擎语义 = 本机真实实测**（mcp `1.28.1`：`FastMCP`+`@mcp.tool()` 注册、`create_initialization_options` 能力自报、进程内环回握手 `initialize→result→initialized`、17 请求方法面（`ClientRequest.root` Union 实测）、三种传输 `import` 面、版本协商接受任意声称档并返 LATEST、`tools/list` Schema、`tools/call` 正常/缺参/错型/未知工具五类响应、`no/such/method` 协议层 `-32602`、`server_version` 缺省=Sdk 版本 1.28.1、帧级双向实录）；**MCP 帧全部在进程内 memory-stream 环回生成**（同一进程、零网络——能测到的是协议帧本身，不是 socket 层）；**错误响应文本（`Error executing tool add: …validation error…`/`Unknown tool: nope`）= pydantic/SDK 本机真实产出**；**版本节奏/生态归属/§7.12 叙事=要素事实**（2024-11 Anthropic 提出、2025 开放标准、OpenAI/谷歌 2025 支持 MCP、三原语说法——无外网未逐版复核）；**未连接任何远程/第三方 MCP Server**（who-speaks-MCP 只用 `hasattr` 本机探测安装的框架包）；墙钟只进 stderr（≈22 s=账 D 重框架 import 主导；三进程浮动但 stdout md5 恒一）。

---

## 📑 本章目录

1. [为什么 07 章的收束关键章是协议？](#0-为什么-07-章的收束关键章是协议)
2. [先把问题拆开：协议在 24 家框架之外干一件别的事](#1-先把问题拆开协议在-24-家框架之外干一件别的事)
3. [把「协议」降维成本机测量（探针设计）](#2-把协议降维成本机测量探针设计)
4. [实验 A · 协议面账（词汇表 + 17 方法 + 3 传输 + 两个角色）](#3-实验-a--协议面账词汇表--17-方法--3-传输--两个角色)
5. [实验 B · 握手帧账（三帧定版本，屏蔽的诚实电视频道）](#4-实验-b--握手帧账三帧定版本屏蔽的诚实)
6. [实验 C · 工具调用与错误分层账（带内 isError vs 带外 JSON-RPC error）](#5-实验-c--工具调用与错误分层账带内-iserror-vs-带外-json-rpc-error)
7. [实验 D · 生态账（who-speaks-MCP + 版本档 + 选择树 8/8）](#6-实验-d--生态账who-speaks-mcp--版本档--选择树-88)
8. [常见坑（本机实测踩过的 7 条）](#7-常见坑本机实测踩过的-7-条)
9. [诚实边界（AAA 自我审查）](#8-诚实边界aaa-自我审查)
10. [参考与衔接](#9-参考与衔接)

---

## 0. 为什么 07 章的收束关键章是协议？

07-应用框架走完了 00 分类学、01 LangChain、02 LangGraph、03 LlamaIndex、04 低代码、05 Haystack、06 Semantic Kernel、07 AutoGen/AG2/MAF、08 CrewAI、10 Agents SDK、09 DSPy——整整十一种「拿来干活的库」。00 开篇那张选型表里有一格被反复点名却一直没展开：**协议层（MCP/A2A/ACP）——覆盖 1 维的特判类型**。09/10 两篇的「后启」都分别留言：`11-MCP协议`（关键章）在等。

协议和框架在 07 章的语境里不是并列的：

- **框架**＝把「状态执行 + 接口组装 + 可观测 + 成本可靠」四类工程惯例打包的库（00 账 B）。你写 `from langgraph import StateGraph`、`from crewai import Crew`——你买的是「帮我管状态」「帮我排角色」。
- **协议**＝不承诺状态、不承诺观测、不承诺成本。它只回答一个问题：**工具接口长什么样，让「我装的工具」和「任何一家 Agent」不用互相认亲。** MCP 把答案钉死成：Server 声明能力 → Client 按 `tools/call` 调 → 结果按三件套回。工具一次实现，处处可用（§7.12）。

所以链路 `013`（手写 MCP Server 并调用）是 07 章全章唯一在「不装任何框架」的前提下要交付的能力：**把「手搓工具协议」（05 章四档：prose→JSON→FC 形状→校验）升格成「行业标准的协议壳」。** 本篇把它量成四本账：

- **账 A 协议面账**：协议词汇表——17 个请求方法（`ClientRequest.root` 联合类型实测）、三种传输（stdio/HTTP-SSE/streamable-HTTP）、两个角色（Server↔Client）、能力自报（`capabilities{tools/prompts/resources}`）。协议的「户口簿」。
- **账 B 握手帧账**：进程内环回，帧级双向实录——`initialize`(带协议版本 + 能力) → `result`(服务端回声 + serverInfo + 协议版本) → `notifications/initialized` → 之后业务请求。外加一个诚实的边缘：客户端谎报 `1999-01-01`，服务端照样接受、协商到最新档——协议的版本协商是「双方谈」，不是「一方验」。
- **账 C 工具调用与错误分层账**：工具的正常回包三件套（`content` 文本 + `structuredContent` 结构化 + `isError` 旗标）；以及分层最关键的观测——**缺参/错型/工具不存在都在 `CallToolResult` 带内**（`isError=True` + 文本），**只有协议层方法名错才出带外 JSON-RPC `-32602`**。工具错误是数据，协议错误才异常。
- **账 D 生态账**：who-speaks-MCP——本机装过的 8 家框架谁带 MCP 接口（`hasattr` 实测：SK/Agents SDK/CrewAI 三家有，LangChain/DSPy/AG2/LlamaIndex/Haystack 没有）；协议版本档 4 个（2024-11-05 首批 → 2025-03-26 默认 → 2025-06-18 → 2025-11-25 当前）；选择树 8 场景断言 8/8。

## 1. 先把问题拆开：协议在 24 家框架之外干一件别的事

「协议」这个词被说烂了。MCP 的独特之处在于它标准化的是**接口本身**，而不是**实现**。拆成三层：

1. **词汇表（账 A）**：参与方就俩——Server（注册工具、声明能力）和 Client（发起请求、消费工具）。再往下是方法面：`initialize`（握手）、`tools/list`（报目录）、`tools/call`（执行）、`resources/*`、`prompts/*`、`completion/complete`、`ping`、`logging/setLevel`、`tasks/*`——实测 17 个。传输有三种（stdio 进程管道、HTTP+SSE、streamable-HTTP），**协议词汇面与传输解耦**：同一套方法在三条管子上走。
2. **握手（账 B）**：Client 先 `initialize`——带上自己认得的协议版本 + 能力；Server 回 `result`——带 `serverInfo`（名 + 版本）+ 自己支持的协议版本 + 能力回声；Client 再补一条 `notifications/initialized`。**三帧定版本**：从此按谈好的协议版本讲话。
3. **错误分层（账 C）**：这是协议对「工具生态」最关键的承诺——**工具业务错误不炸客户端**。缺参 `add{a:3}`、错型 `b:"x"`、工具不存在 `nope`，全部是 `CallToolResult` 带内 `isError=True` 的文本结果。为什么分层重要？因为 LLM 是这套协议的消费主体——一段文本错误它能读、能回、能自愈（08-RAG 13 章「自愈」挂接的就是这个）；而一个异常会打断整轮。

一句话分层：**框架管「怎么干活」，协议管「接口长什么样」；MCP 把 07 章所有框架的工具出口统一成同一条线。**

## 2. 把「协议」降维成本机测量（探针设计）

- **引擎**：`mcp` **1.28.1**（`importlib.metadata.version('mcp')` 实测）真实安装、真实执行。四本账全部走真 API：`FastMCP`、`@mcp.tool()`、`create_initialization_options`、低层 `mcp.server.lowlevel.Server.run`、`mcp.shared.memory.create_client_server_memory_streams`、`mcp.types` 结构类型。
- **传输 = 进程内 memory-stream 环回**：`create_client_server_memory_streams()` 把「客户端写流↔服务端读流」直接对接（零网络、零 socket）。这保证两点：（1）能测到的就是**协议帧本身**（JSON-RPC 2.0 的 method/params/result/error 字段流）；（2）`Rec` 包裹器在写流上拦截 `SessionMessage`，把双向每一帧 `model_dump` 成 JSON 台账——**帧级实录**。
- **手工 JSON-RPC 客户端**：不用高层 `ClientSession` 的友好方法，而是直接构造 `JSONRPCRequest(jsonrpc="2.0", id, method, params)` 发原始帧——这样才能测：谎报协议版本（协商边缘）、未定义方法（协议层错误 `-32602`）、以及精确的帧序。每次协商用**全新 Server 实例**（`mcp` 的 `Server.run` 是一次性的：跑完即终态，复用会串状态——这也是探针踩过的坑）。
- **量程**：四本账——A 协议面（常量三件套 + 17 方法面 + 3 传输 + 角色 + 能力自报）、B 握手帧（正常协商 result + 谎报版本返回 + 帧流全纪录）、C 工具调用（正常 add / 默认参生效 / 五类错误分层）、D 生态（8 家框架属性探测 + 协议版本档 + 8 场景选择树）。
- **确定性**：三个独立进程跑完整探针，stdout md5 恒一 `5bf0d14d`（8233 字节）；`warnings.filterwarnings("ignore")` 前置、`sys.stdout.reconfigure(encoding="utf-8")`、stdout 以二进制整块写出（Windows 文本模式 `\n→\r\n` 会污染 md5）；引擎校验错误树/INFO 日志用 `logging.disable(CRITICAL)` + `redirect_stdout/stderr(StringIO())` 静音（不吞顶层 `_OUT` 缓冲），墙钟只进 stderr（**≈22 s——账 D 重框架 import 主导（semantic_kernel 2.9 s/crewai 3.9 s/dspy 6.3 s/haystack 6.0 s），引擎本体≈1.5 s**；三进程浮动但 stdout 逐字节一致）。

复现实录（严格逐字节 = 探针 stdout 三遍恒一）：

<details>
<summary>📊 探针 stdout 全量实录（md5 <code>5bf0d14d</code> · 8233 字节 · 墙钟仅 stderr）</summary>

````text
MCP 协议实测（mcp_demo.py · 07-应用框架 11-MCP协议 · 真实 mcp 1.28.1 引擎）

==============================================================
账 A  协议面账 —— 协议把『组装 + 工具接口』标准化成什么
--------------------------------------------------------------
  SDK 版本        : mcp 1.28.1（本机实测 import）
  LATEST          : 2025-11-25（主版本号，要素事实+常量）
  DEFAULT_NEGOT   : 2025-03-26
  SUPPORTED       : ['2024-11-05', '2025-03-26', '2025-06-18', '2025-11-25']（按时间序 2024-11-05 → 2025-03-26 → 2025-06-18 → 2025-11-25）
  客户端方法面    : 17 个请求类（ClientRequest.root Union 成员）：
    completion/complete            CompleteRequest
    initialize                     InitializeRequest
    logging/setLevel               SetLevelRequest
    ping                           PingRequest
    prompts/get                    GetPromptRequest
    prompts/list                   ListPromptsRequest
    resources/list                 ListResourcesRequest
    resources/read                 ReadResourceRequest
    resources/subscribe            SubscribeRequest
    resources/templates/list       ListResourceTemplatesRequest
    resources/unsubscribe          UnsubscribeRequest
    tasks/cancel                   CancelTaskRequest
    tasks/get                      GetTaskRequest
    tasks/list                     ListTasksRequest
    tasks/result                   GetTaskPayloadRequest
    tools/call                     CallToolRequest
    tools/list                     ListToolsRequest
  三种传输        : stdio / streamable-http / SSE（client 与 server 模块本机 import 在位）
  两个角色        : Server（声明能力·serve 工具）↔ Client（发起请求·消费工具）
  能力自报        : capabilities{tools/prompts/resources} 协商期一次性广播，之后请求才能发
  字面密度代理    : 17 方法名 + 2 角色名称 est_tok ≈ 51（协议词汇面规模的一个粗代理）

==============================================================
账 B  握手帧账 —— 进程内环回，帧级实录（零网络，真实 mcp 1.28.1 引擎）
--------------------------------------------------------------
  服务端初始化选项 create_initialization_options（实测）：
    server_name  = calc-server
    server_version = 1.28.1（缺省 = Sdk 版本，要素+本机实测）
    capabilities = {"experimental": {}, "prompts": {"listChanged": false}, "resources": {"listChanged": false, "subscribe": false}, "tools": {"listChanged": false}}
  正常版本协商（客户端声称 2025-11-25）：
    → 协商到 protocolVersion = 2025-11-25；serverInfo = {"name": "calc-server", "version": "1.28.1"}
      capabilities = {"experimental": {}, "prompts": {"listChanged": false}, "resources": {"listChanged": false, "subscribe": false}, "tools": {"listChanged": false}}
      instructions = "只做数学与问候"
  谎报版本协商边缘（客户端声称 1999-01-01）：
    → 服务端接受任意声称版本，仍回合法 result，协议版本协商到 LATEST = 2025-11-25
  帧流全记录（双向标签 · kind 去重计数）：
    C->S  notification   notifications/initialized  ×1
    C->S  request        id=1 initialize            ×1
    C->S  request        id=2 tools/list            ×1
    C->S  request        id=3 tools/call            ×1
    C->S  request        id=4 tools/call            ×1
    S->C  result         id=1                       ×1
    S->C  result         id=2                       ×1
    S->C  result         id=3                       ×1
    S->C  result         id=4                       ×1
  帧序结论：initialize(id1) → result → notifications/initialized → 业务请求(id2..) → result —— 每个请求都带 id，响应按 id 配对；通知无 id

==============================================================
账 C  工具调用与错误分层账 —— 工具层带内 isError vs 协议层带外 JSON-RPC error
--------------------------------------------------------------
  工具调用出的『结果』长什么样（正常 add{a:1,b:2}）：
    content = [{"type": "text", "text": "3"}]
    structuredContent = {"result": 3}
    isError = False （三件套：文本 + 结构化结果 + 错误旗标）
  工具注册（tools/list）：2 个 —— add · greet（@mcp.tool() 即注册）
  默认参生效（greet 只传 name，不传 greeting）：服务端给默认 → "嗨，小明"
  五类探测的真实响应（1.28.1 引擎）：
    add{a:3}          -> 带内 isError=True  text="Error executing tool add: 1 validation error for addArguments\nb\n  Field required [type=mis"…
    add{a:3,b:'x'}    -> 带内 isError=True  text="Error executing tool add: 1 validation error for addArguments\nb\n  Input should be a valid "…
    tools/call nope    -> 带内 isError=True  "Unknown tool: nope"
    no/such/method    -> 协议层 JSON-RPC Error：code=-32602  message="Invalid request parameters"
                       （未知方法名报在协议层 = 客户端 SDK 抛异常，不落进 CallToolResult）
  分层结论：
    · 工具层错误（缺参/错型/工具名错）全部落在 CallToolResult 内：isError=True + content.text
      → 对客户端不是 Python 异常，而是一段『文本结果』→ 可直接喂回 LLM
    · 协议层错误（方法名拼错）才生成 JSON-RPC Error：code=-32602（未知方法）
      → 这是带外错误：客户端 SDK 抛异常，与工具结果不同通道
    · 诚实细节：本 SDK 对未知方法返回 -32602『Invalid request parameters』，而非规范建议的 -32601

==============================================================
账 D  生态账 —— who-speaks-MCP：本机装过的框架谁带 MCP？版本节奏？选择树 8/8
--------------------------------------------------------------
  本机属性探测（hasattr，真实 import）：
    Semantic Kernel     : ✓ 带 MCP  as_mcp_server 类方法（v0.32 实测 → tools/list 1:1 注册）
    OpenAI Agents SDK   : ✓ 带 MCP  .mcp 子模块（v0.35：import 名 agents 而非 openai_agents）
    CrewAI              : ✓ 带 MCP  .mcp 属性（v0.34）
    AG2                 : ✗ 无 MCP 接口  无专用 MCP 属性（v0.33）
    DSPy                : ✗ 无 MCP 接口  无（Prompt 编程不走协议，v0.36）
    LlamaIndex          : ✗ 无 MCP 接口  无（v0.29 实测）
    Haystack            : ✗ 无 MCP 接口  无（v0.31 实测）
    LangChain           : ✗ 无 MCP 接口  无（v0.28 实测）
  协议版本档（要素事实）：
    Sdk 认得的版本档 = ['2024-11-05', '2025-03-26', '2025-06-18', '2025-11-25']（2024-11-05 首批 → 2025-03-26 谈判默认档 → 2025-06-18 → 2025-11-25 当前档）
  选择树（8 场景，断言 8/8）：
    场景1：要给远程/外部 Agent 暴露工具（跨进程·跨语言）
       → 引入 MCP Server（协议层是主线）
    场景2：模型只在自己进程内调函数（单程序单语言）
       → function calling 直接调，别上协议
    场景3：要消费现成模块/公司的能力
       → 找他们有没有现成 MCP Server
    场景4：写工具想免去手工维护 DTO/Schema
       → @mcp.tool() 注解即注册，仍属协议面
    场景5：仓库内部多个 Python 进程共享工具
       → stdio/streamable-http 内网直连即可
    场景6：服务端能力还没想全
       → capabilities 诚实自报，少即少报
    场景7：想把工具做成『生态』而非散函数
       → MCP 把工具标准化 → 可替换可组合
    场景8：已有 REST/gRPC 想暴露给 Agent
       → 包一层 MCP Server 做协议翻译，不动业务
    断言 8/8（场景-动作规则 = 作者按 §7 大意整理；统计 = 本机确定性枚举）
  诚实边界：
    引擎语义（协商/工具注册/调用/错误分层/帧流）= 本机真实实测；
    server_version 缺省 = Sdk 版本（1.28.1）= 本机实测要素；
    版本节奏、生态归属、能力自报惯例 = 要素事实（未连任何远程 MCP 服务器，无外网未逐版复核）。
````
</details>

---

## 3. 实验 A · 协议面账（词汇表 + 17 方法 + 3 传输 + 两个角色）

**问题**：MCP 说自己是「协议」——那协议到底规定了多少个动词、多少条管道、几张能力表？

**实测**（stdout 关键行）：

- **常量三件套**：`LATEST_PROTOCOL_VERSION = "2025-11-25"`、`DEFAULT_NEGOTIATED_VERSION = "2025-03-26"`、`SUPPORTED_PROTOCOL_VERSIONS = ["2024-11-05","2025-03-26","2025-06-18","2025-11-25"]`（四个档，全部由 `mcp.types`/`mcp.shared.version` 本机读出）。注意 **SUPPORTED 与 DOC 版本完全同构**：`2024-11-05` 首批、`2025-03-26` 协议 RFC 化后的默认档、`2025-06-18`、`2025-11-25` 当前主版本（要素事实）。
- **17 个客户端请求方法**（从 `ClientRequest.root` 联合类型逐个取 `method` 字段默认值实测——注意不是定义一个写死数组，是**读引擎的类型面**）：
  - 会话：`initialize`（握手）· `ping`（心跳）
  - 工具：`tools/list`（目录）· `tools/call`（执行）
  - 资源：`resources/list` · `resources/templates/list` · `resources/read` · `resources/subscribe` · `resources/unsubscribe`
  - 提示：`prompts/list` · `prompts/get`
  - 补全：`completion/complete`
  - 任务：`tasks/list` · `tasks/get` · `tasks/result` · `tasks/cancel`
  - 日志：`logging/setLevel`
  - 三大原语 = **tools + resources + prompts**（§7.12 说的「工具/资源/提示三大原语」在方法面的落地）。
- **三种传输**：`mcp.client.stdio` / `mcp.client.sse` / `mcp.client.streamable_http`（及 server 侧 stdio）import 探测全在位。**同一套 17 个方法在三条管子上走**——协议词汇面与传输解耦。
- **两个角色**：Server（声明能力 · serve 工具）↔ Client（发起请求 · 消费工具）。**谁接谁的接口是固定的**——Client 永远先 `initialize`，Server 永远先 `result` 回能力。
- **能力自报**：`capabilities{tools,prompts,resources}` —— 协商期一次性广播（`{"prompts": {"listChanged": false}, "resources": {"subscribe": false, "listChanged": false}, "tools": {"listChanged": false}}`），之后请求才能发。**只有声明过的能力才可被调**——这是协议的「诚实」机制：少报少承诺。

**结论**：协议不是一个工具函数，是一张**固定方向的接口表**——17 个动词 × 2 个角色 × 3 条管道，能力分发靠协商期的 `capabilities` 广播。

## 4. 实验 B · 握手帧账（三帧定版本，屏蔽的诚实）

**问题**：协议怎么定「版本」？谁说了算？

**实测**（stdout 关键行）：

- **服务端 `create_initialization_options`**（实测）：`server_name="calc-server"`、`server_version="1.28.1"`——**注意 `server_version` 缺省 = Sdk 版本**（`mcp 1.28.1`），即「服务器报的版本默认就是它用的 SDK 版本」（要素+本机实测）。
- **正常握手三帧**（帧流全记录）：
  1. `C→S  request id=1 initialize`（带 `protocolVersion=2025-11-25` + `capabilities={}` + `clientInfo`）
  2. `S→C  result id=1`（回：`protocolVersion=2025-11-25` + `serverInfo{name,version:"1.28.1"}` + 能力回声 + `instructions`)
  3. `C→S  notification notifications/initialized`（**无 id**——通知与请求的区别：请求要配对 result，通知是单向）。
  - 之后 `id=2 tools/list` → `result`、`id=3/4 tools/call` → `result`。**每个请求都带 id，响应按 id 配对**。
- **谎报版本协商边缘**（客户端声称 `protocolVersion="1999-01-01"`）：**服务端照样接受，返回合法 result，协商到 LATEST `2025-11-25`**。这颠覆一个直觉：协议版本协商不是「客户端必须报对才放行」，而是「双方谈，谈不拢则各自收敛到共识档」——本 SDK 的实现是**接受任意声称版本并返回自己认得的最高档**。

**结论**：握手三帧 = `initialize`(客户端出自己的牌) → `result`(服务端回能力与版本) → `initialized`(客户端确认结束握手)。版本是协商出来的，不是验出来的——这是协议的谈判语义，也是为什么 `mcp` 的 `Server.run` 是一次性的（握手之后服务端状态才可服务业务请求）。

## 5. 实验 C · 工具调用与错误分层账（带内 isError vs 带外 JSON-RPC error）

**问题**：工具出错，协议怎么回？——这决定 LLM 能不能 「读错、回错、自愈」。

**实测**（stdout 关键行）：

- **正常回包三件套**（`tools/call add{a:1,b:2}`）：
  - `content = [{"type":"text","text":"3"}]`（给 LLM 读的文本）
  - `structuredContent = {"result": 3}`（给程序用的结构化结果）
  - `isError = false`（旗标）
- **默认参生效**：`greet{name:"小明"}` 不传 `greeting` → 服务端默认 → `"嗨，小明"`（默认值在**服务端**的 `inputSchema` 里声明、由服务端落地——客户端只需少传）。
- **五类探测**（工具层错误 vs 协议层错误分两层的铁证）：
  | 探测 | 响应层 | 实测响应 |
  |---|---|---|
  | 缺必选参数 `add{a:3}` | 带内 | `isError=True`，text=`"Error executing tool add: 1 validation error for addArguments\nb\n  Field required …"` |
  | 类型错位 `add{a:3,b:"x"}` | 带内 | `isError=True`，text=`"…b\n  Input should be a valid integer, unable to parse string as an integer …"` |
  | 工具不存在 `tools/call{name:"nope"}` | 带内 | `isError=True`，text=`"Unknown tool: nope"` |
  | 协议层未知方法 `no/such/method` | 带外 | JSON-RPC `{"code": -32602, "message": "Invalid request parameters"}` |

**分层结论**：
- **工具层错误 = 数据，不是异常**：缺参/错型/工具名错全部落在 `CallToolResult` 里（`isError=True` + `content.text`）。到客户端不是 Python 异常，是一段**可读文本结果** → **可直接喂回 LLM**（这正是 08-RAG 13 章「自愈」那条线的协议层前提）。
- **协议层错误 = 才到 JSON-RPC error 层**：方法名拼错（`no/such/method`）才生成带外错误，客户端 SDK 抛异常、打断流程。
- **诚实细节**：本 SDK 对未知方法报 `-32602`「Invalid request parameters」，**不是规范文档建议的 `-32601` MethodNotFound**（要素事实 —— SDK 实现与规范建议有一个具体出入；文档写的是建议，实现说了算）。

## 6. 实验 D · 生态账（who-speaks-MCP + 版本档 + 选择树 8/8）

**问题**：07 章十一家框架里，谁真的「会说」MCP？

**实测**（stdout 关键行）：

- **who-speaks-MCP（本机 `hasattr` 属性探测，非臆测）**：
  - ✅ **有 MCP 接口**：Semantic Kernel（`Kernel.as_mcp_server` 类方法，v0.32 实测 `tools/list` 1:1 注册）、OpenAI Agents SDK（`.mcp` 子模块，v0.35：import 名 `agents`）、CrewAI（`.mcp` 属性，v0.34）。
  - ❌ **无 MCP 接口**：AG2（v0.33）、DSPy（v0.36，Prompt 编程不走协议）、LlamaIndex（v0.29）、Haystack（v0.31）、LangChain（v0.28）。
  - 读这张表要说清楚：**「有接口」是引擎事实，「生态认出它」是另一回事。** SK/Agents SDK/CrewAI 有 MCP 通道，不代表 LangChain/Haystack 不能用 MCP——它们能通过 `mcp` 客户端消费一个 MCP Server，只是没有开箱即用的属性。协议的赢面不在某一家框架里，在**跨框架都认这条协议**。
- **协议版本档**：四个档按时间序 `2024-11-05 → 2025-03-26 → 2025-06-18 → 2025-11-25`（要素事实）——协议自己也在演进，`SUPPORTED_PROTOCOL_VERSIONS` 就是演进足迹。
- **选择树 8 场景断言 8/8**：见 stdout 账 D——一句话：跨进程/跨语言/给外部 Agent 暴露工具 → MCP；单进程内调函数 → function calling 直接调；消费现成能力 → 找现成 MCP Server；已有 REST/gRPC → 包一层 MCP Server 做协议翻译。

**结论**：MCP 是「2026 标配技能」（§7.12 实践）——因为工具**一次实现、处处可用**必须有一个「处处」都认的接口，框架各有各的注脚，只有协议是共线。

## 7. 常见坑（本机实测踩过的 7 条）

1. **`SessionMessage` 不在 `mcp.types`**：在 `mcp.shared.message`。类型导入时被 `from mcp.types import ...` 惯性坑到（`ImportError: cannot import name 'SessionMessage'`）。
2. **`FastMCP` 没有 `create_initialization_options` / `.run`**：那是**低层** `server._mcp_server`（`mcp.server.lowlevel.Server`）的。上层 `FastMCP` 是糖，帧级控制要走 `. _mcp_server`。
3. **`Server.run` 一次性**：跑完即终态（版本协商状态被吃掉），第二次握手会拿到空 result——**每次协商用全新实例**。
4. **原始帧必须包 `JSONRPCMessage(root=...)`**：直接 `send(JSONRPCRequest(...))` 服务端收不到（`'JSONRPCRequest' object has no attribute 'root'`）——`SessionMessage(message=JSONRPCMessage(root=req))` 才是传输面要吃的样子。
5. **服务器执行时的校验错误树/INFO-WARNING 巨吵**：不静音会污染 stdout 导致 md5 漂移——`logging.disable(CRITICAL)` + `redirect_stdout/stderr(StringIO())` 双管齐下。
6. **`completion/complete` 不是 `completions/complete`**：方法面实测值就是单数 `completion/complete`（别按直觉写复数）。
7. **未知协议方法报 `-32602` 不是 `-32601`**：本 SDK 的实现（诚实边界：规范建议是 `-32601`，实测 `-32602`「Invalid request parameters」）。

## 8. 诚实边界（AAA 自我审查）

- **引擎语义 = 本机真实实测**：mcp `1.28.1` 的握手（`initialize→result→initialized`）、版本协商（接受任意声称档并返 LATEST）、17 方法面（读 `ClientRequest.root` Union）、三种传输 import 面、`tools/list` Schema、`tools/call` 五类响应（正常/缺参/错型/未知工具/协议层 `-32602`）、`server_version` 缺省=Sdk 版本、帧级双向实录——全部真 API 执行（stdout md5 `5bf0d14d` 三个独立进程逐字节恒一）。
- **传输层 = 进程内 memory-stream 环回**：零网络、零 socket——测的是协议帧本身，不是 TCP/HTTP 传输行为。`stdio` 的真实进程管道、`streamable-http` 的真实 HTTP 握手不在本探针量程（要素事实）。
- **错误响应文本 = 本机真实产出**：缺参/错型文本是 pydantic `2.12` 校验器 + mcp SDK 拼出来的（`Error executing tool add: …`），`Unknown tool: nope` 是 SDK 模板。
- **协议版本档与生态归属 = 要素事实**：四个版本档时间序、Anthropic 2024-11 提出、2025 开放标准、OpenAI/谷歌 2025 支持、三大原语说法、who-speaks-MCP 的「生态」解读——无外网未逐版复核。
- **未连接任何远程/第三方 MCP Server**：who-speaks-MCP 只是 `hasattr` 探测本机装的框架包，不能推断远程生态全貌。
- **选择树 8/8 规则 = 作者按 §7 大意整理**：统计（断言计数）= 本机确定性枚举。
- **墙钟只进 stderr**（≈22 s=账 D 重框架 import 主导；三进程浮动但 stdout md5 恒一）。

## 9. 参考与衔接

- **本课地图**：§7.12（MCP 定位：不是框架是协议；client-server；三原语 tools/resources/prompts；2024-11 Anthropic 提出、2025 开放标准；2026 标配技能）+ §7.1（协议层覆盖 1 维特判）+ §7.13（MCP 底座影响面 9；框架分层=协议层+轻量SDK+图引擎+平台）+ §7.14（决策树）+ §17.5.10（应用编程接口类）+ 链路 `013`（手写 MCP Server 并调用）。
- **选择树 8 场景**（探针断言 8/8，要素事实）：见账 D——一句话：跨进程暴露工具 → MCP；单进程调函数 → function calling；消费现成能力 → 找现成 MCP Server；已有 REST/gRPC → 包一层 MCP 翻译；协议层不要拿来当状态/图/编排用（00 账 B「覆盖 1 维」）。
- **前承**：00-框架分类学（协议层特判 + MCP 底座影响面 9）；05-Function-Tool-Calling（手搓工具协议四档 → 本篇标准协议壳）；06-Semantic-Kernel（`as_mcp_server`→`tools/list` 1:1 先例）；10-OpenAI-Agents（`mcp_servers` 挂接点）；08-RAG 13（自愈=工具错误文本喂回 LLM 的挂靠）；09-DSPy（`gpt-4` 的 Python 工具 ≠ 协议，协议层在它下面）。
- **后启**：`12-框架继承关系与选型决策`（✅ 已交付 v0.38 收束章：完整 DAG + 「该不该引入框架」量化——本篇「MCP 不是框架」是「协议可不引入框架」的判定依据，账 C 协议场景 MCP 域内净分 +0.78）。

> 本篇完工于 2026-09-22（v0.37 批次）；探针代码 `code/notebooks/_tools/mcp_demo.py`；stdout md5 `5bf0d14d`（8233 字节，三独立进程逐字节恒一）。
