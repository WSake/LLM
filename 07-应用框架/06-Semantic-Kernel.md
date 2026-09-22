# 🎯 Semantic Kernel：企业语言栈的编排 SDK —— 把『技能+规划器』重构成 插件/函数/自动调用 三原语（微软 · .NET/Azure 企业栈）

> 对应知识点：知识地图 §7.7（**Semantic Kernel（微软；位置：企业 .NET/云 SDK）**——"特点：与 Azure AI、.NET/C#/Python/Java 深度绑定；把『技能（Skill）+规划器（Planner）』封装；企业级治理与传统开发语言友好。定位：微软技术栈企业的选择；独立开发者用得少"）、§7.1（类 1 **通用编排 SDK**：把模型+工具+检索串成链，代表 LangChain / Haystack / Semantic Kernel——注意 SK 不是 05 那样"通用编排×RAG 框架"的双栖点，它只在类 1）、§7.14（选型：**.NET 企业 → Semantic Kernel**）、§7.8（**AutoGen 分支**：AG2 社区维护；微软 2025 年推出 Microsoft Agent Framework——承接 AutoGen，与 Semantic Kernel 统一）。
> 前置：[00-框架分类学](./00-框架分类学.md)（开篇章：账 B 矩阵里 SK 标格=次主次次·覆盖 4/4 维，账 C 类别 1 代表三家里有它，选择树 10 个场景只有场景⑨专给它——**本篇=把场景⑨".NET 企业栈"展开成四账**）；[05-Haystack](./05-Haystack.md)（账 D 决策表场景⑤的指针："多轮状态恢复不归 Haystack，**.NET 栈归 Semantic Kernel**"+ 账 A `max_runs_per_component` 护栏的对偶参照）；[02-LangGraph-状态图与Checkpoint](./02-LangGraph-状态图与Checkpoint.md)（账 A 的 `recursion_limit` **内核级上限**真抛 `GraphRecursionError`——本篇账 B 的护栏对照：SK 的上限是"配置"不是"内核强制"）；[06-应用开发 05-Function-Tool-Calling](../06-应用开发/05-Function-Tool-Calling.md)（工具调用的入参校验/循环回喂/并行 DAG 账=本篇账 B 的语义源头：SK 的自动函数调用是把同一件事架构进引擎）。
> 动手：`python code/notebooks/_tools/semantic_kernel_demo.py`（**本机真实执行 semantic-kernel 1.44.1 引擎**；LLM 决策/真实模型=预置仿真保确定性；零网络 · 零随机 · stdout md5 `66c105f5…` 三个独立进程恒一）。
> 一句话：**SK 的"函数即一等公民"不是口号，是可复现的机制——三条注册路径（对象类 / 纯函数 / 显式聚合）全部收敛到 `@kernel_function` 装饰器（它就是契约载体：名称/描述/参数 Schema 由它携带，裸函数注册当场 `FunctionInitializationError`）；缺参在运行期按签名强制抛 `KernelInvokeException` 而不是静默吞掉；`FunctionCallContent` 让引擎把工具调用编排成 TOOL 回填闭环、缺参错误文本由引擎按签名生成（自愈有据）；但护栏是『要你自己配的配置』（`FunctionChoiceBehavior.maximum_auto_invoke_attempts`），不是 02 章 `recursion_limit` / 05 章 `max_runs_per_component` 那样的内核级真兜底；`as_mcp_server()` 把整个插件 1:1 导出成 MCP Server（工具名=函数短名）。它不是 Python 栈的平替，是企业栈的入场券——0.x→1.x 四个概念子包整包消失，改名烈度 07 章之最，冲它而来的 Python 开发者多数会失望。**
>
> 📌 诚实边界：**引擎语义 = 本机真实实测**（semantic-kernel 1.44.1 的 Plugin/Function 注册、`k.invoke` 往返、`invoke_function_call` 自动调用、缺参错误文本生成、TOOL 回填、`as_mcp_server`→tools/list 真实注册、import 探测真跑真报错）；但 **LLM 决策 = 非本机实测**——探针里"何时调哪个工具/参数值/何时收手"是作者预置的词面规则桩，真实 LLM（GPT/DeepSeek…）经 SK 连接器输出的 function calling 行为=非本机实测；**0.x→1.x 版本节奏、`FunctionChoiceBehavior` 字段名 = 要素事实**（写作环境无外网、未在线复核官方文档）；**8 场景决策树 = 作者按知识地图 §7.14 与 §7.1 整理（要素事实）**。第三节嵌的每一行「实测」均可 `python code/notebooks/_tools/semantic_kernel_demo.py` 复现，stdout 与正文逐字节一致；墙钟 ≈0.1 s 仅进 stderr。

---

## 📑 本章目录

1. [为什么 06-Semantic-Kernel 值得单独讲一课？](#0-为什么-06-semantic-kernel-值得单独讲一课)
2. [先钉死事实：知识地图给 SK 的定位（§7.7 / §7.1 / §7.14 / §7.8）](#1-先钉死事实知识地图给-sk-的定位-77--71--714--78)
3. [把「函数即一等公民」降维成本机测量（探针设计 + 全量实录）](#2-把函数即一等公民降维成本机测量探针设计--全量实录)
4. [实验 A · 插件/函数一等公民（三路注册收敛 @kernel_function · 运行期签名强制）](#3-实验-a--插件函数一等公民三路注册收敛-kernel_function--运行期签名强制)
5. [实验 B · 自动函数调用编排（并行子调用 · 报错自愈 · 护栏三态）](#4-实验-b--自动函数调用编排并行子调用--报错自愈--护栏三态)
6. [实验 C · 版本存续（0.x → 1.x 概念全名消失 · 改名烈度）](#5-实验-c--版本存续0-x--1-x-概念全名消失--改名烈度)
7. [实验 D · MCP 桥接与选择树（as_mcp_server 1:1 导出 · 8 场景断言）](#6-实验-d--mcp-桥接与选择树as_mcp_server-11-导出--8-场景断言)
8. [拿这张表怎么读本目录（07 章 13 篇的路牌）](#7-拿这张表怎么读本目录07-章-13-篇的路牌)
9. [常见坑（5 个）](#8-常见坑5-个)
10. [诚实边界（AAA 自我审查）](#9-诚实边界aaa-自我审查)
11. [参考与衔接](#10-参考与衔接)

---

## 0. 为什么 06-Semantic-Kernel 值得单独讲一课？

07 章的路牌上，00-框架分类学给 Semantic Kernel 的位置比其他框架更"欠"：账 B 矩阵标格=次主次次 · 覆盖 4/4 维（组装=主业；状态/观测/成本=顺带），选择树 10 个场景**只有一个是专给它的**（**场景⑨ .NET 企业栈 → Semantic Kernel**）。这个坐标的意思是：SK 的展开主题不是"通用编排"（那是 01-章 LangChain 的一格）、不是"图引擎"（02-章）、不是"RAG 管线"（05-章）——**本篇要展开的，是知识地图 §7.7 那句"把『技能（Skill）+规划器（Planner）』封装"的机制**。

这句在 0.x 时代是真的对象（`Skill` 类、`SKFunction` 类、`ContextVariables` 类、`SequentialPlanner` 类——planning 子包）；到 1.x，**全部降维成两样东西：`@kernel_function` 装饰器（技能→函数，函数=一等公民）和 `FunctionCallBehavior.Auto`（规划器→自动函数调用，规划器 LLM 化）**。本篇用真实 1.44.1 引擎把这条演化量出来，加上四件可复现的机制：

- **三路注册=一个契约**（账 A）：`add_plugin(对象类)` / `KernelFunctionFromMethod` + `add_functions(纯函数)` / `KernelPlugin(显式聚合)` **全部收敛到同一个装饰器**——`@kernel_function` 携带名称/描述/参数 Schema；裸函数注册当场 `FunctionInitializationError: Method is not a Kernel function`（装饰器就是入场券）。
- **运行期签名强制**（账 A）：缺参 `invoke` → `KernelInvokeException`『Parameter city is required but not provided in the arguments.』——**SK 管签名、不静默吞**。（对照 06-应用开发 05 账"校验器兜值域"：SK 兜形状与必填，值域语义仍归你的函数体。）
- **引擎真跑自动调用**（账 B）：LLM 出 `FunctionCallContent` → `invoke_function_call` 真执行并 TOOL 回填；**缺参错误文本由引擎按键签名生成**（`Missing required argument(s): ['city']…`），让 stub 自愈闭环有据可依。
- **护栏=配置而不是内核强制**（账 B）：SK 内核对自动调用**没有内置总轮上限**（`invoke_function_call` 手动循环不设防）——对比 02 章 `recursion_limit`（真抛 `GraphRecursionError`）/ 05 章 `max_runs_per_component`（真兜 `PipelineMaxComponentRuns`），SK 的上限是 `FunctionChoiceBehavior.Auto(maximum_auto_invoke_attempts=N)` 这个**设置项**（在连接器自动调用内循环生效）；绕过连接器手动编排=应用自管。
- **改名烈度 07 章之最**（账 C）：0.x 四个概念子包（`orchestration` / `skill_definition` / `planning` / `core_skills`）整包消失；`Kernel` 与 import 名/包名（`semantic_kernel` / `semantic-kernel`）存活——**概念全换、身份存活**，比 LangChain（移子包）/ Haystack（换包名）更狠。
- **企业栈桥**（账 D）：`k.as_mcp_server()` 把整个插件 **1:1** 导出成 MCP Server（tools/list 真实注册、工具名=函数短名、排除也按短名）——"技能封装"今天的一端是 MCP 出口，SK 是发工具到协议层的企业级发射器。

和 05-章 Haystack 的对照正好互补：**Haystack 把"严谨"做成装配期类型契约（`connect()` 立断、待填槽可查、`max_runs` 真兜底），SK 把"企业"做成函数契约（装饰器=入场券、签名=运行时强制、`Auto`=规划器 LLM 化）**——一个给 Python/企业 RAG，一个给 .NET/微软技术栈。04-章"逃离清单③ 深度 RAG 调优"的归处在 05-章补了生产线，本篇再补企业栈这杆：老一代的 Skill/Planner 对象不适合今天的 .NET 工程，1.x 的 Plugin/Function/Auto 才是。

> **本篇与 00-章账 B 的一致性**：00-章账 B 给 SK 标格 = 次主次次 · 覆盖 4/4 维（组件=主业；状态/观测/成本=顺带）——**本篇账 A 的"函数即一等公民"恰是"组装"主业的机制面（把一条链上的环节注册成可调用函数）；账 B 的护栏对照、账 D 的 MCP 出口都挂在这张标的语义上**。跨章数据一致是硬纪律。

## 1. 先钉死事实：知识地图给 SK 的定位（§7.7 / §7.1 / §7.14 / §7.8）

四处来源各给一句可复核的定位：

1. **§7.7 专节盖章**：Semantic Kernel（**微软**；位置：**企业 .NET/云 SDK**）——"与 Azure AI、.NET/C#/Python/Java 深度绑定；把『技能（Skill）+规划器（Planner）』封装；企业级治理与传统开发语言友好。定位：微软技术栈企业的选择；独立开发者用得少"。**"终身有效的企业级答案"两句话各有一个可测量对应**："与传统开发语言友好"→ 本篇账 A 的显式 API（无框架魔法，全是要能读的注册/契约/强制）；"独立开发者用得少"→ §7.7 明确把 SK 钉死在微软技术栈，**这不是 Python 栈的平替**（本篇账 D 场景⑧的警句）。
2. **§7.1 分类学盖章（类 1 通用编排 SDK）**："把模型+工具+检索串成链"代表 LangChain / Haystack / Semantic Kernel。**注意与 05 的差异**：清单里 SK 只在类 1，**不是像 Haystack 那样「通用编排 × RAG 框架」的跨类双栖点**——它的分类学坐标更窄：纯通用编排、纯企业栈。
3. **§7.14 决策树盖章**：".NET 企业 → Semantic Kernel"——§7.14 里**唯一点名 SK 的场景**，也是本篇账 D 决策表的源（场景⑨在 00-章账 D 已断言过一次，本篇用完整四账把它展开）。
4. **§7.8 相邻分支盖章**：AutoGen（2023，微软）开创多 Agent 对话范式；**分支 AG2（社区维护）；微软 2025 年推出 Microsoft Agent Framework（承接 AutoGen，与 Semantic Kernel 统一）**——SK 的"下一步"不是 Python 化，而是 07 章的下一站（07-AutoGen·AG2）与它的统一。这是 07 章路牌上 SK 与 AutoGen 相邻的官方理由。

再钉反向边界：SK **不**解决"有状态、可分支、可恢复的流程"（那是 02-章图引擎，账 B 证明它连"自动调用总轮上限"都不内置）、**不**解决"生产级 RAG 管线的类型契约"（那是 05-章，SK 的检索要自己接线）、**不**承诺低代码（那是 04-章）、**不**发明协议标准（那是 11-章 MCP；SK 的 `as_mcp_server` 是**消费** MCP 的出口，不是自造协议）。**它给的是"把技能+规划器组织成插件+函数+自动调用、且全部显式、可治理"的纪律，不是"任意应用的 Python 兜底平台"。**

## 2. 把「函数即一等公民」降维成本机测量（探针设计 + 全量实录）

一篇"SK 很企业"的口水文不稀奇；稀奇的是把**"函数一等公民"的五个机制（注册收敛 / 签名强制 / 自动调用 / 护栏三态 / 命名演化）变成可复排的本机实验**。与 05-章同级的思路，本篇探针真实安装了 **semantic-kernel 1.44.1** 引擎（网线可达的一次性安装；运行过程零网络零随机），量 4 本可复排的账：

- **账 A 插件/函数一等公民**：三条注册路径（对象类 / 纯函数 + `KernelFunctionFromMethod` / 显式 `KernelPlugin`）都收敛到 `@kernel_function`；函数元数据（fq 名=`plugin-function`、参数契约由类型注解推导 `is_required`/默认值）；`k.invoke` 往返；缺参 → `KernelInvokeException`（运行期签名强制）；
- **账 B 自动函数调用编排**：`FunctionCallContent` 并行双工具（一次推理发 2 枚 → 引擎逐枚执行、TOOL 回填、收敛答复）；报错自愈（stub 漏参 → 引擎生成错误文本 → 修正 1 次收敛 3 轮）；护栏三态（内核无内置上限 → 应用侧 `max_iter` 截断；`FunctionChoiceBehavior.Auto(maximum_auto_invoke_attempts=N)`=引擎侧预算选项）；
- **账 C 版本存续**：0.x 四大概念子包（`orchestration` / `skill_definition` / `planning` / `core_skills`）**真 import 探测**全 FAIL；1.x 存活面（`Kernel` / `functions.kernel_function` / `KernelFunction` / `KernelPlugin` / `KernelArguments` / `contents.ChatHistory` / `connectors.mcp`）全 OK；概念映射表 + 改名烈度与 01/05 对比；
- **账 D MCP 桥接与选择树**：`k.as_mcp_server()` → mcp Server → **tools/list 真实注册**（4 函数 → 4 工具 1:1、工具名=函数短名、`excluded_functions` 按短名排除）；8 场景选择树断言 8/8（§7.14 + §7.1，要素事实）。

确定性纪律：`warnings.filterwarnings("ignore")` 最先执行（stderr 纯净，不污染 md5）；`logging` 打静音；全程序零网络、零随机、不打印对象地址；stub LLM=词面规则预置（决策可复现）。**stdout md5 `66c105f5…` 三个独立进程逐字节恒一**，墙钟 ≈0.1 s 仅进 stderr。

复现实录（严格逐字节 = 探针 stdout 三遍恒一）：

<details>
<summary>📊 探针 stdout 全量实录（md5 <code>66c105f5…</code> · 墙钟仅 stderr）</summary>

```text
semantic_kernel_demo：07-应用框架 · 06-Semantic-Kernel（知识地图 §7.7/§7.1/§7.14）
    『技能+规划器』重构为 插件/函数/自动调用 三原语，函数=一等公民 —— 企业语言栈 SDK：契约 · 自动调用 · 版本 · MCP
[0] 口径：引擎语义=本机真实实测 semantic-kernel 1.44.1；LLM 决策/真实模型=预置仿真（非本机实测）
========================================================================
[账 A] 插件/函数一等公民账：SK 把『工具』建模成 KernelFunction——@kernel_function 装饰器=契约载体
  注册路 1 add_plugin(WeatherPlugin(), plugin_name='weather')     —— 对象类（2 函数）
  注册路 2 KernelFunctionFromMethod(@kernel_function def add) -> add_functions('math2') —— 纯函数（1 函数）
  注册路 3 KernelPlugin(name='cal', functions=[...]) -> add_plugin —— 显式聚合（1 函数）
  插件清单：cal / math2 / weather
  函数元数据（一等公民证据：fq 名=plugin-function · 参数契约由类型注解+默认值推导）
    cal.add_event  fq=cal-add_event  desc=向日程添加一条事件  is_prompt=False
        params: date(str, required=True)
    math2.add  fq=math2-add  desc=两个数相加  is_prompt=False
        params: a(float, required=True) · b(float, required=True)
    weather.get_forecast  fq=weather-get_forecast  desc=查询指定城市未来几天的预报  is_prompt=False
        params: city(str, required=True) · days(int, required=False, default=3)
    weather.get_weather  fq=weather-get_weather  desc=查询指定城市的当前天气  is_prompt=False
        params: city(str, required=True) · unit(str, required=False, default=celsius)
  调用往返（无 LLM 真执行）：
    invoke weather-get_weather(city=北京)                 -> '北京: 20C sunny (celsius)'
    invoke weather-get_weather(city=上海, unit=华氏)       -> '上海: 20C sunny (华氏)'
    缺参 invoke weather-get_weather(unit=华氏)  -> 抛 KernelInvokeException『Parameter city is required but not provided in the arguments.』（运行期签名强制）
  断言：三路注册等价可见 · fq 名=plugin-function · 装饰器携带名称/描述/参数 Schema · 缺参=运行期契约强制
========================================================================
[账 B] 自动函数调用编排账：stub LLM 出 FunctionCallContent → 引擎 invoke_function_call 真执行 → TOOL 回填
  诚实边界：LLM 决策（何时调哪个工具/参数值/何时收手）=词面规则预置（真实 LLM 非本机实测）；
            invoke_function_call 的解析/执行/缺参错误生成/TOOL 回填=引擎本机真实实测
    TOOL#1 [weather.get_weather] -> 北京: 20C sunny (celsius)
    TOOL#2 [cal.add_event] -> ok 已安排 周三
  场景 1 · 并行双工具：一次推理发 2 枚 function_call，引擎逐枚执行并 TOOL 回填，收敛答复
    度量：工具执行 2 次 · function_call 2 枚 · TOOL 消息 2 条 · 推理轮 2 轮 · 形状: user[1] / assistant[2] / tool[1] / tool[1] / assistant[1]
  场景 2 · 报错自愈：stub LLM 参数漏 city -> 引擎不执行函数体、按签名生成错误 TOOL 回填
    TOOL#[错] [weather.get_weather] -> Missing required argument(s): ['city']. Please revise the arguments to match the function signature.
    TOOL#[修] [weather.get_weather] -> 上海: 20C sunny (celsius)
    度量：错误重试 1 次 · 收敛 3 轮 · 错误文本由引擎生成（非 stub 造）· 形状: user[1] / assistant[1] / tool[1] / assistant[1] / tool[1] / assistant[1]
    第 1 轮 stub 又发 function_call -> 引擎照常执行回填 TOOL: 北京#1: 20C sunny (celsius)
    第 2 轮 stub 又发 function_call -> 引擎照常执行回填 TOOL: 北京#2: 20C sunny (celsius)
    第 3 轮 stub 又发 function_call -> 引擎照常执行回填 TOOL: 北京#3: 20C sunny (celsius)
    第 4 轮 stub 又发 function_call -> 引擎照常执行回填 TOOL: 北京#4: 20C sunny (celsius)
    达到应用侧截断 max_iter=4（stub 永不收手）
    护栏账：SK Kernel 无内置『自动调用总轮上限』（调用数不受内核约束）；引擎侧预算=FunctionChoiceBehavior
            .Auto(maximum_auto_invoke_attempts=N)（在连接器自动调用内循环生效）——对比 02 章 recursion_limit / 05 章 max_runs_per_component：
            LangGraph/Haystack 是内核级上限真抛异常，SK 的上限是一份『要你自己配的配置』（若绕过连接器手动循环=应用自管）
========================================================================
[账 C] 版本存续账：0.x→1.x 概念全名消失——SK 是 07 章改名最狠的一家
    semantic_kernel.orchestration.sk_context        -> FAIL ModuleNotFoundError: No module named 'semantic_kernel.orchestra
    semantic_kernel.orchestration.sk_function_base  -> FAIL ModuleNotFoundError: No module named 'semantic_kernel.orchestra
    semantic_kernel.orchestration.context_variables -> FAIL ModuleNotFoundError: No module named 'semantic_kernel.orchestra
    semantic_kernel.skill_definition.sk_function_base-> FAIL ModuleNotFoundError: No module named 'semantic_kernel.skill_def
    semantic_kernel.skill_definition.sk_function_decorators-> FAIL ModuleNotFoundError: No module named 'semantic_kernel.skill_def
    semantic_kernel.planning.sequential_planner     -> FAIL ModuleNotFoundError: No module named 'semantic_kernel.planning'
    semantic_kernel.planning.action_planner         -> FAIL ModuleNotFoundError: No module named 'semantic_kernel.planning'
    semantic_kernel.core_skills                     -> FAIL AttributeError: module 'semantic_kernel' has no attribute 
  0.x 四大基本概念整包消失：orchestration / skill_definition / planning / core_skills
    semantic_kernel.Kernel                          -> OK <ModelMetaclass>
    semantic_kernel.functions.kernel_function       -> OK <module>
    semantic_kernel.functions.KernelFunction        -> OK <ModelMetaclass>
    semantic_kernel.functions.KernelPlugin          -> OK <ModelMetaclass>
    semantic_kernel.functions.KernelArguments       -> OK <type>
    semantic_kernel.contents.ChatHistory            -> OK <ModelMetaclass>
    semantic_kernel.connectors.mcp                  -> OK <module>
  核心身份（Kernel）与 import 名/包名（semantic_kernel / semantic-kernel）存活；变的是『一切描述功能的词』
  概念映射表（0.x -> 1.x）：Skill->KernelPlugin · SKFunction->KernelFunction · @skill_function->@kernel_function
        · ContextVariables->KernelArguments · Planner->FunctionCallBehavior.Auto（规划器 LLM 化=自动函数调用）
        · core_skills 名消失（实测 import FAIL），1.x 真名 core_plugins（TimePlugin/MathPlugin/TextPlugin 都在）
  断言：概念全换、身份存活 · 改名烈度高于 LangChain（移子包）/ Haystack（换包名）连核心概念词都换了
========================================================================
[账 D] 选择树 + MCP 桥接账：07-02 章场景 9『.NET 企业栈』的落点 + 插件一键变 MCP Server
  MCP 桥接：k.as_mcp_server() -> mcp Server -> tools/list 处理器真实注册表（本机真跑）
    kernel 插件函数 4 个 -> MCP 工具 4 个（1:1）
    tool=add          props=['a', 'b']  required=['a', 'b']  desc=两个数相加…
    tool=add_event    props=['date']  required=['date']  desc=向日程添加一条事件…
    tool=get_forecast props=['city', 'days']  required=['city']  desc=查询指定城市未来几天的预报…
    tool=get_weather  props=['city', 'unit']  required=['city']  desc=查询指定城市的当前天气…
    发现：MCP 工具名=函数短名（不带 plugin 前缀）——as_mcp_server 的 excluded_functions 也是按短名排除（文档原话），跨插件同名函数会撞名
    excluded_functions='add'（按短名排除）-> 工具剩 ['add_event', 'get_forecast', 'get_weather']
  选择树 8 场景断言 8/8：
    1.  .NET/C#/Azure 微软企业栈 -> Semantic Kernel（本篇——函数即一等公民的企业语言栈 SDK）
    2.  严谨生产级 RAG（Python/企业） -> Haystack（05 篇）
    3.  复杂状态/循环/恢复/中断 -> LangGraph（02 篇）
    4.  RAG 数据管线专精 -> LlamaIndex（03 篇）
    5.  非工程师快速搭应用 -> 低代码 Dify/Coze（04 篇）
    6.  想少依赖·轻量单 Agent -> OpenAI Agents SDK / 直写（10 篇）
    7.  跨栈工具互操作（一次实现处处可用） -> MCP 协议层（11 篇｜里程碑 013）
    8.  已深度持有 LangChain 资产 -> 同系演进 LangGraph/LlamaIndex——SK 是企业栈入场券、不是 Python 栈平替
  台账断言：MCP 工具数=插件函数数(1:1) · 工具名=函数短名 · props=参数名 · required=必填参数 · 排除按短名
========================================================================
台账汇总（A 插件函数 / B 自动调用编排 / C 版本存续 / D 选择树+MCP）
  A  三路注册 4 函数全部收敛到 @kernel_function 契约 · fq 名=plugin-function · is_required/默认值由签名推导 · 缺参运行期抛 KernelInvokeException
  B  并行：一次推理 2 枚 function_call → 引擎逐枚执行 TOOL#2 → 2 轮收敛 · 自愈：缺参引擎生成 'Missing required argument(s): ['city']'→修正 1 次 3 轮收敛 ·
      护栏：内核无内置总轮上限，引擎侧预算=FunctionChoiceBehavior.maximum_auto_invoke_attempts（连接器内生效，手动循环=应用自管）
  C  0.x 四个概念子包整包消失（orchestration/skill_definition/planning/core_skills）·import 名/包名/Kernel 存活 · Skill→Plugin 等映射=改名最狠
  D  选择树 8/8（.NET 企业→本篇）· as_mcp_server 真实注册 1:1 · 工具名=函数短名 · 排除按短名 · 跨插件同名=撞名风险
一句话：SK 把『技能+规划器』重构成 插件/函数/自动调用 三原语，函数=一等公民（@kernel_function=契约载体、签名=运行时强制、Auto=规划器 LLM 化）——
它是微软企业栈的编排 SDK（.NET/C#/Azure 场景的入场券），Python 栈冲它而来的人多数会失望：改名最狠、护栏要自带、MCP 桥接按短名 1:1 暴露插件
done · 一键复现：python code/notebooks/_tools/semantic_kernel_demo.py
```

</details>

## 3. 实验 A · 插件/函数一等公民（三路注册收敛 @kernel_function · 运行期签名强制）

**目标**：把 §7.7"与传统开发语言友好"降成可复排的机制——">技能+规划器"重构成 Plugin+Function 之后，**函数怎么被注册、契约写在哪、缺参会怎样**，全部要能"读到"、要显式。

**过程**：在真实 1.44.1 引擎上走三条注册路径（`WeatherPlugin` 对象类 / 纯函数 `add` 经 `KernelFunctionFromMethod` / 显式 `KernelPlugin` 聚合 `CalPlugin`），随后 `k.invoke` 往返三次（正常 × 2 + 缺参 × 1）。

**实测**（stdout 关键行）：

```
  注册路 1 add_plugin(WeatherPlugin(), plugin_name='weather')     —— 对象类（2 函数）
  注册路 2 KernelFunctionFromMethod(@kernel_function def add) -> add_functions('math2') —— 纯函数（1 函数）
  注册路 3 KernelPlugin(name='cal', functions=[...]) -> add_plugin —— 显式聚合（1 函数）
  插件清单：cal / math2 / weather
    cal.add_event  fq=cal-add_event  desc=向日程添加一条事件  is_prompt=False
        params: date(str, required=True)
    weather.get_forecast  fq=weather-get_forecast  desc=查询指定城市未来几天的预报  is_prompt=False
        params: city(str, required=True) · days(int, required=False, default=3)
    invoke weather-get_weather(city=北京)                 -> '北京: 20C sunny (celsius)'
    缺参 invoke weather-get_weather(unit=华氏)  -> 抛 KernelInvokeException『Parameter city is required but not provided in the arguments.』（运行期签名强制）
```

**结论**：

1. **三条路全部收敛到同一个装饰器 = "一等公民"的机制面**：对象类里注解方法、裸函数、显式 `KernelPlugin`——三种粒度写法最终都落在 `@kernel_function` 上，而**装饰器就是契约载体**（它把名称/描述/参数 Schema 写进 `__kernel_function_*__` 双下划线元数据）。所以"函数=一等公民"可复现为一句：**同一个 `@kernel_function`，无论函数长在哪，契约都一致**。反过来，**裸函数**（不装饰）经 `KernelFunctionFromMethod` 注册会当场 `FunctionInitializationError: Method is not a Kernel function`——装饰器不是语法糖，是入场券。
2. **参数契约由类型注解 + 默认值推导**：`city: str`（无默认）→ `is_required=True`；`unit: str = "celsius"` → `default_value=celsius`、`required=False`；`days: int = 3` 同理。**契约长在签名上、可机器读**——这正是 MCP/OpenAI tools 导出时 inputSchema 的来源（账 D 会看到 Schema 1:1 传进 tools/list）。
3. **缺参在运行期强制，不静默**：`invoke(unit=华氏)` 少了必填 `city` → 引擎抛 `KernelInvokeException`，内层消息『Parameter city is required but not provided in the arguments.』是**引擎按参数元数据实时生成**的（账 B 会复用这件事）。**SK 管"签名"，语义/值域仍归你的函数体**——这与 06-应用开发 05 章"校验器兜值域"的分工一致：SK 兜形状与必填，值校验是另一层。

## 4. 实验 B · 自动函数调用编排（并行子调用 · 报错自愈 · 护栏三态）

**目标**：SK 的规划器（0.x `SequentialPlanner`）在 1.x 变 `FunctionCallBehavior.Auto`——它到底怎么把"LLM 想调工具"变成"引擎真跑工具"？以及**循环收不收得住**？用 `FunctionCallContent` + `invoke_function_call` 在真实引擎上走三个场景。

**过程**：stub LLM 是词面规则桩（收到用户话 → 出什么 function_call、参数漏不错、何时收手，全部预置可复现）；引擎侧 `invoke_function_call` 逐枚真执行、真生成错误文本、真 TOOL 回填。**诚实边界**：LLM 的"决策"是桩（真实 LLM 经连接器自动调用的行为=非本机实测），引擎的"执行/解析/报错/填充"=本机真实。

**实测**（stdout 关键行）：

```
    TOOL#1 [weather.get_weather] -> 北京: 20C sunny (celsius)
    TOOL#2 [cal.add_event] -> ok 已安排 周三
    度量：工具执行 2 次 · function_call 2 枚 · TOOL 消息 2 条 · 推理轮 2 轮 · 形状: user[1] / assistant[2] / tool[1] / tool[1] / assistant[1]
    TOOL#[错] [weather.get_weather] -> Missing required argument(s): ['city']. Please revise the arguments to match the function signature.
    TOOL#[修] [weather.get_weather] -> 上海: 20C sunny (celsius)
    度量：错误重试 1 次 · 收敛 3 轮 · 错误文本由引擎生成（非 stub 造）· 形状: user[1] / assistant[1] / tool[1] / assistant[1] / tool[1] / assistant[1]
    第 1 轮 stub 又发 function_call -> 引擎照常执行回填 TOOL: 北京#1: 20C sunny (celsius)
    达到应用侧截断 max_iter=4（stub 永不收手）
```

**结论**：

1. **并行子调用是"一次推理、逐枚执行"**：一个 assistant 消息里挂 2 枚 `FunctionCallContent`（`weather.get_weather` + `cal.add_event`）→ 我逐枚 `invoke_function_call` → 引擎各自执行、各自 TOOL 回填（`工具执行 2 次 · function_call 2 枚 · TOOL 消息 2 条`）→ 再加一句收尾文本。**这就是"规划器"的现代形态**：不是 0.x 的 `SequentialPlanner` 类，而是 "LLM 在一条消息里列出工具调用 + 引擎按调用真跑 + 结果回填历史" 的循环。
2. **自愈有引擎背书**：stub 漏了 `city`（`arguments="{}"`）→ 引擎**不执行函数体**，按参数元数据生成错误 TOOL 消息『Missing required argument(s): ['city']. Please revise the arguments to match the function signature.』→ stub 读到后修正参数重发 → 收敛。**"错误文本是引擎生成的、回填进历史的"这一条是硬证据**：SK 的自愈不是应用自己拼错误串，是签名契约在运行期的又一次强制（账 A 同源）。
3. **护栏三态，SK 只给"配置"不给"强制"**：场景 3 里 stub 永不收手，我设应用侧 `max_iter=4` 截断——**`invoke_function_call` 手动循环对调用次数毫无感知**。框架给的预算= `FunctionChoiceBehavior.Auto(maximum_auto_invoke_attempts=N)`（实测字段存在、可配），但它在**连接器的自动调用内循环里生效**；一旦绕过连接器自己编排，上限就要应用自管。**对比 02 章 `recursion_limit`（内核真抛 `GraphRecursionError`）/ 05 章 `max_runs_per_component`（引擎真兜 `PipelineMaxComponentRuns`）——SK 的特征恰恰是：护栏是『要你自己配的配置』而不是内核级兜底。** 这是"企业级"的另一面：框架不替你决定什么该停，治理留给应用。

## 5. 实验 C · 版本存续（0.x → 1.x 概念全名消失 · 改名烈度）

**目标**：沿 01-章"版本存续"账的思路，给 SK 自己的 0.x→1.x 变迁做一张**生存表**——哪些老概念还活着、哪些整包消失，用**真实 import 探测**而不是文档印象。

**过程**：8 个 0.x 时代的目标名（`orchestration.sk_context` / `.sk_function_base` / `.context_variables` / `skill_definition.sk_function_base` / `.sk_function_decorators` / `planning.sequential_planner` / `.action_planner` / 顶层 `core_skills`）+ 7 个 1.x 存活名逐一 `importlib` 探测。

**实测**（stdout 关键行）：

```
    semantic_kernel.orchestration.sk_context        -> FAIL ModuleNotFoundError: No module named 'semantic_kernel.orchestra
    semantic_kernel.planning.sequential_planner     -> FAIL ModuleNotFoundError: No module named 'semantic_kernel.planning'
    semantic_kernel.core_skills                     -> FAIL AttributeError: module 'semantic_kernel' has no attribute
    semantic_kernel.Kernel                          -> OK <ModelMetaclass>
    semantic_kernel.functions.kernel_function       -> OK <module>
    semantic_kernel.functions.KernelFunction        -> OK <ModelMetaclass>
    semantic_kernel.contents.ChatHistory            -> OK <ModelMetaclass>
    semantic_kernel.connectors.mcp                  -> OK <module>
```

**结论**：

1. **0.x 四大基本概念整包子包消失**：`orchestration`（SKContext 的家）、`skill_definition`（SKFunction 与 `@skill_function` 装饰器的家）、`planning`（SequentialPlanner 的家）、顶层 `core_skills`——**8/8 全 FAIL**。"技能+规划器"的 0.x 字面（Skill/SKFunction/ContextVariables/Planner）在 1.x 没有对应的 import 路径。
2. **身份存活、概念全换**：`Kernel`（ModelMetaclass）、`functions.kernel_function`（装饰器本体）、`KernelFunction`/`KernelPlugin`/`KernelArguments`、`contents.ChatHistory`、`connectors.mcp`——7 个 1.x 名全 OK，import 名/包名（`semantic_kernel`/`semantic-kernel`，**绕过了 Haystack 那种 `haystack→haystack-ai` 的包名变迁**）都没动。**变的是"一切描述功能的词"**：Skill→Plugin、SKFunction→Function、ContextVariables→Arguments、Planner→`FunctionCallBehavior.Auto`（规划器 LLM 化已经在账 B 演示过）。
3. **改名烈度 07 章之最**：01-章 LangChain 是"概念永存、库换血、**移子包**"（LLMChain 等整包移除）；05-章 Haystack 是"**换 pip 包名、import 保留**"；SK 是"**核心概念词全部换掉**（Skill 没了、Planner 没了），只剩 Kernel 这个身份"。**三条迁移路径三种死法**——沿任意一处旧教程写 `semantic_kernel.orchestration` 或 `hand skill`，得到的都是 FAIL；所以 SK 的迁移第一课=**查官方 0.x→1.x 迁移指南，别在老教程里猜**。

## 6. 实验 D · MCP 桥接与选择树（as_mcp_server 1:1 导出 · 8 场景断言）

**目标**：给"企业栈的出口"一个可复排的本机证据——SK 的"技能"怎么变成跨栈工具（MCP），以及 .NET 栈在哪一格由哪个框架接走（§7.14 + §7.1，要素事实）。

**过程**：`k.as_mcp_server(server_name=, version=)` 把 AST 已注册的 4 个插件函数整包导出；再发 `ListToolsRequest` 到 mcp 低层 Server 的 tools/list 处理器（`request_ctx` ContextVar 注入）读真实工具清单；再测 `excluded_functions="add"` 的排除语义。随后落 8 场景选择树（作者按 §7.14/§7.1 整理）。

**实测**（stdout 关键行）：

```
    kernel 插件函数 4 个 -> MCP 工具 4 个（1:1）
    tool=get_weather  props=['city', 'unit']  required=['city']  desc=查询指定城市的当前天气…
    excluded_functions='add'（按短名排除）-> 工具剩 ['add_event', 'get_forecast', 'get_weather']
    1.  .NET/C#/Azure 微软企业栈 -> Semantic Kernel（本篇——函数即一等公民的企业语言栈 SDK）
    8.  已深度持有 LangChain 资产 -> 同系演进 LangGraph/LlamaIndex——SK 是企业栈入场券、不是 Python 栈平替
```

**结论**：

1. **插件→MCP 工具 1:1，Schema 跟着走**：4 个插件函数 → 4 个 MCP 工具（`get_weather` 的 props=`['city','unit']`、required=`['city']`，与账 A 的参数元数据逐字节一致）——**"函数=一等公民"到协议层的延续：inputSchema 就是账 A 那份契约**。工具名=**函数短名**（不带 plugin 前缀，`excluded_functions` 文档原话也按短名）；这带来一个诚实风险：**跨插件同名函数在 MCP 面会撞名**（本探针 4 个函数恰好短名互异，避开；真实工程要留意）。
2. **排除按短名**：`excluded_functions="add"` → 工具只剩 3 个（`add_event/get_forecast/get_weather`）——排除语义与暴露语义同一把尺（函数短名），不是 fq 名。这强化了"短名是 MCP 层的命名空间"这一判断。
3. **选择树 8/8（要素事实）**：.NET 企业栈 → 本篇；生产级 RAG → Haystack；状态/循环 → LangGraph；RAG 管线专精 → LlamaIndex；非工程师 → 低代码；轻量单 Agent → Agents SDK / 直写；跨栈互操作 → MCP 协议层；**已持有 LangChain 资产 → 同系演进**（LangGraph/LlamaIndex），**不是 SK**——SK 是企业栈的入场券，不是 Python 栈的平替（§7.7"独立开发者用得少"的选型回声）。**SK 的格是"在微软技术栈里把技能封成可治理的函数、Function 的格"那一格**——别的场景不硬蹭。

## 7. 拿这张表怎么读本目录（07 章 13 篇的路牌）

本页是 07 章"第七站"：00 篇回答了"框架分几类"，01 篇"通用编排家长子"，02 篇"图引擎一格"，03 篇"RAG 专精一格"，04 篇"低代码一格"，05 篇"生产级 RAG 管线一格"，**本页回答"企业 .NET 栈的编排一格"**。后文各篇的路牌（按依赖次序）：

| 后文篇 | 读它的理由（对应本页哪一格） | 状态 |
|---|---|---|
| `00-框架分类学` | 账 B 标格次主次次·覆盖 4/4、账 C 类别 1 代表、选择树场景⑨ .NET 企业栈 | 🔥 已交付（v0.27） |
| `01-LangChain与历史包袱` | 账 C 版本存续的对照母题：概念永存 vs API 换血（SK=概念词全换的最强样本） | 🔥 已交付（v0.28） |
| `02-LangGraph-状态图与Checkpoint` | 账 B 护栏的对偶：内核级 recursion_limit 真抛异常 vs SK 护栏=要自配的配置 | 🔥 已交付（里程碑 012） |
| `03-LlamaIndex` | 账 D 决策表另一臂：RAG 数据管线专精住进 03 | 🔥 已交付（v0.29） |
| `04-Dify-Flowise-Coze-n8n低代码` | 账 D 决策表另一臂：非工程师落 04 | 🔥 已交付（v0.30） |
| `05-Haystack` | 账 A 护栏对偶（max_runs 真兜底）+ 账 D 场景⑤指针： .NET 栈归本篇 | 🔥 已交付（v0.31） |
| `06-Semantic-Kernel`（本篇） | 企业 .NET/云 SDK 的编排一格：插件/函数/自动调用三原语 | 🔥 已交付（v0.32） |
| `07-AutoGen·AG2` | §7.8 背书：微软 2025 推出 Microsoft Agent Framework 承接 AutoGen、与 SK 统一 | 🔥 已交付（v0.33） |
| `08-CrewAI` | 账 A 类 4/账 B：多 Agent 角色化分工（角色·任务·流程三一等公民，真实 crewai 1.15.22 引擎四账） | 🔥 已交付（v0.34） |
| `09-DSPy` | 账 A 类 5：质量敏感评测（评测主业互补角色化） | 🔥 已交付（v0.36） |
| `10-OpenAI-Agents-SDK` | 账 D 场景 6：轻量运行时"回归轻量"先例（一个 Agent 对象 + 一个 Runner 函数） | 🔥 已交付（v0.35） |
| `11-MCP协议` | 账 B"覆盖 1 维"特判、账 C"2025 公共底座"、里程碑 `013`；本篇账 D=协议层的 SK 出口 | 🔥 已交付（v0.37） |
| `12-继承关系与选型决策` | 收束章：完整 DAG + "该不该引入"量化，00 篇 C 账扩版 | 🔥 已交付（v0.38 收束章） |

读法口诀（本页的一页带走）：**遇到"我们在 .NET/C#/Azure 微软技术栈，想把技能封成可治理的函数"先落 Semantic Kernel——函数=一等公民（账 A：三条注册路都收敛到 `@kernel_function`，装饰器携带契约、缺参运行期强制）是一种可量出来的企业级纪律；但别指望它替你兜循环（账 B：护栏=`FunctionChoiceBehavior` 配置项，不是内核强制，手动循环应用自管）、别沿 0.x 老教程写 Skill/Planner（账 C：四个概念子包整包消失）、跨栈用 `as_mcp_server` 1:1 导出（账 D：工具名=函数短名、Schema 跟着走）——它是企业栈入场券，不是 Python 平替，已持有 LangChain 资产的直接走同系演进。**

## 8. 常见坑（5 个）

1. **把 SK 当"Python 通用框架 / LangChain 平替"**（本课第一个坑）：§7.7 定位="微软技术栈企业的选择；独立开发者用得少"。账 D 场景⑧的警句：**已深度持有 LangChain 资产的场景 → LangGraph/LlamaIndex 同系演进，不是 SK**——SK 是 .NET/C#/Azure 的入场券，Python 栈冲它而来多数会失望。
2. **以为"写了函数就有魔法，不装饰也成"**：账 A 实测裸函数经 `KernelFunctionFromMethod` 注册 → `FunctionInitializationError: Method is not a Kernel function`。**`@kernel_function` 装饰器是契约载体也是入场券**——名称/描述/参数 Schema 全由它携带，漏了=注册即炸。
3. **把"缺参会报错"当万能校验**：账 A 的 `KernelInvokeException` 管的是**参数必填**（签名层面）；类型/值域语义仍归你的函数体——`city: str` 传个空串也不会被引擎拦。**SK 兜形状不兜值域**，接 06 章 05 篇的"校验器兜值域"这层还是要自己写（账 A 实测的边界）。
4. **指望"框架会替你兜住自动调用循环"**：账 B 护栏三态证明 `invoke_function_call` 手动循环**没有内置总轮上限**——stub 永不收手就永远跑。`FunctionChoiceBehavior.maximum_auto_invoke_attempts` 是**连接器内循环生效的设置项**，不是内核强制；绕过连接器自己编排=应用自管 max_iter。对标 02 章 `recursion_limit`/05 章 `max_runs_per_component` 那才叫内核级兜底。
5. **沿 0.x 教程写 Skill / ContextVariables / Planner**：账 C 真 import 探测四个子包（`orchestration`/`skill_definition`/`planning`/`core_skills`）8/8 全 FAIL。**概念词全换、只剩 Kernel 这个身份**——迁移第一课=查官方 0.x→1.x 迁移指南，老教程的每一条 import 都可能是这次的 FAIL。

## 9. 诚实边界（AAA 自我审查）

- **引擎语义 = 本机真实实测（semantic-kernel 1.44.1）**：三条注册路径、`k.invoke` 往返与缺参 `KernelInvokeException`、`invoke_function_call` 的执行/TOOL 回填、缺参错误文本生成、`FunctionConnectionBehavior.Auto(maximum_auto_invoke_attempts=…)` 字段可配、`as_mcp_server`→tools/list 真实注册、import 探测 OK/FAIL——全部在本机真实引擎上跑出（有真实报错原文为证）。本节不借鉴文档印象，凡可复现行为都对真实输出。
- **LLM 决策 = 非本机实测（stub 预置）**：探针里"何时调哪个工具、参数值、缺不漏参、何时收手"全部是作者预置的词面规则桩——这是为了保证零网络零随机、stdout 三进程 md5 恒一。**请勿把 stub 的"决策质量"与真实 LLM 的 function calling 能力混淆**：真实 LLM（GPT/DeepSeek…）经 SK 连接器（`AddOpenAIChatCompletion` 等）自动调用 `FunctionCallBehavior` 的行为=非本机实测，账 B 只量"引擎接住函数调用后的执行/回填"，不量"LLM 会不会选对工具"。
- **版本存续行 = 本机真 import 探测 + 版本节奏要素事实**：0.x 目标名是真实 `importlib` 探测（8/8 FAIL、7 个 1.x 名 OK 逐条真实）；但"0.x 曾长这样"、"微软 2025 推出 Microsoft Agent Framework 与 SK 统一"、"pip 包名 semantic-kernel" 的版本节奏 = 要素事实，写作环境无外网、未在线复核官方文档，以官方 release/文档为准。**账户 C 的核心发现（四个概念子包整包消失、Kernel 存活）是本机真实 FAIL/OK 撑住的。**
- **账 D 选择树 = 作者按知识地图 §7.14 与 §7.1 整理（要素事实）**：8 场景决策规则为作者整理定性，非本机可运行断言；MCP 桥接的 tools/list **真实注册**（本机真跑），但**没有起真实 MCP client 走一遍完整握手**（消息层/传输层互操作=非本机实测，只量了 tools/list 这一档）。
- **确定性纪律**：stdout md5 `66c105f5…` 三个独立进程逐字节恒一（每次 `python code/notebooks/_tools/semantic_kernel_demo.py` 复现）；墙钟 ≈0.1 s 仅进 stderr；不打印随机对象 ID/地址/哈希；输出排序保证哈希无关。

## 10. 参考与衔接

- **本页地图**：§7.7（Semantic Kernel：微软/企业 .NET 云 SDK/把技能+规划器封装/企业级治理/独立开发者用得少）+ §7.1（类 1 通用编排 SDK 代表三分一）+ §7.14（.NET 企业 → Semantic Kernel）+ §7.8（AutoGen 分支 AG2 + 微软 2025 Microsoft Agent Framework 与 SK 统一）。
- **选择树 8 场景**（探针断言 8/8，要素事实）：① .NET 企业栈 → Semantic Kernel（本篇）；② 生产级 RAG → Haystack（05）；③ 状态/循环/恢复 → LangGraph（02）；④ RAG 管线专精 → LlamaIndex（03）；⑤ 非工程师 → 低代码（04）；⑥ 轻量单 Agent → Agents SDK/直写（10）；⑦ 跨栈互操作 → MCP（11·里程碑 013）；⑧ 已持 LangChain → 同系演进。
- **前承**：00-框架分类学（账 B 标格次主次次·覆盖 4/4、选择树场景⑨）；05-Haystack（账 A `max_runs` 护栏对偶 + 账 D 场景⑤指针）；02-LangGraph（`recursion_limit` 内核级上限对偶）；06-应用开发 05-Function-Tool-Calling（函数调用入参/循环回喂/并行的语义源头）。
- **后启**：`07-AutoGen·AG2`（✅ 已交付 v0.33：§7.8 官方背书——SK 与 AutoGen 的统一线、微软 2025 Microsoft Agent Framework 承接 AutoGen，配 `_tools/autogen_demo.py` 真实 ag2 1.0.6 四账实测）；`11-MCP协议`（✅ 已交付 v0.37：本篇账 D=协议层的 SK 出口，milestone `013`）；`10-OpenAI-Agents-SDK`（✅ 已交付 v0.35：回归轻量对照——Agent+Runner 一件套、手转交 §7.11 实测，真实 openai-agents 0.17.0 四账）；`12-继承关系与选型决策`（✅ 已交付 v0.38 收束章：完整 DAG + "该不该引入"量化——账 C .NET 场景本页 Semantic Kernel 域内净分 +1.02；账 A 微软双线 AutoGen/SK→MAF 入度 2）。
- **对应里程碑**：`012`（02 章已交付）之后，企业栈这杆由本篇收；里程碑 `013`（手写 MCP Server）已在 `11-MCP协议` 章交付（v0.37）——SK 的 `as_mcp_server` 是"框架代写"的对照参照。

> 本篇完工于 2026-09-22（v0.32 批次）；探针 `code/notebooks/_tools/semantic_kernel_demo.py`；stdout md5 `66c105f5…`。
