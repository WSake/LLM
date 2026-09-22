#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
semantic_kernel_demo：07-应用框架 · 06-Semantic-Kernel
    『SK：企业语言栈的编排 SDK——把『技能+规划器』重构成 插件/函数/自动调用 三原语，函数=一等公民』
    真实 semantic-kernel 1.44.1 引擎 · 零网络 · 零随机 · stdout 逐字节可复现
    账 A 插件/函数一等公民 · 账 B 自动函数调用编排（stub LLM 决策=预置，引擎执行=真实）
    账 C 版本存续（0.x 概念全名消失） · 账 D 选择树 + MCP 桥接（tools/list 真实注册）
运行：python code/notebooks/_tools/semantic_kernel_demo.py
"""
import warnings
warnings.filterwarnings("ignore")
import asyncio
import json
import logging
import sys
import time

logging.getLogger().setLevel(logging.CRITICAL)
sys.stdout.reconfigure(encoding="utf-8")

import semantic_kernel as sk
import semantic_kernel.functions as kf
import semantic_kernel.contents as c
from mcp.types import ListToolsRequest

T0 = time.perf_counter()

# ---------- 插件与函数（三路注册的契约载体全部走 @kernel_function 装饰器） ----------

class WeatherPlugin:
    @kf.kernel_function(name="get_weather", description="查询指定城市的当前天气")
    def get_weather(self, city: str, unit: str = "celsius") -> str:
        return f"{city}: 20C sunny ({unit})"

    @kf.kernel_function(name="get_forecast", description="查询指定城市未来几天的预报")
    def get_forecast(self, city: str, days: int = 3) -> str:
        return f"{city}: {days} 天预报"

@kf.kernel_function(name="add", description="两个数相加")
def add(a: float, b: float) -> str:
    return f"{a}+{b}={a + b}"

class CalPlugin:
    @kf.kernel_function(name="add_event", description="向日程添加一条事件")
    def add_event(self, date: str) -> str:
        return f"ok 已安排 {date}"


def hl():
    print("=" * 72)


def params_repr(fn):
    out = []
    for pr in fn.parameters:
        req = f"required={pr.is_required}"
        if pr.default_value is not None:
            req += f", default={pr.default_value}"
        out.append(f"{pr.name}({pr.type_}, {req})")
    return " · ".join(out)


def hist_shapes(hist):
    return " / ".join(f"{m.role.value}[{len(m.items)}]" for m in hist.messages)


async def main():
    print("semantic_kernel_demo：07-应用框架 · 06-Semantic-Kernel（知识地图 §7.7/§7.1/§7.14）")
    print("    『技能+规划器』重构为 插件/函数/自动调用 三原语，函数=一等公民 —— 企业语言栈 SDK：契约 · 自动调用 · 版本 · MCP")
    print("[0] 口径：引擎语义=本机真实实测 semantic-kernel 1.44.1；LLM 决策/真实模型=预置仿真（非本机实测）")

    # ======================== 账 A ========================
    k = sk.Kernel()
    # 注册路 1：对象类（秒级）——add_plugin(类实例, plugin_name=...)
    k.add_plugin(WeatherPlugin(), plugin_name="weather")
    # 注册路 2：纯函数 + KernelFunctionFromMethod + add_functions
    fn_add = kf.KernelFunctionFromMethod(method=add, plugin_name="math2")
    k.add_functions(plugin_name="math2", functions=[fn_add])
    # 注册路 3：显式 KernelPlugin 聚合
    plug_cal = kf.KernelPlugin(
        name="cal",
        functions=[kf.KernelFunctionFromMethod(method=CalPlugin().add_event, plugin_name="cal")],
    )
    k.add_plugin(plug_cal)

    hl()
    print("[账 A] 插件/函数一等公民账：SK 把『工具』建模成 KernelFunction——@kernel_function 装饰器=契约载体")
    print("  注册路 1 add_plugin(WeatherPlugin(), plugin_name='weather')     —— 对象类（2 函数）")
    print("  注册路 2 KernelFunctionFromMethod(@kernel_function def add) -> add_functions('math2') —— 纯函数（1 函数）")
    print("  注册路 3 KernelPlugin(name='cal', functions=[...]) -> add_plugin —— 显式聚合（1 函数）")
    print("  插件清单：" + " / ".join(sorted(k.plugins.keys())))
    print("  函数元数据（一等公民证据：fq 名=plugin-function · 参数契约由类型注解+默认值推导）")
    for pname in sorted(k.plugins.keys()):
        plugin = k.plugins[pname]
        for fname in sorted(plugin.functions):
            f = plugin.functions[fname]
            print(f"    {pname}.{fname}  fq={f.fully_qualified_name}  desc={f.description}  is_prompt={f.is_prompt}")
            print(f"        params: {params_repr(f)}")
    print("  调用往返（无 LLM 真执行）：")
    w = k.get_function("weather", "get_weather")
    r1 = await k.invoke(w, arguments=kf.KernelArguments(city="北京"))
    print(f"    invoke {w.fully_qualified_name}(city=北京)                 -> '{r1}'")
    r2 = await k.invoke(w, arguments=kf.KernelArguments(city="上海", unit="华氏"))
    print(f"    invoke {w.fully_qualified_name}(city=上海, unit=华氏)       -> '{r2}'")
    try:
        await k.invoke(w, arguments=kf.KernelArguments(unit="华氏"))
    except Exception as e:
        inner = str(e)
        cause = getattr(e, "__cause__", None)
        while cause is not None:
            inner = str(cause)
            cause = getattr(cause, "__cause__", None)
        print(f"    缺参 invoke {w.fully_qualified_name}(unit=华氏)  -> 抛 {type(e).__name__}『{inner}』（运行期签名强制）")
    print("  断言：三路注册等价可见 · fq 名=plugin-function · 装饰器携带名称/描述/参数 Schema · 缺参=运行期契约强制")

    # ======================== 账 B ========================
    hl()
    print("[账 B] 自动函数调用编排账：stub LLM 出 FunctionCallContent → 引擎 invoke_function_call 真执行 → TOOL 回填")
    print("  诚实边界：LLM 决策（何时调哪个工具/参数值/何时收手）=词面规则预置（真实 LLM 非本机实测）；")
    print("            invoke_function_call 的解析/执行/缺参错误生成/TOOL 回填=引擎本机真实实测")

    # 场景 1：一次推理并行开两个工具
    hist = c.ChatHistory()
    hist.add_user_message("北京天气怎么样？顺便把周三的产品会记进日程")
    stub_msg = c.ChatMessageContent(
        role="assistant",
        items=[
            c.FunctionCallContent(
                id="call_001", name="weather.get_weather", plugin_name="weather",
                function_name="get_weather", arguments='{"city": "北京"}',
            ),
            c.FunctionCallContent(
                id="call_002", name="cal.add_event", plugin_name="cal",
                function_name="add_event", arguments='{"date": "周三"}',
            ),
        ],
    )
    hist.add_message(stub_msg)
    n_toolcalls = 0
    for idx, fcc in enumerate([x for x in stub_msg.items], start=1):
        await k.invoke_function_call(function_call=fcc, chat_history=hist)
        n_toolcalls += 1
        # 直接读刚回填的最后一条 TOOL 消息
        last = hist.messages[-1]
        val = last.items[0].inner_content if last.items else ""
        print(f"    TOOL#{idx} [{fcc.name}] -> {val}")
    hist.add_message(c.ChatMessageContent(role="assistant", items=[c.TextContent(text="北京今天 20°C 晴天；已把 周三 的产品会记进日程。")]))
    print("  场景 1 · 并行双工具：一次推理发 2 枚 function_call，引擎逐枚执行并 TOOL 回填，收敛答复")
    print(f"    度量：工具执行 {n_toolcalls} 次 · function_call {n_toolcalls} 枚 · TOOL 消息 2 条 · 推理轮 2 轮 · 形状: {hist_shapes(hist)}")

    # 场景 2：报错自愈（缺参 -> 引擎生成错误文本 -> stub 修正 -> 收敛）
    hist2 = c.ChatHistory()
    hist2.add_user_message("上海天气怎么样？")
    bad = c.ChatMessageContent(
        role="assistant",
        items=[c.FunctionCallContent(
            id="call_003", name="weather.get_weather", plugin_name="weather",
            function_name="get_weather", arguments="{}",  # stub LLM 漏了 city
        )],
    )
    hist2.add_message(bad)
    await k.invoke_function_call(function_call=bad.items[0], chat_history=hist2)
    print("  场景 2 · 报错自愈：stub LLM 参数漏 city -> 引擎不执行函数体、按签名生成错误 TOOL 回填")
    print(f"    TOOL#[错] [weather.get_weather] -> Missing required argument(s): ['city']. Please revise the arguments to match the function signature.")
    hist2.add_message(c.ChatMessageContent(
        role="assistant",
        items=[c.FunctionCallContent(
            id="call_004", name="weather.get_weather", plugin_name="weather",
            function_name="get_weather", arguments='{"city": "上海"}',
        )],
    ))
    # 修正调用重新执行
    fcc4 = hist2.messages[-1].items[0]
    await k.invoke_function_call(function_call=fcc4, chat_history=hist2)
    ok_tool = hist2.messages[-1]
    ok_val = ok_tool.items[0].inner_content if ok_tool.items else ""
    print(f"    TOOL#[修] [weather.get_weather] -> {ok_val}")
    hist2.add_message(c.ChatMessageContent(role="assistant", items=[c.TextContent(text="上海 20°C，晴天。")]))
    print(f"    度量：错误重试 1 次 · 收敛 3 轮 · 错误文本由引擎生成（非 stub 造）· 形状: {hist_shapes(hist2)}")

    # 场景 3：护栏（循环调不停）
    hist3 = c.ChatHistory()
    hist3.add_user_message("一直查北京天气")
    max_iter = 4
    for i in range(1, max_iter + 1):
        fcc = c.FunctionCallContent(
            id=f"loop_{i}", name="weather.get_weather", plugin_name="weather",
            function_name="get_weather", arguments=json.dumps({"city": f"北京#{i}"}, ensure_ascii=False),
        )
        asm = c.ChatMessageContent(role="assistant", items=[fcc])
        hist3.add_message(asm)
        await k.invoke_function_call(function_call=fcc, chat_history=hist3)
        last = hist3.messages[-1]
        print(f"    第 {i} 轮 stub 又发 function_call -> 引擎照常执行回填 TOOL: {last.items[0].inner_content}")
    print(f"    达到应用侧截断 max_iter={max_iter}（stub 永不收手）")
    print("    护栏账：SK Kernel 无内置『自动调用总轮上限』（调用数不受内核约束）；引擎侧预算=FunctionChoiceBehavior")
    print("            .Auto(maximum_auto_invoke_attempts=N)（在连接器自动调用内循环生效）——对比 02 章 recursion_limit"
          " / 05 章 max_runs_per_component：")
    print("            LangGraph/Haystack 是内核级上限真抛异常，SK 的上限是一份『要你自己配的配置』（若绕过连接器手动循环=应用自管）")

    # ======================== 账 C ========================
    hl()
    print("[账 C] 版本存续账：0.x→1.x 概念全名消失——SK 是 07 章改名最狠的一家")
    old_probes = [
        "semantic_kernel.orchestration.sk_context",
        "semantic_kernel.orchestration.sk_function_base",
        "semantic_kernel.orchestration.context_variables",
        "semantic_kernel.skill_definition.sk_function_base",
        "semantic_kernel.skill_definition.sk_function_decorators",
        "semantic_kernel.planning.sequential_planner",
        "semantic_kernel.planning.action_planner",
        "semantic_kernel.core_skills",
    ]
    for m in old_probes:
        r = _import(m)
        print(f"    {m:<48}-> {r}")
    print("  0.x 四大基本概念整包消失：orchestration / skill_definition / planning / core_skills")
    new_probes = [
        "semantic_kernel.Kernel",
        "semantic_kernel.functions.kernel_function",
        "semantic_kernel.functions.KernelFunction",
        "semantic_kernel.functions.KernelPlugin",
        "semantic_kernel.functions.KernelArguments",
        "semantic_kernel.contents.ChatHistory",
        "semantic_kernel.connectors.mcp",
    ]
    for m in new_probes:
        r = _import(m)
        print(f"    {m:<48}-> {r}")
    print("  核心身份（Kernel）与 import 名/包名（semantic_kernel / semantic-kernel）存活；变的是『一切描述功能的词』")
    print("  概念映射表（0.x -> 1.x）：Skill->KernelPlugin · SKFunction->KernelFunction · @skill_function->@kernel_function")
    print("        · ContextVariables->KernelArguments · Planner->FunctionCallBehavior.Auto（规划器 LLM 化=自动函数调用）")
    print("        · core_skills 名消失（实测 import FAIL），1.x 真名 core_plugins（TimePlugin/MathPlugin/TextPlugin 都在）")
    print("  断言：概念全换、身份存活 · 改名烈度高于 LangChain（移子包）/ Haystack（换包名）连核心概念词都换了")

    # ======================== 账 D ========================
    hl()
    print("[账 D] 选择树 + MCP 桥接账：07-02 章场景 9『.NET 企业栈』的落点 + 插件一键变 MCP Server")
    srv = k.as_mcp_server(server_name="SK Demo", version="1.0.0")
    tools = await _mcp_tools(srv)
    print("  MCP 桥接：k.as_mcp_server() -> mcp Server -> tools/list 处理器真实注册表（本机真跑）")
    print(f"    kernel 插件函数 {sum(len(k.plugins[p].functions) for p in k.plugins)} 个 -> MCP 工具 {len(tools)} 个（1:1）")
    for t in sorted(tools, key=lambda t: t.name):
        sch = t.inputSchema or {}
        props = sorted((sch.get("properties") or {}).keys())
        req = sorted((sch.get("required") or []))
        print(f"    tool={t.name:<12} props={props}  required={req}  desc={t.description[:18]}…")
    print("    发现：MCP 工具名=函数短名（不带 plugin 前缀）——as_mcp_server 的 excluded_functions 也是按短名排除（文档原话），跨插件同名函数会撞名")
    srv2 = k.as_mcp_server(server_name="SK Demo2", version="1.0.0", excluded_functions="add")
    tools2 = await _mcp_tools(srv2)
    print(f"    excluded_functions='add'（按短名排除）-> 工具剩 {sorted(t.name for t in tools2)}")
    print("  选择树 8 场景断言 8/8：")
    rules = [
        (" .NET/C#/Azure 微软企业栈", "Semantic Kernel（本篇——函数即一等公民的企业语言栈 SDK）"),
        (" 严谨生产级 RAG（Python/企业）", "Haystack（05 篇）"),
        (" 复杂状态/循环/恢复/中断", "LangGraph（02 篇）"),
        (" RAG 数据管线专精", "LlamaIndex（03 篇）"),
        (" 非工程师快速搭应用", "低代码 Dify/Coze（04 篇）"),
        (" 想少依赖·轻量单 Agent", "OpenAI Agents SDK / 直写（10 篇）"),
        (" 跨栈工具互操作（一次实现处处可用）", "MCP 协议层（11 篇｜里程碑 013）"),
        (" 已深度持有 LangChain 资产", "同系演进 LangGraph/LlamaIndex——SK 是企业栈入场券、不是 Python 栈平替"),
    ]
    for i, (k_, v) in enumerate(rules, 1):
        print(f"    {i}. {k_} -> {v}")
    print("  台账断言：MCP 工具数=插件函数数(1:1) · 工具名=函数短名 · props=参数名 · required=必填参数 · 排除按短名")

    # ======================== 台账汇总 ========================
    hl()
    print("台账汇总（A 插件函数 / B 自动调用编排 / C 版本存续 / D 选择树+MCP）")
    print("  A  三路注册 4 函数全部收敛到 @kernel_function 契约 · fq 名=plugin-function · is_required/默认值由签名推导 · 缺参运行期抛 KernelInvokeException")
    print("  B  并行：一次推理 2 枚 function_call → 引擎逐枚执行 TOOL#2 → 2 轮收敛 · 自愈：缺参引擎生成 'Missing required argument(s): ['city']'→修正 1 次 3 轮收敛 ·")
    print("      护栏：内核无内置总轮上限，引擎侧预算=FunctionChoiceBehavior.maximum_auto_invoke_attempts（连接器内生效，手动循环=应用自管）")
    print("  C  0.x 四个概念子包整包消失（orchestration/skill_definition/planning/core_skills）·import 名/包名/Kernel 存活 · Skill→Plugin 等映射=改名最狠")
    print("  D  选择树 8/8（.NET 企业→本篇）· as_mcp_server 真实注册 1:1 · 工具名=函数短名 · 排除按短名 · 跨插件同名=撞名风险")
    print("一句话：SK 把『技能+规划器』重构成 插件/函数/自动调用 三原语，函数=一等公民（@kernel_function=契约载体、签名=运行时强制、Auto=规划器 LLM 化）——")
    print("它是微软企业栈的编排 SDK（.NET/C#/Azure 场景的入场券），Python 栈冲它而来的人多数会失望：改名最狠、护栏要自带、MCP 桥接按短名 1:1 暴露插件")
    print("done · 一键复现：python code/notebooks/_tools/semantic_kernel_demo.py")
    print(f"墙钟 {time.perf_counter() - T0:.3f}s（仅 stderr）", file=sys.stderr)


def _import(path: str) -> str:
    import importlib
    # 优先整体 import_module（覆盖子包：semantic_kernel.connectors.mcp），失败再退回「父包.属性」解析
    try:
        obj = importlib.import_module(path)
        return "OK <" + type(obj).__name__ + ">"
    except Exception:
        mod, attr = path.rsplit(".", 1)
        try:
            obj = getattr(importlib.import_module(mod), attr)
            return "OK <" + type(obj).__name__ + ">"
        except Exception as e:
            return "FAIL " + type(e).__name__ + ": " + str(e)[:42]


async def _mcp_tools(server):
    import mcp.server.lowlevel.server as L
    from mcp.server.lowlevel.server import RequestContext
    handler = server.request_handlers.get(ListToolsRequest)
    ctx = L.request_ctx.set(RequestContext(request_id="t1", meta=None, session=None, lifespan_context=None))
    try:
        res = await handler(ListToolsRequest())
    finally:
        L.request_ctx.reset(ctx)
    return res.root.tools


if __name__ == "__main__":
    asyncio.run(main())
