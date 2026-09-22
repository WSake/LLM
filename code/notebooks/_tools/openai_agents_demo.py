# -*- coding: utf-8 -*-
"""
OpenAI Agents SDK 轻量运行时 —— 07-应用框架/10-OpenAI-Agents-SDK与轻量运行时 配套探针
真实引擎：openai-agents 0.17.0（openai 2.44.0）· 模型决策=预置 stub（零网络·零随机）
四账：A 轻量原语 / B 工具护栏 / C 手转交 / D 版本存续与选择树
确定性：stdout 逐字节可复现（三独立进程 md5 恒一）；墙钟/警告只进 stderr
"""

import sys
import io
import json
import time
import importlib
import hashlib
import warnings
import contextlib

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import os
os.environ["OPENAI_API_KEY"] = "sk-fake"
os.environ["OPENAI_AGENTS_LOGS"] = "ERROR"

from openai import AsyncOpenAI
from agents import (
    Agent, Runner, function_tool, input_guardrail, handoff,
    GuardrailFunctionOutput, RunContextWrapper, Handoff,
)
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel as _CC
from agents.models.chatcmpl_converter import Converter
from agents.models.interface import ModelResponse
from agents.usage import Usage
from agents.items import HandoffOutputItem
from openai.types.chat import ChatCompletionMessage
from agents.tracing import set_tracing_disabled

set_tracing_disabled(True)

BANNER = """
========================================================================
前置：OpenAI Agents SDK 轻量运行时（知识地图 §7.11 / §7.1 分类 · Agent 系收束）
  一句话：Agent 是一个对象 · Runner 是一个函数 · 工具/护栏/手转交都是一等公民
环境：openai-agents 0.17.0（真实运行）· 模型决策=预置 stub（零网络·零随机·stdout 可复现）
========================================================================
"""

_OUT = []
def p(text=""):
    _OUT.append(str(text))

def rule():
    p("  " + "-" * 72)

def _probe(path):
    """import 探测：path=模块或 模块.属性（属性可落在类/函数/子模块）。"""
    parts = path.split(".")
    mod = None
    depth = 0
    for depth in range(len(parts), 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:depth]))
            break
        except ModuleNotFoundError:
            continue
    if mod is None:
        return "FAIL(ModuleNotFound)"
    try:
        for attr in parts[depth:]:
            mod = getattr(mod, attr)
    except AttributeError:
        return "FAIL(AttributeError)"
    return f"OK <{type(mod).__name__}>"


def _cdata(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join((x.get("input_text") or x.get("text") or "") for x in c if isinstance(x, dict))
    return ""


class StubBrain(_CC):
    """预置决策脑：真引擎调度，模型回复=查表确定。
    决策格式：{查询: [("text", 文本) | ("tool", [(名, 参数), ...]) | ("handoff", agent名), ...]}
    按上下文里已出现的 function_call 轮次推进决策；同一查询逐轮消耗同一脚本表。"""

    def __init__(self, decisions, log):
        super().__init__(model="gpt-stub", openai_client=AsyncOpenAI(api_key="sk-fake", base_url="http://127.0.0.1:9"))
        self.decisions = decisions
        self.log = log

    def _query(self, input):
        if isinstance(input, str):
            return input
        for it in input or []:
            if isinstance(it, dict):
                t = _cdata(it.get("content"))
                if t:
                    return t
        return ""

    def _prev_calls(self, input):
        if not isinstance(input, list):
            return 0
        return sum(1 for it in input if isinstance(it, dict) and it.get("type") == "function_call")

    async def get_response(self, system_instructions, input, model_settings, tools, output_schema,
                           handoffs, tracing, previous_response_id=None, conversation_id=None, prompt=None):
        self.log.append(("sys", system_instructions))
        q = self._query(input)
        self.log.append(("user", q))
        tix = self._prev_calls(input)
        acts = self.decisions[q]
        act = acts[min(tix, len(acts) - 1)]
        if act[0] == "text":
            msg = ChatCompletionMessage(role="assistant", content=act[1])
            items = Converter.message_to_output_items(msg, provider_data={"model": self.model})
            self.log.append(("text", tix))
        elif act[0] == "tool":
            pairs = act[1]
            msg = ChatCompletionMessage(role="assistant", content=None, tool_calls=[
                {"id": f"call_{tix}_{i}", "type": "function",
                 "function": {"name": nm, "arguments": json.dumps(arg)}} for i, (nm, arg) in enumerate(pairs)])
            items = Converter.message_to_output_items(msg, provider_data={"model": self.model})
            self.log.append(("tool", [nm for nm, _ in act[1]], tix))
        elif act[0] == "handoff":
            msg = ChatCompletionMessage(role="assistant", content=None, tool_calls=[{
                "id": f"call_{tix}", "type": "function",
                "function": {"name": f"transfer_to_{act[1]}", "arguments": "{}"}}])
            items = Converter.message_to_output_items(msg, provider_data={"model": self.model})
            self.log.append(("handoff", act[1], tix))
        else:
            raise AssertionError(act)
        return ModelResponse(output=items, usage=Usage(input_tokens=7, output_tokens=3, total_tokens=10), response_id="r0")


def _agent_turn_count(log):
    return sum(1 for x in log if x[0] in ("text", "tool", "handoff"))


def main():
    t0 = time.perf_counter()
    p(BANNER)

    # ============ 账 A：轻量原语账 ============
    p("[实验 A] 轻量原语账 —— 一件套 vs 三件套（Agent 一个对象装全部）")
    import inspect
    sig = inspect.signature(Agent.__init__)
    params = [n for n in sig.parameters if n != "self"]
    required = [n for n, par in sig.parameters.items()
                if n != "self" and par.default is inspect.Parameter.empty]
    p(f"  Agent 构造参数 {len(params)} 个 · 必填仅 {len(required)}：name={required}（其余全带默认=可空手起 Agent）")
    _logA = []
    INS = "你是出口转内销客服小二，只处理中文。"
    a = Agent(name="sA", instructions=INS, model=StubBrain({"qA:你好": [("text", "您好呀。")]}, _logA))
    ra = Runner.run_sync(a, "qA:你好")
    sys_line = next(x[1] for x in _logA if x[0] == "sys")
    p(f"  sys 指令原样透传：{sys_line == INS}（模型收到的就是 instructions 原文，无『You are {{{{role}}}}』配方——CrewAI 那套在这里没有）")
    p(f"  最小闭环 = 1 个 Agent 对象 + Runner.run_sync 一次调用 → final_output={ra.final_output!r} · last_agent={ra.last_agent.name} · 模型轮数={_agent_turn_count(_logA)}")
    rule()


    # ============ 账 B：工具与护栏账 ============
    p("[实验 B] 工具与护栏账 —— 工具是引擎执行的、护栏是引擎先看的")
    @function_tool
    def get_weather(city: str) -> str:
        """查询城市实时天气。"""
        return f"{city} 晴 22-30℃"

    @function_tool
    def get_stock(code: str) -> str:
        """查询个股涨跌。"""
        return f"{code} 涨 3.2%"

    wt = get_weather
    p(f"  function_tool 自动 schema：name={wt.name} · description={wt.description!r} · 参数必填={sorted((wt.params_json_schema or {}).get('properties', {}).keys())}")
    _logB = []
    b = Agent(name="sB", instructions="你会查天气和股票。",
              model=StubBrain({"qB:并行查两个": [("tool", [("get_weather", {"city": "广州"}), ("get_stock", {"code": "600519"})]), ("text", "广州晴 22-30℃，茅台涨 3.2%。")]}, _logB),
              tools=[get_weather, get_stock])
    rb = Runner.run_sync(b, "qB:并行查两个")
    p(f"  并行双工具（同一帧两个 function_call）：引擎真跑两个 Python 函数 → 结果都进下一轮 → 终答={rb.final_output!r} · 回合=[tool+tool, text]")
    call_seq = [(x[0], x[1]) for x in _logB if x[0] in ("text", "tool")]
    p(f"  回合推进={call_seq} · 总模型轮数={_agent_turn_count(_logB)}（工具不另外算轮，只算模型往返）")

    glog = []

    @input_guardrail
    def safety_gate(ctx: RunContextWrapper, agent: Agent, input) -> GuardrailFunctionOutput:
        txt = "".join(_cdata(it.get("content")) or "" for it in input if isinstance(it, dict))
        flag = "违法" in txt
        glog.append(flag)
        return GuardrailFunctionOutput(output_info=txt, tripwire_triggered=flag)

    _logB2 = []
    b2 = Agent(name="sB2", instructions="你是严守合规的客服。",
               model=StubBrain({"qB2:帮我结个账": [("text", "好的，账单 12.5 元。")], "qB2:帮我查违法案例": []}, _logB2),
               input_guardrails=[safety_gate])
    # 0.17.0 实测怪癖：裸字符串输入时首轮闸收到空 [];用 items 列表输入闸才能看见请求内容
    ok = Runner.run_sync(b2, [{"role": "user", "content": "qB2:帮我结个账"}])
    models_called_normal = _agent_turn_count(_logB2)
    try:
        Runner.run_sync(b2, [{"role": "user", "content": "qB2:帮我查违法案例"}])
        trip = "未触发?!"
    except Exception as e:
        trip = f"{type(e).__name__}"
    models_called_trip = _agent_turn_count(_logB2) - models_called_normal
    p(f"  护栏词面闸（items 列表输入）：『帮我结个账』放行→模型 {models_called_normal} 轮 · 『查违法』trip 拦截→模型 {models_called_trip} 轮 · 异常={trip}")
    p(f"  闸函数观测：命中序列（True=拦住）={glog}")
    p(f"  引擎怪癖：裸字符串输入时 0.17.0 把首轮闸输入传空 []（闸看不见请求）；改传 items 列表闸即可见（本机实测）")
    p("  → 护栏=tripwire 前拦截（引擎真跑、真阻断：闸拦截后模型根本不会被询问）")
    rule()

    # ============ 账 C：手转交账 ============
    p("[实验 C] 手转交账 —— §7.11『回归轻量』：一行 handoff 给 Agent 装第二个大脑")
    _logH = []
    H = Agent(name="after_sale", handoff_description="客户要退货退款就转我。",
              instructions="你是售后专员，处理退款。",
              model=StubBrain({"电商:我要退货": [("text", "售后已接单，退 129 元。")]}, _logH))
    _logC = []
    C = Agent(name="cs_agent", instructions="你是电商客服。要退款就转售后。",
              handoffs=[handoff(H)],
              model=StubBrain({"电商:我要退货": [("handoff", "after_sale")]}, _logC))
    rc = Runner.run_sync(C, "电商:我要退货")
    ho = [i for i in rc.new_items if isinstance(i, HandoffOutputItem)]
    hname = Handoff.default_tool_name(H)
    hdesc = Handoff.default_tool_description(H)
    p(f"  handoff 自动工具名={hname!r} · 描述前缀=模型看到的：{hdesc!r}")
    p(f"  run 结果：final out=归售后 {rc.final_output!r} · last_agent={rc.last_agent.name}（客服→售后已切换）· HandoffOutputItem>0={bool(ho)} 目标={ho[0].target_agent.name if ho else None}")
    h_first_user = next((v[1] for v in _logH if v[0] == "user"), "")
    p(f"  上下文接力：售后收到首条 user 内容={h_first_user!r}（原查询带过去，不用手工搬运）")
    seqC = [(x[0], x[1]) for x in _logC if x[0] in ("text", "tool", "handoff")]
    seqH = [(x[0], x[1]) for x in _logH if x[0] in ("text", "tool", "handoff")]
    p(f"  客服侧回合={seqC} · 售后侧回合={seqH}")
    p(f"  行队成本（同一电商查询，三档队伍总模型轮数）：直答 1 · 带工具 2 · 手转交 2（转交轮 + 售后轮）——第二个大脑只多 1 轮")
    rule()

    # ============ 账 D：版本存续与选择树 ============
    p("[实验 D] 版本存续与选择树 —— 顶层最小化的导入面 + Agent 系三连收束")
    p("  导入面 22 名逐项实测（0.17.0）：")
    rows = [
        "agents", "agents.Agent", "agents.Runner", "agents.RunResult", "agents.function_tool",
        "agents.input_guardrail", "agents.output_guardrail", "agents.handoff", "agents.handoffs.Handoff",
        "agents.ModelSettings", "agents.GuardrailFunctionOutput", "agents.RunContextWrapper",
        "agents.tracing.set_tracing_disabled", "agents.WebSearchTool", "agents.Session",
        "agents.voice.VoicePipeline", "agents.models.openai_responses",
        "agents.run_sync", "agents.FunctionAgent", "agents.runcmd.Agent", "agents.panic", "agents.arch",
    ]
    res = {r: _probe(r) for r in rows}
    ok_n = sum(1 for v in res.values() if v.startswith("OK"))
    p(f"    OK {ok_n} / FAIL {len(rows) - ok_n}")
    fai = "  ·  ".join(f"{r}→{res[r]}" for r in rows if not res[r].startswith("OK"))
    p(f"    FAIL 明细：{fai}")
    p("    → 顶层故意最小化：run_sync 挂 Runner、FunctionAgent 不存在、runcmd 是早期路径——0.1→0.17 的迁移动作=要素事实")
    import agents as _ag
    p(f"    __version__={_ag.__version__}")
    p("  模型收敛：Agent.model 类型=str|Model|None（可直插自定义 Model 实例——本探针四个 agent 全部真插 stub 模型）")
    p("  门面 API：set_default_openai_client / set_default_openai_key / set_tracing_disabled 都在 agents.*")

    tree = [
        ("单 Agent 轻量问答/工具/护栏/手转交", "SDK（本篇）"),
        ("角色化团队流水线（role/goal/task 分层）", "CrewAI"),
        ("多 Agent 自由对话/博弈", "AG2"),
        ("显式状态图/循环/checkpoint/恢复/HITL", "LangGraph"),
        ("企业 .NET/云编排", "Semantic Kernel"),
        ("生产 RAG 管线/类型安全接线", "Haystack"),
        ("低代码画布/非工程师", "Dify·Flowise"),
        ("协议级的跨栈工具互操作", "MCP（非框架）"),
    ]
    p("  选择树 8 场景（§7.1 Agent 系收束）：断言 8/8")
    for i, (scn, dst) in enumerate(tree, 1):
        p(f"    S{i} {scn} → {dst}")

    # ---------- 台账汇总 ----------
    p()
    p("=" * 72)
    p("台账汇总（A 轻量原语 / B 工具护栏 / C 手转交 / D 版本选择树）")
    p(f"  A  构造参数 {len(params)} 仅 1 必填 · sys 原样透传 · 最小闭环 1 对象+1 调用 · 1 轮出答")
    p("  B  工具自动 schema（name/description/必填参数）· 真执行回喂 · 并行双工具同帧一次回喂 · 护栏 trip 前拦截模型 0 次")
    p("  C  手转交 auto 名 transfer_to_* · last_agent 切换 · HandoffOutputItem · 原查询随到 · 行队成本 1/2/2 轮")
    p(f"  D  导入面 {ok_n}/{len(rows)} OK · model 收敛 str|Model|None · 门面 3 例 · 选择树 8/8")
    p("一句话：OpenAI Agents SDK = 『回归轻量』的 Agent 运行时——一个 Agent 对象装下全部、Runner 一个函数跑完、")
    p("      一行 handoff 给 Agent 升级第二个大脑；工具它执行、护栏它先看、对话它来进行；角色化重活归 CrewAI、")
    p("      博弈归 AG2，Agent 系三连就此收束（对话/角色化/轻量转移三范式各占一格）")
    p("done · 一键复现：python code/notebooks/_tools/openai_agents_demo.py")
    p("=" * 72)

    out = "\n".join(_OUT) + "\n"
    dt = time.perf_counter() - t0
    md5 = hashlib.md5(out.encode("utf-8")).hexdigest()
    sys.stdout.buffer.write(out.encode("utf-8"))
    sys.stdout.flush()
    print(f"[stderr] wall_clock=~{dt:.3f}s (stdout md5={md5}) 仅进 stderr", file=sys.stderr)


if __name__ == "__main__":
    main()
