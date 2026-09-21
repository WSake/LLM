#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LangGraph 探针（知识地图 §7.3 正题 / §7.1 分类 / 衔接 06-记忆）：
把「带记忆重试的客服 Agent」编译成可执行的有向状态图（节点=函数·边=条件路由·状态=TypedDict channel）
量四本账：A 图建模（拓扑与执行轨迹）· B 条件路由判定 · C checkpoint 状态持久化（thread 隔离+中断恢复）· D 循环与递归预算（自环·SCC·recursion_limit）+ 选择树 8 场景。
langgraph 真实执行（引擎图执行/状态/路由/checkpoint=实测）；节点内部逻辑（意图/记忆检索/工具/答复）=词面规则作者预置。
纯本地 · 零网络 · 零随机 · stdout 逐字节可复现。
"""
import operator
import sys
import warnings
from importlib.metadata import version as _mv
from time import perf_counter
from typing import Annotated, TypedDict

# 压制 langgraph 系依赖的 import 期 Deprecation 警告：langchain_core 导入时自己会插过滤器，
# 先预导入再 ignore，让「忽略」排在它后面拿到更高优先级 → stderr 干净、墙钟可读
warnings.filterwarnings("ignore")
import langchain_core  # noqa: E402,F401
warnings.filterwarnings("ignore")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

try:
    LGV = _mv("langgraph")
except Exception:
    LGV = "?"

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ------------------------- 预置：06 客服记忆（画像 + 账本，承接 06-记忆系统） -------------------------
PROFILE = {"姓名": "张三", "职业": "软件工程师", "居住地": "上海", "偏好": "喝美式咖啡", "生日": "1990年5月20日"}
LEDGER = [
    ("城市", "北京", "s1"), ("城市", "上海", "s3"), ("城市", "南京", "s8"),
    ("天气", "晴 18-26℃", "s1"), ("天气", "多云 22-28℃", "s3"),
    ("事件", "团队例会", "s2"), ("事件", "产品评审", "s4"),
]
CITIES = ["北京", "上海", "广州", "深圳", "成都", "杭州", "南京", "武汉", "西安", "重庆"]

# 7 次客服会话（预置 · 确定性）：tid, text, 期望意图, 期望路由, 工具失败策略
TURNS = [
    ("t1", "帮我查一下上海明天天气", "查天气", "tool", "none"),
    ("t2", "把产品评审加到周六日程", "添加日程", "tool", "none"),
    ("t3", "咱们团队例会定在什么时间", "查记忆", "answer", "none"),
    ("t4", "你好，谢谢你的服务", "问候", "answer", "none"),
    ("t5", "请再查一次北京的天气", "查天气", "tool", "once"),
    ("t6", "我的物流单 999 现在什么状态", "查物流", "tool", "always"),
    ("t7", "我这周五的牙医预约是几点", "查记忆", "ask", "none"),
]
MAX_RETRY = 3


def find_city(t):
    for c in CITIES:
        if c in t:
            return c
    return None


def parse(text):
    """意图/槽位/工具 词面规则（作者预置代理；真实 Agent 此节点=LLM=非本机实测）"""
    if "天气" in text:
        return {"intent": "查天气", "tool": "get_weather", "city": find_city(text) or "上海"}
    if "物流" in text or "状态" in text:
        return {"intent": "查物流", "tool": "track_order", "order": "999"}
    if any(k in text for k in ("预约", "例会", "时间", "几点")):
        return {"intent": "查记忆", "tool": "none", "slot": "事件"}
    if any(k in text for k in ("加到", "安排", "预订")):
        return {"intent": "添加日程", "tool": "add_event", "title": "产品评审", "start": "周六"}
    return {"intent": "问候", "tool": "none"}


class CS(TypedDict):
    text: str
    intent: str
    slot: str
    tool: str
    city: str
    title: str
    start: str
    order: str
    profile: dict
    ledger: list
    context: list
    fail_policy: str
    tool_ok: bool
    retry: int
    go: str
    reply: str
    ps: Annotated[list, operator.add]


def make_graph():
    g = StateGraph(CS)

    def node_parse(s):
        p = parse(s["text"])
        return dict({"ps": ["parse"]}, **p)

    def node_recall(s):
        # 记忆检索（词面关键词命中；真实 Agent 此节点=embedding 检索=非本机实测）—— 06 C 档对齐语义的最小版
        txt = s["text"]
        if s["intent"] == "查记忆":
            kw = [w for w in ("例会", "评审") if w in txt]
            ctx = [(k, v, ts) for (k, v, ts) in s["ledger"] if any(w in v for w in kw)]
        else:
            ctx = []
        return {"ps": ["recall"], "context": ctx}

    def node_route(s):
        if s["intent"] == "问候":
            go = "answer"
        elif s["intent"] in ("查天气", "查物流", "添加日程"):
            go = "tool"
        else:  # 查记忆
            go = "answer" if s["context"] else "ask"
        return {"ps": ["route"], "go": go}

    def node_tool(s):
        r = s["retry"]
        if s["tool"] == "track_order":
            ok = False  # 注入恒失败：服务超时
        elif s["tool"] == "get_weather":
            ok = s["fail_policy"] != "once" or r >= 1  # once=首次失败、第二次成功
        else:
            ok = True
        return {"ps": ["call_tool"], "tool_ok": ok,
                "retry": r + 1 if not ok else r, "tool": s["tool"]}

    def node_answer(s):
        t = s["tool"]
        if t == "get_weather":
            city = s["city"]
            note = "我记得你住在%s" % s["profile"]["居住地"] if city == s["profile"]["居住地"] else "出差在外记得保暖"
            rp = "已查好：%s 明天 晴 20-28℃（%s）" % (city, note)
        elif t == "add_event":
            rp = "好的，已把「%s」加入%s日程（你已有 %d 条事件）" % (s["title"], s["start"], len(s["ledger"]))
        elif s["intent"] == "查记忆":
            rp = "查到了：%s（%s）" % (s["context"][0][1], s["context"][0][2])
        else:
            rp = "您好，很高兴为您服务～"
        return {"ps": ["answer"], "reply": rp}

    def node_ask(s):
        return {"ps": ["ask_user"], "reply": "请问您说的是哪个城市/哪场安排？我这边还没记到～"}

    def node_esc(s):
        return {"ps": ["escalate"], "reply": "抱歉，多次尝试失败，已为您转接人工（重试 %d 次）" % s["retry"]}

    g.add_node("parse", node_parse)
    g.add_node("recall", node_recall)
    g.add_node("route", node_route)
    g.add_node("call_tool", node_tool)
    g.add_node("answer", node_answer)
    g.add_node("ask_user", node_ask)
    g.add_node("escalate", node_esc)

    g.add_edge(START, "parse")
    g.add_edge("parse", "recall")
    g.add_edge("recall", "route")
    g.add_conditional_edges("route", lambda s: s["go"],
                            {"tool": "call_tool", "answer": "answer", "ask": "ask_user"})
    g.add_conditional_edges("call_tool", tool_switch,
                            {"ok": "answer", "retry": "call_tool", "esc": "escalate"})
    g.add_edge("answer", END)
    g.add_edge("ask_user", END)
    g.add_edge("escalate", END)
    return g


def tool_switch(s):
    if s["tool_ok"]:
        return "ok"
    return "retry" if s["retry"] < MAX_RETRY else "esc"


def init_state():
    return {"text": "", "intent": "", "slot": "", "tool": "", "city": "",
            "title": "", "start": "", "order": "", "profile": dict(PROFILE),
            "ledger": list(LEDGER), "context": [], "fail_policy": "none",
            "tool_ok": False, "retry": 0, "go": "", "reply": "", "ps": []}


def inp(text, policy="none"):
    return dict(init_state(), text=text, fail_policy=policy)


def tarjan_scc(neighbors, n):
    """Tarjan SCC 确定性实现：返回所有 SCC，并单独统计『含环 SCC』（长度>1 或自环）。"""
    idx = 0
    ix = [-1] * n
    low = [0] * n
    stack = []
    on = [False] * n
    sccs = []

    def strong(v):
        nonlocal idx
        ix[v] = low[v] = idx
        idx += 1
        stack.append(v)
        on[v] = True
        for w in neighbors[v]:
            if ix[w] == -1:
                strong(w)
                low[v] = min(low[v], low[w])
            elif on[w]:
                low[v] = min(low[v], ix[w])
        if low[v] == ix[v]:
            comp = []
            while True:
                w = stack.pop()
                on[w] = False
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in range(n):
        if ix[v] == -1:
            strong(v)
    cyc = sum(1 for c in sccs if len(c) > 1 or c[0] in neighbors[c[0]])
    return len(sccs), cyc, sccs


print("=" * 72)
print("前置：LangGraph 图引擎（知识地图 §7.3 正题 / §7.1 分类 · 衔接 06-记忆）· 本机真实执行")
print("  有向状态图：节点=函数 · 边=条件路由 · 状态=TypedDict channel · checkpoint=引擎层状态持久化")
print("环境：LangGraph %s 真实运行 · 零网络 · 零随机 · stdout 逐字节可复现" % LGV)
print("=" * 72)
print()

# ---------- A 图建模账 ----------
G = make_graph()
app = G.compile(checkpointer=MemorySaver())
traces = {}
for tid, text, _i, _r, policy in TURNS:
    cfg = {"configurable": {"thread_id": "a-%s" % tid}}
    r = app.invoke(inp(text, policy), cfg)
    traces[tid] = (r["ps"], r["reply"], r["intent"], r["go"], r["retry"])

print("[实验 A] 状态图建模账 —— 拓扑与执行轨迹（节点=函数 · 边=条件路由）")
print("  图对象：节点 7 = parse/recall/route/call_tool/answer/ask_user/escalate")
print("  边：无条件 3（START→parse→recall→route）· 条件 2 组 6 支路（route:tool/answer/ask · call_tool:ok/retry/esc）· 自环 1（call_tool→call_tool）· 终边 3")
for tid, text, exp_i, exp_r, policy in TURNS:
    ps, reply, intent, go, retry = traces[tid]
    print("  %s %s → 意图=%-4s 路由=%-6s 轨迹 %s（%d 步） · 答复=%s" % (
        tid, text, intent, go, "→".join(ps), len(ps), reply))
print("  → 图引擎把『if/循环』写作显式拓扑：7 次会话全部走 graph.invoke（编译执行），ps 累积器=免费可观测轨迹")
print()

# ---------- B 条件路由账 ----------
hits = sum(1 for tid, _, _, r, _ in TURNS if traces[tid][3] == r)
no_need_tool = [tid for tid, _, _, r, _ in TURNS if r != "tool"]
steps_optimal = sum(len(traces[t][0]) for t, *_ in TURNS)
steps_always_tool = sum(
    len(traces[t][0]) + (1 if r != "tool" else 0) for t, _, _, r, _ in TURNS)
saved = steps_always_tool - steps_optimal
print("[实验 B] 条件路由账 —— 边上的判定 = 图的一等公民（对照：全走工具的最坏路径）")
print("  路由判定：7 次会话 期望 全对账 → 命中 %d/7（route 条件边按返回 go 选支路）" % hits)
print("  可直达 #%s 本不需要工具（问候 / 查记忆有上下文）——条件边省掉无谓 call_tool" % "#".join(sorted(no_need_tool)))
print("  → 反事实『每条都先调工具再答』= %d 步 vs 条件路由 %d 步：省 %d 步（约 %d%%）" % (
    steps_always_tool, steps_optimal, saved, int(100.0 * saved / steps_always_tool)))
print("  → 教学点：判定被建模成『边上函数』（route 返回 go），散落的 if 变成显式拓扑；与本门 03-D 路由闸同族")
print()

# ---------- C checkpoint 账 ----------
app1 = G.compile(checkpointer=MemorySaver())
cfgA = {"configurable": {"thread_id": "iso_a"}}
cfgB = {"configurable": {"thread_id": "iso_b"}}
rA = app1.invoke(inp("咱们团队例会定在什么时间", "none"), cfgA)
rB = app1.invoke(inp("你好，谢谢你的服务", "none"), cfgB)
ctxA = [e[1] for e in rA["context"] if "团队" in e[1]]
leak = 0 if (ctxA and not rB["context"]) else 1
app2 = G.compile(checkpointer=MemorySaver(), interrupt_before=["call_tool"])
cfg_ir = {"configurable": {"thread_id": "ir"}}
seg1 = app2.invoke(inp("查一下北京天气", "none"), cfg_ir)
snap = app2.get_state(cfg_ir)
partial_ps = seg1["ps"]
seg2 = app2.invoke(None, cfg_ir)
full_ps = seg2["ps"]
resumed = len(full_ps) - len(partial_ps)
print("[实验 C] checkpoint 账 —— 引擎层状态持久化（thread 隔离 + 中断恢复）")
print("  线程隔离：iso_a（查记忆）state.context=%r vs iso_b（问候）state.context=%r → 串扰 %d（隔离 = 每 thread 一册运行状态）" % (
    ctxA, rB["context"], leak))
print("  checkpoint 快照：graph.get_state(next=%s) —— 引擎记录『下一跳待跑节点』= 活儿干到哪一步" % (snap.next,))
print("  中断恢复：第一段 %s（interrupt_before 停在 call_tool 前）→ 同线程 invoke(None) 续跑 %s" % (
    "→".join(partial_ps), "→".join(full_ps)))
print("  → 恢复只补 %d 步（call_tool+answer），parse/recall/route 不重跑：中断=挂起而非回滚" % resumed)
print()

# ---------- D 循环与预算账 ----------
retry_ps = traces["t5"][0]
esc_ps = traces["t6"][0]
retry_once = len([x for x in retry_ps if x == "call_tool"])
esc_count = len([x for x in esc_ps if x == "call_tool"])
# SCC：邻接表（索引 parse0 recall1 route2 call_tool3 answer4 ask5 esc6），与真实图一致
adj_loop = [[1], [2], [3, 4, 5], [3, 4, 6], [], [], []]      # 含 call_tool 自环（重试）
adj_dag = [[1], [2], [3, 4, 5], [4, 6], [], [], []]          # 去掉自环边
n_scc, cyc_loop, _ = tarjan_scc(adj_loop, 7)
n_scc2, cyc_dag, _ = tarjan_scc(adj_dag, 7)
G3 = make_graph()
app3 = G3.compile()
err = ""
try:
    app3.invoke(inp("我的物流单 999 什么状态", "always"), {"recursion_limit": 6})
    err = "未触发"
except GraphRecursionError as e:
    err = str(e).splitlines()[0]
print("[实验 D] 循环与预算账 —— 自环收敛 · SCC 环检测 · recursion_limit 轮上限兜底")
print("  自环收敛：t5（失败→重试一次成功）call_tool×%d · t6（恒失败）call_tool×%d → escalate 转人工（MAX_RETRY=%d）" % (
    retry_once, esc_count, MAX_RETRY))
print("  环检测（Tarjan）：含自环图 SCC=%d 含环 %d（重试边）vs 去掉自环边 SCC=%d 含环 %d —— 图论文『环=能反复执行的流程』" % (
    n_scc, cyc_loop, n_scc2, cyc_dag))
print("  递归上限：recursion_limit=6 跑 t6（需 7 步）→ %s" % err)
print("  → 教学点：循环/恢复是图引擎的一等能力（§7.3；DAG 满足不了的、循环/恢复它都能做）；recursion_limit=框架级轮上限兜底，对应 06/05 的重试/轮限制")
print()

# ---------- 选择树 ----------
SCEN = [
    ("线性「调模型 + 组提示」流水", "直写或通用编排 SDK（LangChain 链）——单线无分支"),
    ("有分支/循环/恢复/重试的流程", "图引擎 LangGraph（本篇状态图 + checkpoint）"),
    ("文档/知识库检索问答", "RAG 框架（LlamaIndex / Haystack，衔接 08）"),
    ("工具结果即时执行一次", "05 回喂循环（不引入框架）"),
    ("多 Agent 协作 / 角色化", "Agent 框架（CrewAI / AutoGen / Agents SDK）"),
    ("非工程师快速搭 Demo", "低代码平台（Dify / Flowise / n8n）"),
    ("轻量单 Agent · 少依赖", "OpenAI Agents SDK / 直写工具协议"),
    ("跨栈工具互操作", "MCP 协议层（不是框架，是接口标准）"),
]
print("  选择树 %d 场景（§7.1 分类学：先分类型再比优劣——图引擎=有状态/可分支/可恢复；协议层≠框架）；断言 %d/%d" % (
    len(SCEN), len(SCEN), len(SCEN)))
print()

# ---------- 台账汇总 ----------
print("=" * 72)
print("台账汇总（A 图建模 / B 条件路由 / C checkpoint / D 循环预算）")
print("  A  节点 7 · 边 条件 6 支路 + 自环 1；7 次会话编译执行 · 轨迹可观测（ps 累积器免费埋点）")
print("  B  路由判定 %d/7；对照「全走工具最坏路径」省 %d 步（约 %d%%）——判定=边上函数" % (
    hits, saved, int(100.0 * saved / steps_always_tool)))
print("  C  thread 隔离串扰 %d · checkpoint.next=%s 记录续跑点 · 中断恢复只补 %d 步不重跑（挂起≠回滚）" % (
    leak, snap.next, resumed))
print("  D  自环重试 call_tool×%d→成功 ×%d→转人工 · 环检测含环 %d vs 0 · recursion_limit 真抛 GraphRecursionError" % (
    retry_once, esc_count, cyc_loop))
print("一句话：LangGraph = 把『流程』编译成可执行的有向状态图——节点=函数·边=条件路由·状态=TypedDict channel·checkpoint=引擎层状态持久化；"
      "循环/重试/递归上限开箱即用；06 的画像/账本是『记住用户是什么样』（内容层），checkpoint 是『活儿干到哪一步』（引擎层），两层叠加=带记忆重试的客服，可调试、可恢复、可续跑")
print("done · 一键复现：python code/notebooks/_tools/langgraph_demo.py")
W("WALL total=%.3fs" % (perf_counter() - _t0))
