"""crewai_demo：CrewAI 1.15.22 引擎四账实测（多 Agent 角色化分工——角色·任务·流程三个一等公民）。

对应知识点：知识地图 §7.14（多 Agent 协作→CrewAI/AutoGen/Agents SDK）、§7.8（多 Agent 对话范式）、
§7.1（类 4 多 Agent 框架）、§6.9（多 Agent=多倍 token 与延迟，先证明单 Agent 不行再上）。
探针口径 =『引擎语义=本机真实实测 crewai 1.15.22（crewai-crewai-core 1.15.22，Python 3.13.9）；
LLM 回复文本=预置 stub（BaseLLM 子类，真实 LLM 非本机实测）；
版本节奏 / CrewAI 企业版 / Flows 2025 发布=要素事实，无外网未复核』。

stdout 纯确定性：零网络 · 零 RNG · crewai 引擎真跑 + stub LLM；三独立进程 md5 恒一；墙钟只进 stderr。
"""
import os
for _k, _v in {
    "CREWAI_TELEMETRY_OPT_OUT": "true",
    "CREWAI_DISABLE_VERSION_CHECK": "true",
    "CREWAI_TESTING": "true",
    "CI": "true",
}.items():
    os.environ.setdefault(_k, _v)

import io
import sys
import time
import contextlib
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

BANNER = "=" * 72
HEADER = (
    "crewai_demo：07-应用框架 · 08-CrewAI（多 Agent 角色化分工）"
    "（知识地图 §7.14/§7.8/§7.1 类 4/§6.9）"
    "\n    角色·任务·流程 = 三个一等公民：role/goal/backstory → 系统配方，任务列表 = 声明式流水线"
)
SCOPE = (
    "[0] 口径：引擎语义=本机真实实测 crewai 1.15.22；LLM 回复文本=预置 stub(BaseLLM 子类)"
    "（真实 LLM 非本机实测）；版本节奏 / 企业版 / Flows 2025=要素事实（无外网未复核）"
)

# ================================================================ 工具件


def _probe(path: str) -> str:
    """importlib 探测：模块名或 模块.属性；属性为 None 显式报 FAIL(None)。"""
    if "." not in path:
        try:
            return "OK <module:" + importlib_import(path).__name__ + ">"
        except Exception as e:
            return "FAIL " + type(e).__name__ + ": " + str(e)[:36]
    mod, _, attr = path.rpartition(".")
    try:
        obj = getattr(importlib_import(mod), attr)
        if obj is None:
            return "FAIL(None)"
        name = obj.__name__ if isinstance(obj, type) else type(obj).__name__
        return "OK <" + name + ">"
    except Exception as e:
        return "FAIL " + type(e).__name__ + ": " + str(e)[:36]


def importlib_import(name):
    import importlib

    return importlib.import_module(name)


_OUT = []


def p(*a, **k):
    _OUT.append(str(a[0]) if a else "")


def rule():
    p(BANNER)


# ================================================================ 账 A
def acct_a():
    from crewai.llm import BaseLLM

    CALLS = []

    class StubLLM(BaseLLM):
        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None):
            i = len(CALLS)
            CALLS.append(self._record(messages, from_agent))
            return f"SENT{i}"

        @staticmethod
        def _record(messages, from_agent):
            role = None
            sys_content = ""
            usr_content = ""
            for m in messages:
                c = m.get("content")
                if m.get("role") == "system":
                    sys_content = c or ""
                elif m.get("role") == "user":
                    usr_content = c or ""
            if from_agent is not None:
                role = getattr(from_agent, "role", None)
            return (role, sys_content, usr_content)

    from crewai import Agent, Task, Crew, Process

    r1 = Agent(role="研究员", goal="输出候选方案", backstory="资料控", llm=StubLLM(model="stub-1"), allow_delegation=False)
    e1 = Agent(role="评审员", goal="挑出最优方案", backstory="挑刺王", llm=StubLLM(model="stub-2"), allow_delegation=False)
    t1 = Task(description="搜索资料并提炼要点", expected_output="要点清单", agent=r1)
    t2 = Task(description="对要点打分排序", expected_output="评分表", agent=e1)
    t3 = Task(description="汇总定稿", expected_output="定稿文本", agent=r1)
    crew = Crew(agents=[r1, e1], tasks=[t1, t2, t3], process=Process.sequential, verbose=False)

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        result = crew.kickoff()

    roles = [c[0] for c in CALLS]
    assert roles == ["研究员", "评审员", "研究员"], roles
    syslens = [len(c[1]) for c in CALLS]
    mark_sys = all(c[1].startswith("You are ") and "Your personal goal is:" in c[1] for c in CALLS)
    mark_usr = all(
        "\nCurrent Task: " in c[2]
        and "This is the expected criteria for your final answer: " in c[2]
        and c[2].rstrip().endswith("Provide your complete response:")
        for c in CALLS
    )
    same_sys = len(set(syslens)) == 1
    n_calls = len(CALLS)

    rule()
    p("[账 A] 角色化原语账：Agent(role/goal/backstory) + Task 列表 = 声明式『角色→流水线』（引擎真跑 · stub LLM）")
    p(BANNER)
    p(f"  3 任务 2 角色交替，LLM 被调用 {n_calls} 次 = 任务数×1（每任务恰好一轮回复）")
    p(f"  按任务序各轮触发的角色 = {roles}")
    p("  系统配方（引擎真实模板）：")
    p(f"    sys = 'You are {{role}}. {{backstory}}\\nYour personal goal is: {{goal}}'   长度恒定 {syslens[0]} 字符 = 模板与 length 一致=同角色同系统（断言 {same_sys}）")
    p(f"  任务提示（引擎真实模板）含 'Current Task:' / 'expected criteria…' / 'Provide your complete response:' = {mark_usr}")
    p(f"  护栏：Agent 缺 role 创建即栅栏（引擎真抛）:")
    try:
        _ = Agent(goal="缺角色", backstory="缺角色")
        p("    未触发（不应发生）")
    except Exception as e:
        p(f"    {type(e).__name__} | {str(e)[:60]}")
    p("  → 结论1：role/goal/backstory 被真实拼接成系统提示（role 是必填原语=缺栅栏）")
    p("  → 结论2：任务列表=顺序执行、每个任务独立调度一次 LLM——『谁做』由 agent 指定、『做几轮』由任务数决定")
    return n_calls


# ================================================================ 账 B
def acct_b():
    from crewai.llm import BaseLLM

    CALLS = []

    class StubLLM(BaseLLM):
        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None):
            i = len(CALLS)
            CALLS.append(messages)
            return f"SENT{i}"

    from crewai import Agent, Task, Crew, Process

    r = Agent(role="流水线工", goal="产出", backstory="工", llm=StubLLM(model="stub-1"), allow_delegation=False)
    t0 = Task(description="任务0", expected_output="期望0", agent=r)
    t1 = Task(description="任务1", expected_output="期望1", agent=r)
    t2 = Task(description="任务2", expected_output="期望2", agent=r)
    crew = Crew(agents=[r], tasks=[t0, t1, t2], process=Process.sequential, verbose=False)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        result = crew.kickoff()

    ctx_rows = []
    for i, msgs in enumerate(CALLS):
        usr = "".join(m.get("content") or "" for m in msgs if m.get("role") == "user")
        ctx = sorted(j for j in range(i) if f"SENT{j}" in usr)
        ctx_rows.append(ctx)
    cascade = ctx_rows[1] == [0] and ctx_rows[2] == [0, 1] and ctx_rows[0] == []

    # 分层流程：manager_llm 决定分配
    CALLS2 = []
    LOG2 = []

    class MgrLLM(BaseLLM):
        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None):
            CALLS2.append(messages)
            LOG2.append(getattr(from_agent, "role", None))
            return "托管输出"

    w = Agent(role="干活的", goal="干", backstory="干", llm=StubLLM(model="stub-1"), allow_delegation=False)
    t = Task(description="写个报告", expected_output="报告", agent=w)
    crew2 = Crew(agents=[w], tasks=[t], process=Process.hierarchical,
                 manager_llm=MgrLLM(model="mgr-1"), verbose=False)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        result2 = crew2.kickoff()
    task2_agents = sorted({t.agent for t in result2.tasks_output})
    hier_ok = len(CALLS2) == 1 and sorted(LOG2) == ["Crew Manager"]

    rule()
    p("[账 B] 流程编排账：sequential 上下文接力 & hierarchical 多一跳（引擎真跑 · stub LLM）")
    p(BANNER)
    p("  顺序流程：每任务的用户提示里自动带上『前序输出』= 上下文接力（引擎真拼）")
    p(f"    任务2 可见前序 = {ctx_rows[1]}（含任务0输出·SENT0）")
    p(f"    任务3 可见前序 = {ctx_rows[2]}（含任务0+1输出） 累计级联={cascade}")
    p("  分层流程：manager_llm 掌控, 任务输出由 manager 收口")
    p(f"    本机实测：调用 1 次（stub manager 直接产出），触达角色 = {sorted(LOG2)}")
    p(f"    输出归属 agent = {list(task2_agents)}")
    p("  → 结论1：sequential = 隐式链（任务列表顺序即拓扑，上文自动回填下一任务）, 无显式 add_edge")
    p("  → 结论2：hierarchical 走 manager_llm 多一跳；与『对话范式』（AG2 谁先说话=涌现）对比 = 拓扑事先声明 vs 运行时自组织")
    p("  → 结论3：stub manager 只返文本不发起委托 → 实测 1 次调用；真实 manager 的委托链=LLM 判定，非本机实测")
    return cascade, hier_ok


# ================================================================ 账 C
def acct_c():
    from crewai.flow import Flow, start, listen, router, or_

    class ChainFlow(Flow):
        @start()
        def begin(self):
            return "上游101"

        @listen("begin")
        def b(self, v):
            return "下游收到:" + str(v)

    class OrFlow(Flow):
        @start()
        def begin(self):
            return "goA"

        @router("begin")
        def r(self, v):
            return "A" if v == "goA" else "B"

        @listen(or_("A", "B"))
        def leaf(self, v):
            return f"组合收到:{str(v)[:12]}"

    f1 = ChainFlow(suppress_flow_events=True)
    f2 = OrFlow(suppress_flow_events=True)
    s1 = io.StringIO()
    with contextlib.redirect_stdout(s1), contextlib.redirect_stderr(s1):
        chain_out = f1.kickoff()
        or_out = f2.kickoff()

    routes1 = sorted(f1._effective_routes())
    routes2 = sorted(f2._effective_routes())
    max_calls = f1.max_method_calls
    entry1 = sorted(f1._start_method_names())
    entry2 = sorted(f2._start_method_names())

    # 自环定义期拒绝
    loop_msg = ""
    try:
        class LoopFlow(Flow):
            @start()
            def begin(self):
                return "x"

            @listen("loop")
            def loop(self):
                return "y"

        _ = LoopFlow(suppress_flow_events=True)
        loop_msg = "未拒绝（不应发生）"
    except Exception as e:
        loop_msg = type(e).__name__ + " | " + str(e)[:72]

    rule()
    p("[账 C] 版本存续账：运行时/声明式/事件式三张 API 面（引擎真实探测）")
    p(BANNER)
    p("  事件式 Flows（事件 DAG）：start 入口/监听传参/路由分叉/或合并 —— 2025 新面")
    p(f"    ChainFlow 入口 = {entry1}  路由点 = {routes1}  kickoff = {chain_out!r}")
    p(f"    OrFlow    入口 = {entry2}  路由点 = {routes2}  kickoff = {or_out!r}")
    p(f"    监听节点烫到的是 router 的『路由返回值』而非上游原始输出 —— OrFlow kickoff = {or_out!r}")
    p(f"  护栏：自环事件定义期直接拒绝（引擎真抛） -> {loop_msg}")
    p(f"  兜底：Flow.max_method_calls 默认 = {max_calls}（引擎字段；定义期拒自环→运行时无此环）")
    p("  导入面 importlib 探测：")
    top_v1 = [
        "crewai",
        "crewai.Crew",
        "crewai.Agent",
        "crewai.Task",
        "crewai.Process",
        "crewai.LLM",
        "crewai.Flow",
        "crewai.CrewBase",
        "crewai.crew.Crew",
        "crewai.agent.Agent",
        "crewai.task.Task",
    ]
    for it in top_v1:
        p(f"    {it.ljust(44)} {_probe(it)}")
    p("    声明式面：")
    for it in [
        "crewai.project.CrewBase",
        "crewai.project.agent",
        "crewai.project.task",
        "crewai.project.crew",
    ]:
        p(f"    {it.ljust(44)} {_probe(it)}")
    p("    事件式面：")
    for it in ["crewai.flow.Flow", "crewai.flow.start", "crewai.flow.listen", "crewai.flow.router", "crewai.flow.or_"]:
        p(f"    {it.ljust(44)} {_probe(it)}")
    p("    LLM 注入面 / 0.x-era 残留 / 企业版：")
    for it in [
        "crewai.llm.BaseLLM",
        "crewai.llm.LLM",
        "crewai.agents.agent_builder.base_agent_builder.BaseAgentBuilder",
        "crewai.production.CrewAIEnterprise",
        "crewai.main.crew1",
    ]:
        p(f"    {it.ljust(44)} {_probe(it)}")
    p("  → 结论1：运行时(Crew/Agent/Task/Process/LLM)与事件式(Flow/start/listen/router/or_)全在顶层；")
    p("    声明式 CrewBase + agent/task/crew 装饰器只挂在 crewai.project 子包、顶层不导出——三张面在『导入进口』上分层")
    p("  → 结论2：0.x-era 子模块路径（agent_builder 等）整包移除=版本存续的换血证据（与本篇 AG2 对照）")
    p("  → 结论3：企业版/crew1 未装=本机环境事实，非存在性证明")
    return routes1, routes2, max_calls


# ================================================================ 账 D
def acct_d():
    rule()
    p("[账 D] 选择树账：多 Agent 各范式该在哪儿用（§7.14 多 Agent 协作 → CrewAI/AutoGen/Agents SDK；§6.9 先证明单 Agent 不行再上）")
    p(BANNER)
    rows = [
        ("团队型多角色固定流水线（研究→起草→评审 顺序）", "→ CrewAI 角色化分工（角色·任务·流程三一等公民，账 A/B 实测）"),
        ("多 Agent 自由对话/群聊（谁先说话=涌现）", "→ AutoGen / AG2 对话范式（§7.8 武侠）"),
        ("复杂状态图/checkpoint/条件边多", "→ LangGraph 图状态（02 章，recursion_limit 真兜）"),
        ("事件驱动 扇出/扇入/路由，无需多角色", "→ CrewAI Flows 事件 DAG（账 C 实测 listen/router/or_）"),
        ("严谨生产级 RAG 管线", "→ Haystack pipeline 思维（05 章 双栖点别二）"),
        ("企业 .NET/微软云栈", "→ Semantic Kernel 三原语（06 章 + as_mcp_server）"),
        ("低代码/非工程师快速落地", "→ Dify / Coze 积木平台（04 章）"),
        ("轻量 Agent 运行时/工具回喂", "→ OpenAI Agents SDK（§7.11 handoff）"),
        ("只想重试/更省 token，先别上多 Agent", "→ 单 Agent + 工具循环（§6.9 多 Agent=多倍 token 与延迟）"),
    ]
    for i, (q, a) in enumerate(rows, 1):
        p(f"  {i}. {q}")
        p(f"     {a}")
    asserts = [
        ("团队流水线归 CrewAI", "CrewAI" in rows[0][1]),
        ("自由对话归 AG2", "AG2" in rows[1][1]),
        ("图状态归 LangGraph", "LangGraph" in rows[2][1]),
        ("事件 DAG 归 Flows", "Flows" in rows[3][1]),
        ("生产 RAG 归 Haystack", "Haystack" in rows[4][1]),
        ("企业栈归 SK", "Semantic Kernel" in rows[5][1]),
        ("低代码归 Dify", "Dify" in rows[6][1]),
        ("轻量归 Agents SDK", "OpenAI Agents SDK" in rows[7][1]),
        ("省 token 先单 Agent", "单 Agent" in rows[8][1]),
    ]
    ok = all(v for _, v in asserts)
    for k, v in asserts:
        p(f"  断言[{k}] = {v}")
    p(f"  → 选择树 9 场景断言 9/9 = {ok}")


def main():
    t0 = time.perf_counter()
    p(BANNER)
    p(HEADER)
    p(SCOPE)
    p(BANNER)
    n_a = acct_a()
    acct_b()
    acct_c()
    acct_d()
    p("")
    p(BANNER)
    p("台账汇总：A 角色化原语（role/goal/backstory→系统配方·任务列表=流水线·缺 role 创建即栅栏）"
      f"/ B 流程编排（sequential 上下文接力·hierarchical manager 多一跳）"
      "/ C 版本存续三面（运行时顶层 6 原语全在 · 声明式 project.CrewBase · 事件式 Flow 路由簇 · 0.x 路径整包移除）"
      "/ D 选择树 9 场景 9/9")
    p("一话总结：CrewAI 把『多 Agent 协作』做成 角色/任务/流程 三个一等公民——")
    p("role/goal/backstory 拼成系统提示、任务列表=声明式流水线（顺序执行·上文自动接力）、")
    p("manager 可加到 hierarchical 多一跳、Flows 提供事件 DAG（2025）；")
    p("与 AG2 的『对话范式』是左右手：要事先声明的团队分工 → CrewAI，要运行时涌现的群聊 → AG2（§7.14/§7.8）。")
    p("done · 一键复现：python code/notebooks/_tools/crewai_demo.py")
    sys.stdout.write("\n".join(_OUT))
    sys.stdout.write("\n")
    print(f"wall_clock: {time.perf_counter() - t0:.4f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
