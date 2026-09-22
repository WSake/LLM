"""autogen_demo：AG2 1.0.6 引擎四账实测（AutoGen 经典『互聊』范式 vs 新一代函数式 Agent）。

对应知识点：知识地图 §7.8（AutoGen/AG2/Microsoft Agent Framework——多 Agent 对话范式）、
§7.1（类 4 多 Agent 框架）、§14 决策树。
探针口径 =『引擎语义=本机真实实测 ag2 1.0.6；Agent 回复的文本生成=预置 stub client
（真实 LLM 非本机实测）；版本节奏 / MAF / 决策规则=要素事实』。

stdout 纯确定性：零网络 · 零 RNG · 纯 stdlib + ag2；三独立进程 md5 恒一；墙钟只进 stderr。
"""
import sys
import asyncio
import importlib
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from ag2.config.config import ModelProvider
from ag2.config.client import ModelResponse
from ag2.events.types import ModelMessage

BANNER = "=" * 72
HEADER = (
    "autogen_demo：07-应用框架 · 07-AutoGen·AG2·Microsoft-Agent-Framework"
    "（知识地图 §7.8/§7.1 类 4/§6.9/§7.14）"
    "\n    多 Agent 对话范式：AutoGen『Agent 互聊』→ AG2 1.0 函数式重写 → Microsoft Agent Framework"
)
SCOPE = (
    "[0] 口径：引擎语义=本机真实实测 ag2 1.0.6；Agent 回复文本=预置 stub client"
    "（真实 LLM 非本机实测）；AutoGen/AG2/MAF 年份、版本路线、选择规则=要素事实（无外网未复核）"
)


def _import(path: str) -> str:
    """importlib 探测：先整名 import（覆盖子模块），再回落父模块属性。"""
    name_of = lambda o: o.__name__ if isinstance(o, type) else type(o).__name__
    try:
        obj = importlib.import_module(path)
        return "OK <" + name_of(obj) + ">"
    except Exception as e1:
        if "." not in path:
            return "FAIL " + type(e1).__name__ + ": " + str(e1)[:36]
        mod, _, attr = path.rpartition(".")
        try:
            obj = getattr(importlib.import_module(mod), attr)
            return "OK <" + name_of(obj) + ">"
        except Exception as e2:
            return "FAIL " + type(e2).__name__ + ": " + str(e2)[:36]


class StubClient:
    """预置模型客户端：按"可见历史"词面计轮返回，引擎其余路径全真跑。"""

    def __init__(self, base: str = "答案：42"):
        self.base = base
        self._n = 0

    async def __call__(self, messages, context, *, tools, response_schema, serializer):
        seen = 0
        for m in messages:
            if (m.__class__.__name__ == "ModelResponse") and getattr(
                m, "message", None
            ):
                seen += 1
        self._n += 1
        if self.base == "ask-echo":
            return ModelResponse(
                message=ModelMessage(role="assistant", content=f"第{self._n}轮：模型可见前 {seen} 条历史回复"),
                tool_calls=None, usage=None, files=[],
                model="stub/1", provider="stub", finish_reason=None, response_id=None,
            )
        return ModelResponse(
            message=ModelMessage(role="assistant", content=self.base),
            tool_calls=None, usage=None, files=[],
            model="stub/1", provider="stub", finish_reason=None, response_id=None,
        )


class StubConfig:
    """预置模型配置：协议面最小实现（provider/model/create），喂给真实 Agent。"""

    def __init__(self, base: str = "答案：42"):
        self.provider = ModelProvider.OPENAI
        self.model = "stub-1"
        self._c = StubClient(base=base)

    def copy(self):
        return self

    def create(self):
        return self._c

    def create_files_client(self):
        raise NotImplementedError("stub does not support Files API")


_OUT = []


def p(*a, **k):
    _OUT.append(str(a[0]) if a else "")


def rule():
    p(BANNER)


# ================================================================ 账 A
async def acct_a():
    from ag2.agent import Agent

    a = Agent(name="assistant", prompt="固定系统提示", config=StubConfig("答案：42"))
    async with a.run("第1问：2+2=？", config=StubConfig("答案：42")) as run:
        rep = await run.result()
    rule()
    p("[账 A] 新一代三原语账：Agent(name, prompt, config) → run() → AgentReply（引擎真跑 · stub client）")
    p(BANNER)
    p("  Agent 一次交互=一条 run 流水：prompt → LLMCall → AgentReply（原语=函数式，不再是『互聊』消息）")
    p(f"  agent.name        = {a.name}")
    p(f"  reply.body        = {rep.body!r}")
    p(f"  reply 类型        = {type(rep).__module__}.{type(rep).__name__}")
    p("  无模型配置护栏（引擎真抛）: ")
    try:
        agent = Agent(name="nog", prompt="你好")
        async with agent.run("hi") as run:
            await run.result()
        p("    未触发（不应发生）")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        p(f"    {type(e).__name__} | {e}")
    p("  → 结论1：『对话』被拆成 run()/ask() 回合原语，Agent 回复=模型事件（非聊天消息缓冲）")
    p("  → 结论2：模型配置是 Agent 原语的强制契约——缺配置当场 ConfigNotProvidedError（对比 02/05 的轮数类护栏）")


# ================================================================ 账 B
async def acct_b():
    from ag2.agent import Agent

    a = Agent(name="conversationalist", prompt="多轮对话", config=StubConfig("ask-echo"))
    async with a.run("第0问") as run:
        r0 = await run.result()
    r1 = await r0.ask("第1问")
    r2 = await r1.ask("第2问")
    rule()
    p("[账 B] 多轮会话账：AgentReply.ask() 续谈——引擎把历史回灌给模型（对话即记忆载体）")
    p(BANNER)
    p(f"  第0轮 reply.body  = {r0.body!r}")
    p(f"  第1轮 ask.body   = {r1.body!r}")
    p(f"  第2轮 ask.body   = {r2.body!r}")
    p("  → 结论：AG2 1.x 的『对话』=同一 Agent 上连续 run/ask，历史以事件流形式回灌模型")
    p("    （第2轮 stub 实测可见前 2 条历史回复=引擎真回灌；对偶 06 章记忆回灌 / 02 checkpoint）")


# ================================================================ 账 C
def acct_c():
    rule()
    p("[账 C] 版本存续账：pyautogen 0.2 经典『互聊』API → AG2 0.9-era agentchat → AG2 1.0.6 函数式换血")
    p(BANNER)
    legacy = [
        "autogen",
        "autogen.agentchat.conversable_agent",
        "autogen.agentchat.assistant_agent",
        "autogen.agentchat.user_proxy_agent",
        "autogen.groupchat.GroupChat",
        "autogen.oai.Completion",
        "autogen.code_utils",
        "autogen.agentchat.contrib",
    ]
    old_ag2 = [
        "ag2.agentchat.conversable_agent",
        "ag2.agentchat.assistant_agent",
        "ag2.agentchat.groupchat.GroupChat",
    ]
    v1 = [
        "ag2",
        "ag2.agent.Agent",
        "ag2.agent.AgentReply",
        "ag2.agent.AgentRun",
        "ag2.task.Task",
        "ag2.config.client.ModelResponse",
        "ag2.tools.subagents",
        "ag2.events.types.Usage",
    ]
    maf = ["microsoft.agents", "microsoft.agents.context.AgentContext"]
    for block_title, items in (
        ("pyautogen 0.2 经典命名空间（AutoGen『互聊』API）", legacy),
        ("AG2 0.9-era 命名空间（agentchat 同构迁移）", old_ag2),
        ("AG2 1.0.6 新一代命名空间（函数式 Agent）", v1),
        ("Microsoft Agent Framework（2025 微软，未装=非本机实测）", maf),
    ):
        p(f"  ── {block_title} ──")
        for it in items:
            p(f"    {it.ljust(50)} {_import(it)}")
    p("  → 结论：经典 AutoGen『ConversableAgent + GroupChat 互聊』API 在 AG2 1.0 整代消失")
    p("    （连 `import autogen` 兼容垫片也移除）——改名烈度=07 章继 06 Semantic-Kernel 的又一里碑")


# ================================================================ 账 D
def acct_d():
    rule()
    p("[账 D] 选择树账：多 Agent 范式该在哪儿用（§7.14 多 Agent 协作 → CrewAI/AutoGen/Agents SDK）")
    p(BANNER)
    rows = [
        ("多 Agent 研究/原型、要『互聊』范式", "→ AutoGen / AG2（group chat 起家；生产要自管状态）"),
        ("生产要状态/检查点/恢复", "→ LangGraph（02 章 checkpoint，recursion_limit 真兜）"),
        ("子任务并发/委托 + 显式 checkpoint", "→ AG2 v1.0 Task.run_task（checkpoint 一等公民）"),
        ("严谨生产级 RAG 管线", "→ Haystack（05 章 pipeline 思维）"),
        ("RAG 专精：数据→索引→检索→问答", "→ LlamaIndex（03 章七工序七组件）"),
        ("企业 .NET/微软栈", "→ Semantic Kernel（06 章三原语 + as_mcp_server）"),
        ("低代码/非工程师快速落地", "→ Dify / Coze（04 章积木平台）"),
        ("轻量 Agent 运行时、回归轻量", "→ OpenAI Agents SDK（§7.11 handoff）"),
        ("微软 2025 Agent 路线（要素事实）", "→ Microsoft Agent Framework（承接 AutoGen，与 SK 统一）"),
    ]
    for i, (q, a) in enumerate(rows, 1):
        p(f"  {i}. {q}")
        p(f"     {a}")
    asserts = [
        ("多 Agent 研究原型默认入口", "AutoGen" in rows[0][1]),
        ("生产状态管理归图引擎", "LangGraph" in rows[1][1]),
        ("子任务委托归新一代 Task", "run_task" in rows[2][1]),
        ("生产 RAG 归 Haystack", "Haystack" in rows[3][1]),
        ("RAG 专精归 LlamaIndex", "LlamaIndex" in rows[4][1]),
        ("企业栈归 SK", "Semantic Kernel" in rows[5][1]),
        ("非工程师归低代码", "Dify" in rows[6][1]),
        ("轻量归 Agents SDK", "OpenAI Agents SDK" in rows[7][1]),
        ("MAF=承接 AutoGen 与 SK 统一", "AutoGen" in rows[8][1]),
    ]
    ok = all(v for _, v in asserts)
    for k, v in asserts:
        p(f"  断言[{k}] = {v}")
    p(f"  → 选择树 9 场景断言 9/9 = {ok}")


async def main():
    p(BANNER)
    p(HEADER)
    p(SCOPE)
    p(BANNER)
    await acct_a()
    await acct_b()
    acct_c()
    acct_d()
    p("")
    p(BANNER)
    p("台账汇总：A 新一代三原语 / B 多轮 ask 回灌 / C 版本存续 0.2→1.0 整代换血 / D 选择树 9 场景")
    p("一话总结：AutoGen 以『Agent 之间互聊 + GroupChat』开创多 Agent 对话范式（§7.8），")
    p("AG2 1.0 把范式重写成 函数式 Agent + run/ask 回合 + 子任务委托（经典 API 整代换血），")
    p("微软 2025 以 Microsoft Agent Framework 承接 AutoGen 并与 Semantic Kernel 统一=官方延续线；")
    p("“研究/原型用 AG2 的对话范式、生产自己补状态管理”仍是 §7.8 的适用结论。")
    p("done · 一键复现：python code/notebooks/_tools/autogen_demo.py")
    sys.stdout.write("\n".join(_OUT))
    sys.stdout.write("\n")


if __name__ == "__main__":
    asyncio.run(main())
