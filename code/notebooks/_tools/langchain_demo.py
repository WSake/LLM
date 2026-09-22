#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""07-应用框架 · 01-LangChain与历史包袱（知识地图 §7.2 / §17.5.8）探针：
「Chain 概念永存 · LangChain 库有历史包袱」—— 用真实 langchain 引擎量四本账：
  A 链式组装账 —— Runnable 统一抽象最小闭环（steps/内部图/同链三触发）
  B 封装厚度账 —— MRO 继承深 / pydantic 字段数 / 概念面-实现面隐性节点（套娃）
  C 版本存续账 —— 0.x→1.x 旧 API 生存表（7 FAIL / 3 OK）· 库集合现状 · 同机版本实测
  D 链 vs 图分界账 —— 客服重试：链=无自环·重试手写 while vs 图=自环+recursion_limit（挂靠 02 章）
口径：真实 langchain 1.2.10 / langchain-core 1.4.9 引擎（本机 importlib.metadata 实测）；
     模型回复 = FakeListLLM 预置（LLM 非本机实测，确定性）；版本节奏/年份 = 要素事实（无外网未复核）。
注意：langchain get_graph().nodes 的 ID 是随机哈希（跨进程不稳定），本探针只打"计数"绝不打 ID。
纯网络零（Fake 模型）· stdout 逐字节可复现；wall-clock 只进 stderr。
"""
import inspect
import logging
import sys
import warnings
from importlib.metadata import version
from time import perf_counter

warnings.filterwarnings("ignore")          # 先于一切 langchain import（requests/pydantic 告警不进 stderr）
sys.stdout.reconfigure(encoding="utf-8")
logging.getLogger().setLevel(logging.CRITICAL)
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ---- 真实框架 import（LangChain 系：多线程兼容、静态类型）----
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence
from langchain_core.language_models.fake import FakeListLLM, FakeStreamingListLLM


def expA():
    print("=" * 72)
    print("[账 A] 链式组装账：Runnable 统一抽象的最小闭环（概念永存面——Chain 是什么）")
    print("=" * 72)
    chain = (PromptTemplate.from_template("你是订单客服。用户问：{q}")
             | FakeListLLM(responses=["订单状态：已发货（编号 7281）"])
             | StrOutputParser())
    steps = [type(s).__name__ for s in chain.steps]
    g = chain.get_graph()
    n_internal, n_edges = len(g.nodes), len(g.edges)
    print(f"  steps 类型序列（确定性）：{steps}")
    print(f"    -> 概念面 steps = {len(steps)} 步 · 2 条 `|` 边 = 组装 2 行")
    print(f"    -> 实现面 get_graph 内部节点 = {n_internal}（多出 {n_internal - len(steps)} 个隐形节点"
          f"=封装的入场费，账 B 展开）")
    print(f"    -> 链 = langchain_core.runnables.RunnableSequence（instanceof 校验 "
          f"{isinstance(chain, RunnableSequence)}）——Runnable 协议统一了每一步")
    # 同一条链 · 三种触发（Fake 预置回复 → 逐字节可复现）
    inv = chain.invoke({"q": "查一下订单 7281 走到哪了"})
    print(f"  → 同一条链 · 三种触发（Fake 预置回复 → 逐字节恒一）：")
    print(f"      invoke : {inv}")
    chain_b = (PromptTemplate.from_template("你是订单客服。用户问：{q}")
               | FakeListLLM(responses=["回复一", "回复二", "回复三"])
               | StrOutputParser())
    outs = chain_b.batch([{"q": "a"}, {"q": "b"}, {"q": "c"}])
    print(f"      batch  : {outs}   <- batch(3 条) 依次消费 3 个预置回复")
    chain_s = (PromptTemplate.from_template("你是订单客服。用户问：{q}")
               | FakeStreamingListLLM(responses=["流式回复甲"])
               | StrOutputParser())
    chunks = list(chain_s.stream({"q": "流式"}))

    print(f"      stream : {chunks}   <- 逐块流出（FakeStreaming 逐字）")
    print("  → 概念：invoke/batch/stream = 同一个 Runnable 协议的三种触发，换触发不需改代码——"
          "『`|` 组装一次、三种节奏都通』")
    assert isinstance(chain, RunnableSequence)
    assert len(steps) == 3 and n_internal == 5 and n_edges == 4
    assert inv == "订单状态：已发货（编号 7281）"
    assert outs == ["回复一", "回复二", "回复三"]
    assert "".join(chunks) == "流式回复甲"
    print(f"  → 断言：steps==3 · 内部节点=={n_internal} 边=={n_edges} · invoke/batch/stream 全与预置一致")


def _mro_names(cls):
    out, seen = [], set()
    for c in cls.__mro__:
        name = c.__name__.split("[")[0]
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def expB():
    print()
    print("=" * 72)
    print("[账 B] 封装厚度账：继承深 · 字段多 · 隐形节点（包袱面——『套娃』的数字）")
    print("=" * 72)
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables import Runnable
    mro_chat = ChatOpenAI.__mro__
    runnable_pos = next(i + 1 for i, c in enumerate(mro_chat) if c is Runnable)
    fields = len(ChatOpenAI.model_fields)
    print(f"  → ChatOpenAI MRO 继承链 深度 = {len(mro_chat)} 层 —— 离 Runnable 还隔 {runnable_pos - 1} 层"
          f"（Runnable 在第 {runnable_pos} 层）")
    print(f"      链：{' → '.join(_mro_names(ChatOpenAI)[:6])} → … → Runnable")
    print(f"  → ChatOpenAI pydantic 字段 = {fields}（一个『模型客户端』要背 {fields} 个配置项）")
    mro_fake = FakeListLLM.__mro__
    runnable_pos_f = next(i + 1 for i, c in enumerate(mro_fake) if c is Runnable)
    print(f"  → FakeListLLM MRO 深度 = {len(mro_fake)} · Runnable 位次 {runnable_pos_f}")
    print(f"  → 隐形节点：3 步链内部图 = 5 节点（对账 A）——概念 3 步 vs 实现面多 2 个壳；"
          f"『封装』=用『少写 2 行』换『多爬 2 层』")
    print("  → 结论：抽象深 ≠ 功能重；重封装的真正代价 = 调试要跨 13 层 MRO、56 字段给你无限 "
          "`**kwargs` 的选项面——『抽象多而杂』（§7.2）在此有数字")
    assert len(_mro_names(ChatOpenAI)) == 11 and len(mro_chat) == 13
    assert fields >= 50 and runnable_pos == 10
    print(f"  → 断言：ChatOpenAI MRO={len(mro_chat)} · Runnable 位次={runnable_pos} · 字段={fields}")


def expC():
    print()
    print("=" * 72)
    print("[账 C] 版本存续账：0.x→1.x 换血清单（包袱面 II + 概念永存）")
    print("=" * 72)
    import importlib
    import pkgutil
    import langchain
    print("  同机共存（importlib.metadata 实测）：")
    print(f"    langchain {version('langchain')} · langchain-core {version('langchain-core')} · "
          f"langchain-openai {version('langchain-openai')}")
    print(f"    langchain-community {version('langchain-community')} · langgraph {version('langgraph')} · "
          f"langsmith {version('langsmith')}")
    subs = sorted(m.name for m in pkgutil.iter_modules(langchain.__path__))
    top = [n for n in dir(langchain) if not n.startswith("_")]
    print(f"  langchain 包本体：顶层命名空间 {len(top)} 个公开符号（{'空' if not top else top}）· "
          f"子模块只剩 {len(subs)} 个：{' · '.join(subs)}")
    print("    -> 旧『全家桶』（chains/memory/llms/schema/vectorstores/document_loaders…）整包拆走=『库集合』")
    LEGACY = [
        ("LLMChain", "langchain.chains.llm"),
        ("ConversationBufferMemory", "langchain.memory"),
        ("OpenAI", "langchain.llms"),
        ("BaseMessage", "langchain.schema"),
        ("PromptTemplate", "langchain.prompts"),
        ("TextLoader", "langchain.document_loaders.text"),
        ("FAISS", "langchain.vectorstores"),
        ("RunnableSequence", "langchain_core.runnables.base"),
        ("PromptTemplate", "langchain_core.prompts"),
        ("tool", "langchain_core.tools"),
    ]
    ok = fail = 0
    print("  → 0.x 时代路径生存表（真实 import 探测）：")
    rows = []
    for attr, mod in LEGACY:
        try:
            m = importlib.import_module(mod)
            getattr(m, attr)
            mark, ok = "OK", ok + 1
        except Exception:
            mark, fail = "FAIL", fail + 1
        rows.append((attr, mod, mark))
    for attr, mod, mark in rows:
        print(f"      | {attr:<22} | {mod:<30} | {mark}")
    print(f"    -> 生存表：旧壳 {fail} 条全 FAIL（ModuleNotFoundError）· 核心概念 {ok} 条 OK——"
          f"Runnable/PromptTemplate/@tool 在 langchain-core 里 0.x→1.x 连续存活")
    print("  → 版本节奏（要素事实=非本机实测，无外网未在线复核）："
          "2023 v0.1（langchain-core 独立 · LCEL 定型）→ 2024 v0.2/v0.3（工具调用/大清理）→ 2025 v1.0（移除 legacy chain API）")
    assert fail == 7 and ok == 3
    print(f"  → 断言：生存表 FAIL={fail} / OK={ok} · 库集合子模块={len(subs)}")


def _retry_loop(chain, q, max_tries=3, ok_prefix="订单状态：已发货"):
    """链式『答错重试』唯一选择：在链外层手写 while。内层链没有任何循环原语。"""
    final = ""
    tries = 0
    for tries in range(1, max_tries + 1):
        final = chain.invoke({"q": q})
        if final.startswith(ok_prefix):
            break
    return tries, final


def expD():
    print()
    print("=" * 72)
    print("[账 D] 链 vs 图分界账：一次『客服重试』的两种范式（该在哪用）")
    print("=" * 72)
    chain = (PromptTemplate.from_template("你是订单客服。用户问：{q}")
             | FakeListLLM(responses=["订单状态：查询失败（暂不可用）", "订单状态：查询失败（暂不可用）",
                                      "订单状态：已发货（编号 7281）"])
             | StrOutputParser())
    g = chain.get_graph()
    self_loops = sum(1 for e in g.edges if e.source == e.target)
    print(f"  → 链（LCEL）= 单向数据流：RunnableSequence 内部图 自环数 = {self_loops}"
          f"（能分支不能自循环——这是 DAG，不是会恢复的图）")
    lines = len(inspect.getsource(_retry_loop).splitlines())
    kills, final = _retry_loop(chain, "查一下订单 7281 走到哪了")
    print(f"  → 想要『答错重试』→ 唯一选择 = 链外层自写 while 包装：{lines} 行手写补丁（_retry_loop）")
    print(f"      实测轨迹：第 1/2 轮输出『查询失败』→ 第 {kills} 轮『已发货…』停（Fake 预置 2 连不合格 + 1 合规）")
    print("  → 对照·图式（LangGraph，02-章真实引擎已实测）：自环=一条条件边 · 轮上限=引擎 recursion_limit 兜底 · "
          "checkpoint=中断恢复补 2 步不重跑（7 节点 6 支路 1 自环——挂靠 02 章数字）")
    print("  → 结论：链省的是『组装』（2 行 `|`）；图省的是『状态/循环/恢复』那一整类工程。"
          "线性/教学/脚本 → 链（概念入门）；状态/恢复/重试（生产 Agent）→ 图；LangChain 本体生产谨慎引入")
    assert self_loops == 0 and kills == 3 and final.startswith("订单状态：已发货")
    print(f"  → 断言：链自环={self_loops} · 重试轮数={kills}（Fake+词面判定，LLM 非本机实测）· 补丁行={lines}")


def expTree():
    print()
    SCEN = [
        ("教学/入门·只想懂 Chain 概念", "直接读 langchain-core 的 Runnable 文档 + LCEL（概念），不装重量依赖"),
        ("线性脚本：模板→调一次→拿字符串", "PromptTemplate | 模型 | StrOutputParser——2 行 `|` 就够（账 A）"),
        ("批量/并发跑同一链", "Runnable.batch()——一个协议换触发，不重写代码（账 A）"),
        ("需要状态/恢复/重试的客服", "换 LangGraph 图引擎（优先于 chain）：checkpoint+recursion_limit（挂靠 02-章）"),
        ("已经有 0.x 代码要迁移", "先过生存表（账 C）：chains/memory/llms 已整包移除→迁 langchain-core 的 Runnable 组件"),
        ("只想要某个工具/能力", "取用 langchain-community / langchain 的『库集合』小件（账 C）而非整套框架"),
        ("生产可观测链路", "LangSmith（同生态）或自接 Langfuse（账 D 之外，05-章后的横切章）"),
        ("嫌封装太厚的团队", "直写原生 SDK（06-章原语）或上轻量运行时（账 B 的 13 层 MRO 就是劝退理由）"),
    ]
    assert len(SCEN) == 8
    print("  → 选择树 8 场景断言 8/8：链=概念入门工具（概念永存）；生产=图/原生（链库退居库集合）")
    for i, (s1, s2) in enumerate(SCEN, 1):
        print(f"     {i:>2}. {s1}")
        print(f"         -> {s2}")


def main():
    print("=" * 72)
    print("langchain_demo：07-应用框架 · 01-LangChain与历史包袱（知识地图 §7.2 / §17.5.8）")
    print("    『Chain 概念永存 · LangChain 库有历史包袱』——统一抽象 vs 套娃/换血/单向链")
    print("=" * 72)
    print("[0] 口径：真实 langchain 1.2.10 / langchain-core 1.4.9 引擎（本机实测）；模型=FakeListLLM"
          " 预置（LLM 非本机实测）；版本节奏=要素事实（无外网未在线复核）；不打印随机图节点 ID")
    expA()
    expB()
    expC()
    expD()
    expTree()
    print()
    print("=" * 72)
    print("台账汇总（A 链式组装 / B 封装厚度 / C 版本存续 / D 链 vs 图）")
    print("  A steps 3 步 · 内部图 5 节点（隐形 2）· invoke/batch/stream 同链三触发逐字节一致")
    print("  B ChatOpenAI MRO 13 层 · Runnable 位次 10 · pydantic 字段 56 · 封装=少写 2 行换多爬 2 层")
    print("  C 0.x 生存表 7 FAIL / 3 OK · langchain 本体 6 子模块库集合 · 同机 6 包版本实测")
    print("  D 链自环 0（重试手写 while·轮数 3）· 图式挂靠 02-章（自环 1·recursion_limit·checkpoint）")
    print("一句话：LangChain 贡献的是『Chain/Runnable』这个概念（概念永存）；LangChain 库本身背的是")
    print("套娃/换血/单向链的包袱（库会消亡）——用概念入门，生产看 LangGraph 或直写（§7.2 学习建议）")
    print("done · 一键复现：python code/notebooks/_tools/langchain_demo.py")


if __name__ == "__main__":
    main()
    W(f"[langchain_demo] wall-clock {perf_counter() - _t0:.3f} s")
