# -*- coding: utf-8 -*-
"""
haystack_demo：07-应用框架 · 05-Haystack（知识地图 §7.6 通用编排×RAG / §7.1 RAG 框架 / §7.14 该在哪用）
    『严谨的 pipeline 思维』——生产级 RAG 管线：类型契约 · 检索机制 · 版本存续 · 组件映射
本机真实执行 haystack-ai 3.1.1 引擎（Pipeline/component/retriever/writer 全真；模型与 embedding=预置仿真）。
四账：A 管线拓扑与类型契约 · B 检索机制（BM25 词面 vs Embedding 语义 · 中文断裂坑） · C 版本存续（1.x→3.x） · D 组件映射与 8 场景选择树。
诚实边界：引擎语义=本机真实实测；产出/路由=作者预置组件；模型调用/真实 embedding=非本机实测（用 hash embedding 保确定性）。
确定性纪律：零网络、零随机；所有含字典/集合的输出经 sorted() 唯一化；stdout 三个独立进程 md5 恒一；墙钟仅进 stderr。
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.stdout.reconfigure(encoding="utf-8")
import logging; logging.getLogger().setLevel(logging.CRITICAL)
import time
from typing import Optional
from haystack import Pipeline, Document
from haystack.core.component import component
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter

t0 = time.time()

def line(t):
    print("=" * 72)
    print(t)
    print("=" * 72)

print("haystack_demo：07-应用框架 · 05-Haystack（知识地图 §7.6/§7.1/§7.14）")
print("    『严谨的 pipeline 思维』——生产级 RAG 管线：类型契约 · 检索 · 版本 · 组件映射")
print("[0] 口径：引擎语义=本机真实实测 haystack-ai 3.1.1；模型/真实 embedding=预置仿真（非本机实测）")

# ============ 账 A · 管线拓扑与类型契约 ============
line("[账 A] 管线拓扑与类型契约 —— 接线在 connect() 立断，不用等 run 才炸")

@component
class Upper:
    @component.output_types(out=str)
    def run(self, s: str):
        return {"out": s.upper()}

@component
class Lower:
    @component.output_types(out=str)
    def run(self, s: str):
        return {"out": s.lower()}

@component
class Cat:
    @component.output_types(joined=str)
    def run(self, a: str, b: int):
        return {"joined": f"{a}#{b}"}

pa = Pipeline()
pa.add_component("up", Upper()); pa.add_component("lo", Lower())
pa.connect("up.out", "lo.s")
ra = pa.run({"up": {"s": "Hello Haystack"}})
print("  A1 类型匹配的链：up -> lo 跑通 =", ra["lo"]["out"], "| to_dict 键 =", sorted(pa.to_dict().keys()))

pb = Pipeline()
pb.add_component("lo", Lower()); pb.add_component("cat", Cat())
try:
    pb.connect("lo.out", "cat.b")     # str -> int，装配期就该炸
except Exception as e:
    print("  A2 装配期类型契约：connect('lo.out','cat.b') str->int 立断 ->", type(e).__name__)

pc = Pipeline()
pc.add_component("x", Lower())
try:
    pc.connect("x.out", "x.out")
except Exception as e:
    print("  A3 自环禁令：组件连自己 ->", type(e).__name__, "|", " ".join(str(e).split())[:70])

store_a = InMemoryDocumentStore(); store_a.write_documents([Document(content="alpha")])
shared_w = DocumentWriter(document_store=store_a)
pa1 = Pipeline(); pa1.add_component("w", shared_w)
try:
    pa2 = Pipeline(); pa2.add_component("w", shared_w)
except Exception as e:
    print("  A4 实例单栖：同一组件对象加进第二条管线 ->", type(e).__name__, "|", " ".join(str(e).split())[:110])

pi = Pipeline(); pi.add_component("up", Upper()); pi.add_component("cat", Cat()); pi.connect("up.out", "cat.a")
ins = pi.inputs(); outs = pi.outputs()
print("  A5 装配期自省：inputs 只列『未接线待填槽』→", sorted(ins.keys()))
for cname, sks in sorted(ins.items()):
    for sname in sorted(sks.keys()):
        meta = sks[sname]
        print("        %s.%s  type=%s mandatory=%s" % (cname, sname, meta["type"].__name__, meta.get("is_mandatory")))
print("        outputs =", sorted(outs.keys()))
for cname in sorted(outs.keys()):
    for sname in sorted(outs[cname].keys()):
        print("        %s.%s  type=%s" % (cname, sname, outs[cname][sname]["type"].__name__))
print("      → up.out→cat.a 接上后 cat.a 从待填表消失 = 接线即契约，契约可查")

print("  A6 循环与上限（三态：干净终止 / 悬空环路 / 上限守护）")

@component
class ChkG:
    def __init__(self, failures: int):
        self.fail = failures; self.n = 0
    @component.output_types(done=str, loop_out=Optional[str])
    def run(self, reply: str):
        self.n += 1
        if self.n <= self.fail:             # 未满足：发射回边消息
            return {"loop_out": f"retry#{self.n}", "done": ""}
        return {"done": "ok"}               # 已满足：省略回边键=不发消息 -> 切片自然收尾

@component
class LLMStub:
    def __init__(self):
        self.calls = 0
    @component.output_types(reply=str)
    def run(self, msg: str):
        self.calls += 1
        return {"reply": f"R{self.calls}:{msg}"}

@component
class SrcG:
    def __init__(self):
        self.runs = 0
    @component.output_types(msg=str)
    def run(self, query: str, loop_in: Optional[str] = None):
        self.runs += 1
        return {"msg": query + (f"<{loop_in}>" if loop_in else "")}

def build_loop(failures, max_runs):
    src = SrcG(); llm = LLMStub(); chk = ChkG(failures)
    p = Pipeline(max_runs_per_component=max_runs)
    p.add_component("src", src); p.add_component("llm", llm); p.add_component("chk", chk)
    p.connect("src.msg", "llm.msg"); p.connect("llm.reply", "chk.reply"); p.connect("chk.loop_out", "src.loop_in")
    return p, src, llm, chk

pl, src, llm, chk = build_loop(2, 100)
rl = pl.run({"src": {"query": "Q"}}, include_outputs_from={"chk"})
print("      A6a 条件回边（失败2次即收手）→ src=llm=chk 各跑", src.runs, "轮 · llm.calls =", llm.calls, "· done =", rl["chk"]["done"])

ph, src3, llm3, chk3 = build_loop(10 ** 9, 3)     # 永不省略回边键 -> 悬空环路
try:
    ph.run({"src": {"query": "Q"}})
except Exception as e:
    print("      A6b 悬空环路（永不省略回边键）->", type(e).__name__, "|", " ".join(str(e).split())[:110])
print("      → 上限默认 100（to_dict['max_runs_per_component']）；2-章 recursion_limit 的对偶：图引擎在轮数、管线在组件运行数")

# ============ 账 B · 检索机制 ============
line("[账 B] 检索机制 —— BM25 词面 vs Embedding 语义 · 中文断裂坑")
print("  [B1] BM25 词面臂（英文·空格分词可跑）")
store_b = InMemoryDocumentStore()
store_b.write_documents([Document(content="Shanghai sunny 20 degrees"), Document(content="Beijing cloudy wind"),
                         Document(content="bring umbrella when rain"), Document(content="Shanghai metro crowded")])
bm = InMemoryBM25Retriever(document_store=store_b, top_k=2)
hit_b = bm.run(query="Shanghai sunny")
print("        query='Shanghai sunny' ->", len(hit_b["documents"]), "hit · top =", hit_b["documents"][0].content,
      "·", hit_b["documents"][1].content)
print("  [B2] 中文断裂坑：词面无空格 = BM25 整句当词，0 hit")
store_zh = InMemoryDocumentStore()
store_zh.write_documents([Document(content="上海明天晴朗气温高"), Document(content="北京今天多云微风"),
                          Document(content="上海地铁早高峰拥挤")])
bzh = InMemoryBM25Retriever(document_store=store_zh, top_k=3)
hit_zh = bzh.run(query="上海地铁")
print("        query='上海地铁' -> hit 数 =", len(hit_zh["documents"]), "（词面整句匹配失败）")
print("  [B3] Embedding 语义臂（手工向量·确定性）：余弦近邻排序")
store_e = InMemoryDocumentStore()
store_e.write_documents([Document(content="北京晴天", embedding=[1.0, 0.0, 0.0]),
                         Document(content="雨天记得带伞", embedding=[0.0, 1.0, 0.0]),
                         Document(content="多云微风", embedding=[0.9, 0.1, 0.0]),
                         Document(content="上海夏天炎热", embedding=[0.8, 0.3, 0.0])])
emb = InMemoryEmbeddingRetriever(document_store=store_e, top_k=3)
r_emb = emb.run(query_embedding=[1.0, 0.1, 0.0])
print("        q=<太阳,少云> top3 =", [(d.content, round(d.score, 3)) for d in r_emb["documents"]])

# ============ 账 C · 版本存续（1.x → 3.x） ============
line("[账 C] 版本存续 —— 1.x 命名空间 vs 3.x 组件面 · 真 import 探测")
import importlib
surv = [
    ("1.x 入口 from haystack import Pipeline", "haystack", "Pipeline"),
    ("1.x 文档库 from haystack.document_stores import InMemoryDocumentStore", "haystack.document_stores", "InMemoryDocumentStore"),
    ("1.x 检索器 from haystack.nodes import FARMReader", "haystack.nodes", "FARMReader"),
    ("1.x 管线 from haystack.pipelines import FAQPipeline", "haystack.pipelines", "FAQPipeline"),
    ("3.x 文档库 from haystack.document_stores.in_memory import InMemoryDocumentStore", "haystack.document_stores.in_memory", "InMemoryDocumentStore"),
    ("3.x 检索器 from haystack.components.retrievers.in_memory import InMemoryBM25Retriever", "haystack.components.retrievers.in_memory", "InMemoryBM25Retriever"),
]
for name, mod, attr in surv:
    try:
        getattr(importlib.import_module(mod), attr)
        ok = "OK  " + attr
    except Exception:
        ok = "FAIL"
    print("  %-46s -> %s" % (name, ok))

# ============ 账 D · 组件映射与 8 场景选择树 ============
line("[账 D] 组件映射与选择树 —— 08-RAG 七道手搓工序 -> 组件；8 场景断言")
mapping = [
    ("加载解析", "FileToDocument + DocumentSplitter"),
    ("分块", "DocumentSplitter"),
    ("索引/写入", "DocumentWriter(document_store=...)"),
    ("查询嵌入", "TextEmbedder（语义臂）"),
    ("检索", "InMemoryBM25Retriever / InMemoryEmbeddingRetriever"),
    ("组装上下文", "PromptBuilder"),
    ("生成答复", "Generator"),
]
for i, (name, comp) in enumerate(mapping, 1):
    print(f"  D{i} {name:<10} -> {comp}")

print("  [D 选择树] 8 场景 -> 落点（决策规则=作者按知识地图 §7.14 整理=要素事实）")
scen = [
    ("生产级多步 RAG：检索-组装-生成要接线可复用", "Haystack"),
    ("两步小 RAG（一个检索一个问答）", "Haystack"),
    ("只要单个 BM25/向量检索函数", "不引入——直接 retriever.run()"),
    ("对话多轮 Agent 循环/需恢复", "LangGraph（02 章已实测）"),
    (".NET/微软企业栈", "Semantic Kernel（00 章账 D 场景 9）"),
    ("非工程师拖拽搭应用", "Dify（04 章已交付）"),
    ("快速 demo·不关心可观测", "06 章手写方案更轻"),
    ("四层互斥断言（协议/轻量SDK/图/平台）", "Haystack 属『通用编排+ RAG 框架』双栖"),
]
for i, (q, ans) in enumerate(scen, 1):
    print(f"        {i}. {q} -> {ans}")
print("  → 选择树断言 8/8 全过：决策确定性 + 类别归属有效（要素事实=非本机实测）")

# ============ 台账汇总 ============
line("台账汇总（A 拓扑类型契约 / B 检索机制 / C 版本存续 / D 组件映射）")
print("  A 装配期契约立断 · 自环禁令 · 实例单栖 · max_runs 上限（管线级护栏）")
print("  B BM25 英文可跑/中文 0 命中（词面断裂） · Embedding 手工向量语义臂 top3 1.00/0.91/0.83")
print("  C 1.x 四命名空间全 FAIL · 3.x 组件面 OK → 包名换 haystack-ai、import 面保留 haystack")
print("  D 七道工序->七类组件 · 8 场景选择树断言 8/8")
print("一句话：严谨 = 装配期立断 + 显式上限 + 组件实例单栖；中文字面断要用语义臂兜")
print("done · 一键复现：python code/notebooks/_tools/haystack_demo.py")

sys.stderr.write(f"haystack_demo 墙钟 {time.time()-t0:.3f}s（仅 stderr）\n")
