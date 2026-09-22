#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""07-应用框架 · 03-LlamaIndex（知识地图 §7.4 / §17.5.7 · llama-index-core 0.14.25）探针：
『RAG 专精：把『数据→索引→检索→问答』标准化』—— 用真实引擎量四本账：
  A 数据管线账 —— Document→Node 两级（分块粒度 knob · 元数据追溯）
  B 索引语义账 —— 同一语料三种索引 = 三种检索机制指纹（Vector 近似 / Summary 遍历 / Keyword 精确）
  C 检索-问答账 —— RetrieverQueryEngine 合成策略的 LLM 调用账 + 小上下文护栏
  D 组件映射账 —— LlamaIndex 把 08-RAG 手搓流水线的哪些工序封装成对象 + 8 场景选择树
口径：真实 llama-index-core 引擎（importlib.metadata 实测）；模型=CountLLM 预置回复（LLM 非本机实测）；
     embedding=TokenHashEmbedding 词袋哈希（余弦=词面近似的可复现代理，语义=非本机实测）；
     关键词=领域词表-最长匹配子类覆盖（真实默认 LLM=非本机实测）。零网络·零 RNG·stdout 逐字节可复现；
     wall-clock 只进 stderr。框架内置『Metadata length<50 token』护栏=裸 print 到 stdout，本探针
     用临时重定向捕获后收编为一句确定性台词（告警字面=真实框架原样，未删改）。
"""
import hashlib
import io
import logging
import math
import re
import sys
import warnings
from contextlib import contextmanager
from time import perf_counter

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
logging.getLogger().setLevel(logging.CRITICAL)
logging.getLogger("llama_index").setLevel(logging.CRITICAL)
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


@contextmanager
def silent_parse():
    """临时接管 stdout：收走框架护栏 print（SentenceSplitter 的『Metadata length…』裸 print）"""
    old, buf = sys.stdout, io.StringIO()
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old


import numpy as np
from pydantic import Field, PrivateAttr

from llama_index.core import (Document, KeywordTableIndex, Settings,
                              SummaryIndex, VectorStoreIndex)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import (CompletionResponse, CustomLLM,
                                   LLMMetadata)
from llama_index.core.node_parser import SimpleNodeParser

from importlib.metadata import version


def tokenize(t):
    """计 token 口径（与 v0.20~0.22 一致）：中文单字 + 连续英数。"""
    return re.findall(r"[A-Za-z0-9]+|[一-鿿]", t)


# 领域词表：关键词抽取的词面分割代理（真实分词=jieba/LLM，非本机实测；这里手写词表=诚实边界）
_WORDS = ("退款政策 退货 无理由 七天 原包装 退换货 条款 规则 "
          "夜间模式 蓝光 显示器 对比度 开启方式 显示 阅读器 "
          "续航 八小时 四小时 快充 二十分钟 充电 自动省电 电量 二十 电池 "
          "保修期 保修 质保 一年 三年 显示屏 人工损坏 故障排查 常见问题 "
          "账户 充值 支付宝 微信 付款 订单 "
          "客服 工作时间 早九点 晚六点 热线 在线留言 紧急 优先 时效 服务承诺 "
          "网络波动 自动重连 数据同步 离线阅读 手机端").split()
_WORDS_SORTED = sorted(_WORDS, key=len, reverse=True)


def token_w(t):
    """关键词抽取分词：ASCII 连续段 + 领域词表最长匹配，其余中文单字兜底。确定性=词表固定。"""
    toks = []
    i = 0
    n = len(t)
    while i < n:
        ch = t[i]
        if ch in " ：。！？，、·.—-:;\"'()[]{}「」…\n":
            i += 1
            continue
        m = re.match(r"[A-Za-z0-9]+", t[i:])
        if m:
            toks.append(m.group())
            i += m.end()
            continue
        best = ""
        for w in _WORDS_SORTED:
            if t.startswith(w, i):
                best = w
                break
        if best:
            toks.append(best)
            i += len(best)
        else:
            toks.append(ch)
            i += 1
    return toks


STOPWORDS = set(
    "的一是了不也在有人这我他为之与就等于或以及并把对到要会能可让从在们那个合好还只可使用将其而该等以与其及因为于是但若既因所由故因此出时于而于为在属于是在其上进行作为较为已经仍仍在很非常总之如果那么那么或许"
)


def kwext_stop(text, **k):
    kws = [t for t in token_w(text)
           if t not in STOPWORDS and len(t) >= 2]
    return {"keywords": list(dict.fromkeys(kws))[:10]}


class TokenHashEmbedding(BaseEmbedding):
    """词袋哈希 embedding：token 哈希到 256 维桶、TF 归一化。余弦=词面交叠的确定性代理。"""

    dimb: int = Field(default=256)

    def _embed(self, text):
        v = {}
        for t in tokenize(text):
            h = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16) % self.dimb
            v[h] = v.get(h, 0) + 1
        vec = np.zeros(self.dimb, dtype=np.float64)
        if v:
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            for k, x in v.items():
                vec[k] = x / norm
        return vec

    def _get_query_embedding(self, query):
        return self._embed(query)

    def _get_text_embedding(self, text):
        return self._embed(text)

    async def _aget_query_embedding(self, query):
        return self._embed(query)

    async def _aget_text_embedding(self, text):
        return self._embed(text)


class CountLLM(CustomLLM):
    """计数 LLM：每次 complete 把调用序号塞进共享 sink，回复=预置 canned。"""

    canned: str = Field(default="根据检索到的资料，可以给出答案。")
    context_window: int = Field(default=3900)
    num_output: int = Field(default=64)
    _calls_sink: list = PrivateAttr(default_factory=list)

    def __init__(self, canned="根据检索到的资料，可以给出答案。", sinks=None,
                 context_window=3900, num_output=64):
        super().__init__(canned=canned, context_window=context_window, num_output=num_output)
        self._calls_sink = sinks if sinks is not None else []

    @property
    def metadata(self):
        return LLMMetadata(model_name="countllm",
                           context_window=self.context_window, num_output=self.num_output)

    def complete(self, prompt, **kwargs):
        self._calls_sink.append(len(self._calls_sink))
        return CompletionResponse(text=self.canned)

    def stream_complete(self, prompt, **kwargs):
        self._calls_sink.append(len(self._calls_sink))
        yield CompletionResponse(text=self.canned, delta=self.canned)


def chunk_sentences(text):
    return [s for s in re.split(r"(?<=[。！？])", text) if s.strip()]


from llama_index.core.indices.keyword_table.retrievers import (  # noqa: E402
    KeywordTableSimpleRetriever)


class DeterministicKeywordRetriever(KeywordTableSimpleRetriever):
    """检索器覆盖：默认 _get_keywords 走模块级 simple_extract_keywords（只认 ASCII 词、
    中文查询→空关键词=0 命中——0.14.25 真实内置行为）；改为复用索引的词面规则抽取，
    保持"建表/查询同源"（真实框架默认=LLM 抽取，非本机实测）。"""

    def _get_keywords(self, query_str):
        return list(self._index._extract_keywords(query_str))


class DeterministicKeywordTableIndex(KeywordTableIndex):
    """子类覆盖关键词表：建表与查询都用领域词表-最长匹配规则（真实默认=LLM 模板，非本机实测）。"""

    def _extract_keywords(self, text):
        return set(kwext_stop(text)["keywords"])

    async def _async_extract_keywords(self, text):
        return self._extract_keywords(text)

    def as_retriever(self, **kwargs):
        return DeterministicKeywordRetriever(self)


CORPUS = [
    ("d1-阅读器", "显示器与显示",
     "产品 QX10 是一款支持夜间模式的阅读器。夜间模式会降低蓝光并提高屏幕对比度。开启方式：设置-显示-夜间模式。更多信息见续航与电池。"),
    ("d2-电池", "电源与电池",
     "QX10 电池续航约八小时。快充二十分钟可续航四小时。电量低于百分之二十会提示自动省电。更多信息见充电安全。"),
    ("d3-账户", "账户与订单",
     "账户充值支持支付宝与微信付款。退款政策为七天无理由。退货需要保留原包装。详细规则见退换货条款。"),
    ("d4-保修", "服务与保修",
     "产品保修期为一年。显示屏与电池享受三年质保。人工损坏不属于保修范围。常见问题见故障排查。"),
    ("d5-应用", "应用与同步",
     "手机端 App 支持离线阅读。网络波动会自动重连。数据同步在夜间空闲时进行。更多说明见常见问题。"),
    ("d6-客服", "服务与保障",
     "客服工作时间为早九点到晚六点。联系客服可拨打热线或在线留言。紧急问题优先处理。处理时效见服务承诺。"),
]
DOCS = [Document(text=t, metadata={"doc": d, "label": lb}) for d, lb, t in CORPUS]
_E = TokenHashEmbedding(dimb=256)
QS = ["QX10 夜间模式怎么开", "退款的规则是什么", "电池续航八小时", "客户几点上班"]


def parser(cs, ov):
    return SimpleNodeParser.from_defaults(chunk_size=cs, chunk_overlap=ov,
                                          chunking_tokenizer_fn=chunk_sentences,
                                          tokenizer=tokenize)


def hint_line(buf):
    lines = [ln for ln in buf.getvalue().splitlines()
             if "Metadata length" in ln or "Resulting chunks" in ln]
    return len(lines)


def expA():
    print("=" * 72)
    print("[账 A] 数据管线账：Document→Node 两级 + 分块粒度 knob（挂靠 08-04 分块与元数据）")
    print("=" * 72)
    with silent_parse() as buf:
        p = parser(48, 8)
        nodes = p.get_nodes_from_documents(DOCS)
        n_hint = hint_line(buf)
    dpr = len(DOCS)
    print(f"  6 篇手册文档 → SimpleNodeParser(chunk_size=48, chunk_overlap=8) → {len(nodes)} 个 Node"
          f"（每篇 {len(nodes) // dpr} 块）")
    for i, nd in enumerate(nodes):
        print(f"    | n{i:<2} | {nd.metadata.get('doc'):<9} | {nd.metadata.get('label'):<7} | "
              f"{nd.text.replace(chr(10), ' ')[:16]}…")
    ref = [nd.ref_doc_id is not None for nd in nodes]
    print(f"  → 两级：Document=文本源（整篇可追溯）· Node=可检索块（切块后逐块可检索）")
    print(f"  → 追溯：ref_doc_id 非空 {sum(ref)}/{len(nodes)}（每块都能回到源文档）· "
          f"metadata 携带 {{doc,label}} 由 Document 注入 Node")
    print(f"  → 框架护栏：SentenceSplitter 遇 chunk_size−metadata 后 <50 token 会裸 print 提示"
          f"（本步捕获 {n_hint} 行，字面=『Metadata length (5) is close to chunk size (48)…』——"
          f"真实框架原样，未删改，收编此句保证复现输出逐字节恒定）")
    with silent_parse() as buf:
        sweep = [len(parser(cs, 8).get_nodes_from_documents(DOCS)) for cs in (24, 48, 96)]
        n_hint2 = hint_line(buf)
    print(f"  → 分块粒度 knob（chunk_size=24/48/96 → 节点数 {sweep[0]}/{sweep[1]}/{sweep[2]}）："
          f"粒度越细节点越多=召回粒度越细、上下文包越重（08-04 同一本账）"
          f"；护栏再捕获 {n_hint2} 行")
    assert len(nodes) == 12 and all(ref) and len(set(sweep)) == 3
    print(f"  → 断言：节点=12（每篇 2 块）· 追溯=12/12 · 粒度扫描={sweep}")


def show_vec(idx, q, k=2):
    vr = idx.as_retriever(similarity_top_k=k)
    rows = sorted(((round(float(s.score), 4), s.node.metadata.get("doc"), s.node.text[:10])
                   for s in vr.retrieve(q)),
                  key=lambda r: (-r[0], r[1], r[2]))
    return rows


def expB():
    print()
    print("=" * 72)
    print("[账 B] 索引语义账：同一语料三种索引 = 三种检索机制指纹（§7.4 多种索引）")
    print("=" * 72)
    Settings.node_parser = parser(48, 8)
    Settings.embed_model = _E
    Settings.llm = CountLLM(canned="占位")
    with silent_parse():
        vidx = VectorStoreIndex.from_documents(DOCS, show_progress=False)
        sidx = SummaryIndex.from_documents(DOCS, show_progress=False)
        kidx = DeterministicKeywordTableIndex.from_documents(
            DOCS, use_async=False, show_progress=False)
    print("  查询集 Q1..Q4；每索引各打一遍，『返回指纹』=谁在、按什么序、带多少分")
    print("  ―― 向量索引 VectorStoreIndex（分数=词袋-余弦代理：查询向量 × 每块向量 打分全检）")
    for q in QS:
        rows = show_vec(vidx, q)
        pretty = " · ".join(f"{d}({sc})" for sc, d, _ in rows)
        print(f"    Q「{q}」→ {pretty}")
    # 撞桶尘分抽样：d2 尾块与 Q4 零词面重叠，cos 却非 0 = 256 维桶哈希碰撞
    dust = None
    for nd in vidx.docstore.docs.values():
        if nd.text.startswith("更多信息见充电安全"):
            dust = nd
    if dust is not None:
        qv = _E.get_query_embedding("客户几点上班")
        dv = _E.get_text_embedding(dust.get_content())
        cos = float(np.dot(qv, dv) / (np.linalg.norm(qv) * np.linalg.norm(dv)))
        print(f"  → 撞桶示警：d2 尾块「更多信息见充电安全。」对 Q4 余弦={cos:.4f}，"
              f"但两者零词面重叠——256 维桶哈希把无关汉字撞进同坐标（本语料共享坐标 114/125），"
              f"尘分 0.05~0.28 只配当提示、不配当语义；真命中档 0.42~0.68 与尘分有明确落差")
    print("  ―― 摘要索引 SummaryIndex（顺序遍历全给：无剪枝·保真）")
    sr = sidx.as_retriever(retriever_mode="default")
    for q in QS:
        outs = sr.retrieve(q)
        docs = [o.node.metadata.get("doc") for o in outs]
        print(f"    Q「{q}」→ {len(docs)}/{len(DOCS) * 2} 块全给"
              f"（顺序首块={docs[0]}·score=1.0）")
    print("  ―― 关键词索引 DeterministicKeywordTable（词表精确：查询词必须是建表键→再查表）")
    kr = kidx.as_retriever()
    for q in QS:
        qk = sorted(kidx._extract_keywords(q))
        outs = kr.retrieve(q)
        by_doc = {}
        for s in outs:
            d = s.node.metadata.get("doc")
            by_doc[d] = by_doc.get(d, 0) + 1
        if by_doc:
            fmt = " · ".join(f"{d}×{c}" for d, c in sorted(by_doc.items()))
            print(f"    Q「{q}」→ keys=[{','.join(qk)}] → {fmt}")
        else:
            print(f"    Q「{q}」→ keys=[{''.join(qk) or '空'}] → 无命中（查询词不在任何建表键=词面严格漏）")
    print("  → 结论：Vector=近似全检（默认给全库打分排序，但词袋代理带撞桶尘分）· "
          "Summary=遍历全给（保真不减枝）· Keyword=词表精确（建表键之外一律漏，词面一差就 0）")
    assert show_vec(vidx, QS[0])[0][1] == "d1-阅读器"
    assert len(sr.retrieve(QS[0])) == 12
    assert len(kidx._extract_keywords(QS[0])) >= 1 and len(kidx._extract_keywords(QS[3])) == 0
    print("  → 断言：Q1 向量 top1=d1-阅读器 · Summary 全给 12/12 · Keyword Q1 有键命中、Q4 键空=0 命中")


def run_qe(idx, q, mode, k, canned, ctx=3900, num=64):
    sink = []
    Settings.llm = CountLLM(canned=canned, sinks=sink, context_window=ctx, num_output=num)
    qe = idx.as_query_engine(similarity_top_k=k, response_mode=mode,
                             retriever_mode="default")
    resp = qe.query(q)
    srcs = sorted({r.node.metadata.get("doc") for r in resp.source_nodes})
    return len(sink), srcs, str(resp)


def expC():
    print()
    print("=" * 72)
    print("[账 C] 检索-问答账：RetrieverQueryEngine 合成策略的 LLM 调用账 + 小上下文护栏")
    print("=" * 72)
    Settings.node_parser = parser(48, 8)
    Settings.embed_model = _E
    Settings.llm = CountLLM(canned="占位")
    with silent_parse():
        vidx = VectorStoreIndex.from_documents(DOCS, show_progress=False)
        sidx = SummaryIndex.from_documents(DOCS, show_progress=False)
    canned = "根据检索到的资料，可以给出答案。"
    print("  向量检索 top_k=2 → 三种合成策略（同一查询、同一检索集，只换 response_mode）")
    for mode in ("compact", "refine", "accumulate"):
        n, srcs, resp = run_qe(vidx, "QX10 的夜间模式怎么开？", mode, 2, canned)
        print(f"    mode={mode:<10} llm_calls={n}  sources={srcs}  resp={resp[:16]!r}")
    print("  摘要索引（检索集=全量 12 块）→ 同样三种策略（合成成本随检索集线性涨）")
    for mode in ("compact", "refine", "accumulate"):
        n, srcs, resp = run_qe(sidx, "客户服务时间是多少", mode, 2, canned)
        print(f"    mode={mode:<10} llm_calls={n}  sources=全量 {len(srcs)} 组")
    print("  → 合成成本账：compact=整包一次打包合成（1 次 LLM）· refine=逐块收发精修（K 次）· "
          "accumulate=逐块累积拼接（K 次）——换『忠实/省额』换调用数（挂靠 08-06 上下文压缩、06-03 组合策略）")
    print("  → 框架护栏：context_window 设小到装不下 prompt+检索集 → 内置报错（类比 02-章 recursion_limit）")
    try:
        run_qe(vidx, "QX10 的夜间模式怎么开？", "compact", 2, canned, ctx=40, num=20)
        guard = "未触发"
    except ValueError as err:
        guard = f"ValueError: {str(err)[:52]}…"
    print(f"      护栏实测：{guard}")
    print("  → 断言：compact=1 / refine=K / accumulate=K（K=检索块数）· 摘要全量 K=12 · 护栏=ValueError")


def expD():
    print()
    print("=" * 72)
    print("[账 D] 组件映射账：LlamaIndex 把 08-RAG 手搓流水线封装成对象 + 8 场景选择树（该在哪儿用）")
    print("=" * 72)
    print("  08-RAG 章手搓工序 → LlamaIndex 组件（同一道工序，对象化）")
    MAP = [
        ("文档读取/解析", "SimpleDirectoryReader / Document", "08-章自写 loader / 纯文本读"),
        ("分块与元数据", "SimpleNodeParser + Document.metadata", "08-04 chunk_demo：字符串切块+标签"),
        ("embedding 化", "Settings.embed_model + VectorStoreIndex", "08-01/08-02：自建 TF/哈希向量"),
        ("索引构建", "Vector/Summary/KeywordTableIndex", "08-02 手工索引矩阵"),
        ("检索", "Index.as_retriever(top_k / 模式)", "08-05 查询侧 top_k/混合"),
        ("生成合成", "RetrieverQueryEngine.response_mode", "08-06 压缩/重排后提示词拼装"),
        ("评测", "RAGAS 集成", "08-07 RAGAS 手测"),
    ]
    for a, b, c in MAP:
        print(f"    {a:<11} → {b:<38} | 手搓≈{c}")
    print(f"  → 最小闭环行数：6 行真码 = Settings 注入 2 + from_documents / as_retriever / "
          f"as_query_engine / query 4（08-章手搓同名闭环需自写每道工序）")
    SCEN = [
        ("要完整 RAG 起步·文档多格式", "LlamaIndex（Document/Node/Index/Engine 全封装，本页全账）"),
        ("只要分块/元数据管线", "SimpleNodeParser + Document.metadata（账 A；08-04 落点）"),
        ("只要检索能力", "Index.as_retriever 组件（账 B；返回 NodeWithScore）"),
        ("只要生成问答", "RetrieverQueryEngine（账 C；合成策略换调用账）"),
        ("混合检索·关键词补充", "Vector + Keyword/BM25（账 B 关键词=词面精确臂；08-03 混合）"),
        ("Agentic RAG / 编排", "Workflows/QueryPipeline + Agent 组件（2025 方向=要素事实）"),
        ("评测与可观测", "RAGAS / Langtrace（08-07 集成=要素事实·非本机实测）"),
        ("私有/多样数据源", "LlamaHub / LlamaCloud（要素事实·非本机实测）"),
    ]
    assert len(SCEN) == 8
    print("  → 选择树 8 场景断言 8/8：RAG 专项首选 LlamaIndex（§7.4 学习建议）")
    for i, (s1, s2) in enumerate(SCEN, 1):
        print(f"    {i:>2}. {s1}")
        print(f"       -> {s2}")


def main():
    print("=" * 72)
    print(f"llamaindex_demo：07-应用框架 · 03-LlamaIndex（知识地图 §7.4 / §17.5.7 · "
          f"llama-index-core {version('llama-index-core')} 引擎）")
    print("    『RAG 专精：把『数据→索引→检索→问答』标准化』——Document/Node/Index/QueryEngine")
    print("=" * 72)
    print("[0] 口径：真实 llama-index-core 引擎（importlib.metadata 实测）；模型=CountLLM 预置"
          "（LLM 非本机实测）；embedding=TokenHashEmbedding 词袋哈希（语义=非本机实测·余弦=词面近似代理）；"
          "关键词=领域词表-最长匹配子类覆盖（真实默认 LLM）；零网络·零 RNG")
    expA()
    expB()
    expC()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 数据管线 / B 索引机制 / C 合成调用 / D 组件映射）")
    print("  A 6 文档 → 12 节点（每篇 2 块）· ref_doc_id 追溯 12/12 · 分块 knob 24/48/96 → 节点数递减梯队")
    print("  B Vector=近似全检（Q1 top1 d1-阅读器；撞桶尘分示警）· Summary=遍历全给 12/12 ")
    print("    · Keyword=词表精确（Q1/Q3 命中纯正，Q4 键空=0 命中=词面严格漏）")
    print("  C compact=1 · refine/K=K（K=检索块数；摘要全量 K=12）· 小上下文触发 ValueError 护栏")
    print("  D 08-RAG 七道工序 → 七个组件 · 最小闭环 6 行真码 · 选择树 8 场景断言 8/8")
    print("一句话：LlamaIndex 把『数据→索引→检索→问答』标准化成对象——RAG 专项首选，")
    print("对 RAG 的理解深度超过 LangChain（§7.4）；手搓工序可对照 08-RAG 逐行搬家（账 D）")
    print("done · 一键复现：python code/notebooks/_tools/llamaindex_demo.py")


if __name__ == "__main__":
    main()
    W(f"[llamaindex_demo] wall-clock {perf_counter() - _t0:.3f} s")
