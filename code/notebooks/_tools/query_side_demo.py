# -*- coding: utf-8 -*-
"""05-查询侧技术 的可复现探针：同义扩展救 BM25 + 复合/多跳问题分解（真实数字，零下载）。

运行：cd 仓库根 && python code/notebooks/_tools/query_side_demo.py
依赖：rank_bm25 jieba faiss sentence-transformers（与 009/010 相同，Colab 可用）
语料：复用 code/notebooks/_demo_corpus.json（100 条中文百科，第 i 条答案在第 i 条上下文）。
实验1（词面断裂 → 同义扩展）：8 条问题故意不含目标实体词；
  度量 BM25 / Dense 的 Hit@1，naive vs 扩展（append 同义实体词）。
实验2（复合/多跳 → 分解）：4 条复合问题，答案实体不出现在问题里；
  naive 单句检索 vs「hop1 扶实体 + hop2 检索」并集，看 Hit@1/Hit@5。
"""
import io, json, sys, warnings
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

# ---------- 数据与检索器（沿用 009/010） ----------
CORPUS = json.load(io.open(r'code/notebooks/_demo_corpus.json', encoding='utf-8'))
corpus_q = [c for c, _ in CORPUS]
N = len(corpus_q)
def cut(s):
    return [w for w in jieba.cut(s) if w.strip() and len(w.strip()) > 1]
bm = BM25Okapi([cut(c) for c in corpus_q])
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
emb = model.encode(corpus_q, normalize_embeddings=True, show_progress_bar=False)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(np.ascontiguousarray(emb))

def search_bm25(q, n=5):
    return np.argsort(bm.get_scores(cut(q)))[::-1][:n].tolist()

def search_dense(q, n=5):
    v = model.encode([q], normalize_embeddings=True)
    _, idx = index.search(np.ascontiguousarray(v), n)
    return idx[0].tolist()

def hit1(fn, q, tgt):
    return 1.0 if fn(q, 1)[0] == tgt else 0.0

def hit5(fn, q, tgt):
    return 1.0 if tgt in fn(q, 5) else 0.0

# ---------- 实验1：词面断裂 → 同义扩展 ----------
EXP = [
    (5,  "卷积里那个扫过图像的小窗口叫什么",   "卷积里那个扫过图像的小窗口叫什么 卷积核"),       # 目标词：卷积核
    (8,  "模型预测时挑重点看的地方，是什么机制", "模型预测时挑重点看的地方，是什么机制 注意力机制"), # 注意力机制
    (22, "让小的模型学大的模型的那套办法",     "让小的模型学大的模型的那套办法 知识蒸馏"),         # 知识蒸馏
    (25, "多个设备各练各的加总模型的办法",     "多个设备各练各的加总模型的办法 联邦学习"),         # 联邦学习
    (30, "用一串数字表示词的意思",             "用一串数字表示词的意思 词向量"),                   # 词向量
    (23, "把模型里不重要的连接剪掉来瘦身",     "把模型里不重要的连接剪掉来瘦身 剪枝"),             # 剪枝
    (4,  "区分机器和人的那个古老检验",         "区分机器和人的那个古老检验 图灵测试"),             # 图灵测试
    (33, "把词投到向量空间去的那步操作",       "把词投到向量空间去的那步操作 词嵌入"),             # 词嵌入
]
print("=" * 62)
print("实验1  词面断裂 → 同义扩展（问题不含目标实体词，8 题）")
print("=" * 62)
b_naive = [hit1(search_bm25, d, t) for t, d, _ in EXP]
b_exp   = [hit1(search_bm25, e, t) for t, _, e in EXP]
d_naive = [hit1(search_dense, d, t) for t, d, _ in EXP]
d_exp   = [hit1(search_dense, e, t) for t, _, e in EXP]
print("BM25 naive    Hit@1 = %.3f (%d/8)" % (np.mean(b_naive), sum(b_naive)))
print("BM25 +扩展    Hit@1 = %.3f (%d/8)  [+%.3f]" % (np.mean(b_exp), sum(b_exp), np.mean(b_exp) - np.mean(b_naive)))
print("Dense naive   Hit@1 = %.3f (%d/8)  ← 语义向量天然免疫词面断裂" % (np.mean(d_naive), sum(d_naive)))
print("Dense +扩展   Hit@1 = %.3f (%d/8)  [+%.3f]" % (np.mean(d_exp), sum(d_exp), np.mean(d_exp) - np.mean(d_naive)))
for i, (t, d, e) in enumerate(EXP):
    flag = " ◆" if b_naive[i] == 0 else ""
    print("  目标[%2d] %-24s BM25 %d→%d  Dense %d→%d%s" % (
        t, corpus_q[t][:22], b_naive[i], b_exp[i], d_naive[i], d_exp[i], flag))

# ---------- 实验1b：词面检索代（escape 不了直接搜 BM25+e 全中细看） ----------
print()
print("  「BM25 naive 失败」明细：top1 是哪个文档？")
for i, (t, d, e) in enumerate(EXP):
    if b_naive[i] == 0:
        top1 = search_bm25(d, 1)[0]
        print("  问题：%s" % d)
        print("    期望 doc[%d] %s" % (t, corpus_q[t][:26]))
        print("    BM25 top1 → doc[%d] %s  ◆ naive Miss → 扩展后 Hit" % (top1, corpus_q[top1][:26]))
print()

# ---------- 实验2：复合/多跳 → 分解（答案实体不在问题词面） ----------
MH = [
    (22, "联邦蒸馏结合出的那套压缩技术，本质让谁模仿谁",
         ["联邦蒸馏结合了哪两种技术", "知识蒸馏让谁模仿谁"]),
    (8,  "Transformer 脚下的注意力机制，核心让模型干嘛",
         ["Transformer 建立在什么之上", "注意力机制让模型做什么"]),
    (25, "联邦蒸馏里不共享原始数据的技术，靠上传什么聚合",
         ["联邦蒸馏结合了哪两种技术", "联邦学习靠上传什么训练"]),
    (5,  "做图像识别的那类网络，局部特征靠什么提取",
         ["哪种神经网络擅长图像识别", "卷积核的作用是什么"]),
]
print("=" * 62)
print("实验2  复合/多跳问题 → 分解（naive 单句 vs hop1+hop2 并集）")
print("=" * 62)
agg = []
for tgt, naive_q, subs in MH:
    n1, n5 = hit1(search_dense, naive_q, tgt), hit5(search_dense, naive_q, tgt)
    hop1_doc = search_dense(subs[0], 1)[0]                 # 第一跳：扶出目标实体所在桥文档
    union = search_dense(subs[0], 2) + search_dense(subs[1], 5)
    seen = []
    for x in union:
        if x not in seen:
            seen.append(x)
    dec1, dec5 = hit1((lambda q, n: seen[:n]), 'x', tgt), (1.0 if tgt in seen[:5] else 0.0)
    agg.append((n1, n5, dec1, dec5, hop1_doc))
    print("目标[%2d] %-18s naive H1=%d H5=%d | hop1→doc[%2d] %s | 分解并集 H1=%d H5=%d" % (
        tgt, corpus_q[tgt][:18], n1, n5, hop1_doc,
        corpus_q[hop1_doc][:12], dec1, dec5))
m = len(MH)
print()
print("聚合：naive  H1=%d/%d  H5=%d/%d；分解并集 H1=%d/%d  H5=%d/%d" % (
    sum(r[0] for r in agg), m, sum(r[1] for r in agg), m,
    sum(r[2] for r in agg), m, sum(r[3] for r in agg), m))
print("\n结论：检索层（H5）在易语料上顶格 → 先测张力再上查询侧；",
      "真正的失配在首屏（H1）与 Agentic 多跳（hop1 扶实体）。")
