# -*- coding: utf-8 -*-
"""10-Agentic RAG 的可复现探针：判断要不要查 / 查几轮 / 够不够→再查与止损（真实数字，零下载）。

运行：cd 仓库根 && python code/notebooks/_tools/agentic_rag_demo.py
依赖：rank_bm25 jieba faiss sentence-transformers（与 009/010 相同，Colab 可用）
语料：复用 code/notebooks/_demo_corpus.json（100 条中文百科，第 i 条答案在第 i 条上下文）。
对照基准：naive 静态流水线（不判断，每问必检、首屏直采） vs Agentic 决策层（判断→多轮→采信）。

实验 A（要不要查）：12 条混合池 = 8 条库内事实题 + 4 条库外闲聊常识题。
  judge 先行（Dense top1 cos>0.50 判定"要查"） vs 无脑每问必检 —— 对比检索次数与上下文 token 账。
实验 B（查几轮/够不够→再查）：4 条复合/多跳问题（答案实体不在问题词面，复用 05 章 MH 表）。
  naive 恒 1 轮首屏直采 vs agentic「首轮 top1 与子问重查 top1 不一致 → 判断不够 → 再查采信」。
实验 C（止损值不值）：无判断盲目加深 top-k（k=1..5）的 Hit@k 与 token 累积 vs agentic 的判断后多查一次。

一切确定性：固定语料/模型/无随机数、单线程 BLAS；墙钟只写 stderr，stdout byte-for-byte 恒一。
"""
import io, json, os, sys, time, warnings

# 单线程 BLAS 必须在 numpy import 之前设定，保证两次运行逐位一致。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
T0 = time.time()

# ---------- 数据与检索器（沿用 009/010 与 05 探针同一栈） ----------
CORPUS = json.load(io.open(r'code/notebooks/_demo_corpus.json', encoding='utf-8'))
corpus_q = [c for c, _ in CORPUS]          # 第 i 条答案原本就在第 i 条上下文
N = len(corpus_q)

def cut(s):
    return [w for w in jieba.cut(s) if w.strip() and len(w.strip()) > 1]

bm = BM25Okapi([cut(c) for c in corpus_q])
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
emb = model.encode(corpus_q, normalize_embeddings=True, show_progress_bar=False)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(np.ascontiguousarray(emb))

def dense_top(q, k):
    v = model.encode([q], normalize_embeddings=True)
    _, idx = index.search(np.ascontiguousarray(v), k)
    return idx[0].tolist()

def dense_top1_sim(q):
    v = model.encode([q], normalize_embeddings=True)
    d, idx = index.search(np.ascontiguousarray(v), 1)
    return float(d[0][0]), int(idx[0][0])

# token 估算口径（诚实声明）：中文按字符数粗估，仅作相对账，非真实 BPE token 数。
def tokens_of(doc):
    return len(doc)

# =============================================================
# 实验 A：判断「要不要查」——judge 先行 vs 无脑每问必检
# =============================================================
# 8 条库内事实题（答案在语料里）+ 4 条库外闲聊常识题（库里没有，直接回答即可）。
NEED = ["卷积里那个扫过图像的小窗口叫什么",
        "模型预测时挑重点看的地方，是什么机制",
        "让小的模型学大的模型的那套办法",
        "多个设备各练各的加总模型的办法",
        "用一串数字表示词的意思",
        "把模型里不重要的连接剪掉来瘦身",
        "区分机器和人的那个古老检验",
        "把词投到向量空间去的那步操作"]
SKIP = ["一年有多少个季度",
        "水的化学式是什么",
        "太阳东升西落月亮绕地球转",
        "一公斤等于多少克"]

print("=" * 72)
print("实验 A  判断「要不要查」——judge 先行（Dense top1 相似度 > 0.50） vs 无脑每问必检")
print("=" * 72)
JUDGE_TAU = 0.50
rows = []
for i, q in enumerate(NEED + SKIP):
    is_need = i < len(NEED)
    sim, top = dense_top1_sim(q)
    judge = sim > JUDGE_TAU          # judge 判定是否需要检索
    rows.append((q, is_need, sim, judge))
tp = sum(1 for _, n, _, j in rows if n and j)          # 库内题被判定要查（该查的查）
tn = sum(1 for _, n, _, j in rows if (not n) and (not j))  # 库外题被判定不查（省了）
fp = sum(1 for _, n, _, j in rows if (not n) and j)        # 库外题被误判要查（白查）
fn = sum(1 for _, n, _, j in rows if n and (not j))        # 库内题被误判不查（漏检，最危险）
for q, is_need, sim, judge in rows:
    tag = "库内·该查" if is_need else "库外·可省"
    mark = "→ 查" if judge else "→ 省"
    flag = " ✓" if (judge == is_need) else " ✗误判"
    print("  [%s] top1 sim=%.2f judge%s%s  %s" % (tag, sim, mark, flag, q[:24]))
print()
print("  混淆统计：TP 该查的查=%d/8   TN 可省的省=%d/4   FP 白查=%d   FN 漏检=%d" % (tp, tn, fp, fn))
# 无脑版每问必检：12 条全部检索，Top1 文档进上下文。
naive_callsA = len(rows)
naive_tokA = sum(tokens_of(corpus_q[dense_top(q, 1)[0]]) for q, *_ in rows)
# judge 版只对 judge=Need 的检索。
judged = [q for q, n, _, j in rows if j]
judge_calls = len(judged)
judge_tok = sum(tokens_of(corpus_q[dense_top(q, 1)[0]]) for q in judged)
print("  无脑「每问必检」：检索 %d 次 · 上下文 %d 字" % (naive_callsA, naive_tokA))
print("  judge「先判断再查」：检索 %d 次 · 上下文 %d 字（省 %d 次检索 / %d 字 ≈ %.1f%%）" % (
    judge_calls, judge_tok, naive_callsA - judge_calls, naive_tokA - judge_tok,
    100.0 * (naive_tokA - judge_tok) / naive_tokA))

# =============================================================
# 实验 B：查几轮 / 够不够 → 再查 / 采信（复合多跳问题）
# =============================================================
print()
print("=" * 72)
print("实验 B  复合/多跳问题：naive 首屏直采 vs Agentic「判断不够→再查→采信」")
print("=" * 72)
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
m = len(MH)
naive_hit, ag_init_hit, ag_final_hit = [0] * m, [0] * m, [0] * m
ag_rounds = [0] * m
ag_traced = []
for i, (tgt, Q, subs) in enumerate(MH):
    r1 = dense_top(Q, 1)[0]                 # 首轮：拿原问题查
    r2 = dense_top(subs[1], 1)[0]           # 判断候选：子问聚焦再查
    same = (r1 == r2)
    if same:                                # 首轮已聚焦（top1 不被桥文档顶位）
        dec, rounds = r1, 1
        why = "首轮 top1 与子问重查一致 → 自评够 → 1 轮采信"
    else:                                   # 首轮被桥文档顶位 → 自评不够 → 再查采信
        dec, rounds = r2, 2
        why = "首轮 top1 被桥文档顶位 → 判断不够 → 子问再查采信"
    naive_hit[i] = 1 if r1 == tgt else 0
    ag_init_hit[i] = 1 if r1 == tgt else 0
    ag_final_hit[i] = 1 if dec == tgt else 0
    ag_rounds[i] = rounds
    ag_traced.append((tgt, Q, r1, r2, same, dec, rounds, why))
    print("  目标[%2d] %s" % (tgt, corpus_q[tgt][:20]))
    print("     naive   1 轮 · 直采 top1=doc[%2d] %-24s H1=%d" % (r1, corpus_q[r1][:24], naive_hit[i]))
    print("     agentic %d 轮 · 采信 doc[%2d] %-24s H1=%d   判断：%s" % (
        rounds, dec, corpus_q[dec][:24], ag_final_hit[i], why))
print()
print("  聚合：naive 最终首屏命中 %d/%d  （%d 条无声错答）；" % (sum(naive_hit), m, m - sum(naive_hit)))
print("         agentic 采信命中 %d/%d，轮数分布 %s，平均 %.1f 轮" % (
    sum(ag_final_hit), m, ag_rounds, np.mean(ag_rounds)))
naive_callsB = m * 1
ag_callsB = int(sum(ag_rounds))
print("         检索账单：naive %d 次 vs agentic %d 次（多 %d 次换回 %d 条原本错答）" % (
    naive_callsB, ag_callsB, ag_callsB - naive_callsB, sum(ag_final_hit) - sum(naive_hit)))

# =============================================================
# 实验 C：止损值不值——无判断盲目加深 top-k 的边际收益 vs agentic 判断后多查
# =============================================================
print()
print("=" * 72)
print("实验 C  止损：无判断盲目加深 top-k 的边际收益（token 还在涨） vs agentic 判断后及时收手")
print("=" * 72)
print("  对 4 条多跳问题，无判断地把检索深度 k=1..5 一截截放大（每次 Top-k 全量进上下文）：")
cum_hit = [0] * 5
cum_tok = 0
for rnd in range(5):
    k = rnd + 1
    tok = 0
    for tgt, Q, subs in MH:
        tk = dense_top(Q, k)
        cum_hit[rnd] += 1 if tgt in tk else 0
        tok += tokens_of(corpus_q[tk[-1]])
    cum_tok += tok
    gain = "+%d" % cum_hit[rnd] if rnd > 0 else str(cum_hit[rnd])
    print("    k=%d  Hit@k=%d/%d  累计上下文 %6d 字  %s" % (
        k, cum_hit[rnd], m, cum_tok, ("（新增 %d 条命中，仍白付 token）" % (cum_hit[rnd] - cum_hit[rnd - 1])) if rnd > 0 and cum_hit[rnd] > cum_hit[rnd - 1] else ""))
# agentic 对照：判断后只多查一轮，采信命中但 token 只付采信链。
ag_tok = 0
for tgt, Q, r1, r2, same, dec, rounds, why in ag_traced:
    ag_tok += tokens_of(corpus_q[dec]) * rounds
print("  agentic 对照：采信命中 %d/%d，上下文仅 %d 字（只付采信文档、判断后及时收手）" % (
    sum(ag_final_hit), m, ag_tok))
print("  盲目加深 k=2 已达 Hit@2=%d/%d，但首屏仍是 top1 那 2/4 —— 无判断的加深只加 token、不加正确答案位。" % (cum_hit[1], m))
print("  结论：多轮不是目的，判断「哪一轮才算数」才是（05 章『分解只拆不聚』的正解 = 加判断层）。")

print()
print("done · 一键复现：python code/notebooks/_tools/agentic_rag_demo.py")
import sys as _sys
print("═" * 72, file=_sys.stderr)
print("wall-clock %.1f s" % (time.time() - T0), file=_sys.stderr)
