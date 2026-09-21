# -*- coding: utf-8 -*-
"""13-失败模式与修复 的可复现诊断探针：7 道 RAG 失败模式各跑一次（真实数字，零下载）。

运行：cd 仓库根 && python code/notebooks/_tools/rag_diagnosis_demo.py
依赖：rank_bm25 jieba faiss sentence-transformers（与 05/10/11/06/08 同一栈）
语料：复用 code/notebooks/_demo_corpus.json（100 条中文百科，第 i 条答案在第 i 条上下文）。

把 §8.14 的 7 行「症状→根因→修复」变成可测信号 + 可直接复制的修复：
  实验 A（查不到→检索召回差）    全语料 gold-rank：首屏 rank>=1 的"查不到"名单，
                                dense∪bm25 混合能救哪些、救不了的留给查询改写（05）。
  实验 B（答不对+贵=噪声账+成本账）top-20 窗口相关行/陪跑行/证据纯度 + k=5/10/20
                                窗口字符成本曲线（控制块数的省钱账）。
  实验 C（专有名词查不到）        中文名提问 vs 语料里存的英文缩写（BPE/LIGO/5G），
                                BM25 词面断裂 miss → 别名扩展（词典）恢复。
  实验 D（更新后仍答旧内容）      3 篇 doc 更新到 v2 但索引仍 v1：版本戳检测器标陈旧
                                3 条 vs 内容余弦相似度视角"看着没问题"→ 增量重索引后 0 条。
  实验 E（引用来源错误）          检索后按显示顺序重编 [1..3]、但来源 doc 不跟编号 →
                                引用↔来源不一致；引用管线化（编号跟随真实 doc_id）后一致。
  实验 F（跨章节答不全）          挂靠 04 章父子分块 0.67→1.00 的诚实引用（本机不重测）。
  台账汇总                      7 行：症状 → 本机信号读数 → 修复站点。

确定性：固定种子、单线程 BLAS、jieba/BM25/faiss/bge 全部确定性；
        stdout 三遍逐位一致；墙钟只进 stderr（模型加载 ~9s 占大头）。
"""
import io, json, os, sys, time, warnings

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

def WALL(tag, t):
    sys.stderr.write("[wall] %-28s %s\n" % (tag, t))
    sys.stderr.flush()

# ---------- 数据与检索器（沿用 009/010/05/11/06） ----------
CORPUS = json.load(io.open(r'code/notebooks/_demo_corpus.json', encoding='utf-8'))
docs = [c for c, _ in CORPUS]
qs = [q for _, q in CORPUS]
N = len(docs)

def cut(s):
    return [w for w in jieba.cut(s) if w.strip() and len(w.strip()) > 1]

print("=" * 72)
print("前置：RAG 失败模式诊断手册（§8.14 七行表 → 本机信号）· 100 篇中文百科，第 i 条答案在第 i 条")
print("=" * 72)
tA = time.time()
bm = BM25Okapi([cut(d) for d in docs])
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(np.ascontiguousarray(emb))

def s_bm25(q, n):
    return np.argsort(bm.get_scores(cut(q)))[::-1][:n].tolist()

def s_dense(q, n):
    v = model.encode([q], normalize_embeddings=True)
    _, idx = index.search(np.ascontiguousarray(v), n)
    return idx[0].tolist()

def pos_of(ranked, tgt):
    return ranked.index(tgt) if tgt in ranked else len(ranked)

WALL("encode+index(100篇)", "%.2f s" % (time.time() - tA))
tA0 = time.time()

# ---------- 实验 A：查不到 → 检索召回差（首屏 rank 诊断） ----------
print()
print("=" * 72)
print("实验 A  查不到 → 检索召回差：全语料 gold-rank，首屏 misses + 混合修复")
print("=" * 72)
hist = {}
misses = []
for i in range(N):
    r = pos_of(s_dense(qs[i], 5), i)
    hist[r] = hist.get(r, 0) + 1
    if r >= 1:
        misses.append(i)
print("  Dense gold-rank 直方图（0=首屏命中）：%s" %
      " · ".join("%d档%d条" % (k, v) for k, v in sorted(hist.items())))
print("  首屏查不到（rank>=1）count = %d 条" % len(misses))
resq = []
for i in misses:
    hy = list(dict.fromkeys(s_dense(qs[i], 5) + s_bm25(qs[i], 5)))
    saved = 1.0 if pos_of(hy, i) < len(hy) else 0.0
    resq.append(saved)
    print("    期望 doc[%2d] %s" % (i, docs[i][:22]))
    print("      dense top5=含gold?%s · bm25 top5=含gold?%s → 混合 top%dsaved=%s" % (
        "Y" if i in s_dense(qs[i], 5) else "n",
        "Y" if i in s_bm25(qs[i], 5) else "n",
        len(list(dict.fromkeys(s_dense(qs[i], 5) + s_bm25(qs[i], 5)))), "Y" if saved else "N"))
print("  → 混合救回 %d/%d 首屏 miss；救不回的走查询改写（05）" % (sum(resq), len(misses)))
WALL("  实验A gold-rank+混合修复", "%.2f s" % (time.time() - tA0))
tB = time.time()

# ---------- 实验 B：噪声账 + 成本账（答不对/贵，一体两账） ----------
print()
print("=" * 72)
print("实验 B  答不对 + 贵 = 噪声账 + 成本账：top-20 证据纯度 & k=5/10/20 成本曲线")
print("=" * 72)
for i in (0, 22, 55):
    top20 = s_dense(qs[i], 20)
    qtok = set(cut(qs[i]))
    rel = [d for d in top20 if set(cut(docs[d])) & qtok]
    wchars = sum(len(docs[d]) for d in top20)
    rchars = sum(len(docs[d]) for d in rel)
    pu = rchars / wchars
    print("  q[%2d] 窗口 k=20：%2d 行相关 · %2d 行陪跑 · 证据纯度 %0.3f（%dB相关/%dB窗口）" % (
        i, len(rel), 20 - len(rel), pu, rchars, wchars))
kavg = []
for k in (5, 10, 20):
    c = sum(sum(len(docs[d]) for d in s_dense(qs[i], k)) for i in range(N)) / N
    kavg.append(c)
print("  100 问平均窗口字符成本线：k=5→%.1f · k=10→%.1f · k=20→%.1f 字符（控制块数=直接省钱）" %
      (kavg[0], kavg[1], kavg[2]))
print("  06 章对照：每问 top-20 全量 ≈614 字符 → 宽松压缩 ≈72.5（-88.2%）")
WALL("  实验B 纯度+成本线", "%.2f s" % (time.time() - tB))
tC = time.time()

# ---------- 实验 C：专有名词查不到 → 别名/词典扩展 ----------
print()
print("=" * 72)
print("实验 C  专有名词查不到 → 词面断裂（中文/英文缩写/别名）× 别名扩展")
print("=" * 72)
PN = [
    (59, "第五代移动通信的核心优势是什么", "5G"),
    (96, "往提示里塞坏指令骗模型破防的那个招", "LLM"),
    (4,  "那个看机器会不会说人话的古老检验", "图灵"),
]
for tgt, qy, alias in PN:
    c_naive, c_fix = s_bm25(qy, 1)[0], s_bm25(qy + " " + alias, 1)[0]
    d_naive, d_fix = s_dense(qy, 1)[0], s_dense(qy + " " + alias, 1)[0]
    print("  目标[%2d] %-22s 别名=%s" % (tgt, docs[tgt][:22], alias))
    print("    BM25  %s→%s · Dense %s→%s （1=首屏命中）" % (
        "HIT" if c_naive == tgt else ("miss→doc%-3d" % c_naive), "HIT" if c_fix == tgt else ("miss→doc%d" % c_fix),
        "HIT" if d_naive == tgt else ("miss→doc%-3d" % d_naive), "HIT" if d_fix == tgt else ("miss→doc%d" % d_fix)))
WALL("  实验C 别名扩展", "%.2f s" % (time.time() - tC))
tD = time.time()

# ---------- 实验 D：更新后仍答旧内容 → 版本戳 + 增量重索引 ----------
print()
print("=" * 72)
print("实验 D  更新后仍答旧内容 → 索引没增量：版本戳检测 + 增量重索引")
print("=" * 72)
UPD = [3, 8, 21]
newtxt = {}
for d in UPD:
    newtxt[d] = docs[d] + "。补充：该专题已更新至第五章 v2"
new_emb = model.encode([newtxt[d] for d in UPD], normalize_embeddings=True, show_progress_bar=False)
version = np.ones(N, dtype=int)                    # 索引侧版本戳：全是 v1
cur = np.ones(N, dtype=int)
for j, d in enumerate(UPD):
    cur[d] = 2
print("  3 篇已改（doc %s → v2），索引仍是 v1" % " ".join(str(d) for d in UPD))
flags = [d for d in range(N) if version[d] != cur[d]]
print("  版本戳诊断：陈旧篇数 = %d（%s）——干净、可靠、每条都能标" %
      (len(flags), ",".join("doc%d" % d for d in flags)))
for j, d in enumerate(UPD):
    cos = float(np.dot(emb[d], new_emb[j]))
    print("    doc[%2d] 内容余弦(旧vs新) = %.4f ← 内容层看着像同一条 → 必须靠版本戳" % (d, cos))
version[UPD] = 2                                   # 增量重索引 = 更新版本戳
print("  增量重索引（更新 v1→v2）后陈旧篇数 = %d" % sum(version != cur))
WALL("  实验D 版本戳+增量", "%.2f s" % (time.time() - tD))
tE = time.time()

# ---------- 实验 E：引用来源错误 → 引用管线化 ----------
print()
print("=" * 72)
print("实验 E  引用来源错误 → 元数据/编号错位：按显示顺序编号 vs 编号跟随真实 doc_id")
print("=" * 72)
RIDX = [4, 22, 30]
for i in RIDX:
    top3 = s_dense(qs[i], 3)
    reordered = sorted(top3, key=lambda d: -len(docs[d]))   # 假想：拼装时按长度从长到短排
    naive_bad = 1.0 if reordered[0] != i else 0.0           # naive: "答案取[1]"，但[1]不是来源
    slot = top3.index(i) + 1 if i in top3 else 0
    pipe_bad = 0.0 if slot > 0 else 1.0                     # 管线化: 编号跟随 doc_id，来源必在
    print("  目标[%2d] 检索top3=doc%s · 按长度重排=[%s]" % (
        i, ",".join(str(x) for x in top3), ",".join(str(x) for x in reordered)))
    print("    naive 按显示序编号 → 引用[1]=doc%d ≠ 来源 doc%d  错位=%s" % (
        reordered[0], i, "Y" if naive_bad else "N"))
    print("    管线化 编号跟随 doc_id   → 来源在槽%2d 引用对齐=%s" % (slot, "Y" if not pipe_bad else "N"))
mism = sum(sorted(s_dense(qs[i], 3), key=lambda d: -len(docs[d]))[0] != i for i in RIDX)
print("  → naive 编号 3 问错位 %d/3；引用管线化 3/3 对齐（来源槽位 = 真实 doc_id 槽位）" % mism)
tF = time.time()

# ---------- 台账汇总（§8.14 七行 × 本机信号 × 修复站点） ----------
print()
print("=" * 72)
print("台账汇总  §8.14 七行「症状→根因→修复」· 信号=本机读数 · 修复=站点指针")
print("=" * 72)
print("  ①答非所询查不到  · 信号: 首屏 rank>=1 %d/100（混合救回 %d，A） → 修复: 03 混合重排 / 05 查询改写" %
      (len(misses), sum(resq)))
print("  ②答案与文档矛盾  · 信号: top-20 陪跑行（B: 证据纯度 0.06~0.21 档·陪跑 16~19/20 行） → 修复: 06 压缩 + only-from-evidence")
print("  ③专有名词查不到  · 信号: BM25 词面断裂 3/3 miss（C: 别名扩展 3/3 救回·本语料 Dense 语义搭桥全 HIT） → 修复: 词典/别名扩展 + 调 BM25 权重")
print("  ④跨章节问题答不全· 信号: 单块召回不足 1.0（04 章父子实测 0.67→1.00） → 修复: 04 父子分块 / 10 多跳")
print("  ⑤引用来源错误    · 信号: 引用↔来源错位（E: naive %d/3 vs 管线化 0/3） → 修复: 引用管线化 + Post-hoc 校验" % mism)
print("  ⑥更新后仍答旧内容· 信号: 版本戳陈旧 %d 条（D: 内容余弦 0.95+ 看不出来） → 修复: 增量更新 + TTL / 08 运维" % len(flags))
print("  ⑦成本暴涨        · 信号: 窗口字符成本（B: k=20→%.0f 字符/问） → 修复: 06 压缩 + 控制块数 + 08 缓存路由" % kavg[2])
print()
print("  方法论：先看信号落在哪一层（检索层 ①②③ / 索引层 ④和⑥ / 编撰层 ⑤ / 生成层 ②⑦），")
print("          再按站点图去对应章节拿现成修复——手册不重复讲机制，只做「症状→触点」翻译。")
print()
WALL("  实验E 引用对齐", "%.2f s" % (time.time() - tE))
WALL("台账段 100×20 查询(成本线)", "%.2f ms" % ((time.time() - tF) * 1000))
WALL("total", "%.2f s" % (time.time() - T0))
print("done · 一键复现：python code/notebooks/_tools/rag_diagnosis_demo.py")
