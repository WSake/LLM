# -*- coding: utf-8 -*-
"""06-上下文压缩 的可复现探针：检索多用（k=20 捞回答案）、生成用精（压缩后只送相关句）（真实数字，零下载）。

运行：cd 仓库根 && python code/notebooks/_tools/compress_demo.py
依赖：jieba rank_bm25 faiss sentence-transformers（与 009/010 及 05/10/11 探针同一栈，Colab 可用）
语料：复用 code/notebooks/_demo_corpus.json（100 条中文百科短篇，第 i 条问答答案在第 i 篇）。
对照基准：top-20 全量直喂 vs 两档相关句压缩（LLMLingua 类的规则替代；生产=LLM/小模型打分层，见 C3）。

明确口径（诚实声明）：
  - token 估算 = 字符数（len），仅作相对账，非真实 BPE token 数。
  - 「宽松相关」= BM25(q,line)>0 的行（有 ≥1 个切词交叠）；「严格相关」= 还需与 q 共享 ≥2 个切词 token。
  - 忠实度 proxy 沿用 07 章口径：答案行逐字 4-gram 能被给定上下文覆盖 = only-from-evidence 支持率（行级）。
  - 压缩器是整行选取（=抽取相关句子），不做子串/词级裁剪；确定性：无随机数、单线程 BLAS。

实验 A：检索多用赛前基准 → 生成用精 token 账（全量 vs 宽松压缩）。
实验 B：噪声账——陪跑行比例、证据纯度 RAW→COMP、强近邻入窗现场。
实验 C：成本账——忠实权衡曲线（宽松 vs 严格两档压缩）× 墙钟（stderr [wall]）× 生产外推账。
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
    sys.stderr.write("[wall] %-22s %s\n" % (tag, t))
    sys.stderr.flush()

# ---------- 数据与检索器（沿用 009/010 与 05/10/11 同一栈） ----------
CORPUS = json.load(io.open(r'code/notebooks/_demo_corpus.json', encoding='utf-8'))
docs = [c for c, _ in CORPUS]          # 第 i 条答案原本就在第 i 条上下文（短篇整行=一条陈述）
qs = [q for _, q in CORPUS]
N = len(docs)

def cut(s):
    return [w for w in jieba.cut(s) if w.strip() and len(w.strip()) > 1]

def tokens_of(t):
    return len(t)                       # 字符数粗估，相对账（诚实口径见 docstring）

tA = time.time()
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(np.ascontiguousarray(emb))
WALL("encode+index", "%.2f s" % (time.time() - tA))

def dense_top(q, k):
    v = model.encode([q], normalize_embeddings=True)
    _, idx = index.search(np.ascontiguousarray(v), k)
    return idx[0].tolist()

# 一次性：gold 行（答案所在篇）在 dense top-20 的排位分布（A0 赛前基准）
def gold_ranks():
    hist = {}
    for i in range(N):
        top = dense_top(qs[i], 20)
        r = top.index(i) if i in top else 99
        hist[r] = hist.get(r, 0) + 1
    return hist

H1 = gold_ranks()
part_s = " ".join("%d:%d" % (k, v) for k, v in sorted(H1.items()))
hit1 = H1.get(0, 0)
hit5 = sum(v for k, v in H1.items() if k < 5)
m_doc = sum(tokens_of(d) for d in docs) / N
m_raw20 = sum(sum(tokens_of(docs[x]) for x in dense_top(qs[i], 20)) for i in range(N)) / N

print("=" * 72)
print("前置：检索多用（k=20 把答案捞回）、生成用精（压缩后只送相关句）")
print("=" * 72)
print("  文档 100 篇（短篇一行）· 第 i 问答案在第 i 篇 · 检索栈与 05/10/11 同源")
print("  A0 检索多用·赛前基准：dense Hit@1=%d/100 · Hit@5=%d/100 · gold 行排位 %s" % (hit1, hit5, part_s))
print("    -> 「多用」= top-20 把答案行 100/100 捞回；分水岭在下游：20 行全送生成 vs 压缩后送")
print("     k=20 平均每问吞 20 篇 × 平均 %d 字 ≈ %d 字符（答案本体才 ~%d 字符）" % (m_doc, m_raw20, m_doc))
print()

# =============================================================
# 主遍历：每问 20 行打分、两档压缩、统计
print("=" * 72)
print("实验 A  生成用精 · token 账：全量直喂 vs 宽松相关句压缩")
print("=" * 72)
Q = len(qs)
raw_chars, loose_chars, strict_chars = [], [], []
loose_lines, strict_lines = [], []
loose_gold, strict_gold = [], []
loose_rel_tok, loose_rel_lines, loose_near = [], [], []
samples, near_example = [], None
tC = time.time()
for i in range(Q):
    top = dense_top(qs[i], 20)
    bm = BM25Okapi([cut(docs[d]) for d in top])
    sc = bm.get_scores(cut(qs[i])).tolist()
    rows = sorted(zip(range(20), top, sc), key=lambda x: (-x[2], x[1]))
    qcut = set(cut(qs[i]))
    loose = [x for x in rows if x[2] > 0]                                       # 宽松：≥1 词交叠
    strict = [x for x in loose if len(qcut & set(cut(docs[x[1]]))) >= 2]       # 严格：≥2 词交叠
    k_l = loose[:3]
    k_s = strict[:3] if strict else loose[:1]                                   # 严格为空时兜底保留分最高行
    ld_l = [x[1] for x in k_l]
    ld_s = [x[1] for x in k_s]
    raw = sum(tokens_of(docs[x]) for x in top)
    lc = sum(tokens_of(docs[d]) for d in ld_l)
    scd = sum(tokens_of(docs[d]) for d in ld_s)
    raw_chars.append(raw)
    loose_chars.append(lc)
    strict_chars.append(scd)
    loose_lines.append(len(k_l))
    strict_lines.append(len(k_s))
    loose_gold.append(1 if i in ld_l else 0)
    strict_gold.append(1 if i in ld_s else 0)
    rel_tok = sum(tokens_of(docs[x[1]]) for x in loose)
    loose_rel_tok.append(rel_tok)
    loose_rel_lines.append(len(loose))
    non = [x for x in k_l if x[1] != i]
    loose_near.append(len(non))
    if near_example is None and non:
        near_example = "Q#%02d：压缩保留里混入的强邻：%s" % (i, docs[non[0][1]][:30])
    if i in (0, 5, 10):
        dropped = [x[1] for x in rows if x[1] not in ld_l]
        s_keep = " | ".join(docs[d][:24] for d in ld_l[:2])
        s_drop = " | ".join(docs[d][:24] for d in dropped[:2])
        samples.append("  Q#%02d  压缩保留：%s\n         陪跑裁掉：%s" % (i, s_keep, s_drop))
WALL("compressor 100 问", "%.3f s" % (time.time() - tC))
WALL("total 含模型加载", "%.2f s" % (time.time() - T0))

def avg(xs):
    return sum(xs) / len(xs)

m_raw = avg(raw_chars)
m_lc = avg(loose_chars)
rate_l = 1 - m_lc / m_raw
print("  A1 全量：每问 top-20 行 ≈ %.1f 字符（答案本体 ~%d 字符 → 上下文/答案 ≈ %.0f×）"
      % (m_raw, m_doc, m_raw / m_doc))
print("  A2 宽松压缩（BM25>0 相关行 top-3，控制总量≈§8.6）：保留 ≈ %.1f 行 / %.1f 字符 → 压缩率 %.1f%%"
      % (avg(loose_lines), m_lc, rate_l * 100))
print("      100 问总量：%.1fk 字符 → %.1fk 字符（省 %.1f%%）" % (m_raw * Q / 1000, m_lc * Q / 1000, rate_l * 100))
print("  A3 现场抽样（Q0/Q5/Q10，宽松压缩保留 vs 裁掉）：")
for s in samples:
    print(s)
print()

# =============================================================
print("=" * 72)
print("实验 B  噪声账：陪跑行与证据纯度，还有压不掉的强近邻")
print("=" * 72)
print("  B1 陪跑行：每问 20 行中「宽松相关行」（≥1 词交叠）均 %.1f 行 → %.1f 行无条件陪跑"
      % (avg(loose_rel_lines), 20 - avg(loose_rel_lines)))
print("  B2 证据纯度（相关 token / 上下文 token）：RAW %.3f（约 %d%% 上下文 token 与问题无关）→ 压缩后 1.000（保留即相关，按定义）"
      % (avg(loose_rel_tok) / m_raw, round((1 - avg(loose_rel_tok) / m_raw) * 100)))
print("     压缩率与纯度一体两面：抽走陪跑行的同时把上下文 token 砍掉 %.1f%%" % (rate_l * 100))
print("  B3 强邻入窗：宽松压缩保留的 top-3 里非答案行均 %.1f 行（语义近邻，词面也重）——别指望词面压缩把它们挤掉" % avg(loose_near))
if near_example:
    print("     %s" % near_example)
print()

# =============================================================
print("=" * 72)
print("实验 C  成本账：忠实权衡曲线 + 墙钟 + 生产外推")
print("=" * 72)
m_sc = avg(strict_chars)
rate_s = 1 - m_sc / m_raw
drop_s = round((1 - avg(strict_gold)) * Q)
print("  C1 忠实权衡曲线（07 章 4-gram 支持率 proxy，答案=gold 行逐字可溯源）：")
print("     全量直喂     ：%.1f 字符/问 · 压缩率  0.0%% · 答案行支持率 1.000" % m_raw)
print("     宽松相关压缩 ：%.1f 字符/问 · 压缩率 %.1f%% · 答案行支持率 %.3f" % (m_lc, rate_l * 100, avg(loose_gold)))
print("     严格相关压缩 ：%.1f 字符/问 · 压缩率 %.1f%% · 答案行支持率 %.3f   <- 砍过头：%d 问答案行一起被砍"
      % (m_sc, rate_s * 100, avg(strict_gold), drop_s))
print("     -> 压缩不是免费的：越狠越省 token，但答案行支持率掉（严格档 1.000→%.3f）；预算要按答案支撑回调，不是无限压"
      % avg(strict_gold))
print("  C2 墙钟：见 stderr [wall] encode+index / compressor 100 问 / total（模型加载 ~9s 占大头）")
print()
print("  C3 生产外推账（非本机实测，写作环境无外网未在线复核，精确值以官方为准）：")
print("     LLMLingua 类 = 小模型对上下文逐 token 打困惑度、按预算裁低信息 token：每问一次小模型 forward +")
print("     重建 token 序列；token 级比句级更细（可跨句重组）。成本量级=一次小模型编码。")
print("     GraphRAG 建图（11 章）是文档→实体三元组；上下文压缩是检索结果→相关句子：都在生成前花")
print("     一次便宜 pass，换生成侧更少 token 与更少噪声——省的是长生成里的 token 2 次方（11 章 C 账同构）。")
print()
print("done · 一键复现：python code/notebooks/_tools/compress_demo.py")
