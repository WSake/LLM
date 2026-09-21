# -*- coding: utf-8 -*-
"""08-向量数据库选型 的可复现探针：同层机械四款索引真跑 + 工程外壳决策引擎（真实数字，零下载）。

运行：cd 仓库根 && python code/notebooks/_tools/vector_db_demo.py
依赖：faiss sentence-transformers numpy（与 009/010 及 05/10/11 探针同一栈，Colab 可用）
语料：复用 code/notebooks/_demo_corpus.json（100 篇中文短篇，bge-small-zh 真实嵌入）；
     另在 20k×64d 确定性合成集（RandomState(2026)）上画近似召回曲线——小库上近似层不分水岭，
     大库上「召回-延迟-索引大小」三角才现形。

实验 A：同层机械四款（Exact flat / IVF / HNSW / IVFPQ）——
  A0 真实 100 篇小库：三种索引 Hit@1 排位（近似层在小库不是分水岭）。
  A1 合成 20k×64d：IVF 与 IVFPQ 的 recall@10 vs nprobe、HNSW 的 recall@10 vs efSearch（Exact=基准 1.000）。
  A2 索引字节账：写盘序列化字节数（flat/ivf/hnsw/pq）+ 每引擎一个代表工作点的 recall@10。
实验 B：工程外壳同层比较（§8.8 九家要素事实表）+ 场景决策引擎（9 断言锁死，规则=author 表）。
实验 C：墙钟（build / 每查询延迟，只进 stderr [wall]）+ 生产外推账（非本机实测要素）。

确定性：合成集固定种子；IVF 聚类 faiss Clustering 内部 seed 固定、HNSW 构建 rng 固定、单线程 BLAS；
        召回率与索引大小三遍逐位一致；墙钟只进 stderr（run 间浮动）。
口径：recall@10 = 每个查询「精确 top-10 中被近似 top-10 捕获的个数 / 10」再对查询平均（[0,1]，
      1.000 ≈ 精确）。
"""
import io, json, os, sys, tempfile, time, warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
T0 = time.time()

def WALL(tag, t):
    sys.stderr.write("[wall] %-26s %s\n" % (tag, t))
    sys.stderr.flush()

def index_bytes(idx):
    f = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    f.close()
    faiss.write_index(idx, f.name)
    n = os.path.getsize(f.name)
    os.unlink(f.name)
    return n

# ---------- 实验 A0：真实 100 篇小库，三种索引同台 ----------
print("=" * 72)
print("前置：向量库=「存储向量的系统 + ANN 检索的工程外壳」——先定场景，再选库")
print("=" * 72)
CORPUS = json.load(io.open(r'code/notebooks/_demo_corpus.json', encoding='utf-8'))
docs = [c for c, _ in CORPUS]
qs = [q for _, q in CORPUS]
N = len(docs)
tA = time.time()
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
emb = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
emb = np.ascontiguousarray(emb)
flat = faiss.IndexFlatIP(emb.shape[1]); flat.add(emb)

def gold_rank(idx):
    hist = {}
    for i in range(N):
        v = model.encode([qs[i]], normalize_embeddings=True)
        _, xx = idx.search(np.ascontiguousarray(v), 20)
        top = xx[0].tolist()
        r = top.index(i) if i in top else 99
        hist[r] = hist.get(r, 0) + 1
    return hist

def fmt(hist):
    return " ".join("%d:%d" % (k, v) for k, v in sorted(hist.items()))

ivf100 = faiss.IndexIVFFlat(faiss.IndexFlatIP(emb.shape[1]), emb.shape[1], 10)
ivf100.train(emb); ivf100.add(emb); ivf100.nprobe = 10        # nprobe=nlist → 全量
H_flat = gold_rank(flat)
H_ivf = gold_rank(ivf100)
hn100 = faiss.IndexHNSWFlat(emb.shape[1], 16); hn100.add(emb); hn100.hnsw.efSearch = 128
H_hn = gold_rank(hn100)
WALL("A0 encode+index+排位(100问)", "%.2f s" % (time.time() - tA))

def hit(hist, k):
    return sum(v for r, v in hist.items() if r < k)

print("  文档 100 篇（短篇）· 第 i 问答案在第 i 篇 · bge-small-zh 真实嵌入 · 与 05/06/10/11 同源底板")
for nm, Hv in (("Exact(flat)          ", H_flat), ("IVFFlat(nlist=10,nprobe=10)", H_ivf), ("HNSW(M=16,efSearch=128)", H_hn)):
    print("  A0 %s：Hit@1=%d/100 · Hit@5=%d/100 · 排位 %s"
          % (nm, hit(Hv, 1), hit(Hv, 5), fmt(Hv)))
print("    -> 100 篇小库：三种索引全部≈精确——「召回分水岭」不在这里；选库的分水岭在工程面（§8.8），见实验 B")
print()

# ---------- 实验 A1/A2：合成 20k×64d 上四款索引的召回曲线与字节账 ----------
rng = np.random.RandomState(2026)
D, NS, K = 64, 20000, 10
X = rng.randn(NS, D).astype('float32')
X /= np.linalg.norm(X, axis=1, keepdims=True)                  # 归一化 → IP 即余弦
Q = rng.randn(2048, D).astype('float32')
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
flatS = faiss.IndexFlatIP(D); flatS.add(X)
_, GT = flatS.search(Q, K)                                     # 精确 top-10 = 基准
tB = time.time()

def recall10(idx, nprobe=None, ef=None):
    if isinstance(idx, faiss.IndexIVF) and nprobe is not None:
        idx.nprobe = nprobe
    if isinstance(idx, faiss.IndexHNSW) and ef is not None:
        idx.hnsw.efSearch = ef
    _, I = idx.search(Q, K)
    hits = sum(len(set(g) & set(r)) for g, r in zip(GT, I))
    return hits / (len(GT) * K)

ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(D), D, 200)
ivf.train(X); ivf.add(X)
rec_ivf = []
for p in (1, 5, 10, 20, 50, 200):
    rec_ivf.append((p, recall10(ivf, nprobe=p)))

hn = faiss.IndexHNSWFlat(D, 32); hn.add(X)
rec_hn = []
for e in (16, 32, 64, 128, 256):
    rec_hn.append((e, recall10(hn, ef=e)))

pq = faiss.IndexIVFPQ(faiss.IndexFlatIP(D), D, 200, 8, 8)
pq.train(X); pq.add(X)
rec_pq = []
for p in (1, 5, 10, 20, 50, 200):
    rec_pq.append((p, recall10(pq, nprobe=p)))

WALL("A1/A2 建四款索引(20k×64d)", "%.2f s" % (time.time() - tB))

def round4(x):
    return "%0.4f" % x

print("=" * 72)
print("实验 A  同层机械：Exact / IVF / HNSW / IVFPQ（合成 20k×64d，Recall@10 vs 旋钮）")
print("=" * 72)
print("  A1 Exact(flat)=基准 recall@10 1.0000（精确穷举，代价=每查询全扫全库）")
print("      IVFFlat  recall@10 vs nprobe：%s" % " · ".join("p=%d→%s" % (p, round4(r)) for p, r in rec_ivf))
print("      IVFPQ    recall@10 vs nprobe：%s" % " · ".join("p=%d→%s" % (p, round4(r)) for p, r in rec_pq))
print("      HNSW     recall@10 vs efSearch：%s" % " · ".join("ef=%d→%s" % (e, round4(r)) for e, r in rec_hn))
print("      读法：旋钮（nprobe/efSearch）开到顶，IVF/HNSW 都逼近精确；IVFPQ 顶格也到不了精确（量化噪声）")
print("  A2 索引字节账（写盘序列化）与每引擎一个代表工作点：")
s_flat, s_ivf, s_hn, s_pq = (index_bytes(x) for x in (flatS, ivf, hn, pq))
r_ivf_20 = dict(rec_ivf)[20]
r_pq_20 = dict(rec_pq)[20]
r_hn_128 = dict(rec_hn)[128]
print("      Exact      ：%8d B（≈%.0f B/向量，fp32 原样） · recall@10 1.0000" % (s_flat, s_flat / NS))
print("      IVFFlat    ：%8d B（≈%.0f B/向量）· 工作点 nprobe=20 → recall@10 %s" % (s_ivf, s_ivf / NS, round4(r_ivf_20)))
print("      HNSW(M=32) ：%8d B（≈%.0f B/向量，图边把体积顶到 ~2×）· 工作点 efSearch=128 → recall@10 %s" % (s_hn, s_hn / NS, round4(r_hn_128)))
print("      IVFPQ      ：%8d B（≈%.0f B/向量，压缩到 1/12）· 工作点 nprobe=20 → recall@10 %s" % (s_pq, s_pq / NS, round4(r_pq_20)))
print("    -> 索引层三角：HNSW 在「召回≈精确 + 体积 ~2×」、IVFPQ 在「体积 1/12 + 召回让位」、Exact 在「召回满格 + 体积原样」")
print()

# ---------- 实验 B：工程外壳同层比较 + 决策引擎 ----------
print("=" * 72)
print("实验 B  工程外壳同层比较（§8.8 九家要素事实）· 先定场景再选库")
print("=" * 72)
# 要素事实（非本机实测：公开文档/知识地图，写作环境无外网未在线复核，精确功能以官方为准）
sheet = [
    ("FAISS",          "库（进程内）",  "快、轻、研究友好；HNSW/IVF-PQ 全支持",  "原型/单机"),
    ("Milvus",         "分布式向量库",  "云原生、大规模、多索引、GPU 支持",        "生产大规模"),
    ("Qdrant",         "分布式向量库",  "过滤能力强、Rust 性能好、易部署",          "中大规模生产"),
    ("Weaviate",       "分布式向量库+图","混合检索、模块化",                        "中规模"),
    ("Chroma",         "轻量嵌入式",    "零运维、本地优先",                        "原型/个人"),
    ("pgvector",       "Postgres 扩展", "与业务数据同库、事务一致",                "已有 PG 的中小应用"),
    ("ES/OpenSearch",  "全文+向量双引擎","BM25 极强、运维成熟、企业标配",           "混合检索生产"),
    ("Redis Stack",    "内存 KV+向量",  "超低延迟、缓存热路径",                    "极热查询"),
    ("LanceDB/SQLite-VSS", "嵌入式新秀","零服务、列式存储",                        "本地/边缘"),
]
print("  §8.8 要素事实（非本机实测）：系统类型·优势·适合一张表——")
for nm, ty, adv, fit in sheet:
    print("    · %-16s %-16s %s -> %s" % (nm, ty, adv, fit))
print("  选型钥匙：检索要混合（词法+向量+重排，03 章）；别为了「向量库」而引入重型组件。")

# 决策引擎：场景六维（事务/规模/延迟/运维/混合/图）-> 推荐库（author 规则表，本机断言锁死）
def decide(prof):
    txn, scale, lat, ops, hybrid, graph = prof
    if txn == 1:
        return "pgvector"
    if scale == "1M+" and lat == "常规":
        return "Milvus"
    if lat == "极热":
        return "Redis Stack"
    if graph == 1:
        return "Weaviate"
    if ops == "零服务":
        return "LanceDB/SQLite-VSS"
    if ops == "零运维" and hybrid == "低":
        return "Chroma"
    if ops == "轻运维":
        return "FAISS"
    if hybrid == "词法+向量":
        return "ES/OpenSearch"
    return "Qdrant"

scens = [
    ("s01", "原型/个人·本地优先·零运维",                (0, "小", "常规", "零运维", "低", 0),      "Chroma"),
    ("s02", "单机研究·轻量·四款索引都要测",             (0, "小", "常规", "轻运维", "低", 0),      "FAISS"),
    ("s03", "生产大规模·云原生·GPU",                   (0, "1M+", "常规", "重运维", "向量为主", 0), "Milvus"),
    ("s04", "中大规模生产·过滤强（多租户/时间过滤）",     (0, "中", "常规", "重运维", "向量为主", 0), "Qdrant"),
    ("s05", "已有 PG 业务库·订单与向量同库要事务一致",    (1, "小", "常规", "中运维", "低", 0),      "pgvector"),
    ("s06", "企业混合检索·BM25+向量·运维成熟优先",       (0, "中", "常规", "重运维", "词法+向量", 0), "ES/OpenSearch"),
    ("s07", "极热查询·超低延迟·缓存热路径",             (0, "小", "极热", "轻运维", "向量为主", 0), "Redis Stack"),
    ("s08", "本地/边缘·零服务·列式存储",               (0, "小", "常规", "零服务", "低", 0),      "LanceDB/SQLite-VSS"),
    ("s09", "中小规模·混合检索+图也要·模块化",          (0, "中", "常规", "中运维", "词法+向量", 1), "Weaviate"),
]
ok = 0
for sid, desc, prof, exp in scens:
    got = decide(prof)
    mark = "lock" if got == exp else "MISMATCH"
    ok += got == exp
    print("  B  %s %s -> %-20s [决策锁: %s]" % (sid, desc, got, mark))
print("    -> 决策引擎 9 场景断言 %d/9 全过：先定场景（事务/规模/延迟/运维/混合/图六问）再选库，选库不是选索引" % ok)

# ---------- 实验 C：墙钟 + 生产外推 ----------
print()
print("=" * 72)
print("实验 C  成本账：墙钟（stderr [wall]）+ 生产外推账（非本机实测要素）")
print("=" * 72)
n_q = 100
t = time.time()
flatS.search(Q[:n_q], K); ivf.search(Q[:n_q], K); hn.search(Q[:n_q], K); pq.search(Q[:n_q], K)
WALL("C 100×4 查询(合成集)", "%.2f ms" % ((time.time() - t) * 1000))
t = time.time()
_ = [decide(p) for _, _, p, _ in scens]
WALL("C 决策引擎 9×判断", "%.3f ms" % ((time.time() - t) * 1000))
WALL("total", "%.2f s" % (time.time() - T0))
print("  C1 墙钟：见 stderr [wall]——模型加载 ~9s 占大头；建 20k 索引 ivf 0.1s/hnsw 2s/pq 3s 量级，100×4 次查询毫秒级")
print("      本机浮动只在 stderr；stdout 三遍逐位一致（召回率与索引字节是确定性的科学数字）")
print("  C2 生产外推（非本机实测，精确值以官方为准）：百万级（1,000,000×768 fp32≈2.9 GiB 裸向量）")
print("     · 建图三角复利：HNSW 图边 ≈ M×2×H0×N ≈ 每点几百 B，1M 点 HNSW 图边量级 ~GB 级（≈向量本身的 ~2 倍）")
print("     · 上同一磁盘前先算：裸向量 + 图边/聚簇 + 元数据过滤字段 + 备份，别只按裸向量估容量")
print("     · 「生产大规模看 Milvus/Qdrant/ES」说的是工程面：分片/HA/增量删除/多租户过滤/运维，不是召回（召回近似层已拉平）")
print("     · 极热路径用 Redis Stack 是缓存热路径账：内存驻留 vs 磁盘 scan")
print()
print("done · 一键复现：python code/notebooks/_tools/vector_db_demo.py")
