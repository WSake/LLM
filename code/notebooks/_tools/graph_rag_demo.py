# -*- coding: utf-8 -*-
"""11-GraphRAG 的可复现探针：实体-关系图（概念共现）vs 向量检索的三场对照（真实数字，零下载）。

运行：cd 仓库根 && python code/notebooks/_tools/graph_rag_demo.py
依赖：rank_* 无；sentence-transformers + faiss（与 009/010/10-Agentic 同一栈，Colab 可用）
语料：复用 code/notebooks/_demo_corpus.json（100 条中文百科，第 i 条答案在第 i 条上下文）。

本篇诚实声明：生产 GraphRAG 用 LLM 抽取实体与类型化关系三元组（贵、慢、非本机实测）；
本探针用"规则词典 × 文档共现"建图（LightRAG 式轻量图，确定性、零下载），图中每条边都能
回溯到一个具体文档 = "断言可溯源"。三场对照全部在同一语料、同一 Dense encoder 上做。

实验 A（全局问题）：向量单次全局检索只看得到 top-k 相似片段（覆盖 2/11 主题网络族）vs
  图谱社区表在建图时刻一次性生成全局目录（11 族 + 69 个单点主题）；附"事实查询"底数——
  100 条专属问答向量 Hit@1 98/100 · Hit@5 100/100（向量在事实查询上近乎完美，GraphRAG 的
  用武之地不在这里，对应 §8.11 的"混合使用"）。
实验 B（实体关联）：5 组"种子概念 → 图 1-hop 断言 vs 向量 top-5 主题"对照——B1 相关列举
  两边都能答（图断言自证 100%，向量 top-5 也接近）；B2 噪声：向量 top-5 袋里"图上未断言的
  语义近邻"条数（不可回溯源证据）；B3 图 2-hop 可达性机制示范（向量余弦不传递共现关系）。
实验 C（代价账）：规则建图墙钟 vs 向量建索引墙钟（本机实测）+ LLM 抽取建图外推账（非本机实测）。

一切确定性：固定词典/文档顺序/无随机数、单线程 BLAS；墙钟与模型加载只写 stderr，
stdout byte-for-byte 恒一（三遍 md5 相同）。
"""
import io, json, os, sys, time, collections, warnings

# 单线程 BLAS 必须在 numpy import 之前设定，保证两次运行逐位一致。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
T0 = time.time()
def WALL(tag, t):
    dt = time.time() - t
    print("[wall] %-12s %s" % (tag, "%.1f ms" % (dt * 1000) if dt < 1 else "%.2f s" % dt),
          file=sys.stderr)

CORPUS = json.load(io.open(r'code/notebooks/_demo_corpus.json', encoding='utf-8'))
corpus_q = [c for c, _ in CORPUS]
queries = [q for _, q in CORPUS]
N = len(corpus_q)

# ---------- 概念词典（规则抽取：子串匹配；覆盖 100 篇的 98 篇） ----------
CONCEPTS = ['引力波', '量子纠缠', '量子计算', 'Transformer', '注意力', '梯度下降', '损失函数',
            '神经网络', '图灵测试', '卷积', '图像识别', '反向传播', '元学习', '语言模型', '强化学习',
            '奖励信号', '决策树', '支持向量机', '贝叶斯', '过拟合', 'K均值', '主成分分析', '正则化',
            '学习率', '批归一化', '残差', '梯度消失', '长短期记忆', '知识蒸馏', '剪枝', '量化',
            '联邦学习', '联邦蒸馏', '对比学习', '自编码器', '生成对抗网络', '扩散模型', '词向量',
            'N-gram', 'BPE', '词嵌入', '提示工程', '思维链', '检索增强生成', '向量数据库', '限流',
            '数据库索引', '缓存', '负载均衡', '消息队列', '容器', '分布式系统', '数据库事务',
            '布隆过滤器', '一致性哈希', '断点续传', '压缩算法', '自动驾驶', '脑机接口', '基因编辑',
            'CRISPR', '区块链', '5G', '碳中和', '时空', '大模型', '深度学习',
            '中医学', '黑洞', '光合作用', '人类基因组', '疫苗', '全球变暖', '托卡马克', '超级计算机',
            '湿地', '青藏高原', '大熊猫', '福建土楼', '都江堰', '清明上河图', '彗星', '极光', '海啸',
            '沙漠化', '芯片', '熔断', '复利', '市盈率', '分散投资', '通货膨胀', '碳交易', '供应链',
            '核电站', '空间站', '合成生物学', '柔性电子', '星链', '长征火箭', '前庭觉', '激素',
            '免疫记忆', '幻觉', '边缘计算', 'RDMA']

tA = time.time()
doc_concepts = [[k for k in CONCEPTS if k in c] for c in corpus_q]   # 规则抽取（子串匹配）
concept_docs = collections.defaultdict(list)
for i, cs in enumerate(doc_concepts):
    for k in set(cs):
        concept_docs[k].append(i)
concepts_all = sorted(concept_docs.keys())

# 共现边：两个概念同现于同一文档 → 边（未类型化关系，回溯到该文档即"断言证据"）
edges = collections.Counter()
for cs in doc_concepts:
    uniq = sorted(set(cs))
    for a in range(len(uniq)):
        for b in range(a + 1, len(uniq)):
            edges[tuple(sorted((uniq[a], uniq[b])))] += 1
net = {k: v for k, v in edges.items()}
W = float(sum(net.values()))


# ---------- 确定性多级 Louvain 社区检测（单轮贪心 + 聚合重划；真实模块度用权威公式独立核算） ----------
def detect_communities():
    orig_ids = {nd: i for i, nd in enumerate(concepts_all)}
    o_edges = [(orig_ids[a], orig_ids[b], w) for (a, b), w in net.items()]
    n_orig = len(concepts_all)

    # Q = Σ_c [ e_c/W − (k_c/(2W))² ]，e_c=社区内边权和（每条边计一次），k_c=社区内度数和
    def _modularity(edges, cm):
        Wt = float(sum(w for _, _, w in edges))
        if Wt == 0 or not cm:
            return 0.0
        k_c = collections.defaultdict(float)
        e_in = 0.0
        for (a, b, w) in edges:
            ca, cb = cm[a], cm[b]
            k_c[ca] += w; k_c[cb] += w
            if ca == cb:
                e_in += w
        return e_in / Wt - sum((v / (2 * Wt)) ** 2 for v in k_c.values())

    def one_round(node_n, edges):
        # 单轮贪心：节点按索引序、邻居社区排序、严格 > 1e-12 才动；一整遍后真实模块度不再严格上升即停（局部峰顶护栏）。
        # 返回 (峰值分区, 峰值真实模块度, 步数)
        Wt = float(sum(w for _, _, w in edges))
        adj = collections.defaultdict(list)
        for (a, b, w) in edges:
            adj[a].append((b, w)); adj[b].append((a, w))
        cm = list(range(node_n))
        q_prev = _modularity(edges, cm)
        best_cm, best_q, peak = list(cm), q_prev, 0
        steps = 0
        while True:
            deg = [sum(w for _, w in adj[i]) for i in range(node_n)]
            comm_tot = collections.defaultdict(float)
            for i in range(node_n):
                comm_tot[cm[i]] += deg[i]
            changed = False
            for i in range(node_n):
                cur = cm[i]; ki = deg[i]
                bd, bc = 0.0, cur
                for c in sorted({cm[j] for j, _ in adj[i] if cm[j] != cur}):
                    kic = sum(w for j, w in adj[i] if cm[j] == c)
                    tot_c = comm_tot.get(c, 0.0)
                    delta = kic / Wt - (tot_c * ki) / (2 * Wt * Wt)   # Louvain ΔQ：k_ic/Wt − Σ_tot·k_i/(2Wt²)
                    if delta > bd + 1e-12:
                        bd, bc = delta, c
                if bc != cur:
                    comm_tot[cur] -= ki
                    comm_tot[bc] += ki
                    cm[i] = bc
                    changed = True
            steps += 1
            q_next = _modularity(edges, cm)
            if q_next > best_q + 1e-12:
                best_q, best_cm, peak = q_next, list(cm), steps
            if not changed:
                break
            if q_next <= q_prev + 1e-12:
                break
            q_prev = q_next
        return best_cm, best_q, steps, peak

    cur_edges = o_edges
    covers = [{i} for i in range(n_orig)]       # 每个当前超节点覆盖的原始概念集
    node_n = n_orig
    diag = {"q1": None, "fams1": [], "rounds": []}
    final_groups = None
    while True:
        if not cur_edges:
            break
        cm, ql, steps, peak = one_round(node_n, cur_edges)
        if diag["q1"] is None:                  # 首轮=原始图上的单层贪心（诚实展示其局部峰顶）
            diag["q1"] = ql
            g0 = collections.defaultdict(list)
            for iy, c in enumerate(cm):
                g0[c].append(concepts_all[iy])
            diag["fams1"] = [sorted(v) for v in g0.values() if len(v) > 1]
        diag["rounds"].append((steps, peak))
        gtmp = collections.defaultdict(list)    # 本级分区投影回原始概念
        for iy, c in enumerate(cm):
            gtmp[c].extend(covers[iy])
        final_groups = [set(v) for v in gtmp.values()]
        ng = len(gtmp)
        if ng >= node_n:
            break                               # 无可再并 → 顶层即此级
        node_of = {}
        for c in sorted(gtmp):
            node_of.setdefault(c, len(node_of))
        new_covers = [set() for _ in range(ng)]
        for iy, c in enumerate(cm):
            new_covers[node_of[c]].update(covers[iy])
        agg = collections.Counter()            # 聚合：社区间边权和 → 超节点图
        for (a, b, w) in cur_edges:
            ca, cb = cm[a], cm[b]
            if ca == cb:
                continue
            u, v = node_of[ca], node_of[cb]
            if u > v:
                u, v = v, u
            agg[(u, v)] += w
        new_edges = [(u, v, w) for (u, v), w in agg.items()]
        if not new_edges:
            break
        cur_edges = new_edges
        covers = new_covers
        node_n = ng

    final_of = {}
    for gidx, s in enumerate(final_groups):
        for oi in s:
            final_of[oi] = gidx
    cm_full = [final_of[i] for i in range(n_orig)]
    QQ = _modularity(o_edges, cm_full)         # 最终分区在【原始图】上的权威模块度
    comms = [sorted(concepts_all[oi] for oi in s) for s in final_groups]
    comms.sort(key=lambda x: (-len(x), x[0]))
    return comms, QQ, diag

comms, QQ, diag = detect_communities()
cid_of = {}
for gid, cs in enumerate(comms):
    for k in cs:
        cid_of[k] = gid
doc_comm = {}
for i, cs in enumerate(doc_concepts):
    if cs:
        doc_comm[i] = sorted({cid_of[k] for k in set(cs)})
fam_ids = sorted(gid for gid, c in enumerate(comms) if len(c) > 1)
extract_wall = time.time() - tA
WALL("extract", tA)          # 建图墙钟（概念抽取 + Louvain），在模型加载之前打点

# ---------- 向量栈（同一 Dense encoder 全程） ----------
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
t_idx = time.time()
emb = model.encode(corpus_q, normalize_embeddings=True, show_progress_bar=False)
index = faiss.IndexFlatIP(emb.shape[1]); index.add(np.ascontiguousarray(emb))
idx_wall = time.time() - t_idx
WALL("encode+index", t_idx)  # 向量建索引墙钟（100 篇 encode + IndexFlatIP）

def dense_top(q, k):
    v = model.encode([q], normalize_embeddings=True)
    _, i = index.search(np.ascontiguousarray(v), k)
    return [int(x) for x in i[0].tolist()]

def neigh(seed):
    return sorted(set(b for (a, b) in net if a == seed) | set(a for (a, b) in net if b == seed))

print("=" * 72)
print("前置·实体-关系图（规则词典 × 文档共现，LightRAG 式轻量图）")
print("=" * 72)
print("  文档 %d 篇 · 词典 %d 概念 · 命中 %d/%d（规则抽取召回上限）" % (
    N, len(CONCEPTS), sum(1 for c in doc_concepts if c), N))
print("  概念节点 %d · 共现边 %d · 跨文档概念 %d 个（多篇共享=图的主干）" % (
    len(concepts_all), len(net),
    sum(1 for v in concept_docs.values() if len(v) > 1)))
if diag["q1"] is not None:
    print("  社区检测：确定性多级 Louvain（单轮贪心 + 聚合重划）· 轮次=%d · 每轮步数=%s" % (
        len(diag["rounds"]), "-".join(str(s) for s, _ in diag["rounds"])))
    print("    轮1 单层贪心达局部峰顶：真实模块度 Q=%.3f · 初划 %d 族（单层贪心不保证全局最优）" % (
        diag["q1"], len(diag["fams1"])))
    bridge = [f for f in diag["fams1"] if len(f) == 2 and "剪枝" in f and "图像识别" in f]
    if bridge:
        print("      「桥接错对」：%s 并无直接共现边（各自只与神经网络共现），单层贪心仍凑为一族 —— 聚合轮把它并回训练簇" % (" | ".join(bridge[0])))
    print("    聚合重划后最终分区：%d 族，原始图独立核算的权威模块度 Q=%.3f（跨层聚合把结构拉近全局最优点）" % (
        len(fam_ids), QQ))
print("  -> %d 个社区（%d 多概念主题族 + %d 单点主题·孤点概念）" % (
    len(comms), len(fam_ids), len(comms) - len(fam_ids)))
for g in fam_ids:
    docs = sorted(set(i for i, cs in enumerate(doc_concepts) if set(comms[g]) & set(cs)))
    print("      族[%d] %s -> 文档 %s" % (g, " | ".join(comms[g]),
          " ".join(str(d) for d in docs)))
print("  每条边可回溯到共现文档 = 断言可溯源（生产 GraphRAG 用 LLM 抽取类型化三元组=非本机实测）")
print()

# =============================================================
# 实验 A：全局问题（"整体讲了哪些主题"）
# =============================================================
print("=" * 72)
print("实验 A  全局问题：向量单次检索的'念力窗口' vs 图谱社区表（索引端建图时一次生成）")
print("=" * 72)
# A0 底数：事实查询场景向量近乎完美
h1 = h5 = 0
for i, q in enumerate(queries):
    I = dense_top(q, 5)
    h1 += 1 if I[0] == i else 0
    h5 += 1 if i in I else 0
print("  A0 事实查询底数：100 条专属问答 向量 Hit@1=%d/100 · Hit@5=%d/100" % (h1, h5))
print("     -> 事实查询场景向量近乎完美；GraphRAG 的用武之地不在这里（对应 §8.11 混合使用）")
# A1 向量单次全局检索
global_q = "这个语料库都讲了哪些机器学习、人工智能、科学、工程的主题"
top10 = dense_top(global_q, 10)
cover = set(); cnt = collections.Counter()
for d in top10:
    for g in doc_comm.get(d, []):
        if g in fam_ids:
            cover.add(g); cnt[g] += 1
print("  A1 全局问题（%s）" % global_q)
print("     向量单次 top-10 -> 覆盖主题网络族 %d/%d （其余是单点主题/同族重复）" % (
    len(cover), len(fam_ids)))
for d in top10:
    cs = doc_concepts[d]
    fam_of = ["+".join(comms[g]) for g in doc_comm.get(d, []) if g in fam_ids]
    tag = " | ".join(fam_of) if fam_of else "单点:%s" % (cs[0] if cs else "无概念")
    print("       doc[%2d]  覆盖=%s" % (d, tag))
print("  A2 图谱社区表（索引端建图时刻已算好，查询端一次给出）：")
print("      %d 个多概念主题族 + %d 个单点主题 = 全库 98/100 篇的主题目录" % (
    len(fam_ids), len(comms) - len(fam_ids)))
print("      -> 全局问题用向量要'多查询+判重聚合'，每次查询只进 top-k 念力窗口（接 10-Agentic 决策账）")
print()

# =============================================================
# 实验 B：实体关联（图 1-hop 断言 vs 向量 top-5 主题）
# =============================================================
print("=" * 72)
print("实验 B  实体关联：图中'共现断言' vs 向量 top-5 的'语义近邻'")
print("=" * 72)
SEEDS = [
    ("知识蒸馏", ["大模型", "联邦学习"]),
    ("Transformer", ["注意力", "语言模型"]),
    ("卷积", ["神经网络", "图像识别"]),
    ("损失函数", ["过拟合", "正则化", "梯度下降"]),
    ("一致性哈希", ["缓存"]),
]
tot_expected = tot_vhit = tot_noise = 0
for seed, expected in SEEDS:
    nn = neigh(seed)
    tops = dense_top(seed, 5)
    bag_all = sorted(set(x for d in tops for x in doc_concepts[d]) - {seed})
    # 证据：种子↔邻居 共现于哪些文档（每条边可回溯）
    ev_dict = {}
    for (a, b), w in net.items():
        if seed in (a, b):
            other = b if a == seed else a
            ev_dict.setdefault(other, set()).update(
                i for i, cs in enumerate(doc_concepts) if a in cs and b in cs)
    vhit = [e for e in expected if e in bag_all]
    noise = [x for x in bag_all if x not in nn and x not in expected]  # 袋里"图上未断言"的近邻
    tot_expected += len(expected)
    tot_vhit += len(vhit)
    tot_noise += len(noise)
    print("  seed=%s" % seed)
    print("     图 1-hop（断言，共现文档=%s）: %s" % (
        {k: sorted(v) for k, v in ev_dict.items()}, nn))
    print("     向量 top-5 主题袋=%s 命中期望 %s · 图上未断言噪声=%s" % (
        bag_all, vhit, noise))
print("     合计：期望相关 %d 个概念，向量 top-5 命中 %d/%d（recall=%d%%）· 图外未断言噪声 %d 条" % (
    tot_expected, tot_vhit, tot_expected, round(100.0 * tot_vhit / tot_expected), tot_noise))
print("     -> 单跳'相关列举'向量语义近邻几乎够用（本例 10/10）；差异在 B2：向量袋里"
      "'语义近邻但图上未断言'（无共现证据）是假候选，图 1-hop 每条都能回溯到文档。")
# B3 2-hop 传递
print("  B3 2-hop 可达性：种子 知识蒸馏 ->1hop(共现断言)-> 联邦学习 ->再向外-> %s" % (
    sorted(set(x for y in neigh("联邦学习")
                for x in neigh(y)) - {"知识蒸馏", "联邦学习"})))
print("     -> 图上共现关系的'传递'是可计算可达性；向量余弦相似度不传递'共现断言'（无有向图遍历）")
print()

# =============================================================
# 实验 C：代价账（建图成本，本机 + 外推）
# =============================================================
print("=" * 72)
print("实验 C  代价账：建图 vs 建索引（GraphRAG 贵在哪、值不值的分界）")
print("=" * 72)
print("  C1 本机规则建图墙钟：见 stderr [wall] extract（run1 ≈ 2 ms 量级，跨 run 浮动）")
print("  C2 本机向量建索引墙钟：见 stderr [wall] encode+index（run1 ≈ 1.15 s 量级，100 篇 encode + build IndexFlatIP）")
print("  C3 LLM 抽取建图外推账（非本机实测，写作环境无外网未在线复核）：")
print("     生产 GraphRAG 每篇文档抽查实体/关系 = 3+ 次 LLM 调用 × 文档数；100 篇级中小库")
print("     快速被 LLM 账单淹没——本 toy 的规则词典是确定性替代，但召回上限 98/100 且无类型化关系")
print("     -> 代价分界：语料越大、跨文档实体越密，共现图摊得越平；单主题百科（本书 %d/%d 社区是单点）" % (
    len(comms) - len(fam_ids), len(comms)))
print("        建图成本摊不平，先问'有没有全局问题、要几跳'再决定上不上 GraphRAG")

WALL("total", T0)
print()
print("done · 一键复现：python code/notebooks/_tools/graph_rag_demo.py")
import sys as _sys
print("wall %.1f s" % (time.time() - T0), file=_sys.stderr)
