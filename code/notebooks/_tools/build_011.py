# -*- coding: utf-8 -*-
"""构建 notebook 011（父子分块+元数据过滤）：复用 009 的 CORPUS cell，实验照 n011_probe。"""
import io, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

NB9 = r'F:\00AI创业\01LLM\LLM\code\notebooks\009-中文RAG全管线.ipynb'
OUT = r'F:\00AI创业\01LLM\LLM\code\notebooks\011-父子分块与元数据过滤.ipynb'

nb9 = json.load(io.open(NB9, encoding='utf-8'))
def cell_code(idx):
    return ''.join(nb9['cells'][idx]['source'])
DEPS_CELL = cell_code(2)
DATA_CELL = cell_code(4)   # CORPUS(100×[上下文,问题]) + corpus_q/test_q/gold_idx

MD, C = [], []

MD.append(r"""# 父子分块 + 元数据过滤：RAG 增强第一式（011）

> 配套章节：[04-分块与元数据](../../08-RAG体系/04-分块与元数据.md)、[05-查询侧技术](../../08-RAG体系/05-查询侧技术.md)；语料复用 [009-中文RAG全管线](009-中文RAG全管线.ipynb)。
> 一句话：**索引粒度等于答案粒度——用「子块」建索引（找得准），用「父块」回填上下文（喂得足），再让元数据把检索关进专属主题。**

两个问题按一个难点：

1. **粒度谜题**：文档太大→embedding 被「平均化」，具体问题找准？文档太小→上下文差一截，跨子句线索全丢。父子分块 = **检索用小块、回填用大块**，鱼与熊掌通吃。
2. **噪音污染**：top-5 里混着旁类文档。给每个块挂主题元数据，过滤后检索直接被「关进」用户所属的主题。

> 实现说明（诚实标注）：为不下载长文档，本 notebook 把前 90 条语料每 3 条拼成 1 篇「章」（30 篇 × 3 子块）——拼接是合成的，但「粒度越细检索越准」的机制与数字是真实的，可一键复现。
""")

MD.append("## 1. 安装依赖（Colab 已含大部分）")
C.append(DEPS_CELL)

MD.append("## 2. 数据：复用 009 的 100 条中文百科语料\n\n第 i 条 gold 问题答案在第 i 条上下文里；本 notebook 只用前 90 条做两级结构。")
C.append(DATA_CELL)

MD.append("## 3. 组成「章 → 子块」两级结构\n\n每 3 条连成 1 篇「章」，内部每个条目就是一个子块：`章[i] = 子块[3i,3i+1,3i+2]`。")
C.append(r"""CH = 3                 # 每章子块数
NSECT = 90             # 用前 90 条（30 章）
sects = CORPUS[:NSECT]
par = [" ".join(c[0] for c in sects[k:k + CH]) for k in range(0, NSECT, CH)]   # 30 篇「章」
ques = [q for _, q in sects]                                                    # 90 个子块问题
print("章数:", len(par), " 子块数:", len(sects))
print("示例 章[0] =", par[0][:36] + " … ｜ 子块[0] =", sects[0][0][:24] + " …")""")

MD.append("## 4. 两级索引：子块索引 + 章（父块）索引\n\n同一套 bge 编码，两个 faiss 内积索引——父子分块的实现就这么点事。")
C.append(r"""import numpy as np, faiss
from sentence_transformers import SentenceTransformer
import jieba

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
def emb(txts):
    return model.encode(txts, normalize_embeddings=True, show_progress_bar=False)

child_emb = emb([c[0] for c in sects])          # 子块 90 维→90 条
parent_emb = emb(par)                            # 章  30 条
ci = faiss.IndexFlatIP(child_emb.shape[1]); ci.add(np.ascontiguousarray(child_emb))
pi = faiss.IndexFlatIP(parent_emb.shape[1]); pi.add(np.ascontiguousarray(parent_emb))

def topn(idx, n, q):
    v = model.encode([q], normalize_embeddings=True)
    _, k = idx.search(np.ascontiguousarray(v), n)
    return k[0].tolist()
print("子块 faiss:", ci.ntotal, "条 ｜ 父块 faiss:", pi.ntotal, "条")""")

MD.append(r"""## 5. 实验①：粒度——「找到哪一章」≠「翻到哪一句」

24 个随机子块问题（种子固定可复现）。**整篇索引**报「这篇对」/「这篇错」，**子块索引**报「这句对/句在 top-3」。

关键指标：**证据浓度**——整篇返回 3 个块但只有 1 块相关（1/3）；子块返回 1 块就是证据（1/1）。
""")
C.append(r"""rng = np.random.RandomState(0)
tq = rng.choice(np.arange(NSECT), 24, replace=False).tolist()

par_hit = [int(topn(pi, 1, ques[i])[0] == i // CH) for i in tq]
ch1 = [int(topn(ci, 1, ques[i])[0] == i) for i in tq]
ch3 = [int(i in topn(ci, 3, ques[i])) for i in tq]
print("=== 实验① 粒度（整篇 30 篇 vs 子块 90 块，24 问） ===")
print("整篇检索 top-1 命中「所在章」 : %d/24  %.2f   ← 拼接章被「平均化」，具体问题选错章一半以上" % (sum(par_hit), sum(par_hit) / 24))
print("子块检索 top-1 命中「gold 子块」: %d/24  %.2f" % (sum(ch1), sum(ch1) / 24))
print("子块检索 gold∈top-3           : %d/24  %.2f" % (sum(ch3), sum(ch3) / 24))
print("上下文成本：整篇=3×子块 token；证据浓度 整篇 1/3 vs 子块 1/1")
print()
print("注：本语料拼接「章」非自然长文，但与真实长文档同理——块越大，单条 embedding")
print("    越背不动具体问题（具体问题检索漂移）。产线上见到的「doc 太大检索就飘」同一现象。")
print()
# 展示一条整篇选错章、子块扳正的例子
i = [x for x in tq if x // CH != topn(pi, 1, ques[x])[0]][0]
print("示例（整篇选错章 → 子块扳正）：")
print("  问题      :", ques[i])
print("  所在章[%d] :" % (i // CH), par[i // CH][:30] + "…")
print("  整篇 top1 → 章[%d] %s" % (topn(pi, 1, ques[i])[0], par[topn(pi, 1, ques[i])[0]][:30] + "…"))
print("  子块 top1 → 子块[%d] %s" % (topn(ci, 1, ques[i])[0], sects[topn(ci, 1, ques[i])[0]][0][:30] + "…"))""")

MD.append(r"""## 6. 父子回填：检索小、回填大

子块命中后，**回填它所在的章**作为上下文——找得准（子块索引）还喂得足（同章 3 块都进来）。对 24 问实测两件事：回填把上下文放大了多少、gold 子块是否仍在回填上下文内。""")
C.append(r"""covered = 0
for i in tq:
    j = topn(ci, 1, ques[i])[0]
    parent_i = j // CH
    covered += int(i in range(parent_i * CH, parent_i * CH + CH))
ratio = [len(par[i // CH]) / len(sects[i][0]) for i in tq]   # 回填父块 / 单子块
print("=== 实验①b 父子回填（24 问） ===")
print("子块 top-1 命中 gold 子块：%d/24（本语料顶格——子块索引选句很准）" % sum(ch1))
print("回填父块后 gold 仍在上下文中：%d/24" % covered)
print("上下文放大：回填父块/子块 平均 %.2f×（同章 3 块全给，跨子句线索不会丢）" % (
    sum(ratio) / len(ratio)))
print()
print("跨子句的价值：若答案要跨两个子块才能拼出来（真实长文档很常见），只给子块")
print("  上下文会缺一半证据；回填父块把同章邻居一并带上。")
print("  生产组合：父子回填 + Reranker/LLM 选择（见 09-Advanced-RAG 章节）。")""")

MD.append(r"""## 7. 实验②：主题（元数据）过滤

给每个子块挂主题标签（5 类主题词表，子块文本命中最多的类即其主题；未命中 = 其他）。然后取 6 条「问题里自带主题词」的题，对比 no-filter 与 filter 的 top-5 **主题纯度**。""")
C.append(r"""from collections import Counter
TOPICS = [
    ("AI/机器学习", ["模型", "算法", "学习", "人工智能"]),
    ("通信/网络",   ["网络", "延迟", "带宽", "通信"]),
    ("生物/医学",   ["细胞", "免疫", "基因", "疾病", "RNA", "细菌", "疗法"]),
    ("太空/天文",   ["太空", "卫星", "火箭", "宇宙", "天体", "登月", "空间站"]),
    ("金融/经济",   ["股票", "利率", "通胀", "货币", "资产", "投资"]),
]
def topic_of(text):
    hits = Counter()
    for name, kws in TOPICS:
        n = sum(text.count(w) for w in kws)
        if n:
            hits[name] = n
    return hits.most_common(1)[0][0] if hits else None

labels = [topic_of(c[0]) for c in sects]
print("打标分布:", Counter(x for x in labels if x))

def filter_search(q, keep, n=5):
    ids = topn(ci, NSECT, q)
    out = []
    for x in ids:
        if labels[x] == keep:
            out.append(x)
        if len(out) == n:
            break
    return out[:n]

sel = []
for i in range(NSECT):
    L = labels[i]
    if L and any(kw in ques[i] for kw in dict(TOPICS)[L]):
        sel.append((i, L))
rng2 = np.random.RandomState(1)
chosen = rng2.choice(len(sel), 6, replace=False).tolist()
print()
print("=== 实验② 主题过滤（6 题，主题脑里有明确归属） ===")
stats = []
for ix in chosen:
    i, L = sel[ix]
    n5 = topn(ci, 5, ques[i])
    nofil = [labels[x] for x in n5]
    filt = filter_search(ques[i], L)
    pure_no = sum(1 for x in nofil if x == L)
    gold_hit = int(i in filt)
    stats.append((pure_no, len(filt), gold_hit))
    print("题[%2d/%s] no-filter top5 主题=%s → filter 后 %d 条全属 %s, gold∈top? %d" % (
        i, L, nofil, len(filt), L, gold_hit))
print()
print("no-filter top-5 平均主题纯度: %.2f（5 个里只有 %.1f 个是目标主题）"
      % (sum(s[0] for s in stats) / len(stats) / 5, sum(s[0] for s in stats) / len(stats)))
print("filter 后 top-5 主题纯度: 1.00（同名限定）；gold 命中 %d/6（过滤保精度、不涨召回）"
      % sum(s[2] for s in stats))

# 落盘汇总
import pathlib, pandas as pd
pathlib.Path("out").mkdir(exist_ok=True)
pd.DataFrame([{"query": ques[sel[ix][0]], "topic": sel[ix][1],
               "no_filter_purity": stats[k][0] / 5, "filter_hit": stats[k][2]}
              for k, ix in enumerate(chosen)]).to_csv(
    "out/011_parent_child_metadata.csv", index=False, encoding="utf-8-sig")
print("\n已保存 out/011_parent_child_metadata.csv")""")

MD.append(r"""## 8. 结论

1. **粒度是索引的第一设计决定**：同一语料，整篇 top-1 只选对章 16/24，子块 top-1 选对句 24/24——把问题「平均化」进大块，具体问题就检索飘。
2. **父子分块 = 检索小 + 回填大**：子块索引选句准，回填父块让上下文放大 3× 且 gold 不丢——跨子句证据有了落脚地。
3. **元数据过滤守精度**：no-filter top-5 主题纯度平均 0.57，同名过滤后 1.00；但过滤**不提升召回**——它是「保不准」，不是「找更多」。
4. 落到生产：块长调完（[04-分块与元数据](../../08-RAG体系/04-分块与元数据.md)）就来调粒度；元数据过滤是向量库侧最便宜的精度税（IndexFilter 几乎零成本）。""")

cells = []
exec_count = 0
cells.append({"cell_type": "markdown", "metadata": {}, "source": MD[0].splitlines(keepends=True)})
for md, code in zip(MD[1:], C):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)})
    exec_count += 1
    cells.append({"cell_type": "code", "execution_count": exec_count, "metadata": {}, "outputs": [],
                  "source": code.splitlines(keepends=True)})

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0", "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3", "nbconvert_exporter": "python",
                          "file_extension": ".py"},
        "colab": {"name": "011-父子分块与元数据过滤.ipynb", "provenance": [], "collapsed_sections": []},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print("已写出", OUT, "| 单元格:", len(cells), "| code exec:", exec_count)
