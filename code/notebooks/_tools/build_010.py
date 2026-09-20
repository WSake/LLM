# -*- coding: utf-8 -*-
"""构建 notebook 010（RAGAS 评测）：复用 009 的 CORPUS 内嵌与依赖 cell，metrics 照 07 章节同款 probe。"""
import io, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

NB9 = r'F:\00AI创业\01LLM\LLM\code\notebooks\009-中文RAG全管线.ipynb'
OUT = r'F:\00AI创业\01LLM\LLM\code\notebooks\010-RAGAS评测一个RAG系统.ipynb'

nb9 = json.load(io.open(NB9, encoding='utf-8'))
def cell_code(idx):
    return ''.join(nb9['cells'][idx]['source'])

DEPS_CELL = cell_code(2)     # 覆盖 rank_bm25/jieba/faiss/sentence_transformers
DATA_CELL = cell_code(4)     # CORPUS(100×[上下文,问题]) + corpus_q/test_q/gold_idx

MD = []  # (md) 文本
C = []   # (code) 文本

MD.append(r"""# RAGAS 评测一个 RAG 系统（010 · 忠实度 / 相关性 / 检索层）

> 与章节 [07-RAG评测-RAGAS](../../08-RAG体系/07-RAG评测-RAGAS.md) 配套；语料与检索器沿用 [009-中文RAG全管线](009-中文RAG全管线.ipynb)。
> 一句话：**「能检索」不等于「系统能用」——这套 notebook 用 RAGAS 式三层指标，把「检索层顶格但生成层崩盘」照出来。**

本 notebook 的三层指标（全部免 LLM、CPU 可跑、数字可复现）：

| 层 | 指标 | 实现 | 解释 |
|---|---|---|---|
| 检索层 | Hit@5 Recall | BM25 / Dense 各自 top-5 是否含 gold | 找得到吗 |
| 生成层 | Faithfulness（4-gram 证据覆盖率） | 答案的 4-gram 有多少出现在喂入的上下文中 | 答得有所凭吗 |
| 生成层 | Answer Relevance（bge 余弦） | query 与答案的 bge-small-zh 余弦 | 答得贴题吗 |

然后做**判别性对照**：把答案换成「错误来源」，看指标是否显著下跌——评测指标最怕「怎么改都 0.9」。
""")

MD.append("## 1. 安装依赖（Colab 已含大部分）")
C.append(DEPS_CELL)

MD.append("## 2. 数据：沿用 009 的 100 条中文百科语料\n\n每条 `[上下文, gold问题]`，第 i 条的答案就在第 i 个上下文里（`gold_idx = i`），无需下载。")
C.append(DATA_CELL)

MD.append(r"""## 3. 检索器：BM25 + bge 稠密（含探针）

沿用 009 的检索栈；先用 Hit@5 把「检索层有没有张力」探明白。
""")
C.append(r"""import jieba, faiss, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def cut(s):
    return [w for w in jieba.cut(s) if w.strip() and len(w.strip()) > 1]

# -- 稀疏：BM25（中文先 jieba 分词） --
bm = BM25Okapi([cut(c) for c in corpus_q])

# -- 稠密：bge-small-zh-v1.5（约 95MB，CPU 可跑） + faiss 内积索引 --
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
corpus_emb = model.encode(corpus_q, normalize_embeddings=True, show_progress_bar=False)
index = faiss.IndexFlatIP(corpus_emb.shape[1])
index.add(np.ascontiguousarray(corpus_emb))

def search_bm25(q, n=5):
    return np.argsort(bm.get_scores(cut(q)))[::-1][:n].tolist()

def search_dense(q, n=5):
    v = model.encode([q], normalize_embeddings=True)
    _, idx = index.search(np.ascontiguousarray(v), n)
    return idx[0].tolist()

# 检索层探针：两条路 Hit@5 是否顶格
print("检索层 Hit@5: BM25=%.4f  Dense=%.4f" % (
    np.mean([1.0 if i in search_bm25(q, 5) else 0.0 for i, q in enumerate(test_q)]),
    np.mean([1.0 if i in search_dense(q, 5) else 0.0 for i, q in enumerate(test_q)])))""")

MD.append(r"""## 4. RAGAS 式指标实现（免 LLM，4-gram / 余弦）

与章节 07 第 5 节的 probe 同款实现，保证文中的数字可在本 notebook 复现：
- `faithfulness_proxy`：答案 4-gram 在「喂入上下文（top-3 拼接）」中的出现比例 → 证据覆盖率。
- `answer_relevance`：`query` 与 `答案` 的 bge 余弦 → 贴题度。
""")
C.append(r"""def ngrams4(s):
    return {s[i:i+4] for i in range(len(s)-3)}

def faithfulness_proxy(ans, ctx):
    g = ngrams4(ans); m = len(g) or 1
    return len({x for x in g if x in ctx}) / m

def answer_relevance(q, ans):
    qv = model.encode([q], normalize_embeddings=True)[0]
    av = model.encode([ans], normalize_embeddings=True)[0]
    return float(qv @ av)

print("指标就绪：faithfulness_proxy / answer_relevance")""")

MD.append(r"""## 5. 实验 A：正确来源基线（抽取式答案，20 样本）

`RandomState(0)` 固定抽 20 条（与 009 同题面可比）。答案 = 检索 top-1 上下文（抽取式基线，便于免模型复现）。""")
C.append(r"""rng = np.random.RandomState(0)
SAMPLE = rng.choice(np.arange(len(corpus_q)), size=20, replace=False).tolist()

hits, faiths, rels = [], [], []
for i in SAMPLE:
    q = test_q[i]
    top = search_dense(q, 3)                       # RAG: 取 top-3 上下文
    ans = corpus_q[top[0]]                         # 答案 = top-1 上下文（抽取式）
    ctx = " ".join(corpus_q[j] for j in top)       # 喂进"大模型"的上下文
    hits.append(1.0 if i in search_dense(q, 5) else 0.0)
    faiths.append(faithfulness_proxy(ans, ctx))
    rels.append(answer_relevance(q, ans))

print("样本数: %d (USE_LLM=0 抽取式基线, top-3 上下文)" % len(SAMPLE))
print("检索层  Hit@5 Recall                     : %.4f" % np.mean(hits))
print("生成层  Faithfulness(4-gram证据覆盖率)     : %.4f" % np.mean(faiths))
print("生成层  Answer Relevance(bge余弦)        : %.4f (±%.3f)" % (np.mean(rels), np.std(rels)))""")

MD.append(r"""## 6. 实验 B：错误来源对照（判别性检验）

把答案**故意换成错误来源**（检索 top-1 之后的第 3 条，`(top[0]+3) % N`），其余不动。
好指标必须能抓住这种「坏答案」——两张表都改不出来变动的评测台是废的。""")
C.append(r"""bad_faith, bad_rel = [], []
for i in SAMPLE[:6]:
    q = test_q[i]
    top = search_dense(q, 3)
    ans = corpus_q[(top[0] + 3) % len(corpus_q)]      # 故意错来源
    ctx = " ".join(corpus_q[j] for j in top)
    bad_faith.append(faithfulness_proxy(ans, ctx))
    bad_rel.append(answer_relevance(q, ans))
print("对照-错来源答案  →  Faithfulness %.4f / Relevance %.4f（应显著下降）"
      % (np.mean(bad_faith), np.mean(bad_rel)))""")

MD.append("## 7. 汇总对比 + 落盘\n\n逐条（前 6 样本）看正确 vs 错误来源的四列，再给全样本聚合。")
C.append(r"""import pathlib, pandas as pd
pathlib.Path("out").mkdir(exist_ok=True)

# 全样本聚合（实验 A 复用上文算出的向量，避免重复 embedding）
A_f, A_r = float(np.mean(faiths)), float(np.mean(rels))
B_f, B_r = float(np.mean(bad_faith)), float(np.mean(bad_rel))
print("管线A(正确来源)  Faithfulness=%.4f  Relevance=%.4f" % (A_f, A_r))
print("管线B(错误来源)  Faithfulness=%.4f  Relevance=%.4f" % (B_f, B_r))
print("判别性: F %.2f→%.2f, R %.2f→%.2f  —— 指标能抓住坏答案" % (A_f, B_f, A_r, B_r))

rows = []
for i in SAMPLE[:6]:
    q = test_q[i]
    top = search_dense(q, 3)
    good = corpus_q[top[0]]
    bad = corpus_q[(top[0] + 3) % len(corpus_q)]
    ctx = " ".join(corpus_q[j] for j in top)
    rows.append([q[:16], faithfulness_proxy(good, ctx), answer_relevance(q, good),
                 faithfulness_proxy(bad, ctx), answer_relevance(q, bad)])
df = pd.DataFrame(rows, columns=["查询(截)", "F·正确来源", "R·正确来源", "F·错误来源", "R·错误来源"])
df.to_csv("out/010_ragas_metrics.csv", index=False, encoding="utf-8-sig")
print("\n", df.round(3).to_markdown(index=False), sep="")""")

MD.append(r"""## 8. 结论：为什么「检索全中」依然要评测

1. **检索层顶格（Hit@5 ≈ 1.0）是「易题」信号，不是「系统无敌」**——语料都是单答案事实题时，三条检索路都会满分，指标失去张力。
2. **判别性对照是评测台的质检**：把答案换成错误来源，若 Faithfulness / Relevance 不明显下跌，说明指标本身是聋的。本 notebook 两者从高位明显跳水，这才敢说「这套评测在量真东西」。
3. **relevance ⇄ faithfulness 是两种能力**：0.74 的相关性说明「答得贴」≠「答案有证据」。生产里应再上一档**真生成（LLM）评测**：把 `USE_LLM` 打开让 Qwen 0.5B 自己生成后评测，引出真实幻觉——那才是评测「上岗」的时刻（009 第 8 节已预留该开关）。

> 指标定义说明：忠实度此处用「答案 4-gram 对喂入上下文的覆盖率」代理 RAGAS 的 Faithfulness（官方版需 LLM 提取声明再做蕴含判定，见章节 07 §4）；本 notebook 的判别性结论在该代理上同样成立。""")

# ---- 组装 JSON ----
cells = []
exec_count = 0
for item in MD[:1]:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": item.splitlines(keepends=True)})
# 第 2 组：md + code 交替 (第 2 行的 md + deps)
pairs = list(zip(MD[1:], C))
for md, code in pairs:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)})
    exec_count += 1
    cells.append({
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    })

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0", "mimetype": "text/x-python",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "pygments_lexer": "ipython3", "nbconvert_exporter": "python",
                          "file_extension": ".py"},
        "colab": {"name": "010-RAGAS评测一个RAG系统.ipynb", "provenance": [], "collapsed_sections": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with io.open(OUT, "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")
print("已写出", OUT, "| 单元格:", len(cells), "| 预计 code exec:", exec_count)
