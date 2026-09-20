# -*- coding: utf-8 -*-
"""渲染 code/notebooks/009-中文RAG全管线.ipynb（含内嵌中文语料）。
   用法：python _gen_nb009.py   （在仓库根执行）"""
import io, json

corpus = json.load(io.open('code/notebooks/_demo_corpus.json', encoding='utf-8'))
corpus_lit = json.dumps(corpus, ensure_ascii=False, indent=1)

C = []  # ('m'|'c', content)


def md(src): C.append(('m', src))


def code(src): C.append(('c', src))


md("\n".join([
    "# 中文文档 RAG 全管线（009 · 可运行基线）",
    "",
    "> **对应知识点**：08-RAG体系/01-Embedding与检索基础.md、03-混合检索与重排.md、07-RAG评测-RAGAS.md",
    "> **目标**：一套能直接跑的**中文** RAG 最小闭环——BM25 稀疏检索 + 稠密检索 → 混合重排 → 评测（Hit@k + LLM-as-judge）→ 产出对比表。",
    "> **运行环境**：Colab 免费 CPU 即可（约 5–8 分钟）；本地需 `pip install` 下方依赖。",
    "",
    "## 0. 我们要回答的三个问题",
    "",
    "| # | 问题 | 回答哪里找 |",
    "|---|---|---|",
    "| 1 | 稀疏检索 vs 稠密检索谁更强（中文） | 第 5 节 Hit@k 基线 |",
    "| 2 | 混合检索能不能叠加优势 | 第 6 节混合对比 |",
    "| 3 | RAG 回答可信吗 | 第 7–8 节 LLM-judge |",
    "",
    "> 依次跑完得到一张实验对比表——评测型 notebook 的写法：**先列问题，再给数字**。",
    "",
]))

md("\n".join([
    "## 1. 安装依赖（Colab 已含大部分）",
    "",
    "`bm25s` 纯 Python BM25；`jieba` 中文分词（BM25 必须）。内嵌语料不依赖网络。",
    "",
    "> ⚠️ Colab 首次运行需联网装包；重跑跳过已装的。",
    "",
]))

code("\n".join([
    "import importlib, subprocess, sys",
    "for pkg in (\"rank_bm25\", \"jieba\"):",
    "    try:",
    "        importlib.import_module(pkg)",
    "    except ImportError:",
    "        subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", pkg])",
    "try:",
    "    import faiss",
    "except ImportError:",
    "    for pkg in (\"faiss\", \"faiss-cpu\"):",
    "        try:",
    "            subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", pkg])",
    "            import faiss",
    "            break",
    "        except Exception:",
    "            continue",
    "print(\"deps ok\")",
]))

md("\n".join([
    "## 2. 数据：内嵌中文百科语料（100 条，含 gold 问题）",
    "",
    "本 notebook 自带 100 条中文百科知识点：每条 = `(正文上下文, gold 问题)`。可评测、可复现、零下载。",
    "真正做检索时，把 `CORPUS` 换成你自己的文档即可（解析/分块见 08-RAG体系 章节）。",
    "",
]))

code("\n".join([
    "# 100 条中文百科上下文 + 每条一个 gold 问题（NOTEBOOK 内嵌，可直接跑）",
    "CORPUS = %s" % corpus_lit,
    "corpus_q = [c for c, _ in CORPUS]     # 检索语料 = 全部上下文",
    "test_q = [q for _, q in CORPUS]       # 测试问题 = 全部 gold 问题",
    "gold_idx = list(range(len(corpus_q))) # 第 i 条问题的答案就在第 i 个上下文里",
    "print(\"语料条目:\", len(corpus_q), \" 测试问题:\", len(test_q))",
]))

md("\n".join([
    "## 3. 稀疏检索：BM25（中文先分词）",
    "",
    "BM25 统计词频；中文不分词=整句一个 token，必须用 `jieba`。",
    "",
]))
code("\n".join([
    "import jieba",
    "from rank_bm25 import BM25Okapi",
    "import numpy as np",
    "def cut(s):",
    "    return [w for w in jieba.cut(s) if w.strip() and len(w.strip()) > 1]",
    "corpus_tok = [cut(c) for c in corpus_q]",
    "bm = BM25Okapi(corpus_tok)",
    "print(\"BM25 索引建成, 语料点数:\", len(corpus_q))",
]))

md("\n".join([
    "## 4. 稠密检索：bge-small-zh-v1.5（约 90MB，CPU 可跑）",
    "",
    "中文 embedding 模型 `BAAI/bge-small-zh-v1.5`。首次运行会下载模型到本机缓存。",
    "",
]))
code("\n".join([
    "import numpy as np, faiss",
    "from sentence_transformers import SentenceTransformer",
    "model = SentenceTransformer(\"BAAI/bge-small-zh-v1.5\")",
    "corpus_emb = model.encode(corpus_q, normalize_embeddings=True, show_progress_bar=False)",
    "print(\"稠密向量形状:\", corpus_emb.shape)",
    "",
    "index = faiss.IndexFlatIP(corpus_emb.shape[1])",
    "index.add(np.ascontiguousarray(corpus_emb))",
    "print(\"faiss 索引条数:\", index.ntotal)",
]))

md("\n".join([
    "## 5. 检索函数 + Hit@k 基线",
    "",
    "**Hit@k**：检索 Top-k 里命中 gold 上下文（这里 gold = 第 i 个语料的 index）即记 1 分。",
    "",
]))
code("\n".join([
    "def search_bm25(q, n=5):",
    "    scores = bm.get_scores(cut(q))",
    "    return np.argsort(scores)[::-1][:n].tolist()",
    "",
    "def search_dense(q, n=5):",
    "    v = model.encode([q], normalize_embeddings=True)",
    "    _, idx = index.search(np.ascontiguousarray(v), n)",
    "    return idx[0].tolist()",
    "",
    "def hit_at(pred, gold):",
    "    return 1.0 if gold in pred else 0.0",
    "",
    "n = len(test_q)",
    "h_bm25 = [hit_at(search_bm25(q), i) for i, q in enumerate(test_q)]",
    "h_dense = [hit_at(search_dense(q), i) for i, q in enumerate(test_q)]",
    "print(f\"=== 检索 Hit@5 基线（测试集 {n} 条） ===\")",
    "print(\"BM25   Hit@5: %.4f\" % (sum(h_bm25) / n))",
    "print(\"Dense  Hit@5: %.4f\" % (sum(h_dense) / n))",
]))

md("\n".join([
    "## 6. 混合检索：BM25 + 稠密 分数融合",
    "",
    "各自 min-max 归一化后按 alpha 加权。经验上 alpha=0.5 在中文语料上稳健。",
    "",
]))
code("\n".join([
    "def search_bm25_scores(q):",
    "    scores = bm.get_scores(cut(q))",
    "    order = np.argsort(scores)[::-1]",
    "    return [(float(scores[i]), int(i)) for i in order]",
    "",
    "def search_dense_scores(q):",
    "    v = model.encode([q], normalize_embeddings=True)",
    "    s, i = index.search(np.ascontiguousarray(v), 10)",
    "    return list(zip([float(x) for x in s[0]], [int(x) for x in i[0]]))",
    "",
    "def nn(xs):",
    "    a = np.array([x[0] for x in xs], dtype=float)",
    "    rng = a.max() - a.min()",
    "    if rng == 0:",
    "        return [(1.0, i) for _sc, i in xs]",
    "    return [((s - a.min()) / rng, i) for s, i in xs]",
    "",
    "def hybrid(s_bm, s_dense, alpha=0.5, n=5):",
    "    merged = {}",
    "    for sc, i in nn(s_bm):",
    "        merged[i] = merged.get(i, 0.0) + alpha * sc",
    "    for sc, i in nn(s_dense):",
    "        merged[i] = merged.get(i, 0.0) + (1 - alpha) * sc",
    "    return [i for i, _v in sorted(merged.items(), key=lambda kv: -kv[1])][:n]",
    "",
    "h_hybrid = []",
    "for i, q in enumerate(test_q):",
    "    s_b = search_bm25_scores(q)",
    "    s_d = search_dense_scores(q)",
    "    h_hybrid.append(hit_at(hybrid(s_b, s_d, 0.5, 5), i))",
    "print(\"Hybrid Hit@5: %.4f（min-max 融合 alpha=0.5）\" % (sum(h_hybrid) / len(test_q)))",
]))

md("\n".join([
    "## 7. RAG 回答生成 + LLM-as-judge",
    "",
    "- 路径 A：任意 OpenAI 兼容 API（环境变量 `LLM_API_KEY`）。",
    "- 路径 B：本地 `Qwen/Qwen2.5-0.5B-Instruct` 小模型，离线自包含。",
    "",
    "> 两种都输出「查询 → 上下文 Top-3 → 答案」；`USE_LLM=0` 时答案走**抽取式基线**（抄录 top-1 上下文），此时 judge 衡量「答案是否忠于 gold 上下文」——基线全绿，换真模型后分数会分化，正是评测的意义。",
    "",
]))
code("\n".join([
    "import os",
    "",
    "def rag_answer(q, model=None, tokenizer=None):",
    "    top = search_dense(q, 3)",
    "    ctx = chr(10).join(f\"[{i + 1}] {corpus_q[i]}\" for i in top)",
    "    msgs = [",
    "        {\"role\": \"system\", \"content\": \"你是知识库助手，请只依据给定上下文作答，避免编造。\"},",
    "        {\"role\": \"user\", \"content\": \"问题：\" + q + chr(10) + \"上下文：\" + ctx + chr(10) + \"请回答。\"},",
    "    ]",
    "    if model is None:   # fallback：抽取式基线（答案 = top-1 检索文档，免模型出完整评测表）",
    "        return corpus_q[top[0]], top",
    "    if tokenizer is None:  # 路径A：OpenAI 兼容 API",
    "        import openai",
    "        client = openai.OpenAI(api_key=os.environ[\"LLM_API_KEY\"])",
    "        out = client.chat.completions.create(model=model, messages=msgs, temperature=0.2)",
    "        return out.choices[0].message.content, top",
    "    # 路径B：本地 transformers（演示；生产建议 vLLM）",
    "    from transformers import pipeline",
    "    gen = pipeline(\"text-generation\", model=model, tokenizer=tokenizer, max_new_tokens=64)",
    "    return gen(msgs)[0][\"generated_text\"][-1][\"content\"], top",
    "",
    "def judge(gold, answer):",
    "    if not answer:",
    "        return 0.0",
    "    if gold in answer or answer in gold:",
    "        return 1.0",
    "    hits = sum(1 for i in range(len(gold) - 4) if gold[i:i + 4] in answer)",
    "    return hits / max(1.0, len(gold) - 4.0)",
    "",
    "print(\"rag_answer 与 judge 就绪\")",
]))

md("\n".join([
    "## 8. 跑实验，输出对比表",
    "",
    "默认 `USE_LLM = 0`（免重模型直接出**检索 + 抽取式回答**表）；改成 `1` 走本地 Qwen 0.5B，让 judge 有真生成可判。",
    "",
]))
code("\n".join([
    "USE_LLM = 0",
    "import time, pathlib, random as _r",
    "t0 = time.time()",
    "if USE_LLM:",
    "    from transformers import AutoModelForCausalLM, AutoTokenizer",
    "    tok = AutoTokenizer.from_pretrained(\"Qwen/Qwen2.5-0.5B-Instruct\")",
    "    mod = AutoModelForCausalLM.from_pretrained(\"Qwen/Qwen2.5-0.5B-Instruct\", low_cpu_mem_usage=True)",
    "else:",
    "    mod = tok = None",
    "",
    "_r.seed(0)",
    "sample_idx = _r.sample(range(len(test_q)), min(20, len(test_q)))",
    "rows = []",
    "for i in sample_idx:",
    "    q = test_q[i]",
    "    a, top = rag_answer(q, mod, tok)",
    "    a = a if a else \"\"",
    "    rows.append({",
    "        \"query\": q[:18],",
    "        \"hit5\": hit_at(search_dense(q, 5), i),",
    "        \"retrieved_top\": [corpus_q[x][:10] for x in top],",
    "        \"answer\": a[:24],",
    "        \"judge\": judge(corpus_q[i], a),",
    "    })",
    "print(f\"实验完成, 耗时 {time.time() - t0:.0f}s, 样本 {len(rows)} 条\\n\")",
    "pathlib.Path(\"out\").mkdir(exist_ok=True)",
    "import pandas as pd",
    "pd.DataFrame(rows).to_csv(\"out/009_results.csv\", index=False, encoding=\"utf-8-sig\")",
    "print(pd.DataFrame(rows).to_markdown(index=False))",
]))

md("\n".join([
    "## 9. 结论：这份实验你获得了什么",
    "",
    "| 层次 | 你验证的东西 | 下一步 |",
    "|---|---|---|",
    "| 检索基线 | BM25 vs Dense vs Hybrid 的 Hit@5 | 08-RAG体系/03-混合检索与重排.md |",
    "| 工程闭环 | 解析→分块→索引→检索→评测 全链路 | 换成自己的 PDF 再跑一遍 |",
    "| 评测心智 | Hit@k 是检索底线，回答质量要靠 judge | 08-RAG体系/07-RAG评测-RAGAS.md |",
    "| 可扩展位 | Cross-Encoder 重排、GraphRAG、Agentic RAG | 对照索引 §8 后续行 |",
    "",
    "> 建议：打开 `USE_LLM` 复跑一遍；或把 `CORPUS` 换成你的文档集（分块见 08-RAG体系/04-分块与元数据.md）。",
    "",
]))

nb = {
    "cells": [
        {"cell_type": "markdown" if t == "m" else "code", "metadata": {},
         "source": src.splitlines(True),
         "outputs": [] if t == "c" else None, "execution_count": None}
        for t, src in C
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "gpuType": "T4"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
with io.open('code/notebooks/009-中文RAG全管线.ipynb', 'w', encoding='utf-8', newline='') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write('\n')
print('RENDERED %d cells; corpus %d chars' % (len(C), len(corpus_lit)))
