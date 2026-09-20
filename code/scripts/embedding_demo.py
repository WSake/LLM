# -*- coding: utf-8 -*-
"""Embedding 查表层演示（配合《09-Embedding与词表》）。

A1 查表=一行: emb 的前向就是"按 token id 取行";与 one-hot 矩阵乘等价（maxerr~1e-16）
A2 稀疏访问: 语料/序列只有 token 个不同 id,128k 词表每次只碰 3% 的行（前向成本∝len 不∝vocab）
A3 权重共享（tied）: embedding 与输出头共用同一矩阵,参数量省一半; logits=T@E 数值验证
A4 语义入口: 用同一语料统计 PPMI+SVD 造 mini 词向量,最近邻词肉眼可见统计结构

纯 CPU 秒级;语料内嵌、零下载;全部数字一键复现。
"""
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.set_printoptions(precision=4)

rng = np.random.RandomState(0)

# 与 08 章 tokenizer_demo 同一语料（中英短句），保证"语义入口"演示的语料可追溯
ZH = [
    "大语言模型正在改变软件工程", "注意力机制让模型能处理长序列",
    "训练数据需要清洗与去重", "多头注意力每一头看到不同的关系",
    "残差连接与归一化让深层网络稳定训练", "检索增强生成把知识外挂到模型之外",
    "指令微调让模型学会跟人类对话", "MoE 把稀疏专家路由给每个 token",
    "位置编码给序列注入顺序信息", "最大似然目标等价于预测下一个词",
    "推理代价大部分花在生成 token 上", "模型的参数量决定了它的记忆容量",
    "词表大小决定了文本的压缩效率", "训练一个模型需要海量文本与算力",
    "路由与专家分工是 MoE 的核心思路", "长上下文需要更大的缓存与更宽的窗口",
    "语料质量比语料数量更能决定模型好坏", "把问题拆成子问题再逐次求解是推理的关键",
]
EN = [
    "attention is all you need", "the model predicts the next token",
    "byte pair encoding merges frequent pairs",
    "a transformer stacks attention and feed forward layers",
    "vocabulary size controls the compression ratio",
    "training on more data produces stronger models",
    "the token vocabulary maps text to integers",
    "beam search explores several continuations at once",
    "knowledge distillation transfers ability to smaller models",
    "quantization reduces memory by shrinking the data type",
]
CORPUS = [s for s in (ZH + EN) for _ in range(2)]


def units_of(s):
    """把一句切成基础单元：ASCII 留作一个词，中文逐字。"""
    out, buf = [], []
    for c in s:
        if c.isascii() and (c.isalpha() or c.isdigit()):
            buf.append(c)
            continue
        if buf:
            out.append("".join(buf).lower()); buf = []
        out.append(c)
    if buf:
        out.append("".join(buf).lower())
    return out


def main():
    # ------------------------------------------------
    print("=" * 66)
    print("A1 · 查表=一行：emb 前向就是按 id 取行")
    print("=" * 66)
    # 恒等验证用小型 V（lookup==one-hot@W 对标任何 V 都成立）
    V, H = 1000, 64
    W = rng.randn(V, H)
    idx = rng.randint(0, V, 8)                 # 8 个 token id
    lookup = W[idx]                            # 生产实现: lookup（零 FLOP）
    onehot_M = np.eye(V)[idx].astype(np.float64)   # 教学实现: one-hot 展开 (8, V)
    onehot = onehot_M @ W                          # (8, H)
    err = np.abs(lookup - onehot).max()
    print(f"  V={V} H={H}：lookup = W[id] 与 one-hot@W 恒等（maxerr={err:.1e}）")
    print(f"  → 查表只是'按索引取行'，连乘法都不用；one-hot 展开是把 V 维全算一遍的浪费")
    print(f"    在 128k 词表上 exactly 不可行：eye(128k,128k) 就要 128 GiB 内存（本机直接 OOM）")
    print(f"    8 个 id 取出 8 行: shape {lookup.shape}")

    # ------------------------------------------------
    print()
    print("=" * 66)
    print("A2 · 稀疏访问：前向成本 ∝ 序列长度，不 ∝ 词表大小")
    print("=" * 66)
    V = 128256
    for b, s in [(1, 4096), (8, 4096)]:
        touch = b * s
        frac = touch / V
        # 理论 FLOP: one-hot 展开要 (b*s)×V×H 一次矩阵乘; lookup 只要取行(零 FLOP)
        flop_onehot = 2.0 * touch * V * 4096
        print(f"  batch={b} seq={s}：触碰 {touch:,} 行 = 词表 {V:,} 的 {frac:.2%}（lookup 0 FLOP；one-hot 展开 {flop_onehot/1e12:.1f} TFLOP 白算）")

    # ------------------------------------------------
    print()
    print("=" * 66)
    print("A3 · 权重共享（tied）：embedding 与输出头用同一矩阵")
    print("=" * 66)
    V, H = 32000, 4096
    tied = V * H
    untied = 2 * V * H
    print(f"  V={V}×H={H}：共享 {tied/1e6:.1f}M 参数  vs  不共享 {untied/1e6:.1f}M（省 {1-tied/untied:.0%}）")
    print(f"  账算 LLaMA-2-7B 全参 ~7B, 共享后 embedding+head 只付一份 {tied/1e6:.0f}M（≈全参的 {tied/7e9*100:.1f}%）")
    # 数值检查: tied 网络里 logits = E @ h^T（用同一矩阵两边）
    rng2 = np.random.RandomState(3)
    E = rng2.randn(V, 64) / np.sqrt(64)     # 迷你 embedding
    h = rng2.randn(64)                       # 最后一层 hidden
    logits = E @ h                           # 输出 logits（无 bias 简化）
    top3 = np.argsort(logits)[-3:][::-1]
    print(f"  迷你验证: logits=E·h（tied 头即词表矩阵转置应用），argmax id = {top3[0]}, top-3 = {top3}（该 id 的词表行在 embedding 里, 共享成立）")

    # ------------------------------------------------
    print()
    print("=" * 66)
    print("A4 · 语义入口：同一语料 PPMI+SVD 造 mini 词向量, 最近邻肉眼可见")
    print("=" * 66)
    toks = []
    for s in CORPUS:
        toks += units_of(s)
    # 词汇表（本语料基础单元）
    vocab_arr = sorted({t for t in toks})
    ix = {t: i for i, t in enumerate(vocab_arr)}
    n = len(vocab_arr)
    # ±2 窗口共现计数
    co = np.zeros((n, n), dtype=np.float64)
    for s in CORPUS:
        u = [ix[t] for t in units_of(s)]
        for i, a in enumerate(u):
            for j in range(max(0, i - 2), min(len(u), i + 3)):
                if j != i:
                    co[a, u[j]] += 1
    total = co.sum()
    pi = co.sum(1) / total
    pj = co.sum(0) / total
    ppmi = co / total
    mask = ppmi > 0
    ppmi[mask] = np.log2(ppmi[mask] / np.outer(pi, pj)[mask])
    ppmi = np.maximum(ppmi, 0.0)                     # PPMI 负值截断
    # 降到 32 维
    u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
    emb = (u[:, :32] * np.sqrt(s[:32]))              # 单位化后再做余弦
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    probes = ["模", "型", "数", "训", "token", "the"]
    print(f"  词表 {n} 个基础单元（中文逐字 + 英文整词）, ±2 窗口 PPMI → SVD-32 维")
    print(f"  近邻（余弦相似度 top-2, 排除自身）:")
    for q in probes:
        if q not in ix:
            continue
        sim = emb @ emb[ix[q]]
        order = np.argsort(sim)[::-1]
        nbrs = [(vocab_arr[k], float(sim[k])) for k in order if k != ix[q]][:2]
        print(f"  『{q}』 → " + "  ".join(f"{w}({v:.2f})" for w, v in nbrs))

    print()
    print("done · 一键复现：python code/scripts/embedding_demo.py")


if __name__ == "__main__":
    main()
