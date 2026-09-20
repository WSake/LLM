# -*- coding: utf-8 -*-
"""Logits 与采样演示（配合《11-Logits与采样-Temperature-TopK-TopP》）。

A1 温度影谱：T∈{0.2,1,2,5} 下 softmax 分布如何从"收缩"变"平坦"，熵/等效困惑度
A2 Top-k：截断集合的覆盖概率与重归一后的分布变化
A3 Top-p：动态涌现集（nucleus）大小——logits 尖锐时集合小、平坦时集合大
A4 重复惩罚：贪心被"同一 token"卡死 vs 压分后脱困
A5 温度 vs 多样性：同一分布不同温度采样序列的确定性/多样性（实践题"温度 0 vs 1"）

纯 CPU 秒级、零下载；全部数字一键复现。注意：迷你合成 logits 是"结构演示"，不代表真实模型分布。
"""
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
V = 30
rng = np.random.RandomState(11)


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def entropy(p):
    p = p[p > 0]
    return -(p * np.log(p)).sum()


def main():
    # 一个"模型对下一个 token 的原始 logits"（合成，定种子）；top-3 是"领先词"
    z0 = rng.randn(V) * 1.5
    top3 = np.argsort(z0)[::-1][:3]
    print("=" * 72)
    print("A1 · 温度影谱：同一套 logits，T 从 0.2 → 5.0")
    print("=" * 72)
    print(f"  合成 logits V={V}（种子固定）；T=1 时 top-3 概率 + 熵 + 等效困惑度 exp(H)：")
    for T in (0.2, 1.0, 2.0, 5.0):
        p = softmax(z0, T)
        H = entropy(p)
        top = np.argsort(p)[::-1][:3]
        three = " ".join(f"{p[i]:.3f}" for i in top)
        print(f"    T={T:<4}  top1={p[top[0]]:.3f}  top-3 {three}  H={H:.3f} nats  PPL≈{np.exp(H):5.1f}")

    # ------------------------------------------------
    print()
    print("=" * 72)
    print("A2 · Top-k：硬截断保最大 k 个，其余概率是否值得丢")
    print("=" * 72)
    p1 = softmax(z0, 1.0)
    order = np.argsort(p1)[::-1]
    print(f"    T=1 原分布前 5：{('  '.join(f'{p1[order[j]]:.3f}@{order[j]}' for j in range(5)))}")
    for k in (1, 5, 10):
        keep = np.zeros(V)
        keep[order[:k]] = p1[order[:k]]
        keep /= keep.sum()
        print(f"    top-{k:<2} 覆盖原概率 {p1[order[:k]].sum():.3f} → 截断后漏掉 {1-p1[order[:k]].sum():.3f}，"
              f"重归一 top1 = {keep[order[0]]:.3f}（原 {p1[order[0]]:.3f}）")

    # ------------------------------------------------
    print()
    print("=" * 72)
    print("A3 · Top-p：动态涌现集，logits 的锐利度决定集合大小")
    print("=" * 72)
    for label, zz in (("尖锐", rng.randn(V) * 3), ("平坦", rng.randn(V) * 0.1)):
        p = softmax(zz, 1.0)
        srt = np.sort(p)[::-1]
        cs = np.cumsum(srt)
        row = []
        for p_ in (0.7, 0.9, 0.95, 0.99):
            nuc = int(np.searchsorted(cs, p_) + 1)
            row.append(f"p={p_:.2f}→{nuc:>2} 词")
        print(f"    logits {label}：  " + "  ".join(row))
    print("   → 尖锐分布覆盖 p=0.9 只需 8 词，平坦要 27 词——top-p 的集合随置信度浮动")

    # ------------------------------------------------
    print()
    print("=" * 72)
    print("A4 · 重复惩罚：贪心卡死在一个 token 时的脱困机制")
    print("=" * 72)
    z_sticky = z0.copy()
    z_sticky[0] = 3.0                      # token0 领先（余者~N(0,1)·1.5）→ 贪心一直选它
    beta = 4.0                             # 压分幅度：一次足以把 3.0 拉出榜首

    def greedy_seq(penalty: float, n: int):
        seq, seen = [], set()
        for _ in range(n):
            z = z_sticky.astype(np.float64)
            for t in seen:
                z[t] -= penalty              # 减法式重复惩罚
            c = int(np.argmax(z))
            seq.append(c)
            seen.add(c)
        return seq

    print(f"  构造 logits：token0=3.0 其余~N(0,1)·1.5 → 贪心必选 0")
    print(f"    无惩罚（β=0） ：前 8 步 = {greedy_seq(0, 8)}  ← 纯重复")
    print(f"    惩罚（β={beta:.0f}）：前 8 步 = {greedy_seq(beta, 8)}  ← 压分后把机会让给其他词")

    # ------------------------------------------------
    print()
    print("=" * 72)
    print("A5 · 温度 vs 多样性（实践题：温度 0 vs 1）")
    print("=" * 72)
    N, L = 200, 5
    for T in (0.0, 0.2, 0.7, 1.0, 1.5):
        # T=0 => 贪心；否则用 RNG 按分布采样
        if T == 0.0:
            seqs = {tuple([int(np.argmax(softmax(z0, 1.0))) for _ in range(L)])}
            p = np.zeros(V); p[int(np.argmax(z0))] = 1.0     # 贪心 = 把概率全给 argmax
        else:
            r = np.random.RandomState(0)
            seqs = set()
            for _ in range(N):
                seq = tuple(int(r.choice(V, p=softmax(z0, T))) for _ in range(L))
                seqs.add(seq)
            p = softmax(z0, T)
        H = entropy(p)
        print(f"    T={T:<4}  {len(seqs):>3}/{N} 条唯一序列（多样性 {len(seqs)/N*100:4.0f}%）  PPL≈{np.exp(H):5.1f}")

    print()
    print("done · 一键复现：python code/scripts/sampling_demo.py")


if __name__ == "__main__":
    main()
