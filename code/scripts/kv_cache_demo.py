# -*- coding: utf-8 -*-
"""KV Cache 显存账本演示（配合《10-KV-Cache与显存账本》）。

A1 容量账（§17.2.11 实践题）：手算 LLaMA 系 7B 在 1k–128k 的 KV 字节
A2 触摸实测：迷你 attention 上 prefill vs cached-decode 的单步时间（缓存生效 + prefill 缩放）
A3 PagedAttention 玩具：连续分配 vs 分页分配的碎片账 + 前缀缓存剩量

纯 CPU 秒级；A1/A3 是参数化账算（配置公开、代码复现），A2 是本机实测。
"""
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
rng = np.random.RandomState(7)


def kv_bytes_per_token(n_layers, kv_heads, head_dim, dtype_bytes=2):
    """单 token 的 KV 缓存字节数 = 2(K,V) × 层数 × KV头数 × 头维 × 字节。"""
    return 2 * n_layers * kv_heads * head_dim * dtype_bytes


def kv_gi(n_layers, kv_heads, head_dim, seq, dtype_bytes=2):
    return (2 * n_layers * kv_heads * head_dim * dtype_bytes * seq) / 2**30


def main():
    # ------------------------------------------------
    print("=" * 72)
    print("A1 · 手算 7B 系 KV 账（§17.2.11 实践题：32k 长度下 KV 显存）")
    print("=" * 72)
    configs = [
        ("LLaMA-2-7B 风 MHA (Hk=32)", 32, 32, 128),
        ("LLaMA-3-8B 风 GQA (Hk=8 )", 32, 8, 128),
    ]
    seqs = [1024, 2048, 8192, 32768, 131072]
    print("模型".ljust(28) + "".join(f"{s//1024:>7}k" for s in seqs) + "  单token/位")
    for name, l, hk, d in configs:
        row = name.ljust(28)
        for s in seqs:
            row += f"{kv_gi(l, hk, d, s):7.2f}G"
        row += f"  {kv_bytes_per_token(l, hk, d)//1024} KiB"
        print(row)
    print("  核对 03 章 8k：MHA = 4.00 GiB，GQA-8 = 1.00 GiB，逐格相等")
    mha32k, gqa32k = kv_gi(32, 32, 128, 32768), kv_gi(32, 8, 128, 32768)
    w = 14.0  # 7B fp16 ≈ 14 GB
    print(f"  权重 fp16 ≈ {w:.0f} GB；KV@32k = {mha32k:.1f} GiB(MHA) / {gqa32k:.1f} GiB(GQA)")
    print(f"  → MHA 32k 占(权重+KV) 的 {mha32k/(w+mha32k)*100:.0f}%；GQA-8 32k 占 {gqa32k/(w+gqa32k)*100:.0f}%")

    # ------------------------------------------------
    print()
    print("=" * 72)
    print("A2 · 迷你 attention 实测：prefill vs cached decode（T=128, hidden 64）")
    print("=" * 72)
    H = 64

    def attention_step(x, Wq, Wk, Wv, Wo, kvc=None):
        """单头因果 attention。x=(T,H)；kvc=(k,v) 历史缓存；None=全量重算。"""
        q = x @ Wq
        k = x @ Wk
        v = x @ Wv
        if kvc is not None:
            k = np.vstack([kvc[0], k])
            v = np.vstack([kvc[1], v])
        Tq, Tk = q.shape[0], k.shape[0]
        s = (q @ k.T) / np.sqrt(H)
        # 因果掩码：q 的 i 行只能看 k 的前 (i + Tk - Tq + 1) 个
        mask = np.zeros_like(s)
        for i in range(Tq):
            mask[i, i + (Tk - Tq) + 1:] = -np.inf
        a = np.exp(s + mask - (s + mask).max(1, keepdims=True))
        att = a / a.sum(1, keepdims=True)
        return att @ v, (k, v)

    T = 128
    Wq, Wk, Wv, Wo = [rng.randn(H, H) / np.sqrt(H) for _ in range(4)]
    x_full = rng.randn(T, H)

    def timeit(f, rep=15):
        best = np.inf
        for _ in range(rep):
            t0 = time.perf_counter()
            f()
            best = min(best, time.perf_counter() - t0)
        return best * 1e6  # µs

    t_prefill = timeit(lambda: attention_step(x_full, Wq, Wk, Wv, Wo))
    _, kvc = attention_step(x_full, Wq, Wk, Wv, Wo)
    t_cached = timeit(lambda: attention_step(x_full[-1:], Wq, Wk, Wv, Wo, kvc))
    x_next = np.concatenate([x_full, rng.randn(1, H)])
    t_recompute = timeit(lambda: attention_step(x_next, Wq, Wk, Wv, Wo))
    print(f"  prefill（一次算 {T} 位置）        {t_prefill:8.0f} µs")
    print(f"  cached decode（只算 1 新 token）   {t_cached:8.0f} µs")
    print(f"  recompute decode（拼回全量重算）   {t_recompute:8.0f} µs")
    print(f"  → 同一步 decode，带缓存比全量重算快 {t_recompute/t_cached:.1f}×")
    print("  prefill 时间随 T 缩放（每次只喂不同长度, 取 15 次最小）:")
    for tt in (64, 128, 256, 512):
        xt = rng.randn(tt, H)
        tpre = timeit(lambda: attention_step(xt, Wq, Wk, Wv, Wo))
        print(f"    T={tt:4d}  prefill {tpre:7.0f} µs")

    # ------------------------------------------------
    print()
    print("=" * 72)
    print("A3 · PagedAttention 玩具：连续分配 vs 分页分配（账算模拟）")
    print("=" * 72)
    per_token = kv_bytes_per_token(32, 8, 128)  # LLaMA-3-8B 风 128 KiB/token
    page_tokens, max_len, n_req = 16, 2048, 200
    seqs = rng.randint(256, max_len + 1, n_req)  # [256, 2048] 均匀
    used = seqs.sum() * per_token
    contig = n_req * max_len * per_token                          # 每请求按 max_len 开溜
    paged = sum((int(s + page_tokens - 1) // page_tokens) * page_tokens for s in seqs) * per_token
    print(f"  200 请求, seq∈[256,2048] 均匀, k/d 字节 {per_token//1024} KiB/token, 页 = {page_tokens} token：")
    print(f"    实际用到（sum seq）      {used/2**30:8.2f} GiB")
    print(f"    连续分配（全开 max_len） {contig/2**30:8.2f} GiB   → 内部碎片 {contig/used*100-100:.0f}%")
    print(f"    分页分配（按需按页）     {paged/2**30:8.2f} GiB   → 内部碎片 {paged/used*100-100:.0f}%")
    print(f"  → 分页把内部碎片压到每请求最多 1 页（{page_tokens} token）")
    prefix_pages = (512 + page_tokens - 1) // page_tokens
    share = n_req * prefix_pages * page_tokens * per_token
    print(f"  前缀缓存（512 token 系统提示, 200 请求共享）≈ 省 {share/2**30:.2f} GiB = 全部 KV 的 {share/paged*100:.0f}%")

    print()
    print("done · 一键复现：python code/scripts/kv_cache_demo.py")


if __name__ == "__main__":
    main()
