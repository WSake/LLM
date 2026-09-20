# -*- coding: utf-8 -*-
"""从 0 实现 BPE + 字节级 BPE（BBPE）演示（配合《08-Tokenizer-BPE-SentencePiece-BBPE》）。

A0 语料账：中英混合小语料，字符集大小、总字符/字母数
A1 压缩比 vs 词表：char-BPE 在 [base, +50, +100, +200, +400] merges 下 token 数与压缩比
A2 中英逐句：同一套 BPE 下"每句 token 数/每字 token 数"对照（中文≈每字1 token起步，英文≈整词合并）
A3 BBPE 字节级：UTF-8 字节流上训练，中文每字≈3 字节 token → 膨胀 3× 量级，多字节 merge 后回落
A4 词表×embedding 账：vocab×hidden 参数与 fp16 显存，LLaMA/Qwen 风格配置账算 + 占全模型比例

本机纯 CPU 秒级；语料内嵌、零下载；所有数字一键复现。
"""
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ZH = [
    "大语言模型正在改变软件工程",
    "注意力机制让模型能处理长序列",
    "训练数据需要清洗与去重",
    "多头注意力每一头看到不同的关系",
    "残差连接与归一化让深层网络稳定训练",
    "检索增强生成把知识外挂到模型之外",
    "指令微调让模型学会跟人类对话",
    "MoE 把稀疏专家路由给每个 token",
    "位置编码给序列注入顺序信息",
    "最大似然目标等价于预测下一个词",
    "推理代价大部分花在生成 token 上",
    "模型的参数量决定了它的记忆容量",
    "词表大小决定了文本的压缩效率",
    "训练一个模型需要海量文本与算力",
    "路由与专家分工是 MoE 的核心思路",
    "长上下文需要更大的缓存与更宽的窗口",
    "语料质量比语料数量更能决定模型好坏",
    "把问题拆成子问题再逐次求解是推理的关键",
]

EN = [
    "attention is all you need",
    "the model predicts the next token",
    "byte pair encoding merges frequent pairs",
    "a transformer stacks attention and feed forward layers",
    "vocabulary size controls the compression ratio",
    "training on more data produces stronger models",
    "the token vocabulary maps text to integers",
    "beam search explores several continuations at once",
    "knowledge distillation transfers ability to smaller models",
    "quantization reduces memory by shrinking the data type",
]

# 每句重复 2 遍，让共现统计更厚（不影响"对/错"，只影响数字大小）
CORPUS = [s for s in (ZH + EN) for _ in range(2)]


def train_bpe(lines, max_merges):
    """Greedy byte pair encoding。

    用整数码表示符号；每步统计相邻二元组频次，取总频次最高、且同频时字典序最小的
    二元组做一次合并；返回 (merges, char2code, seqs) 用于 encode。
    """
    chars = sorted({c for s in lines for c in s})
    table = {c: i for i, c in enumerate(chars)}
    seqs = [[table[c] for c in s] for s in lines]
    merges = []  # [(a, b), ...] 训练顺序
    next_id = len(chars)
    for _ in range(max_merges):
        cnt = Counter()
        for s in seqs:
            for a, b in zip(s, s[1:]):
                cnt[(a, b)] += 1
        if not cnt:
            break
        best = max(cnt, key=lambda p: (cnt[p], -p[0], -p[1]))  # 频最高;并列取字典序最小
        seqs = [new_seqs_row(s, best, next_id) for s in seqs]
        merges.append(best)
        next_id += 1
    return merges, table, seqs


def new_seqs_row(s, best, new_id):
    """把序列 s 中所有连续 equal(best) 的相邻对合并成 new_id（贪心左到右）。"""
    out, i = [], 0
    while i < len(s):
        if i + 1 < len(s) and (s[i], s[i + 1]) == best:
            out.append(new_id)
            i += 2
        else:
            out.append(s[i])
            i += 1
    return out


def encode(s, table, merges):
    """把字符串按 merges（训练顺序 = 优先级）编成 token 码列表。

    每次成功应用 merge 时用其『训练时分配的真实 id』（len(chars)+rank）作结果码，
    这样深层 merge（合并后的码再参与合并）能在下一轮校正配到。
    """
    base = len(table)
    ids = [table[c] for c in s if c in table]
    rank = {p: i for i, p in enumerate(merges)}
    code_for = {p: base + i for i, p in enumerate(merges)}
    while True:
        # 找序列里出现且 merge 序号最小的一对
        pick = None
        for a, b in zip(ids, ids[1:]):
            r = rank.get((a, b))
            if r is not None and (pick is None or r < pick[0]):
                pick = (r, (a, b))
        if pick is None:
            break
        _, pr = pick
        ids = new_seqs_row(ids, pr, code_for[pr])
    return ids


def main():
    # ------------------------------------------------
    print("=" * 66)
    print("A0 · 语料账：本演示的内嵌中英小语料")
    print("=" * 66)
    tot_chars = sum(len(s) for s in CORPUS)
    charset = sorted({c for s in CORPUS for c in s})
    print(f"  句子 {len(CORPUS)} 条（{len(ZH)} 中文 + {len(EN)} 英文，各重复 2 遍）")
    print(f"  字符集 {len(charset)} 个 → base 词表 {len(charset)} 起步")
    print(f"  总字符数 {tot_chars}（不合并时的参照：每字符 1 token 即 {tot_chars}）")

    # ------------------------------------------------
    print("\nA1 · 压缩比 vs 词表：BPE 就是'把最常共现的相邻对合并'")
    print("=" * 66)
    base = len(set(''.join(CORPUS)))
    for n_merge in [0, 50, 150, 300, 500]:
        merges, table, _ = train_bpe(CORPUS, n_merge)
        used = len(merges)
        toks = sum(len(encode(s, table, merges)) for s in CORPUS)
        v = base + used
        note = ""
        if toks == len(CORPUS):
            note = "   ← 退化：每句恰好 1 token（词表把整句吞了, 小语料的尽头）"
        elif used < n_merge:
            note = f"  ← 语料可合并对已耗尽（预算 {n_merge} 只用掉 {used}）"
        print(f"  vocab {v:>4}（base+{used:>3} merges）  token 数 {toks:>6}   压缩比 {tot_chars/toks:.2f} chars/token{note}")

    # ------------------------------------------------
    print()
    print("=" * 66)
    print("A2 · 同一套词表：中英每句各切出几个 token")
    print("=" * 66)
    merges, table, _ = train_bpe(CORPUS, 300)
    probes = ["大语言模型正在改变软件工程", "注意力机制让模型能处理长序列",
              "训练数据需要清洗与去重",
              "attention is all you need",
              "the model predicts the next token",
              "quantization reduces memory"]
    print(f"（同一套词表：base+{len(merges)} merges = {base + len(merges)} vocab，两栏位于同一统计）")
    for s in probes:
        ids = encode(s, table, merges)
        # 回显每个 token 的内容（把码映射回符号组合较繁琐，改为只报计数+比例）
        zh = any('一' <= c <= '鿿' for c in s)
        per_char = len(ids) / len(s)
        print(f"  {'中文' if zh else '英文'}『{s}』 chars={len(s)} tokens={len(ids)}  per-char={per_char:.2f}")

    # ------------------------------------------------
    print()
    print("=" * 66)
    print("A3 · BBPE 字节级：UTF-8 地上做 BPE，中文每字≈3 字节 token")
    print("=" * 66)
    blines = [s.encode('utf-8') for s in CORPUS]
    bchars = sorted({b for s in blines for b in s})
    btable = {b: i for i, b in enumerate(bchars)}
    bseqs = [[btable[b] for b in s] for s in blines]

    def train_bytes(max_merges):
        seqs = bseqs[:]
        next_id = len(bchars)
        mrgs = []
        for _ in range(max_merges):
            cnt = Counter()
            for s in seqs:
                for a, b in zip(s, s[1:]):
                    cnt[(a, b)] += 1
            if not cnt:
                break
            best = max(cnt, key=lambda p: (cnt[p], -p[0], -p[1]))
            seqs = [new_seqs_row(s, best, next_id) for s in seqs]
            mrgs.append(best)
            next_id += 1
        return mrgs

    bmerges = train_bytes(300)
    bbase = len(bchars)
    brank = {p: i for i, p in enumerate(bmerges)}
    bcode = {p: bbase + i for i, p in enumerate(bmerges)}

    def bencode(b):
        ids = [btable[x] for x in b]
        while True:
            pick = None
            for a, c in zip(ids, ids[1:]):
                r = brank.get((a, c))
                if r is not None and (pick is None or r < pick[0]):
                    pick = (r, (a, c))
            if pick is None:
                break
            _, pr = pick
            ids = new_seqs_row(ids, pr, bcode[pr])
        return ids

    n_bytes = sum(len(s) for s in blines)
    btok = sum(len(bencode(s)) for s in blines)
    print(f"  字节语料总字节 {n_bytes}；字节集 {len(bchars)} 个（UTF-8 编码后去重；中文一律 3 字节/字）")
    print(f"  BBPE base+{len(bmerges)} merges 词表 {len(bchars) + len(bmerges)}（与 A2 的 300 merges 对齐同上）")
    print(f"  全语料 BBPE token {btok} → 压缩比 {n_bytes/btok:.2f} bytes/token")
    side = [("大语言模型正在改变软件工程", True),      # (句, 是否中文)
            ("attention is all you need", False)]
    print(f"  同句对照（char-BPE@300 与 BBPE@300 同一句，两种表示）：")
    for s, zh in side:
        n_utf = len(s)
        nb = len(s.encode('utf-8'))
        ctok = len(encode(s, table, merges))          # char-BPE@300 的 token 数（同一套词表）
        nb = len(s.encode('utf-8'))
        nt = len(bencode(s.encode('utf-8')))
        per = f"{nb/n_utf:.1f} 字节/字（中文 UTF-8）" if zh else f"{nb/n_utf:.1f} 字节/字（ASCII）"
        print(f"  『{s}』 {n_utf} 字/{nb} 字节 → BBPE {nt} tokens"
              f"（char-BPE 是 {ctok} tokens，膨胀 {nt/ctok:.1f}×；{per}）")

    # ------------------------------------------------
    print()
    print("=" * 66)
    print("A4 · 词表×embedding 账：词表是显存的第一笔账")
    print("=" * 66)
    def emb(vocab, hidden, bits=16):
        params = vocab * hidden
        return params, params * bits / 8 / 1e9   # fp16: 每参数 2 字节, 除以 1e9 得 GB
    for vocab, hidden, tag in [(32000, 4096, "LLaMA-2 7B 风格"),
                               (128256, 8192, "LLaMA-3 8B 风格"),
                               (151936, 4096, "Qwen1.5-7B 风格（账算）"),
                               (100278, 5120, "DeepSeek-V2 风格（账算）")]:
        p, gb = emb(vocab, hidden)
        print(f"  vocab {vocab:>7} × hidden {hidden:>5} = {p/1e6:>8.1f}M 参数  {gb:>6.2f} GB(fp16)  {tag}")
    p, gb = emb(151936, 4096)
    print(f"  → Qwen1.5-7B 风格 embedding {p/1e6:.0f}M 参数 ≈ 全模型 7.5B 参数的 {p/7.5e9*100:.1f}%")
    print("  （配置取公开档案的账算，非本机实测；显存按 fp16=每参数 2 字节）")

    print()
    print("done · 一键复现：python code/scripts/tokenizer_demo.py")


if __name__ == "__main__":
    main()
