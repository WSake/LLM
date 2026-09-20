# -*- coding: utf-8 -*-
"""
pretrain_data_demo.py
────────────────────────────────────────────────────────────────────────────
04-训练体系 01-预训练数据工程-采集-清洗-去重 的复现脚本（04 目录第 0 个脚本）。

演示预训练数据管线的三段，全部为真实计算并打印：
  A 质量过滤   启发式规则（长度/有效字符占比/重复字符占比/HTML 残留）
              从"粗爬池"里删掉垃圾，量删了多少、留下多少 token；
  B 去重       字符 4-gram MinHash 签名（K=64 个随机权重）→ LSH(B=16,R=4)
              候选对 → Jaccard≥0.6 判定近重复 → 贪心保序去重，量 token 节省；
  C 配比       三源混合：按"篇数"采样 vs 按"token 预算"采样，量实际占比差异。

纯 CPU、固定随机种子（模块级单例 np.random.RandomState(42)），一键可复现。
复现声明：run1==run2 科学数字逐位一致，只有墙钟在跑次间浮动。
"""
import sys
import re
import collections
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

R = np.random.RandomState(42)
EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────
# 1. 语料构造：一个"粗爬池"，里面既有干净文本也有网上爬下来的脏东西
#    （近重复变体用编辑操作程序化生成，保证"重复"的定义可解释）
# ──────────────────────────────────────────────────────────────────────────

CLEAN_ZH = [
    "黄河发源于青藏高原巴颜喀拉山脉，向东注入渤海。",
    "长江是中国最长的河流，干流全长约六千三百余公里。",
    "泰山位于山东省中部，是五岳之首，主峰海拔约一千五百米。",
    "农历新年在中国也叫春节，是家人团聚最重要的传统节日。",
    "汉语普通话有四个基本声调，加上轻声一共五个。",
    "地球自转一周是一天，也就是二十四小时。",
    "水的沸点在标准大气压下是一百摄氏度。",
    "光的传播速度约为每秒三十万公里，比声音快很多。",
    "圆周长与直径的比值是一个常数，叫圆周率。",
    "秦始皇统一六国后建立了中国历史上第一个中央集权王朝。",
    "中国的地势整体西高东低，呈三级阶梯状分布。",
    "北京是中国的首都，位于华北平原北部。",
    "小麦和水稻是中国最主要的两种粮食作物。",
    "月球的引力只有地球的六分之一。",
    "人体骨骼成年后共有二百零六块。",
    "隋朝修建的大运河连接了南北交通。",
    "三星堆遗址出土了大量青铜面具和象牙制品。",
    "二十四节气是中国古人总结的农事历法。",
    "西湖位于浙江省杭州市，以湖光山色著称。",
    "太阳是一颗黄矮星，位于银河系猎户臂上。",
]

CLEAN_EN = [
    "The nervous system transmits signals through neurons at high speed.",
    "Photosynthesis converts sunlight into chemical energy stored in glucose.",
    "A polynomial of degree n has at most n real roots.",
    "Plate tectonics explains the drift of continents over geological time.",
    "The Amazon rainforest produces a large share of the world's oxygen.",
    "Machine translation quality improves dramatically with larger parallel corpora.",
    "Bees communicate the location of flowers by performing a waggle dance.",
    "The Richter scale is logarithmic, so each step is ten times stronger.",
    "Average pooling smooths features while max pooling keeps sharp activations.",
    "A turing machine can simulate any algorithm given unbounded tape.",
    "Whales are mammals that evolved from land-dwelling ancestors.",
    "The square root of two is irrational, a proof by contradiction.",
    "Catalysts lower activation energy without being consumed by the reaction.",
    "Saturn has a ring system made mostly of ice particles.",
    "RNA interference silences genes at the messenger RNA level.",
    "The Doppler effect shifts pitch when the source is moving toward you.",
    "Gold is dense and malleable, so a small bar weighs surprisingly much.",
    "Enzymes are proteins that catalyze biochemical reactions inside cells.",
    "The moon's far side lacks the dark maria common on the near side.",
    "Binary search requires sorted input and runs in logarithmic time.",
]


def make_near_dups(base_docs, edits_rng):
    """对文档做编辑操作生成近重复变体（换字/插空格/截断/重复片段）。
    用独立 rng 流，避免污染主流程的随机序列。"""
    out = []
    for d in base_docs:
        r = edits_rng.rand()
        if r < 0.30:                      # 换一个同义/近形字
            if len(d) > 6:
                i = int(edits_rng.choice(len(d)))
                out.append(d[:i] + ("о" if i % 3 == 0 else " ") + d[i + 1:])
        elif r < 0.60:                    # 每一处标点后插一个空格，复用字符
            out.append(" ".join(d))
        else:                             # 截掉末 4~10 个字符
            cut = int(edits_rng.choice([4, 5, 6, 8]))
            out.append(d[:-cut] if len(d) > cut + 4 else d * 2)
    return out


def make_noise(noise_rng):
    """重新设计为三类真实网页噪声，各 10 条：
    ① 纯符号串（≤20 字符）→ 命中 low_valid；
    ② 单字符/单语气词重复堆砌（8~40 个）→ 命中 high_repeat；
    ③ 混合乱码（字母/数字/符号混杂）→ 大部分命中 low_valid，少量长串漏网（演示
       真实过滤器做不到 100% 召回——漏网的进到 B 段但不构成重复簇）。"""
    noise = [
        "<div class=ad>点击此处领取 100 元红包</div>",
        "<p>欢迎访问本站</p> 备案号 京ICP备20260000号",
        "哈",
        "ok",
    ]
    # ① 纯符号
    punct_pool = list("#%&*@!^~+=<>/|")
    for _ in range(10):
        n = int(noise_rng.randint(6, 20))
        noise.append("".join(noise_rng.choice(punct_pool) for _ in range(n)))
    # ② 高重复堆砌
    ch_pool = list("哈哈哈哈哈艹水水水水啊啊啊啊赞赞赞")
    for _ in range(10):
        ch = noise_rng.choice(ch_pool)
        k = int(noise_rng.randint(8, 40))
        noise.append(ch * k)
    # ③ 混合乱码：>=6 成符号，少量反而 <6 成（模拟漏网）
    gram = punct_pool + list("aeiou0123456789")
    for _ in range(10):
        n = int(noise_rng.randint(12, 36))
        s = "".join(noise_rng.choice(gram) for _ in range(n))
        noise.append(s)
    return noise


def build_crawl_pool(rng):
    """粗爬池 = 干净中英 + 近重复变体 + 噪声；顺序打乱但固定。"""
    near = make_near_dups(CLEAN_ZH[:20] + CLEAN_EN[:10], rng)   # 30 条变体
    noise = make_noise(rng)                                     # 34 条垃圾（4 手写 + 30 程序化）
    doc_pool = list(zip(CLEAN_ZH, ["clean"] * len(CLEAN_ZH)))
    doc_pool += list(zip(CLEAN_EN, ["clean"] * len(CLEAN_EN)))
    doc_pool += list(zip(near, ["neardup"] * len(near)))
    doc_pool += list(zip(noise, ["noise"] * len(noise)))
    idx = rng.permutation(len(doc_pool))
    pool = [(doc_pool[i][0], doc_pool[i][1], int(i)) for i in idx]
    return pool


# ──────────────────────────────────────────────────────────────────────────
# 2. 公共工具
# ──────────────────────────────────────────────────────────────────────────

def count_tokens(text):
    """中文字符 + 英文/数字词元，统一计为 token 的近似口径。"""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    latin = len(re.findall(r"[A-Za-z0-9]+", text))
    return cjk + latin


def valid_ratio(text):
    """有效字符占比：汉字/字母/数字 / 全部非空白字符。"""
    chars = re.sub(r"\s", "", text)
    if not chars:
        return 0.0
    good = sum(
        1 for ch in chars
        if ("\u4e00" <= ch <= "\u9fff") or ch.isalnum()
    )
    return good / len(chars)


def repeat_ratio(text):
    """重复字符占比：连续出现 >=4 次的字符 run 覆盖的字符数 / 非空白字符数。
    用"连续 run"而不是"全局限频"——健康英文里 t/s/e 本就高频，跑 run 口径不误伤."""
    chars = re.sub(r"\s", "", text)
    n = len(chars)
    if not n:
        return 0.0
    heavy, i = 0, 0
    while i < n:
        j = i
        while j < n and chars[j] == chars[i]:
            j += 1
        if j - i >= 4:
            heavy += (j - i)
        i = j
    return heavy / n


def shingles(text, k=4):
    """字符 4-gram。先去掉空白与常见标点、转小写，保证近重复可匹配。"""
    norm = re.sub(r"[\s，。！？、；：“”（）()'\".,;:!?·]", "", text).lower()
    return [norm[i:i + k] for i in range(len(norm) - k + 1)]


# ──────────────────────────────────────────────────────────────────────────
# 3. 段 A：质量过滤
# ──────────────────────────────────────────────────────────────────────────

def quality_filter(docs):
    kept, dropped_by = [], collections.Counter()
    for text, kind, oid in docs:
        if count_tokens(text) < 6:                       # 太短（<6 token）
            dropped_by["too_short"] += 1
            continue
        if valid_ratio(text) < 0.5:                      # 有效字符不足一半
            dropped_by["low_valid"] += 1
            continue
        if repeat_ratio(text) > 0.40:                    # 重复字符堆砌
            dropped_by["high_repeat"] += 1
            continue
        if ("<" in text) or (">" in text):               # HTML 残留
            dropped_by["html"] += 1
            continue
        kept.append((text, kind, oid))
    return kept, dropped_by


# ──────────────────────────────────────────────────────────────────────────
# 4. 段 B：MinHash + LSH 近重复去重
# ──────────────────────────────────────────────────────────────────────────

def minhash_signatures(docs, K=64, BANDS=16):
    """给每条文档算 K 维 MinHash 签名。
    用全集 gram 的随机权重向量 W (M,K)，签名 = 文档 gram id 集合在每一列的最小权重。
    (Broder/MinHash 的标准随机权重实现。)
    """
    # 建全集 gram → id 映射。注意：必须 sorted 遍历，
    # 否则 Python 跨进程 set 迭代序（哈希种子）不同 → id 分配漂移 → 签名不同。
    grams = [set(shingles(d)) for d in docs]
    id_map = {}
    for gs in grams:
        for g in sorted(gs):
            id_map.setdefault(g, len(id_map))
    M = len(id_map)
    W = R.rand(M, K)                      # 每维一个随机权重
    sigs = []
    for gs in grams:
        ids = [id_map[g] for g in gs]
        sigs.append(W[ids].min(axis=0))
    return np.asarray(sigs), grams


def lsh_candidates(sigs, BANDS=16, ROWS=4):
    """把 K 维签名切成 BANDS 个 band（每个 ROWS 维），同桶即候选。
    返回候选对（无序）。实测 BANDS=16,ROWS=4 → 理论阈值 ≈ (1/16)^(1/4)=0.5。"""
    buckets = collections.defaultdict(list)
    for i, s in enumerate(sigs):
        for b in range(BANDS):
            key = (b, tuple(s[b * ROWS:(b + 1) * ROWS]))
            buckets[key].append(i)
    pairs = set()
    for items in buckets.values():
        if len(items) > 1:
            items = sorted(items)
            for x in range(len(items)):
                for y in range(x + 1, len(items)):
                    pairs.add((items[x], items[y]))
    return pairs


def jaccard(gsa, gsb):
    inter = len(gsa & gsb)
    union = len(gsa | gsb)
    return inter / (union + EPS)


def greedy_dedupe(docs, grams, jacc_threshold=0.6):
    """保留同簇里原序最靠前的文档，其余判为近重复。"""
    n = len(docs)
    keep = []
    removed_ids = set()
    for i in range(n):
        if i in removed_ids:
            continue
        keep.append(docs[i])
        for j in range(i + 1, n):
            if j in removed_ids:
                continue
            if jaccard(grams[i], grams[j]) >= jacc_threshold:
                removed_ids.add(j)
    return keep, removed_ids


# ──────────────────────────────────────────────────────────────────────────
# 5. 段 C：三源配比（篇数采样 vs token 预算采样）
# ──────────────────────────────────────────────────────────────────────────

SRC_WEB = (CLEAN_ZH[:8] + CLEAN_EN[:8] + [
    "天气转凉请大家注意添加衣物，谨防感冒。",
    "今日特价活动到店可享第二件五折，数量有限。",
]) * 12
SRC_KNOW = (CLEAN_ZH + CLEAN_EN) * 3
SRC_CODE = [
    "def mean(xs):\n    return sum(xs) / len(xs)",
    "for i in range(10):\n    print(i * i)",
    "import numpy as np\nw = np.random.RandomState(0).rand(3, 3)",
    "class Node:\n    def __init__(self, val):\n        self.val = val",
    "with open('data.txt') as f:\n    rows = f.readlines()",
    "result = [x for x in range(20) if x % 2 == 0]",
    "def softmax(z):\n    e = np.exp(z - z.max())\n    return e / e.sum()",
    "try:\n    do_something()\nexcept OSError:\n    pass",
    "for k in range(3):\n    acc += (y[k] - pred[k]) ** 2",
    "w = w - lr * g",
] * 4


def pick_by_count(srcs, plan):
    """按篇数配比（plan 为归一化比例），返回实际 token 构成。"""
    picks = []
    for s, ratio in zip(srcs, plan):
        k = int(round(ratio * 100))
        picks += [s[i % len(s)] for i in range(k)]
    return picks


def pick_by_token_budget(srcs, plan, budget=2000):
    """按 token 预算配比：每源分得 plan[i]*budget 个 token，文档随机满取。"""
    picks, spent = [], []
    for s, ratio in zip(srcs, plan):
        target = ratio * budget
        take, tot = [], 0
        order = list(range(len(s)))
        np.random.RandomState(7).shuffle(order)
        for i in order:
            if tot >= target:
                break
            take.append(s[i])
            tot += count_tokens(s[i])
        picks += take
        spent.append(tot)
    return picks, spent


# ──────────────────────────────────────────────────────────────────────────
# 6. main：三段落 + 段间落盘统计
# ──────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    pool = build_crawl_pool(R)
    kinds = collections.Counter(k for _, k, _ in pool)
    tok_before = sum(count_tokens(d) for d, _, _ in pool)

    print("=" * 66)
    print("pretrain_data_demo：质量过滤 → 去重 → 配比（三段实测）")
    print("=" * 66)
    print(f"粗爬池：共 {len(pool)} 条 / {tok_before} token")
    print(f"  clean {kinds['clean']} · neardup {kinds['neardup']} · noise {kinds['noise']}")
    print()

    # ── 段 A：质量过滤 ──
    kept, dropped = quality_filter(pool)
    tok_after_a = sum(count_tokens(d) for d, _, _ in kept)
    noise_kept = sum(1 for d, k, _ in kept if k == "noise")
    print(f"[A] 质量过滤          剩 {len(kept)}/{len(pool)} 条，token {tok_before}→{tok_after_a}")
    print(f"    删除分类: {dict(dropped)}")
    print(f"    垃圾召回率（noise 被删 {kinds['noise'] - noise_kept}/{kinds['noise']}）: "
          f"{100 * (kinds['noise'] - noise_kept) / kinds['noise']:.1f}%")
    print()

    # ── 段 B：MinHash + LSH 去重 ──
    texts_a = [d for d, _, _ in kept]
    sigs, grams_b = minhash_signatures(texts_a, K=64, BANDS=16)
    cands = lsh_candidates(sigs, BANDS=16, ROWS=4)
    # 候选对里真正过阈值的近重复对
    real = [(i, j) for (i, j) in cands if jaccard(grams_b[i], grams_b[j]) >= 0.6]
    jac_vals = sorted(jaccard(grams_b[i], grams_b[j]) for (i, j) in cands)
    keep_b, removed_ids = greedy_dedupe(texts_a, grams_b, jacc_threshold=0.6)
    tok_before_b = sum(count_tokens(d) for d in texts_a)
    tok_after_b = sum(count_tokens(d) for d in keep_b)
    print(f"[B] MinHash+LSH 去重   候选对 {len(cands)} 对，过 Jaccard≥0.6 阈值 {len(real)} 对")
    print(f"    近重复变体被删 {len(removed_ids)} 条："
          f"token {tok_before_b}→{tok_after_b}（省 {100 * (1 - tok_after_b / (tok_before_b + EPS)):.1f}%）")
    if jac_vals:
        print(f"    被候选对里 Jaccard 中位 {jac_vals[len(jac_vals)//2]:.2f} · 最大 {jac_vals[-1]:.2f}")
    print()

    # ── 段 C：配比 ──
    srcs = [SRC_WEB * 3, SRC_KNOW, SRC_CODE * 2]          # 篇数不齐，token 更不齐
    plan = [0.70, 0.20, 0.10]
    names = ["网页", "百科/教科书", "代码"]
    print("[C] 三源配比（目标 70/20/10）")
    # C1 按篇数
    picked_c1 = pick_by_count(srcs, plan)
    per_src = []
    for s, name in zip(srcs, names):
        tok = sum(count_tokens(d) for d in s)
        per_src.append((name, len(s), tok))
    print(f"    每源篇数/token："
          + " · ".join(f"{n} {c} 篇/{t} tok" for n, c, t in per_src))
    c1 = [0, 0, 0]
    seg, lo = [70, 20, 10], 0
    for i in range(3):
        c1[i] = sum(count_tokens(d) for d in picked_c1[lo:lo + seg[i]])
        lo += seg[i]
    tot1 = sum(c1) + EPS
    print(f"    [C1 按篇数 100 篇采样] 实际 token 占比 "
          + f"{c1[0]/tot1:.1%}/{c1[1]/tot1:.1%}/{c1[2]/tot1:.1%}"
          + f"  (网页被代码稀释)")
    # C2 按 token 预算
    picked_c2, spent = pick_by_token_budget(srcs, plan, budget=2000)
    tot2 = sum(spent) + EPS
    print(f"    [C2 按 token 预算 2000 采样] 实际 token 占比 "
          + f"{spent[0]/tot2:.1%}/{spent[1]/tot2:.1%}/{spent[2]/tot2:.1%}"
          + f"  (贴近目标)")
    print()
    print(f"墙钟 {1000 * (time.time() - t0):.0f}ms（科学数字逐位一致，只有墙钟浮动）")


if __name__ == "__main__":
    main()
