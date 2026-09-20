# -*- coding: utf-8 -*-
"""
sft_data_demo.py
────────────────────────────────────────────────────────────────────────────
04-训练体系 03-SFT数据-指令集构建 的复现脚本（04 目录第 2 个引擎脚本）。

演示 SFT 指令集的三个问题：
  A 指令对构建：把三种来源（模板算术合成 / 百科常识改写 / 人工翻译金标准）
      统一改造成 (指令 q, 回答 a) 对，量出"可验证 vs 开放式"的构成比；
  B 指令集质量三件套：
      ① 指令去重（q 完全一致即删——合成模板表单一批次高碰撞的照妖镜）；
      ② 指令多样性（字符 4-gram 唯一率——指令"说人话"的覆盖面指标）；
      ③ 回答长度分布（<10 token 的短回答占比——短回答≈可验证任务同构）。
  C 三主题配比：条数占比与 token 占比是两个打架的视角——算术 60 条占了
      多数样本，每条却只有约 5 token；按 token 预算 (40/30/30) 重采样后
      token 占比被拉回预算附近，并如实核算"池喂不饱预算"的池地板缺口
      （接 01 章数据墙 / 02 章合成扩池）。

纯 CPU、固定随机种子，一键可复现。
复现声明：run1==run2==run3 科学数字逐位一致，只有墙钟在跑次间浮动。
"""
import sys
import re
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

R = np.random.RandomState(42)
EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────
# 1. 三种来源语料
# ──────────────────────────────────────────────────────────────────────────

OPS = {"加": "+", "减": "-", "乘": "*"}
AR_FORMS = [
    "计算：{a}{op}{b}等于多少？",
    "{a}{op}{b}是多少？",
    "{a}{op}{b}=？",
]

KNOW_ZH = [
    "黄河发源于青藏高原巴颜喀拉山脉，向东注入渤海。",
    "长江是中国最长的河流，干流全长约六千三百余公里。",
    "泰山位于山东省中部，是五岳之首，主峰海拔约一千五百米。",
    "地球自转一周是一天，也就是二十四小时。",
    "水的沸点在标准大气压下是一百摄氏度。",
    "光的传播速度约为每秒三十万公里，比声音快很多。",
    "月球的引力只有地球的六分之一。",
    "人体骨骼成年后共有二百零六块。",
    "太阳系有四颗类地行星，从内到外依次是水星、金星、地球和火星。",
    "光合作用把二氧化碳和水转化为氧气和葡萄糖。",
    "秦始皇在公元前二百二十一年统一六国，建立中央集权制度。",
    "眼睛通过晶状体的调节实现对焦，看清远近不同的物体。",
]
KNOW_EN = [
    "The nervous system transmits signals through neurons at high speed.",
    "Plate tectonics explains the drift of continents over geological time.",
    "Binary search requires sorted input and runs in logarithmic time.",
    "Saturn has a ring system made mostly of ice particles.",
    "The Gilgamesh epic was written on clay tablets in cuneiform script.",
    "Ozone in the stratosphere absorbs most of the sun's ultraviolet radiation.",
]
TRANS = [
    ("The cat sits on the mat.", "猫坐在垫子上。"),
    ("Water boils at one hundred degrees.", "水在一百度时沸腾。"),
    ("The river flows into the sea.", "这条河汇入大海。"),
    ("Light travels faster than sound.", "光比声音传播得快。"),
    ("The moon has about one sixth of Earth's gravity.", "月球引力约为地球的六分之一。"),
    ("Spring is the season of new life.", "春天是万物复苏的季节。"),
    ("The machine converts heat into motion.", "这台机器把热转化成运动。"),
    ("Knowledge is a kind of wealth.", "知识是一种财富。"),
    ("The scientist tested the hypothesis with a controlled experiment.",
     "科学家用受控实验检验了这项假设。"),
    ("Silver is a better conductor of electricity than iron.", "银的导电性比铁好。"),
    ("The palace stands on the eastern bank of the river.", "宫殿坐落在河的东岸。"),
    ("Practice gives the athlete steady hands and a calm mind.",
     "练习给了运动员稳定的手和平静的心。"),
]


def make_arith(rng):
    op = rng.choice(list(OPS))
    a = int(rng.randint(2, 13))
    b = int(rng.randint(2, 13))
    c = {"加": a + b, "减": a - b, "乘": a * b}[op]
    form = AR_FORMS[int(rng.randint(len(AR_FORMS)))]
    q = form.format(a=a, b=b, op=op)
    return q, str(c)


def build_instruct_set(rng, n_arith=60):
    """A 段：造 (q,a) 对。返回 (样本列表, 各源计数)。"""
    samples = []
    for _ in range(n_arith):
        q, a = make_arith(rng)
        samples.append(dict(topic="算术", q=q, a=a, verifiable=True))
    for s in KNOW_ZH:
        samples.append(dict(topic="常识", q=f"解释这句话的意思：{s}", a=s,
                            verifiable=False))
    for s in KNOW_EN:
        samples.append(dict(topic="常识", q=f"What does this sentence mean? {s}", a=s,
                            verifiable=False))
    for en, zh in TRANS:
        samples.append(dict(topic="翻译", q=f"把这句话译成中文：{en}", a=zh,
                            verifiable=False))
    return samples


# ──────────────────────────────────────────────────────────────────────────
# 2. 工具
# ──────────────────────────────────────────────────────────────────────────

def count_tokens(text):
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    latin = len(re.findall(r"[A-Za-z0-9]+", text))
    return cjk + latin


def shingles(text, k=4):
    norm = re.sub(r"[\s=，。！？、；：”“（）()\-*+'.,;:!?]", "", text).lower()
    return [norm[i:i + k] for i in range(len(norm) - k + 1)]


# ──────────────────────────────────────────────────────────────────────────
# 3. B 段：质量三件套
# ──────────────────────────────────────────────────────────────────────────

def audit(samples):
    n = len(samples)
    # ① 指令去重
    seen, dup = set(), 0
    dedup = []
    for s in samples:
        if s["q"] in seen:
            dup += 1
            continue
        seen.add(s["q"])
        dedup.append(s)
    # ② 指令多样性：全部 q 的 4-gram 唯一率
    all_grams = []
    for s in dedup:
        all_grams += shingles(s["q"])
    uniq = len(set(all_grams))
    # ③ 回答长度分布
    lens = [count_tokens(s["a"]) for s in dedup]
    short = sum(1 for L in lens if L < 10)
    return {
        "n": n, "dup": dup, "n_out": len(dedup),
        "n_grams": len(all_grams), "uniq_grams": uniq,
        "diversity": uniq / (len(all_grams) + EPS),
        "short": short, "avg_len": sum(lens) / (len(lens) + EPS),
        "max_len": max(lens),
    }


# ──────────────────────────────────────────────────────────────────────────
# 4. C 段：三主题配比（条数视角 vs token 视角 + token 预算重采样）
# ──────────────────────────────────────────────────────────────────────────

def topic_balance(samples, target=(0.40, 0.30, 0.30), budget=900, seed=7):
    """按 token 预算重采样（同 01 章 C2 口径）。返回 (每类 spend, 缺口)。"""
    groups = {}
    for s in samples:
        groups.setdefault(s["topic"], []).append(s)
    rows = []
    for t, ratio in zip(TOPIC, target):
        g = groups.get(t, [])
        target_tok = ratio * budget
        order = list(range(len(g)))
        np.random.RandomState(seed).shuffle(order)
        tot = 0
        for i in order:
            if tot >= target_tok:
                break
            tot += count_tokens(g[i]["q"]) + count_tokens(g[i]["a"])
        rows.append({"topic": t, "spend": tot,
                     "shortfall": max(0.0, target_tok - tot)})
    return rows


TOPIC = ["算术", "常识", "翻译"]


def view(data):
    """双视角：每主题 (条数, 条数占比, token 数, token 占比)。"""
    cnt = Counter(s["topic"] for s in data)
    tok = Counter()
    for s in data:
        tok[s["topic"]] += count_tokens(s["q"]) + count_tokens(s["a"])
    n, T = len(data), sum(tok.values()) + EPS
    return [{"topic": t, "n": cnt.get(t, 0), "n_share": cnt.get(t, 0) / n,
             "tok": tok.get(t, 0), "tok_share": tok.get(t, 0) / T} for t in TOPIC]


def main():
    t0 = time.time()

    print("=" * 66)
    print("sft_data_demo：指令对构建 → 质量三件套 → 三主题配比（SFT 指令集）")
    print("=" * 66)

    # ── 段 A：构建 ──
    data = build_instruct_set(R)
    by_topic = Counter(s["topic"] for s in data)
    ver = sum(1 for s in data if s["verifiable"])
    print(f"[A] 指令对构建       共 {len(data)} 条 (q, a)（指令, 回答）对"
          f"：{dict(by_topic)}")
    print(f"    可验证（算术）{ver} 条 / 开放式（常识+翻译）{len(data)-ver} 条")
    e1 = data[0]
    print(f"    例：q={e1['q']}  →  a={e1['a']}（{e1['topic']}）")
    print()

    # ── 段 B：质量三件套 ──
    r = audit(data)
    print(f"[B] 质量三件套       指令去重 {r['dup']} 条 → 剩 {r['n_out']}/{r['n']}")
    print(f"    指令多样性：4-gram 唯一率 {100*r['diversity']:.1f}%"
          f"（{r['uniq_grams']}/{r['n_grams']}）")
    print(f"    回答长度：平均 {r['avg_len']:.1f} token / <10 token 占 "
          f"{100*r['short']/r['n_out']:.1f}% / 最长 {r['max_len']}")
    print()

    # ── 段 C：配比 ──
    rows = topic_balance(data)
    spend = {x["topic"]: x["spend"] for x in rows}
    total = sum(spend.values()) + EPS
    print("[C] 三主题配比：条数占比 ≠ token 占比（loss 里真实权重按 token）")
    print(f"    目标 token 预算 900：算术 40 / 常识 30 / 翻译 30，重采样后：")
    for v in view(data):
        bl = spend[v["topic"]] / total
        print(f"    {v['topic']:4s} {v['n']:3d} 条 / 条数占比 {100*v['n_share']:.1f}% "
              f"→ token 占比 原始 {100*v['tok_share']:.1f}% → 平衡后 "
              f"{100*bl:.1f}%")
    sf = [(x["topic"], x["shortfall"]) for x in rows if x["shortfall"] > 0]
    if sf:
        msg = " / ".join(f"{t}缺 {s:.0f} token" for t, s in sf)
        print(f"    池地板缺口（喂不满预算）：{msg} → 只能扩池（合成/采集）")
    print()
    print(f"墙钟 {1000*(time.time()-t0):.0f}ms（科学数字逐位一致，只有墙钟浮动）")


if __name__ == "__main__":
    main()
