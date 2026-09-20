# -*- coding: utf-8 -*-
"""
pref_data_demo.py
────────────────────────────────────────────────────────────────────────────
04-训练体系 04-Preference数据与Reward-Model 的复现脚本（04 目录第 3 个引擎脚本）。

演示偏好数据（(指令, chosen 好回答, rejected 差回答) 三元组）的三个问题：
  A 偏好对构造：两路来源——自动域（算术，规则重算自动判 chosen/rejected，
      便宜且答案恒正确）vs 开放域（常识，人工标 chosen/rejected，可能标反）；
  B 偏好数据质检三件套：
      ① 自洽性复核（自动域规则重算 0 冲突；开放域启发式「chosen 不该比
        rejected 短」——抓污染会同时漏报+误报，如实量出来）；
      ② 长度偏置（chosen 是否系统性更长——偏好数据最著名的共线）；
      ③ 近同弱标（chosen/rejected 回答几乎相同的"弱标注"占比）；
  C 长度捷径审计：把"长度是质量信号"当唯一证据投票判 chosen——算术域等长
      零信号（50%）、常识域 83.3%——对照真值 100%，量出"RM 只要学长度就能
      拿到多少分"（verbose 偏置的机制起点）。

纯 CPU、固定随机种子，一键可复现。
复现声明：run1==run2==run3 科学数字逐位一致，只有墙钟在跑次间浮动。
"""
import sys
import re
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

R = np.random.RandomState(11)
EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────
# 1. 语料与生成器
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
    "光合作用把二氧化碳和水转化为氧气和葡萄糖。",
    "秦始皇在公元前二百二十一年统一六国，建立中央集权制度。",
    "眼睛通过晶状体的调节实现对焦，看清远近不同的物体。",
    "太阳系有四颗类地行星，从内到外依次是水星、金星、地球和火星。",
    "月球的引力只有地球的六分之一。",
    "光的传播速度约为每秒三十万公里，比声音快很多。",
    "人体骨骼成年后共有二百零六块。",
]
KNOW_EN = [
    "The nervous system transmits signals through neurons at high speed.",
    "Plate tectonics explains the drift of continents over geological time.",
    "Binary search requires sorted input and runs in logarithmic time.",
    "Saturn has a ring system made mostly of ice particles.",
    "The Gilgamesh epic was written on clay tablets in cuneiform script.",
    "Ozone in the stratosphere absorbs most of the sun's ultraviolet radiation.",
]
KNOWS = KNOW_ZH + KNOW_EN


def flip_digit(txt):
    """把末位数字改 ±1 且保持位数（chosen/rejected 等长，供算术域用）。"""
    d = int(txt[-1])
    delta = -1 if d == 9 else (1 if d == 0 else R.choice([-1, 1]))
    return txt[:-1] + str(d + delta)


def make_arith_pair(rng):
    op = rng.choice(list(OPS))
    a = int(rng.randint(2, 13))
    b = int(rng.randint(2, 13))
    if op == "减":
        b = int(rng.randint(1, a))        # 保证 a-b > 0，不出负数答案
    c = {"加": a + b, "减": a - b, "乘": a * b}[op]
    form = AR_FORMS[int(rng.randint(len(AR_FORMS)))]
    q = form.format(a=a, b=b, op=op)
    return q, str(c), flip_digit(str(c))


def make_know_rej(s, idx):
    """开放域 rejected 两种形态：短型（chosen 更长）/ 长型（rejected 更长）。"""
    if idx % 9 < 8:          # 8/9 是短型：rejected 删尾
        cut = max(1, len(s) * 60 // 100)
        return s[:cut] + "..."
    return s * 2             # 1/9 长型：rejected 比 chosen 长（现实里的标注噪声源）


def build_pairs(rng, n_arith=60):
    pairs = []
    for _ in range(n_arith):
        q, ok, bad = make_arith_pair(rng)
        pairs.append(dict(domain="算术", q=q, chosen=ok, rejected=bad,
                          src="自动", polluted=False))
    for i, s in enumerate(KNOWS):
        pairs.append(dict(domain="常识", q=f"解释这句话的意思：{s}",
                          chosen=s, rejected=make_know_rej(s, i),
                          src="人工", polluted=False))
    # 注入 3 条标反污染（2 短型 + 1 长型，idx 0/9 短型、8 长型），作审计靶子
    for k in (0, 9, 8):
        p = pairs[len(pairs) - len(KNOWS) + k]
        p["chosen"], p["rejected"] = p["rejected"], p["chosen"]
        p["polluted"] = True
    return pairs


# ──────────────────────────────────────────────────────────────────────────
# 2. 工具
# ──────────────────────────────────────────────────────────────────────────

def count_tokens(text):
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    latin = len(re.findall(r"[A-Za-z0-9]+", text))
    return cjk + latin


def ans_ngrams(text, k=4):
    sh = set()
    if len(text) < k:                      # 短串兜底：用字符级，避免双空集恒 1.0
        return set(text)
    for i in range(len(text) - k + 1):
        sh.add(text[i:i + k])
    return sh


def jaccard(a, b):
    sa, sb = ans_ngrams(a), ans_ngrams(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / (len(sa | sb) + EPS)


# ──────────────────────────────────────────────────────────────────────────
# 3. B 段：质检三件套
# ──────────────────────────────────────────────────────────────────────────

def audit(pairs):
    arith_ok = arith_diff = near = 0
    susp, know_long, know = [], 0, 0
    gap_sum = gap_n = 0.0
    for i, p in enumerate(pairs):
        lc, lr = count_tokens(p["chosen"]), count_tokens(p["rejected"])
        if p["domain"] == "算术":
            m = re.search(r"(\d+)\s*([加减乘])\s*(\d+)", p["q"])
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            ok = {"加": a + b, "减": a - b, "乘": a * b}[op]
            arith_ok += int(str(ok) == p["chosen"])
            arith_diff += int(lc != lr)
        else:
            know += 1
            if lc > lr:
                know_long += 1
            else:
                susp.append((p["polluted"], p["q"][:6]))
            gap_sum += max(lc, lr) - min(lc, lr)
            gap_n += 1
        near += int(jaccard(p["chosen"], p["rejected"]) >= 0.85)
    return dict(arith_ok=arith_ok, arith_diff=arith_diff,
                susp=susp, know_long=know_long, know=know,
                near=near, n=len(pairs), gap=gap_sum / gap_n)


# ──────────────────────────────────────────────────────────────────────────
# 4. C 段：长度捷径审计（长度投票 vs 真值）
# ──────────────────────────────────────────────────────────────────────────

def length_vote(pairs):
    by = {"算术": [0.0, 0], "常识": [0.0, 0]}
    for p in pairs:
        lc, lr = count_tokens(p["chosen"]), count_tokens(p["rejected"])
        v, tot = by[p["domain"]]
        if lc > lr:
            by[p["domain"]] = [v + 1, tot + 1]
        elif lc < lr:
            by[p["domain"]] = [v, tot + 1]
        else:
            by[p["domain"]] = [v + 0.5, tot + 1]     # 平局保守计 0.5
    return by


# ──────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("=" * 66)
    print("pref_data_demo：偏好对构造 → 质检三件套 → 长度捷径审计（偏好数据）")
    print("=" * 66)

    # ── 段 A：构造 ──
    data = build_pairs(R)
    c = Counter(p["domain"] for p in data)
    pol = sum(1 for p in data if p["polluted"])
    src_o = sum(1 for p in data if p["src"] == "人工")
    print(f"[A] 偏好对构造       共 {len(data)} 对 (指令, chosen, rejected)="
          f"{dict(c)}；注入标反污染 {pol} 条")
    print(f"    自动域（可验证·规则重算判 chosen）{len(data)-src_o} 对 / 开放域"
          f"（人工标）{src_o} 对")
    e = data[0]
    print(f"    例：trace q={e['q']}  chosen={e['chosen']}  rejected={e['rejected']}")
    print()

    # ── 段 B：质检 ──
    r = audit(data)
    print("[B] 质检三件套")
    print(f"    ① 自洽复核：算术域规则重算对齐 {r['arith_ok']}/60（0 冲突）；"
          f"开放域启发式「chosen 比 rejected 短→可疑」抓 {len(r['susp'])} 候 "
          f"→ 真污染 {sum(s[0] for s in r['susp'])} · 误报 "
          f"{sum(not s[0] for s in r['susp'])}（长回答误伤）")
    print(f"    ② 长度偏置：算术域 chosen/rejected 等长 {60-r['arith_diff']}/60"
          f"（长度零信息）；常识域 chosen 更长 {r['know_long']}/{r['know']}="
          f"{100*r['know_long']/r['know']:.1f}%（平均长差 {r['gap']:.0f} token）")
    print(f"    ③ 近同弱标：chosen/rejected 回答 Jaccard≥0.85 占 "
          f"{100*r['near']/r['n']:.1f}%（{r['near']}/{r['n']}）")
    print()

    # ── 段 C：长度捷径 ──
    lv = length_vote(data)
    print("[C] 长度捷径审计：把'长度'当质量信号，RM 能拿到多少判对率")
    print("    （真值 = 构造时预知的正确答案；注入的 3 条标反连真值都不守恒）")
    for d in ("算术", "常识"):
        hit, tot = lv[d]
        tag = "——长度无信号" if d == "算术" else "——长度与标签共线"
        print(f"    {d:4s} 真值 100% vs 长度投票 {100*hit/tot:.1f}%"
              f" ({hit:.0f}/{tot:.0f}){tag}")
    print()
    print(f"墙钟 {1000*(time.time()-t0):.0f}ms（科学数字逐位一致，只有墙钟浮动）")


if __name__ == "__main__":
    main()
