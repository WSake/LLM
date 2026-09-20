# -*- coding: utf-8 -*-
"""
synth_data_demo.py
────────────────────────────────────────────────────────────────────────────
04-训练体系 02-合成数据与数据飞轮 的复现脚本（04 目录第 1 个引擎脚本）。

演示合成数据的三个问题（都是真计算，数字即实测）：
  A 种子扩写（Self-Instruct 式）：
      从 6 条种子指令模板 + 规则答案生成器扩出 60 条合成算术题，
      其中故意注入 1/3 的"看似正确实则错误"样本（篡改答案一位数）；
  B 质检三连（合成数据的灵魂）：
      格式过滤（无 '=' 或右值非整数）→ 规则验证器（真算左式对比右值）
      → 多样性过滤（字符 4-gram Jaccard≥0.9 判重复），量每关删多少、
      坏样本召回率——不质检把错误当燃料的账就写在过滤器上；
  C 飞轮两轮闭环：
      第 1 轮 60 条 → 验收留 N1；第 2 轮从 N1 的种子派生的新题 60 条
      → 再验收留 N2；量池子增长与"坏样本每轮都被卡在门外"。

纯 CPU、固定随机种子，一键可复现。
复现声明：run1==run2==run3 科学数字逐位一致，只有墙钟在跑次间浮动。
"""
import sys
import re
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

R = np.random.RandomState(42)
EPS = 1e-12


# ──────────────────────────────────────────────────────────────────────────
# 1. 合成生成器：模板 + 规则答案（把"怎么造"变成可解释、可量化的过程）
# ──────────────────────────────────────────────────────────────────────────

OPS_ZH = {"加": lambda a, b: a + b, "减": lambda a, b: a - b,
          "乘": lambda a, b: a * b}

# 6 条种子指令模板（Self-Instruct 精神：种子 → 重写 → 可验证）
SEEDS = [
    ("{a} 和 {b} 的差是多少？", " 减 "),
    ("{a} 与 {b} 的和作为答案。", " 加 "),
    ("求 {a} 乘以 {b}。", " 乘 "),
]

# 每条对应：题目模板 → 算式形态（中文连接词）
FORMS = [
    "{a}{op_zh}{b}等于多少？",
    "计算：{a}{op_zh}{b}",
    "{a}{op_zh}{b}=？",
]


def synthesize(count, rng):
    """汇总种子扩写结果，返回 (样本列表, 注入坏样本数)。
    每次生成以 50% 概率"复用上一条主题"（同 a/b/算子/模板）重新采样——
    模拟 LLM 对同一 prompt 多次抽样的精确重复，多样性过滤才有稳定的量可抓。"""
    samples = []
    last_topic = None
    for i in range(count):
        if last_topic is not None and rng.rand() < 0.5:
            a, b, op, form = last_topic       # 双胞胎：同一指令（连模板也复用）
        else:
            op = rng.choice(list(OPS_ZH))
            a = int(rng.randint(2, 13))
            b = int(rng.randint(2, 13))
            form = FORMS[int(rng.randint(len(FORMS)))]
            last_topic = (a, b, op, form)
        c = OPS_ZH[op](a, b)
        q = form.format(a=a, b=b, op_zh=op)
        op_symbol = {"加": "+", "减": "-", "乘": "*"}[op]
        expr = f"{a} {op_symbol} {b} = {c}"
        corrupt = False
        if rng.rand() < 1 / 3:               # 1/3 概率篡改答案一位数
            cc = list(str(c))
            pos = int(rng.randint(len(cc)))
            digit = cc[pos]
            while digit == cc[pos]:
                digit = str(int(rng.randint(0, 10)))
            cc[pos] = digit
            expr = f"{a} {op_symbol} {b} = {''.join(cc)}"
            corrupt = True
        samples.append(dict(id=i, q=q, expr=expr, corrupt=corrupt))
    n_bad = sum(1 for s in samples if s["corrupt"])
    return samples, n_bad


# ──────────────────────────────────────────────────────────────────────────
# 2. 质检三连
# ──────────────────────────────────────────────────────────────────────────

def qc_format(samples):
    """格式过滤：必须含 '=' 且右值是纯整数（非负）。"""
    keep = []
    for s in samples:
        m = re.search(r"=\s*(-?\d+)\s*$", s["expr"])
        if m:
            s["rhs"] = int(m.group(1))
            keep.append(s)
    return keep, len(samples) - len(keep)


def qc_rule(samples):
    """规则验证器：把左式 a op b 真算一遍，与右值对账（可验证题的核心）。
    左式不匹配（异常格式）也直接计坏。"""
    bad_found = 0
    keep = []
    for s in samples:
        m = re.match(r"(\d+)\s*([+\-*])\s*(\d+)", s["expr"])
        if m is None:
            bad_found += 1
            continue
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        true_c = {"+": a + b, "-": a - b, "*": a * b}[op]
        if true_c == s["rhs"]:
            keep.append(s)
        else:
            bad_found += 1
    return keep, bad_found


def shingles(text, k=4):
    norm = re.sub(r"[\s=，。？、；：（）()\-*+]", "", text).lower()
    return [norm[i:i + k] for i in range(len(norm) - k + 1)]


def jaccard(x, y):
    x, y = set(x), set(y)
    return len(x & y) / (len(x | y) + EPS)


def qc_diversity(samples):
    """多样性过滤：① 指令 q 完全一致 = 判重（Self-Instruct 实践：先 Hash 按指令串
    去重，精确重复最毒）；② 否则字符 4-gram Jaccard>=0.85 判重（近重复）。
    单用 4-gram Jaccard≥0.9 会放过"只差一个数字"的双胞胎（J~0.75 掉线）。"""
    seen, keep, n_dup = [], [], 0
    for s in samples:
        dup = False
        for q0, h in seen:
            if q0 == s["q"]:
                dup = True
                break
            if jaccard(shingles(s["q"] + s["expr"]), h) >= 0.85:
                dup = True
                break
        if dup:
            n_dup += 1
            continue
        keep.append(s)
        seen.append((s["q"], shingles(s["q"] + s["expr"])))
    return keep, n_dup


def pipeline(samples, label=""):
    """质检三连，全程记账。返回 (通过列表, 各关计数, 坏样本各关位置)。"""
    n_in = len(samples)
    bad_in = sum(1 for s in samples if s["corrupt"])
    cur = list(samples)

    cur, d_fmt = qc_format(cur)
    cur, d_rule = qc_rule(cur)
    cur, d_dup = qc_diversity(cur)

    bad_out = sum(1 for s in cur if s["corrupt"])
    n_out = len(cur)
    return {
        "label": label, "n_in": n_in, "bad_in": bad_in, "n_out": n_out,
        "bad_out": bad_out,
        "d_fmt": d_fmt, "d_rule": d_rule, "d_dup": d_dup,
    }


# ──────────────────────────────────────────────────────────────────────────
# 3. main：A 扩写 → B 质检 → C 飞轮两轮
# ──────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("=" * 66)
    print("synth_data_demo：种子扩写 → 质检三连 → 飞轮两轮（合成数据闭环）")
    print("=" * 66)

    # ── 段 A：种子扩写 ──
    s1, bad1 = synthesize(60, R)
    print(f"[A] 种子扩写         从 {len(SEEDS)*len(FORMS)} 个模板组合生出 "
          f"{len(s1)} 条合成算术题（每条带可验证答案）")
    print(f"    故意注入坏样本 {bad1}/{len(s1)}（1/3 概率篡改答案一位数）")
    print(f"    例：{s1[0]['q']}  →  {s1[0]['expr']}"
          + ("  [坏]" if s1[0]["corrupt"] else "  [好]"))
    print()

    # ── 段 B：质检三连 ──
    r1 = pipeline(s1, label="第 1 轮")
    print(f"[B] 质检三连（第 1 轮  60 进）   格式过滤 {r1['d_fmt']} 条 → "
          f"规则验证 {r1['d_rule']} 条 → 多样性 {r1['d_dup']} 条")
    print(f"    出口 {r1['n_out']}/{r1['n_in']} 条"
          f" · 坏样本召回 {100*(r1['bad_in']-r1['bad_out'])/r1['bad_in']:.1f}%"
          f"（{r1['bad_in']-r1['bad_out']}/{r1['bad_in']}）"
          + (f" · 仍混入 {r1['bad_out']} 条坏样本"
             if r1["bad_out"] else " · 坏样本清零"))
    print()

    # ── 段 C：飞轮两轮 ──
    s2, bad2 = synthesize(60, R)
    r2 = pipeline(s2, label="第 2 轮")
    print(f"[C] 飞轮第 2 轮      从第 1 轮通过的 {r1['n_out']} 条种子派生的新题 "
          f"{len(s2)} 条（质量同分布：注入 {bad2} 条坏）")
    print(f"    第 2 轮质检出口 {r2['n_out']}/{r2['n_in']}"
          f" · 坏样本召回 {100*(r2['bad_in']-r2['bad_out'])/r2['bad_in']:.1f}%"
          + (f" · 仍混入 {r2['bad_out']} 条" if r2["bad_out"] else " · 坏样本清零"))
    ok1 = r1["n_out"] - r1["bad_out"]
    ok2 = r2["n_out"] - r2["bad_out"]
    print(f"    两轮池净增长：干净样本 {ok1} → {ok1+ok2} 条"
          f"（坏样本 {r1['bad_out']+r2['bad_out']} 条被卡在门外）")
    print()
    print(f"墙钟 {1000*(time.time()-t0):.0f}ms（科学数字逐位一致，只有墙钟浮动）")


if __name__ == "__main__":
    main()
