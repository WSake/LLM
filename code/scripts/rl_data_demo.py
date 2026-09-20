# -*- coding: utf-8 -*-
"""
rl_data_demo.py
────────────────────────────────────────────────────────────────────────────
04-训练体系 05-RL数据与可验证奖励 的复现脚本（04 目录第 4 个引擎脚本）。

RLVR（RL with Verifiable Rewards）的"数据"不是 (q, 答案) 对，而是
(q, 候选输出, 可计算奖励) 的三元流。本篇测奖励信号的三个问题：
  A 奖励函数构造：同一个任务、三种打分粒度——二值精确匹配 / 数值接近度
      部分 credit / 分维 credit。量"奖励有信息量"的样本占比：reward 在
      4 候选间标准差>0 才算这题能给 RL 提供梯度（reward 全恒零=没信息）。
  B 奖励质检：① 一致性——纯规则函数同输入必同输出（确定性 100%、三跑
      逐位一致）；② 两种偷懒 checker 的假阳性/假阴性——只查末位的放行
      多少错答案（假阳性），只认"小于 99"的误杀多少对答案（假阴性）。
  C 难度配比（V1 二值口径）：易档近乎全对、难档近乎全错，reward 都几乎
      恒定=没梯度；中间档才有信息。量"信息样本占比" vs 难度曲线，并做
      只取中档的重采样对比。

纯 CPU、固定随机种子，一键可复现。
复现声明：run1==run2==run3 科学数字逐位一致，只有墙钟在跑次间浮动。
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

R = np.random.RandomState(13)
EPS = 1e-12

# 三档命中概率：精确命中 h（RLVR 里 c==t 才算对）
PAR = {"易": (0.92, 9, 20, 3), "中": (0.55, 21, 45, 9), "难": (0.08, 90, 99, 9)}


# ──────────────────────────────────────────────────────────────────────────
# 1. 任务与候选生成（模拟模型采样：精确命中/近失/中程/远错）
# ──────────────────────────────────────────────────────────────────────────

def gen_cands(true, h, n):
    cands = []
    for _ in range(n):
        r = R.rand()
        if r < h:                                          # 精确命中
            cands.append(true)
        elif r < h + 0.10:                                 # 近失（±1）
            cands.append(true + int(R.choice([-1, 1])))
        elif r < h + 0.30:                                 # 中程偏差
            cands.append(true + int(R.choice([-1, 1]) * R.randint(3, 25)))
        else:                                              # 远错
            cands.append(true + int(R.choice([-1, 1]) * R.randint(60, 300)))
    return cands


def make_batch(difficulty, n):
    h, lo, hi, bmax = PAR[difficulty]
    qs, trues, batches = [], [], []
    for _ in range(n):
        a, b = int(R.randint(lo, hi + 1)), int(R.randint(2, bmax))
        true = a * b
        qs.append(f"{a}×{b}=？")
        trues.append(true)
        batches.append(gen_cands(true, h, 4))
    return list(zip(qs, batches, trues))


# ──────────────────────────────────────────────────────────────────────────
# 2. 三版奖励函数（打分粒度 → 信息量）
# ──────────────────────────────────────────────────────────────────────────

def r_v1(c, t):            # 二值：精确匹配
    return 1.0 if c == t else 0.0


def r_v2(c, t):            # 数值接近度 partial credit
    return max(0.0, 1.0 - abs(c - t) / (t + EPS))


def r_v3(c, t):            # 分维：在真值±10 内给半对，否则 0
    return 0.5 + 0.5 * int(abs(c - t) <= 10)


def info_frac(batch, rfun):
    """reward 在单题 4 候选间标准差>0 的题占比＝能喂梯度的样本比例。"""
    n = frac = 0.0
    for q, cands, t in batch:
        vals = [rfun(c, t) for c in cands]
        n += 1
        if np.std(vals) > 0:
            frac += 1
    return frac / (n + EPS)


# ──────────────────────────────────────────────────────────────────────────
# 3. B 段：两种偷懒 checker 的假阳/假阴
# ──────────────────────────────────────────────────────────────────────────

def lazy_tail(c, t):     # 偷懒① 只查末位：c 与 t 模 10 同余即放行
    return abs(c - t) % 10 == 0


def lazy_small(c, t):    # 偷懒② 只认 ≤99：大数一律误杀
    return c <= 99


def audit(batch, checker):
    pos = neg = fp = fn = 0
    for q, cands, t in batch:
        for c in cands:
            exact = (c == t)
            verdict = checker(c, t)
            pos += int(exact); neg += int(not exact)
            fp += int((not exact) and verdict)   # 假阳性：错答案被放行
            fn += int(exact and (not verdict))   # 假阴性：对答案被误杀
    return pos, neg, fp, fn


# ──────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("=" * 66)
    print("rl_data_demo：奖励构造 → 奖励质检 → 难度配比（RL 数据·可验证奖励）")
    print("=" * 66)

    # ── 段 A：奖励函数构造 ──
    mid = make_batch("中", 60)
    i1, i2, i3 = (info_frac(mid, r) for r in (r_v1, r_v2, r_v3))
    scores = {k: [r(c, t) for _, cs, t in mid for c in cs]
              for k, r in (("v1", r_v1), ("v2", r_v2), ("v3", r_v3))}
    nbin = {k: len(set(v)) for k, v in scores.items()}
    print("[A] 奖励函数构造     60 道两位数乘法 × 4 候选 = 240 个输出，三种打分粒度:")
    print(f"    V1 二值（精确匹配）       分数档数 {nbin['v1']:2d}（{{0,1}}，"
          f"错 1 与错 300 同罪）· 信息样本 {100*i1:.1f}%")
    print(f"    V2 数值接近度 partial     分数档数 {nbin['v2']:2d}（连续折线，"
          f"越近真值越高分）· 信息样本 {100*i2:.1f}%")
    print(f"    V3 分维（±10 内半对）    分数档数 {nbin['v3']:2d}（"
          f"只拆出『半对/对』一档）· 信息样本 {100*i3:.1f}%")
    q, cands, t = mid[0]
    print(f"    例：{q} 真值 {t}，候选 {cands}")
    print(f"        V1={[round(r_v1(c, t), 2) for c in cands]}   "
          f"V2={[round(r_v2(c, t), 2) for c in cands]}   "
          f"V3={[round(r_v3(c, t), 2) for c in cands]}")
    print()

    # ── 段 B：奖励质检 ──
    pos, neg, fp, fn = audit(mid, lazy_tail)
    pos2, neg2, fp2, fn2 = audit(mid, lazy_small)
    print("[B] 奖励质检")
    print(f"    ① 一致性：规则函数确定性 100%（构造保证——r_v1/v2/v3 是"
          f"无随机状态的纯函数，同输入必同输出，三跑逐位一致）；"
          f"LLM-as-judge 的游移打分在本靶场不出现（非本机实测行业常态）")
    print(f"    ② 偷懒① 只查末位：判对 {pos} / 判错 {neg}"
          f" → 假阳性 {fp}（错答案末位撞车被放行）· 假阴性 {fn}")
    print(f"    ③ 偷懒② 只认≤99：判对 {pos2} / 判错 {neg2}"
          f" → 假阳性 {fp2} · 假阴性 {fn2}（超 99 的正确答案被误杀）")
    print()

    # ── 段 C：难度配比（V1 二值口径） ──
    print("[C] 难度配比：信息带在哪一档（V1 二值口径）")
    rows = {}
    for d in ("易", "中", "难"):
        b = make_batch(d, 40)
        frac = info_frac(b, r_v1)
        rows[d] = frac
        mark = "（近乎全对，reward 饱和=没梯度）" if d == "易" else \
               ("（中间带，reward 抓得住）" if d == "中" else
                "（近乎全错，reward 恒定=没梯度）")
        print(f"    {d:2s}档 40 题 → 信息样本 {100*frac:.1f}%{mark}")
    all3 = make_batch("易", 40) + make_batch("中", 40) + make_batch("难", 40)
    frac_all = info_frac(all3, r_v1)
    frac_mid = rows["中"]
    print(f"    均匀三档混批信息样本 {100*frac_all:.1f}% → 只取中档 {100*frac_mid:.1f}%"
          f"（+{100*(frac_mid-frac_all):.1f}pp，取材搬向信息带）")
    print()
    print(f"墙钟 {1000*(time.time()-t0):.0f}ms（科学数字逐位一致，只有墙钟浮动）")


if __name__ == "__main__":
    main()
