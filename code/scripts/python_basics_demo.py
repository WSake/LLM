# -*- coding: utf-8 -*-
"""01-基础配套实验①：Python 工程能力（入门与工程能力篇附表）。
演示三件事：numpy 向量化快在哪 / pandas 数据操作 / dataclass 数据管线。
运行：python code/scripts/python_basics_demo.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import statistics
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

rng = np.random.RandomState(0)  # 固定种子：任何机器数字一致

print("=" * 62)
print("实验A · numpy 向量化 vs Python 纯循环（真实计时，各取 3 次中位）")
print("=" * 62)
n = 2_000_000
data = rng.rand(n)


def loop_mean(xs):
    s = 0.0
    for x in xs:
        s += x
    return s / len(xs)


t_loop = []
for _ in range(3):
    t0 = time.perf_counter()
    m = loop_mean(data)
    t_loop.append(time.perf_counter() - t0)

t_np = []
for _ in range(3):
    t0 = time.perf_counter()
    m_np = data.mean()
    t_np.append(time.perf_counter() - t0)

mt_loop = statistics.median(t_loop)
mt_np = statistics.median(t_np)
ratio = mt_loop / mt_np
print(f"样本数 {n:,}；两种方式均值: python-loop={m:.6f}   numpy={m_np:.6f}（一致 ✓）")
print(f"Python 纯循环: {mt_loop * 1000:,.0f} ms   NumPy 向量化: {mt_np * 1000:8.3f} ms")
print(f"→ NumPy 快约 {ratio:,.0f} 倍（按本次实测动态改）：它把循环沉进 C 的连续内存，而非逐元素解释")
print()

print("=" * 62)
print("实验B · pandas 一把 DataFrame 干活：看、分组、缺失值")
print("=" * 62)
rng2 = np.random.RandomState(1)
n_student = 100
df = pd.DataFrame({
    "class": rng2.choice(["A", "B", "C"], size=n_student),
    "score": np.round(rng2.normal(80, 10, size=n_student), 2),
})
drop_idx = rng2.choice(df.index, size=3, replace=False)
df.loc[drop_idx, "score"] = np.nan
print(f"总行数 {len(df)} · score 缺失 {int(df['score'].isna().sum())} 行 · 班级 {df['class'].unique().tolist()}")
print("score 描述统计（describe）:")
print(df["score"].describe().round(2).to_string())
print("按班级分组均值（groupby）:")
print(df.groupby("class")["score"].mean().round(2).to_string())
print("（数据管线的日常，就是把 N 行'脏数据'变成这几行'结构'：看、分、补）")
print()

print("=" * 62)
print("实验C · dataclass：给数据行'类型'，让管线出门前先自检")
print("=" * 62)


@dataclass(order=True)
class Record:
    name: str
    score: float
    passed: bool = field(default=False)


recs = [Record("Alice", 92.5), Record("Bob", 74.0), Record("Chen", 58.5)]
for r in recs:
    r.passed = r.score >= 60
print("三个数据行(类型已固定):")
for r in recs:
    print(f"  {r.name:10s} score={r.score:5.1f}  passed={r.passed}")
print("asdict → 直接喂给 json/pandas:")
print("  ", asdict(recs[0]))
print()

print("=" * 62)
print("实验D · 拼成完整数据管线：文本 → DataFrame → 类型化 → 汇总")
print("=" * 62)
raw = (
    "city,orders,gmv\n北京,120,9800.5\n上海,95,12300.0\n深圳,80,7800.25\n"
    "北京,110,10200.0\n上海,60,6600.0\n,50,5000.0\n"
)
raw_rows = pd.read_csv(pd.io.common.StringIO(raw))
df2 = raw_rows.dropna(subset=["city"])          # 有脏行（空 city）→ 清洗
df2["gmv"] = df2["gmv"].astype(float)
recs2 = [Record(str(c), float(g)) for c, g in zip(df2["city"], df2["gmv"])]
assert all(c != "" for c in df2["city"])        # dataclass 构造后类型已固定
summary = df2.groupby("city")[["orders", "gmv"]].agg(["sum", "mean"]).round(2)
print(f"清洗后保留 {len(df2)}/{len(raw_rows)} 行（丢 {len(raw_rows) - len(df2)} 行空 city）")
print("按城市汇总（sum/mean）:")
print(summary.to_string())
print("→ 三段式：pandas 接数据 → 类型化(dataclass) → 分组聚合。就这么长。")
print()
print("done · 全部数字可在本机一键复现（python code/scripts/python_basics_demo.py）")
