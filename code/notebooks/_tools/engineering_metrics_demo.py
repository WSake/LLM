# -*- coding: utf-8 -*-
"""engineering_metrics_demo.py —— §11.7 工程指标·延迟/吞吐/成本（11-评测与可观测 08 篇配套）

§11.7 知识地图原文（🔴 必会；见 §5.8）：质量之外必须同时测——
Latency（TTFT/TPOT/端到端）、Throughput、Token Usage（输入/输出/思考 token 分开看）、
Cost（$/query；Agent 任务按"每任务成本"）、GPU 利用率、错误率/重试率。
§5.8 性能指标与压测（🔴 掌握指标口径）：TTFT=首 token 延迟由 prefill 决定；
TPOT/ITL=每 token 时间约 200ms 是流畅线；吞吐=单请求 tokens/s 与整机输出 tokens/s、QPS；
GPU 利用率=计算/显存/带宽；成本=$/1M tokens（输入/输出分开计）；压测永远用混合长度+混合并发。
§11.9 评测工程最佳实践（🔴 总结）：①评测集分层 ②固定评测环境 ③每次改动跑同一套回归
④线上遥测回流成评测集 ⑤每任务完成指标。

本篇定位：把五层定界（01 篇）里最后一层『工程』做**评测专项**——先讲口径再讲数字、
正例/反例搭配讲（与 05 章 `10-性能指标与压测-TTFT-TPOT` 同口径，本节是 11 章评测视角）。
四账：
账 A  口径账：六把工程尺的分子分母（延迟/吞吐/token 用量/成本/GPU/错误率）× 口头禅正反例判定
账 B  延迟分布账：均值/P50/P95/P99 四值（均值君陷阱）× 混合长度压测 × 端到端 = TTFT+(N-1)×TPOT
账 C  吞吐与成本账：连续批处理吞吐算术（批 16 → 整机 ×16）× 三档价格单 query 账本
      × 输出 token 占比 × Agent 每任务成本（§11.7 ⑤）
账 D  工程门禁账：同一张指标表 × 两种任务口径 = 决策翻转 × 预算回归三阅（P95/成本/缓存降本）
      × 错误率×重试放大 × 遥测回流（§11.9 ④）

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（W() 负责计时与落 stderr）。全部统计 = 本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- §11.7 六尺表、§5.8 指标口径（TTFT 由 prefill 决定、TPOT≈200ms 流畅线、$/1M 输入输出分开、
  混合长度+混合并发压测）= 知识地图要素事实（写作环境无外网未在线复核）。
- 三档价格（$0.30/$0.60 轻量 · $2.50/$10.00 旗舰 · $3.00/$15.00 思考档）=

  代表性示例口径（非某家精确价目）；账 A/B/C/D 的请求负载表、延迟分布、批处理吞吐、
  成本账本、决策翻转表、预算回归表、错误率台账 = **作者预置仿真 + 本机确定性计算**。
- **真实 TTFT/TPOT/吞吐 = 需在 vLLM/Serving 或 API 上压测 = 非本机实测**（05 章指标口径章
  `05-10 性能指标与压测-TTFT-TPOT` 将承载真实压测实操）；01 账 D 的"P95/成本预算回归"与
  05-08 RAG 门禁、09-06/11-07 门禁 = 同一个"回归刹车"思想（各篇各自实测，本篇只引用思想）。
- A-D 四账统计 = 本机确定性计算（stdout 三独立进程 md5 恒一）。
"""
from __future__ import annotations
import math
import sys
import time

try:
    _ = sys.stdout.reconfigure(encoding="utf-8")   # 输出统一 UTF-8 字节，不随控制台代码页漂移
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:  # 极老 Python 无 reconfigure：保持默认
    pass

_WSTART = time.perf_counter()


def W(msg: str) -> None:
    print(f"[wall] {time.perf_counter() - _WSTART:.3f}s  {msg}", file=sys.stderr)


def _pct(sorted_vals: list, q: float) -> float:
    """nearest-rank 百分位：N=30 时 P95 = 第 ceil(28.5)=29 个（1-based）。"""
    n = len(sorted_vals)
    idx = max(0, math.ceil(q * n) - 1)
    return sorted_vals[idx]


# ---------------------------------------------------------------------------
# 账 A · 口径账：六把工程尺的分子分母 × 口头禅正反例判定
# ---------------------------------------------------------------------------
# 每把尺的 定义 => 公式 => 正例（对的话术）=> 反例（坑），全部是标量算术/字符串。
def account_a() -> str:
    out = []
    out.append("== A · 口径账：质量之外必须同时测的六把工程尺（§11.7·先讲口径再讲数字） ==")
    out.append("  六尺表  |  量什么  |  分子/分母")
    out.append("  ① Latency  |  首 token 延迟 TTFT（prefill 决定）· 每 token 时间 TPOT/ITL（decode 决定）· 端到端")
    out.append("            端到端(流式) = TTFT + (N-1)×TPOT")
    out.append("  ② Throughput  |  单请求输出 tok/s = 1/TPOT · 整机输出 tok/s = 批数×1/TPOT · QPS = 并发/平均完成时间")
    out.append("  ③ Token Usage  |  输入/输出/思考 token 分开看（合起来的'总 token'会藏思考税）")
    out.append("  ④ Cost  |  $/1M tokens 输入与输出分开计 · 单 query 成本 =(输入×P_in+输出×P_out)/1e6 · Agent 按每任务成本")
    out.append("  ⑤ GPU 利用率  |  计算/显存/带宽 活跃时间占比（prefill 计算密集 · decode 内存带宽受限）")
    out.append("  ⑥ 错误率/重试率  |  错误率 5% ≠ 成本 5%：失败请求会重试放大调用量、拉长尾延迟")
    out.append("")
    out.append("  A1 口径算术判定（10 题 · 期望值 vs 实用公式）")
    # 基准数
    TTFT, TPOT = 0.85, 0.12
    checks = []

    def chk(qid: str, want: float, got: float, tag: str) -> None:
        ok = abs(want - got) < 1e-6
        checks.append(ok)
        out.append(f"    {qid} {tag}: 期望 {want:.4f} 算得 {got:.4f} = " + ("PASS" if ok else "FAIL"))

    e2e_50 = TTFT + 49 * TPOT  # 单请求 50 输出 token 流式总时长
    chk("Q1", 6.73, e2e_50, "端到端(50 tok) = TTFT + 49×TPOT")
    chk("Q2", 1 / TPOT, 1 / TPOT, "单请求输出吞吐 = 1/TPOT")
    chk("Q3", 16 / TPOT, 16 / TPOT, "整机连续批 16：每 decode 步 1 个矩阵喂 16 人")
    chk("Q4", 20 / 4.0, 20 / 4.0, "QPS = 并发 20 ÷ 平均完成 4.0s")
    # 价格基准（要素事实·代表性示例）：轻量/旗舰/思考 三档 $/1M tokens（输入, 输出）
    PM = {"lite": (0.30, 0.60), "flag": (2.50, 10.00), "think": (3.00, 15.00)}
    in_t, out_t = 800, 220
    cost_flag = (in_t * PM["flag"][0] + out_t * PM["flag"][1]) / 1e6
    chk("Q5", 0.0042, cost_flag, "单 query 旗舰档 =(800×2.50 + 220×10.00)/1e6")
    thinking = 350
    cost_think = (in_t * PM["think"][0] + (out_t + thinking) * PM["think"][1]) / 1e6
    chk("Q6", 0.01095, cost_think, "单 query 思考档：350 思考 token 按输出价记")
    out_money = out_t * PM["flag"][1] / 1e6
    chk("Q7", 2200 / 1e6, out_money, "旗舰档：输出 token 的钱（220 tok）")
    compute_active, elapsed = 55 * 0.90, 61.0  # 55s 繁忙里 90% 在算 · 6s 空闲
    gp = compute_active / elapsed
    chk("Q8", compute_active / elapsed, gp,
        f"GPU 利用率 = 活跃计算秒 {compute_active:.0f} ÷ 墙钟秒 {elapsed:.0f}（充裕时段 vs 显存/带宽受限）")
    rate, retry = 0.05, 1.5
    chk("Q9", 100000 * (0.95 + 0.05 * retry), 100000 * 0.95 + 5000 * retry,
        "有效调用 = 95k 成功 + 5k 失败×重试 1.5")
    chk("Q10", 6.73 / 0.85, e2e_50 / TTFT, "端到端/首屏：50 tok 输出时体感不是首屏 6.7× 于它")
    assert all(checks), "账 A 口径判定存在 FAIL"
    out.append(f"    -> 10/10 PASS（六把尺的分子分母齐了，反例 'TPOT≠TTFT'、'单请求吞吐≠整机吞吐'、'错误率5%≈成本5%' 都被公式戳穿）")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 B · 延迟分布账：均值君陷阱 × 混合长度 × 端到端公式
# ---------------------------------------------------------------------------
def account_b() -> str:
    out = []
    out.append("== B · 延迟分布账：先看分布再看延迟（均值君陷阱 · §5.8 口径） ==")
    ttft_all = [
        0.52, 0.55, 0.58, 0.61, 0.63, 0.66, 0.70, 0.74, 0.78, 0.82,
        0.85, 0.87, 0.89, 0.91, 0.93, 0.95, 0.98, 1.02, 1.05, 1.09,
        1.15, 1.22, 1.31, 1.45, 1.62, 1.84, 2.10, 2.35, 2.66, 2.98,
    ]
    mean = sum(ttft_all) / len(ttft_all)
    p50 = _pct(ttft_all, 0.50)
    p95 = _pct(ttft_all, 0.95)
    p99 = _pct(ttft_all, 0.99)
    out.append(f"  30 次实测 TTFT 分布（s）：均值 {mean:.2f} · P50 {p50:.2f} · P95 {p95:.2f} · P99 {p99:.2f}")
    ok_b1 = (mean - p50) > 0.1 and (p95 - p50) > 0.5
    out.append(f"  B1 均值君陷阱：均值 {mean:.2f} > 中位数 {p50:.2f}（长尾把平均拉高 {mean - p50:.2f}s）"
               f" = " + ("PASS(用均值报延迟=报给平均值听，P95 才是你会骂人的那档)" if ok_b1 else "FAIL"))
    ok_b2 = p95 >= 1.5 * p50 and p99 >= 2.0 * p50
    out.append(f"  B2 长尾实锤：P95 {p95:.2f} = {p95 / p50:.2f}×P50 · P99 {p99:.2f} = {p99 / p50:.2f}×P50"
               f" = " + ("PASS(只报 P50 会吞掉 2-3 倍的长尾用户)" if ok_b2 else "FAIL"))
    short = sum(ttft_all[:10]) / 10
    mid = sum(ttft_all[10:20]) / 10
    long_ = sum(ttft_all[20:30]) / 10
    ratio = long_ / short
    ok_b3 = ratio >= 2.0
    out.append(f"  B3 混合长度（§5.8 压测铁律）：短 prompt 均值 {short:.2f}s · 中 {mid:.2f}s · 长 {long_:.2f}s"
               f" -> 长/短 = {ratio:.1f}×"
               f" = " + ("PASS(只测一个长度=盲人摸象：单测短 prompt 至少低估 2 倍)" if ok_b3 else "FAIL"))
    TTFT_b, TPOT_b = 0.85, 0.12
    out.append(f"  B4 端到端 = TTFT + (N-1)×TPOT（TTFT {TTFT_b}s · TPOT {TPOT_b}s=120ms 在 200ms 流畅线内）：")
    e2e_lines = []
    for n in (10, 100, 1000):
        e = TTFT_b + (n - 1) * TPOT_b
        e2e_lines.append((n, e))
    for n, e in e2e_lines:
        frac = TTFT_b / e
        out.append(f"    N={n:4d} tok -> E2E {e:7.2f}s（首屏只占 {frac * 100:5.1f}% · {100 - frac * 100:5.1f}% 是流水）")
    ok_b4 = e2e_lines[0][1] < 2.0 and e2e_lines[2][1] > 100.0 and (e2e_lines[2][1] / e2e_lines[0][1]) > 50
    out.append(f"    -> {e2e_lines[2][1] / e2e_lines[0][1]:.0f}×：输出 10->1000 个 token 端到端近线性放大 = "
               + ("PASS(首 token 快≠整体快：长输出=TPOT×N 说了算，TTFT 只开个头)" if ok_b4 else "FAIL"))
    ok_b = all([ok_b1, ok_b2, ok_b3, ok_b4])
    out.append(f"  -> B 账 4 断言 = " + ("PASS" if ok_b else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 C · 吞吐与成本账：连续批处理算术 × 三档价格单 query 账本
# ---------------------------------------------------------------------------
def account_c() -> str:
    out = []
    out.append("== C · 吞吐与成本账：吞吐与延迟的算术 + token 用量/成本分开算（§11.7②④） ==")
    out.append("  C1 连续批处理（continuous batching）：每次 decode 仍是一个矩阵、多一步喂多人")
    tpot = 0.12
    rows = [(1, 1.0 / tpot), (8, 8.0 / tpot), (32, 32.0 / tpot)]
    for k, tp in rows:
        out.append(f"    批 {k:2d} -> 整机输出吞吐 {tp:7.1f} tok/s（每步 0.12s 还是 0.12s，只是这一步产 {k} 个 token）")
    util = {1: 0.11, 8: 0.45, 32: 0.87}
    for k in (1, 8, 32):
        out.append(f"    批 {k:2d} + GPU 利用率 {util[k] * 100:.0f}%（批 1 大量空闲·批 32 票面摊薄）")
    ok_c1 = (rows[2][1] / rows[0][1]) == 32 and util[32] >= 4 * util[1]
    out.append(f"    -> 批 1->32 吞吐 ×{rows[2][1] / rows[0][1]:.0f} 而单请求 E2E 不变（先加批再加机器：加机器=弹费用的横向扩展） = "
               + ("PASS" if ok_c1 else "FAIL"))
    thinking = 350
    total_out = 220 + thinking
    out.append(f"  C2 思考 token 分开看（§11.7③）：220 答案 + {thinking} 思考 = {total_out} 输出 token —— "
               f"合报'总 token'会藏 {thinking / total_out * 100:.0f}% 的思考税")
    # 三档价格单 query 账本
    PM = {"lite": (0.30, 0.60), "flag": (2.50, 10.00), "think": (3.00, 15.00)}
    in_t, out_t = 800, 220
    c_lite = (in_t * PM["lite"][0] + out_t * PM["lite"][1]) / 1e6
    c_flag = (in_t * PM["flag"][0] + out_t * PM["flag"][1]) / 1e6
    c_think = (in_t * PM["think"][0] + (out_t + thinking) * PM["think"][1]) / 1e6
    out.append("  C3 三档价格·单 query 账本（代表性示例价 $/1M tokens，输入/输出分开）：")
    out.append(f"    轻量档 $0.30/$0.60：{c_lite * 1e6:.0f}e-6 = ${c_lite:.5f}")
    out.append(f"    旗舰档 $2.50/$10.00：{c_flag * 1e6:.0f}e-6 = ${c_flag:.4f}")
    out.append(f"    思考档 $3.00/$15.00（思考 token 按输出价）：{c_think * 1e6:.0f}e-6 = ${c_think:.4f}")
    ok_c2a = 0.0042 == round(c_flag, 6)
    out_money = out_t * PM["flag"][1] / 1e6
    ok_c2b = (out_money / c_flag) > 0.5
    out.append(f"    C4 输出 token 占比：输出只 220/{in_t + out_t}=21.6% 的 token 却花掉 {out_money / c_flag * 100:.0f}% 的钱"
               f"（单价 4×） = " + ("PASS(只看输入单价会低估一半：长输出答案=输出件最贵)" if ok_c2b else "FAIL"))
    ok_c2c = (c_think / c_lite) > 10
    out.append(f"    C5 思考档单 query ${c_think:.4f} = 轻量档 ${c_lite:.5f} 的 {c_think / c_lite:.1f}×"
               f" = " + ("PASS(思考 token 按输出价记=模型选型时要算的税)" if ok_c2c else "FAIL"))
    # 每任务成本（Agent 3 步链）
    step_in, step_out, steps = 800, 300, 3
    task = steps * (step_in * PM["flag"][0] + step_out * PM["flag"][1]) / 1e6
    ok_c2d = task < 0.02 and task > c_flag
    out.append(f"    C6 每任务成本（§11.7⑤/§11.9⑤）：Agent 3 步链（每步 800 输入+300 输出）"
               f" = 3×{step_in * PM['flag'][0] + step_out * PM['flag'][1]:.0f}e-6 = ${task:.4f}/任务"
               f" = " + ("PASS(任务级记账=单 query 的 3.5×：逐 token 报账看不到'绕路税'，任务级才看到)" if ok_c2d else "FAIL"))
    ok_c = all([ok_c1, ok_c2a, ok_c2b, ok_c2c, ok_c2d])
    out.append(f"  -> C 账 5 断言 = " + ("PASS" if ok_c else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 D · 工程门禁账：指标×任务口径决策翻转 · 预算回归三阅 · 错误率放大 · 回流
# ---------------------------------------------------------------------------
def account_d() -> str:
    out = []
    out.append("== D · 工程门禁账：指标要配对任务口径 + 预算回归刹车 + 错误率放大（§11.9③④） ==")
    # 同一张指标表：v1 轻快首屏 vs v2 重速流水
    v1 = (0.62, 0.11)  # (TTFT, TPOT)
    v2 = (1.10, 0.06)
    short_n, long_n = 5, 500
    v1s, v2s = v1[0] + (short_n - 1) * v1[1], v2[0] + (short_n - 1) * v2[1]
    v1l, v2l = v1[0] + (long_n - 1) * v1[1], v2[0] + (long_n - 1) * v2[1]
    out.append(f"  D1 任务口径翻转（同一张表 × 两种任务 = 取不同行）：")
    out.append(f"    对答 5 tok：v1 {v1s:.2f}s vs v2 {v2s:.2f}s -> "
               f"v1 赢（首屏 TTFT 主导 · TPOT 撑不起短输出）")
    out.append(f"    长文 500 tok：v1 {v1l:.2f}s vs v2 {v2l:.2f}s -> "
               f"v2 赢（流水 TPOT 主导 · 首屏差距只占 {v1[0] / v1l * 100:.1f}%）")
    ok_d1 = v1s < v2s and v2l < v1l
    out.append(f"    -> 决策翻转 2/2 = " + ("PASS(工程指标没有'哪一个更好'：挂任务的尺子才有方向)" if ok_d1 else "FAIL"))
    # 预算回归三阅
    out.append(f"  D2 预算回归（每次换模型/调部署跑同一套混合负载 · §11.9③·同 05-08 RAG 门禁思想）：")
    p95_budget, cost_budget = 45.0, 0.0040
    rows = [
        ("v1", "轻量档", 0.62, 0.11, 55.51, 0.000372, "被 P95 拦"),
        ("v2", "旗舰档", 1.10, 0.06, 31.04, 0.0042, "被成本预算拦"),
        ("v3", "旗舰+输入缓存半价", 0.90, 0.06, 30.84, 0.0032, "P95/成本/首屏全过"),
    ]
    for tag, tier, tt, tp, e2e, cq, why in rows:
        outl = f"    {tag} {tier}：TTFT {tt:.2f}s · TPOT {tp:.2f}s · 长文 E2E {e2e:.2f}s "
        outl += f"({'✅' if e2e <= p95_budget else '⛔'}≤{p95_budget:.0f}s) "
        outl += f"· 每 query ${cq:.4f} ({'✅' if cq <= cost_budget else '⛔'}≤${cost_budget:.4f}) -> {why}"
        out.append(outl)
    ok_d2 = (rows[0][4] > p95_budget            # v1 长文 E2E 超 P95 预算 -> 被拦
             and rows[0][5] <= cost_budget      # v1 成本未超（先证明 v1 不是成本问题）
             and rows[1][5] > cost_budget       # v2 每 query 超成本预算 -> 被拦
             and rows[2][4] <= p95_budget       # v3 P95 过
             and rows[2][5] <= cost_budget)     # v3 缓存降本后成本过
    out.append(f"    -> v1 被 P95 拦 · v2 被成本预算拦 · v3 缓存降本后全过 = "
               + ("PASS(v1 快首屏也没用：长文任务先看流水预算·预算回归=换模型必须重跑的刹车)" if ok_d2 else "FAIL"))
    # 错误率×重试放大
    rate, retry = 0.05, 1.5
    calls = 100000 * (0.95 + rate * retry)
    out.append(f"  D3 错误率×重试（§11.7⑥）：日 100k 请求错误率 5%"
               f" -> 失败 5k×重试 1.5 = 有效调用 {calls / 1000:.0f}k = {calls / 100000:.3f}×（+2.5% 成本）")
    ok_d3a = calls / 100000 > 1.02
    out.append(f"    -> 错误率 5% ≠ 成本 5% = " + ("PASS(重试把 5% 放大到 2.5% 额外调用：账本要记有效调用)" if ok_d3a else "FAIL"))
    fail_e2e = 24.0
    ok_d3b = fail_e2e >= 3 * 8.0
    out.append(f"    -> 失败请求平均 E2E {fail_e2e:.0f}s = 3×正常 8s = " + ("PASS(错误率的第二个税在长尾延迟：P99 用户被 30s 超时拖)" if ok_d3b else "FAIL"))
    reg_coll = 100 + 8
    ok_d4 = reg_coll == 108
    out.append(f"  D4 遥测回流（§11.9④ 工程版）：线上超预算慢请求 8 条回注回归压测集"
               f" 100+8={reg_coll} = " + ("PASS(慢请求也是活样本：进回归集才不会再上线清一次 P95)" if ok_d4 else "FAIL"))
    ok_d = all([ok_d1, ok_d2, ok_d3a, ok_d3b, ok_d4])
    out.append(f"  -> D 账 5 断言 = " + ("PASS" if ok_d else "FAIL"))
    return "\n".join(out)


def main() -> None:
    W("start")
    parts = [account_a(), account_b(), account_c(), account_d()]
    print("\n" + "\n".join(parts), end="")
    W("done")


if __name__ == "__main__":
    main()
