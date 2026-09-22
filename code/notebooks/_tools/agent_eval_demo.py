#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""09-Agent体系 · 06-Agent评测（知识地图 §9.12 + §11.5 + §17.7）探针：
§9.12 原话："好的 Agent 评测 = 固定环境 + 可验证的终结条件 + 成本指标；
不要用'看起来像'的 LLM-judge 替代环境真实验证。"——把 05 账 D 的『提交前评测闸』
放大成整套 Agent 评测台。量四本账：
  A 评测基准谱系账 —— §9.12 三类基准（通用 / 环境 / 代码）12 个成名基准一张表：
    真实环境·真实问题只有 3/12（GAIA 真实问题·τ²-bench 真实 API·SWE-bench 真实 issue），
    可自动验证 11/12（OSWorld 桌面像素需人工/视觉=唯一例外）——瓶颈在"环境真实性"而非"自动验证"
  B 评测指标账 —— §9.12 六指标（成功率/步数/时长/token/工具正确率/安全事件）：
    同一客服 Agent 两台调法，单看成功率 A 9/10 > B 8/10，却 token 2× 且 2 起安全事件
    ——单指标骗人，六指标一起看才看见"成功率的代价"
  C 环境真值 vs LLM-judge 账 —— 同 20 条轨迹：环境真值 12/20 通过，
    LLM-judge 判准 17/20=0.85（FP 2=放 2 个坏上线·FN 1=饿死 1 个好）
    ——judge 便宜但会漏，环境真值贵但唯一可靠（挂 02-20 验证器 78/80）
  D 评测台装配账 —— 里程碑 014：任务集 + 固定环境 + 可验证终结检查器 + 成本计
    + 五列汇总报告，跑 A/B/C 三版 Agent 给『上不上生产』判定（成功率/成本/安全 AND 闸）
口径：12 基准清单与年份/六指标定义/评测台三件套 = 按 §9.12 与公开资料整理的要素事实
     （非本机实测）；轨迹/判定/打分/雪账/三版 Agent = 作者预置仿真；A-D 统计 = 本机确定性计算
     （纯 stdlib · 零随机 · 零网络 · stdout 逐字节可复现；wall-clock 只进 stderr）。
关联：05 账 D（提交前评测闸→本篇整套评测台）· 02-20 验证器 · 07-02 / LINKS。
"""
import sys
from time import perf_counter

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ================= 账 A：评测基准谱系账 =================
# (名称, 类别, 年份, 真实环境/真实问题?, 可自动验证?, 侧重成本?)
BENCHMARKS = [
    ("GAIA", "通用", 2023, True, True, False),            # 真实问题，人类级难题
    ("AgentBench", "通用", 2023, False, True, False),      # 8 模拟环境任务集
    ("τ-bench (工具调用)", "通用", 2024, False, True, True),  # 零售/航空业务模拟
    ("τ²-bench", "通用", 2025, True, True, True),          # 真实 API + 双系统评测
    ("Mind2Web", "环境", 2023, False, True, False),        # 网页指令（真实网页截图→固定环境）
    ("WebArena", "环境", 2024, False, True, False),        # 自托管网站模拟
    ("VisualWebArena", "环境", 2024, False, True, False),  # 加视觉
    ("OSWorld", "环境", 2024, False, False, False),        # 桌面，像素任务需人工/视觉判
    ("Terminal-Bench", "环境", 2025, False, True, False),  # 终端壳模拟
    ("SWE-bench", "代码", 2023, True, True, False),        # 100% 真实 GitHub issue
    ("RepoBench", "代码", 2023, False, True, False),       # 仓库级任务
    ("LiveCodeBench", "代码", 2024, False, True, True),    # 新题防泄漏+成本报告
]


def expA():
    by_year = {}
    per_cat = {"通用": [0, 0], "环境": [0, 0], "代码": [0, 0]}
    for _, cat, yr, real, auto, _c in BENCHMARKS:
        by_year.setdefault(yr, [0, 0])
        by_year[yr][0] += 1
        by_year[yr][1] += 1 if auto else 0
        per_cat[cat][0] += 1
        per_cat[cat][1] += 1 if auto else 0
    n = len(BENCHMARKS)
    real_n = sum(1 for b in BENCHMARKS if b[3])
    auto_n = sum(1 for b in BENCHMARKS if b[4])
    cost_n = sum(1 for b in BENCHMARKS if b[5])
    print("=" * 72)
    print("[账 A] 评测基准谱系账：§9.12 三类基准一张表——12 成名基准，环境真实验证只有 3 个")
    print("=" * 72)
    print(f'  {"基准":<22}{"类别":<5}{"年":<6}{"真环境":<7}{"自动验证":<8}{"成本侧重"}')
    for name, cat, yr, real, auto, cost in BENCHMARKS:
        print(f'  {name:<22}{cat:<5}{yr:<6}{"✓" if real else "—":<7}{"✓" if auto else "—":<8}{"✓" if cost else "—"}')
    print(f"\n  合计 {n} 个基准：真实环境/真实问题 {real_n}/{n}（GAIA 真实问题·τ² 真实 API·SWE-bench 真实 issue）"
          f"· 可自动验证 {auto_n}/{n}（OSWorld 桌面像素需人工/视觉=唯一不可自动）· 侧重成本 {cost_n}/{n}")
    for yr in sorted(by_year):
        t, a = by_year[yr]
        print(f"  · {yr} 年：{t} 个（自动验证 {a}/{t}）")
    # 2025 两个全部自动 · 2024 含 OSWorld（桌面像素）"四舍五入"自动率不算满
    y25_auto = by_year[2025][1] == by_year[2025][0]
    assert real_n == 3 and auto_n == 11 and cost_n == 3
    assert y25_auto
    print("  → 结论一：评测的瓶颈不在『自动验证』——11/12 已可在模拟环境里自动核答案；"
          "真正稀缺的是『环境真实性』：只有 3/12（GAIA/τ²/SWE-bench）是真实问题/真实 API/真实 issue，"
          "模拟环境（WebArena/OSWorld）保证固定=可复现，但『真』要靠基准设计")
    print("  → 结论二：2025 的新基准（τ²-bench 真实 API + Terminal-Bench 终端壳）自动验证率 2/2，"
          "且『侧重成本』开始出现在基准里（τ-bench·τ²-bench·LiveCodeBench 均带成本报告）"
          "——成本指标进基准=§9.12 好评测公式的『成本指标』正在成为默认")
    print("  → 断言：真实环境 3/12 · 自动验证 11/12 · 成本侧重 3/12 · 2025 新基准全自动")


# ================= 账 B：评测指标账 =================
# 同一客服 Agent 的两台调法在 10 任务上的六指标
RUNS = {
    "A · 稳重型": dict(pass_n=9, steps=13, since=260, tokens=12000, tool_ok=9, tool_tot=9, safety=0),
    "B · 激进型": dict(pass_n=8, steps=9, since=150, tokens=6000, tool_ok=5, tool_tot=6, safety=2),
}
METRICS = [("任务成功率", "pass_n"), ("完成步数", "steps"), ("完成时长(s)", "since"),
           ("token 成本", "tokens"), ("工具调用正确率", "_tool_acc"), ("安全事件数", "safety")]


def expB():
    print()
    print("=" * 72)
    print("[账 B] 评测指标账：§9.12 六指标六向量——单指标骗人，六指标一起看才看见代价")
    print("=" * 72)
    a, b = RUNS["A · 稳重型"], RUNS["B · 激进型"]
    a["_tool_acc"] = a["tool_ok"] / a["tool_tot"]
    b["_tool_acc"] = b["tool_ok"] / b["tool_tot"]
    print(f'  {"指标":<18}{"A 稳重型":>14}{"B 激进型":>14}{"谁占优"}')
    for label, key in METRICS:
        av, bv = a[key], b[key]
        fmt = lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
        avs, bvs = fmt(av), fmt(bv)
        winner = ""
        if label == "任务成功率":
            winner = "A" if av > bv else "B"
        elif label in ("完成步数", "完成时长(s)", "token 成本", "安全事件数"):
            winner = "B" if av > bv else "A"
        else:
            winner = "A" if av > bv else "B"
        print(f'  {label:<18}{avs:>14}{bvs:>14}  {winner}')
    print("\n  只看成功率的陷阱：A 9/10 > B 8/10，好像 A 更好；六指标齐看一眼推翻——")
    print(f"  · token 成本 A {a['tokens']:,} = B {b['tokens']:,} 的 2×：A 的成功率是多花一倍的量换的（成本指标登场）")
    print(f"  · 安全事件 A {a['safety']} vs B {b['safety']}：A 没出事，B 的 2 次失败里有 1 次是工具越权=成功率之外的安全面")
    print(f"  · 工具正确率 A {a['_tool_acc']:.2f} > B {b['_tool_acc']:.2f}：B 的 1 次成功是碰运气（工具调用虚高步）")
    print(f"  · 步数/时长 B 更少（{b['steps']}步·{b['since']}s < {a['steps']}步·{a['since']}s）但那是省在跳过校验上")
    assert a["tokens"] == 2 * b["tokens"] and a["safety"] == 0 and b["safety"] == 2
    print("  → §9.12 六指标：任务成功率 / 完成时长·步数 / token 成本 / 工具调用正确率 / 安全事件数 / "
          "人工通过率——单指标是视图，六指标才是账本；成本与安全最容易被只看成功率的评测骗过")
    print("  → 断言：token 2× · 安全 A0 B2 · 工具正确率 A>B · A 成功率更高但代价全在成本+安全")


# ================= 账 C：环境真值 vs LLM-judge 账 =================
ENV_GT = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 12 真过 8 真败
JUDGE = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]  # 11 TP + 1 FN + 2 FP


def expC():
    print()
    print("=" * 72)
    print("[账 C] 环境真值 vs LLM-judge 账：同 20 条轨迹，判准 0.85 vs 1.00——『看起来像』会放行坏 Agent")
    print("=" * 72)
    n = len(ENV_GT)
    tp = sum(1 for g, j in zip(ENV_GT, JUDGE) if g == 1 and j == 1)
    tn = sum(1 for g, j in zip(ENV_GT, JUDGE) if g == 0 and j == 0)
    fp = sum(1 for g, j in zip(ENV_GT, JUDGE) if g == 0 and j == 1)
    fn = sum(1 for g, j in zip(ENV_GT, JUDGE) if g == 1 and j == 0)
    acc = (tp + tn) / n
    print(f"  轨迹 {n} 条 · 环境真值（可验证终结条件）通过 {sum(ENV_GT)}/{n} 失败 {sum(1 for g in ENV_GT if g==0)}/{n}")
    print(f"  LLM-judge 判定：判准 {acc:.2f}（{tp+tn}/{n}）· TP {tp} · TN {tn} · FP {fp} · FN {fn}")
    print(f"  · FP={fp}：judge 说『通过』其实真值失败的 2 条→坏 Agent 被放行上线（评估最贵错误）")
    print(f"  · FN={fn}：judge 说『失败』其实真值通过的 1 条→好 Agent 被埋没（开发期误杀）")
    assert acc == 0.85 and fp == 2 and fn == 1
    print("  → §9.12 原话『不要用“看起来像”的 LLM-judge 替代环境真实验证』的数字版：judge 判准 0.85≠1.00，"
          "漏放 2 个坏 Agent；环境真值=固定环境里的可验证终结条件（如测试/API 返回/断言），唯一可靠，"
          "当终结判据换不得 judge（2026 趋势=环境验证进生产，§9.14 趋势①）")
    print("  → 断言：judge 判准 0.85 · FP=2 放行坏 · FN=1 误杀好")


# ================= 账 D：评测台装配账 =================
# 里程碑 014：任务集 + 固定环境 + 可验证终结检查器 + 成本计 + 汇总五列 + 上生产判定
def run_eval(pass_n, tokens, safety):
    """返回评测报告摘要：五列 + 生产判定。成功率=pass_n/10，预算=8000 token。"""
    sr = pass_n / 10
    cost_ok = tokens <= 8000
    safe_ok = safety == 0
    sr_ok = sr >= 0.85
    verdict = "✅ 上生产" if (sr_ok and cost_ok and safe_ok) else ("⛔ 不上生产" if True else "")
    why = []
    if not sr_ok:
        why.append(f"成功率 {sr:.2f}<0.85")
    if not cost_ok:
        why.append(f"token {tokens:,}>8000 预算")
    if not safe_ok:
        why.append(f"安全事件 {safety}>0")
    return sr, tokens, safety, verdict, why


def expD():
    print()
    print("=" * 72)
    print("[账 D] 评测台装配账：里程碑 014——任务集+固定环境+可验证终结检查器+成本计 → 上不上生产")
    print("=" * 72)
    print("  评测台三件套（§9.12）：固定环境（隔离容器跑 Agent）· 可验证终结条件（每任务一个检查器：")
    print("  测试通过/API 返回/断言=环境真值，不是 judge）· 成本指标（token/步数/时长）")
    print()
    print(f'  {"Agent 版本":<14}{"成功率":>8}{"token":>10}{"安全事件":>8}  判定')
    cases = [("C · 优化版", 9, 7200, 0), ("A · 稳重型", 9, 12000, 0), ("B · 激进型", 8, 6000, 2)]
    res_map = {}
    for name, pn, tok, sf in cases:
        res = run_eval(pn, tok, sf)
        res_map[name] = res
        print(f'  {name:<14}{res[0]:>8.2f}{res[1]:>10,}{res[2]:>8}  {res[3]}  '
              f'（{"·".join(res[4]) if res[4] else "全闸通过"}）')
    assert res_map["C · 优化版"][3] == "✅ 上生产"
    assert all(res_map[n][3] == "⛔ 不上生产" for n in ("A · 稳重型", "B · 激进型"))
    print("\n  复盘三份报告：")
    print("  · C 优化版：调好 A 的预算（降 token 到 7200）过全闸——评测台验证『修复有效』，不只是打分")
    print("  · A 稳重型：成功率 0.90 却 token 12,000 超预算——评测台把『成功率的代价』变成可执行红线（账 B 升级）")
    print("  · B 激进型：token 最省但成功率 0.80 不足 + 安全事件 2——评测台拦住『看起来快』的 Agent")
    print("  → 评测台不是打分器：三件套的『可验证终结+成本』把 05 账 D 的提交前闸放大器成整套 Agent 评测台"
          "（§11.5 联动：Agent 评测=成功率/步数/成本/安全性四件套）")
    print("  → 断言：上生产=C 优化版（全闸过）· A/B 各有拦截理由")


def main():
    print("=" * 72)
    print("agent_eval_demo：09-Agent体系 · 06-Agent评测（基准谱系·指标·真值vs judge·评测台）")
    print("    关联：地图 §9.12（三类基准+六指标+好评测公式）｜§11.5｜05 账D（提交前闸）｜02-20 验证器")
    print("=" * 72)
    print("[0] 口径：基准清单/年份/六指标/评测台三件套=要素事实（§9.12 与公开资料·非本机实测）；")
    print("    轨迹/判定/打分/三版 Agent=作者预置仿真；A-D 统计=本机确定性计算（纯 stdlib·零随机·零网络）")
    expA()
    expB()
    expC()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 基准谱系 / B 指标 / C 真值vs judge / D 评测台）")
    print("  A 真实环境/真实问题 3/12（GAIA/τ²/SWE-bench 才真）·自动验证却 11/12——瓶颈在『真环境』不在『自动核分』")
    print("  B 单指标骗人：A 成功率 0.90>B 0.80 但 token 2× 且 B 有 2 起安全事件——六指标才是账本")
    print("  C LLM-judge 判准 0.85（FP 2 放行坏·FN 1 误杀好）vs 环境真值 1.00——终结条件换不得 judge")
    print("  D 评测台三件套跑三版：只有 C 优化版过成功率+成本+安全全闸——评测台是门禁不是打分器")
    print("一句话：不评测的 Agent 不可信（§9.12）——但评测要用『固定环境+可验证终结条件+成本指标』"
          "造真台子，单指标和 LLM-judge 都是昂贵的捷径。")
    print("done · 一键复现：python code/notebooks/_tools/agent_eval_demo.py")


if __name__ == "__main__":
    main()
    W(f"[agent_eval_demo] wall-clock {perf_counter() - _t0:.3f} s")
