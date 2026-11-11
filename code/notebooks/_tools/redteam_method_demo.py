# -*- coding: utf-8 -*-
"""
红队方法探针 —— 四步流程判别 + 三工具选型 6 场景 + 越狱流量台账 + 三层修复闭环
纯 stdlib · 零网络 · 零随机 · 零第三方 · stdout 三独立进程逐字节一致

账 A  红队四步：范围→侦察→执行→报告，前序产物缺失即阻断
账 B  三工具选型：PyRIT / Garak / HarmBench 判别 6 场景
账 C  流量台账：三批 100 条，命中率演进 0.62→0.53→0.33（呼应 11-07 量表）
账 D  三层修复闭环：提示词层/护栏层/微调层，全发生才算闭环
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

_t0 = time.perf_counter()
_PASS = 0
_FAIL = 0


def check(name, cond):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"[PASS] {name}")
    else:
        _FAIL += 1
        print(f"[FAIL] {name}")


# ---------------------------------------------------------------- 账 A 四步流程
STEPS = {
    1: ("范围界定", "资产清单+威胁模型"),
    2: ("侦察生成", "攻击面收集 → 向量池"),
    3: ("执行攻击", "向量池 → 命中/未命分录"),
    4: ("报告修复", "命中录 → 修复项+回归计划"),
}


def run_redteam(order):
    """按给定步骤顺序跑，前序产物缺失即阻断。返回 (完成到的步骤, 阻断点)"""
    produced = set()
    for step in order:
        # 前序产物检查：步骤 n 需要 STEPS[n-1] 已产出
        need_prev = STEPS[step - 1][1] if step > 1 else None
        prev_done = (step - 1) in produced
        if need_prev and not prev_done:
            return step - 1, step
        produced.add(step)
    return 4, None


def stage_label(steps):
    return "→".join(STEPS[k][0] for k in steps)


# ---------------------------------------------------------------- 账 B 三工具选型
def pick_tool(auto, bench, scan):
    """auto=要自动化攻击+评分; bench=要标准基准横向比; scan=要快速合规扫描"""
    if auto and bench:
        return "PyRIT+HarmBench"
    if auto:
        return "PyRIT"
    if bench:
        return "HarmBench"
    if scan:
        return "Garak"
    return "Garak"


def sec_a():
    print("账 A · 红队四步流程判别")
    ranks = [(1, 2, 3, 4), (1, 3, 2, 4), (1, 2, 4, 3)]
    for r in ranks:
        done, blocked = run_redteam(r)
        print(f"  步骤序 {stage_label(r)} → 完成到 [{STEPS[done][0]}]"
              f"{('· 阻断在 [' + STEPS[blocked][0] + '] 缺前序') if blocked else ' · 全流程可跑'}")
    ok_a = run_redteam((1, 2, 3, 4)) == (4, None)
    ok_b = run_redteam((1, 3, 2, 4))[1] == 3      # 跳过侦察 → 执行被阻断
    ok_c = run_redteam((1, 2, 4, 3))[1] == 4      # 跳过执行 → 报告被阻断
    check("账A 正序全通 + 乱序双阻断", ok_a and ok_b and ok_c)


def sec_b():
    print("账 B · 三工具选型 6 场景")
    SC = [
        ((1, 0, 0), "PyRIT", "自家产品持续自动红队+评分"),
        ((0, 1, 1), "HarmBench", "与论文 18 家模型横向比鲁棒性"),
        ((0, 0, 1), "Garak", "上线前快速合规扫描"),
        ((1, 1, 0), "PyRIT+HarmBench", "自动化流水线还要基准评分"),
        ((0, 0, 0), "Garak", "人工红队先跑概念验证"),
        ((1, 0, 1), "PyRIT", "自动红队+附带扫描能力"),
    ]
    ok = True
    for (a, b, s), want, note in SC:
        got = pick_tool(a, b, s)
        ok = ok and got == want
        print(f"  auto={a} bench={b} scan={s} → {got}  （{note}）")
    check("账B 选型 6 场景全命中", ok)


# ---------------------------------------------------------------- 账 C 流量台账
def batch(hits, total=100):
    """命中下标集合固定手写 → 完全确定。返回命中率（保留 3 位）"""
    rate = len(hits) / total
    return round(rate, 4)


def sec_c():
    print("账 C · 越狱流量台账（每批 100 条，命中下标固定手写）")
    batches = {
        "初版": set(range(0, 62)),          # 62 命中，越狱率 0.62（呼应 11-07 初始）
        "提示词加固": set(range(0, 53)),     # 53 命中，0.53
        "护栏接入": set(range(0, 33)),       # 33 命中，0.33（呼应 11-07 0.333）
    }
    rates = []
    for name, hits in batches.items():
        r = batch(hits)
        rates.append(r)
        print(f"  {name}: 命中 {len(hits):>3}/100 → 越狱率 {r}")
    ok = rates == [0.62, 0.53, 0.33]
    check("账C 越狱率 0.62→0.53→0.33 与 11-07 量表同源", ok)


# ---------------------------------------------------------------- 账 D 三层修复闭环
def fix_loop(prompt_patch, guardrail, finetune, gate):
    """三层修复各自到位 + 回归闸通过 → 闭环"""
    layers = [prompt_patch, guardrail, finetune]
    closed = all(layers) and gate
    return closed


def sec_d():
    print("账 D · 三层修复闭环")
    cases = [
        ("三层全上+回归通过", (1, 1, 1, 1), True),
        ("只改提示词", (1, 0, 0, 1), False),
        ("提示词+护栏缺微调", (1, 1, 0, 1), False),
        ("三层全上但回归未过", (1, 1, 1, 0), False),
    ]
    ok = True
    for name, args, want in cases:
        got = fix_loop(*args)
        ok = ok and (got == want)
        print(f"  {name} → {'闭环✅' if got else '未闭环❌'}")
    check("账D 三层全上+回归过才算闭环", ok)


if __name__ == "__main__":
    print("红队方法探针 · 纯 stdlib · 零网络 · 零随机 · 零第三方 · stdout 三独立进程逐字节一致")
    print("=" * 74)
    sec_a()
    sec_b()
    sec_c()
    sec_d()
    print("=" * 74)
    print(f"PASS {_PASS} / FAIL {_FAIL}")
    if _FAIL:
        sys.exit(1)
    print(f"wall-clock: {time.perf_counter() - _t0:.3f}s", file=sys.stderr)
