#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""09-Agent体系 · 05-Coding-Agent（知识地图 §9.11 + §17.7.7 编程智能体）探针：
把 04 的行动推理范式在代码域做极端落地（§9.11 原文：代码任务是"可验证的"（测试/编译器）
→ 自动评测与强化学习燃料都齐了 = Agent 的最佳落地；先做"能过测试的函数"，再做"能改仓库的 Agent"）。
量四本账：
  A 测试回路账 —— Action=Edit、Observation=测试/编译输出、Reflexion=看报错改代码：
    看报错（refl）2 次尝试收敛 3/3 vs 盲改（blind）4 次——报错读取是收敛的唯一注释
  B 仓库导航账 —— RAG over codebase（§9.11 核心技术第 1 项）：同一任务三种找法
    全仓直读 100% tok·2/2 命中 vs 仓库索引 top-6 26%·1/2 漏依赖 vs top-8 44%·2/2
    vs 符号/grep 切片 6%·2/2——把『全仓』变『相关切片』，检索与词面互为兜底
  C 并行多分支账 —— §9.11 核心技术『并行探索多分支』：同一 bug 两个候选假设
    顺序 6 步 vs 并行 3 步（墙钟省 50%），测试运行次数同 2=算力换墙钟（挂靠 04 账C ToT）
  D 评测闸与安全台账 —— 可验证终结条件（§9.9/§9.12 SWE-bench 思路）：
    3 候选补丁仅 1/3 过全套测试被接受·失败 2/3 回退不落盘；7 个操作风险分诊
    高风险 3/7（改配置/删分支/发布）人工确认拦截=最小权限（§9.13）
口径：代码任务/测试结果/报错文本/仓库文件与检索命中/路径成本/操作分级 = 作者按 §9.11 预置
     （仿真=非真实 Coding Agent 运行，不 import 任何编码框架）；A-D 全部统计 = 本机确定性计算
     （纯 stdlib · 零随机 · 零网络 · stdout 逐字节可复现；wall-clock 只进 stderr）；
     SWE-bench/代表产品/年份 = 要素事实（非本机实测）。挂靠 04 账A·B·C / 02-20 / 09-02。
"""
import sys
from time import perf_counter

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ================= 账 A：测试回路账 =================
def run_tests(version, flip=False):
    """确定性『测试』：v0 比较方向反 → 1/3；盲改两回合仍半对；换比较方向 → 3/3。"""
    if flip:
        return (3, 3)                       # 反转比较方向后全套通过
    if version == 0:
        return (1, 3)                       # 实得 [5,4,3] vs 期望 [3,4,5]
    if version in (1, 2):
        return (1, 3) if version == 1 else (2, 3)
    return (3, 3)


def refl_fixer():
    """Reflexion 式：跑测试→读报错『期望[3,4,5] 实得[5,4,3]』→定位"比较方向反"→改→通过。"""
    log = []
    p0, t0 = run_tests(0)
    log.append(f"v0 测试 {p0}/{t0} 失败 · 报错读入『test2 期望[3,4,5] 实得[5,4,3]』→ 判定：比较方向反")
    p1, t1 = run_tests(0, flip=True)
    log.append(f"v1 反转比较方向 → 测试 {p1}/{t1} 通过（终局）")
    return 2, 2, log, [f"{p0}/{t0}", f"{p1}/{t1}"]


def blind_fixer():
    """盲改式：每次看 pass 数但不用报错内容，按固定猜测序试错。"""
    log = []
    guesses = ["加 try/except", "改用 sorted() 重建", "反转比较方向"]
    passed_seq = []
    for i, g in enumerate([None] + guesses, start=0):
        p, t = run_tests(i)
        passed_seq.append(f"{p}/{t}")
        verb = "初始 v0" if i == 0 else f"盲改『{g}』"
        if p == t:
            log.append(f"{verb} → 测试 {p}/{t} 通过（终局）")
            return len(passed_seq), len(passed_seq), log, passed_seq
        log.append(f"{verb} → 测试 {p}/{t} 失败（只看通过数，报错内容未用）")
    raise AssertionError("blind must converge")


def expA():
    print("=" * 72)
    print("[账 A] 测试回路账：Action=Edit · Observation=测试/编译输出 · Reflexion=看报错改代码")
    print("=" * 72)
    ra, rt, rlog, rseq = refl_fixer()
    ba, bt, blog, bseq = blind_fixer()
    print(f"  ① 看报错（Reflexion 式）：测试输出 {rseq[0]} → {rseq[1]} · 尝试 {ra} 次 · 跑测试 {rt} 次")
    for m in rlog:
        print(f"     · {m}")
    print(f"  ② 盲改（只看通过数）：测试输出 {' → '.join(bseq)} · 尝试 {ba} 次 · 跑测试 {bt} 次")
    for m in blog:
        print(f"     · {m}")
    assert (ra, rt) == (2, 2) and (ba, bt) == (4, 4)
    assert rseq[-1] == "3/3" and bseq[-1] == "3/3"
    assert ra < ba
    print(f"  → 都收敛 3/3，但看报错 {ra} 次尝试 < 盲改 {ba} 次：Observation 是收敛的唯一注释"
          f"——测试/编译输出就是代码域的『执行器真值』（04 账A 的 A==O 不变式在此兑现）")
    print("  → 断言：refl 尝试 2 < blind 4 · 双方终局 3/3 · 每条修改后必有测试观察")


# ================= 账 B：仓库导航账 =================
REPO_FILES = 18
REPO_TOK = 100          # 全仓归一化 token=100
HIT = 2                 # 目标任务需命中 2 处：排序函数 + 依赖常量


def navigation(strategy):
    """同一『改排序函数』任务的三种找法 → (读 token, 命中数)。"""
    if strategy == "read_all":
        return (REPO_TOK, HIT)                 # 全仓直读 100%
    if strategy == "index_top6":
        return (26, 1)                          # 语义近邻，漏依赖常量（检索非词面）
    if strategy == "index_top8":
        return (44, HIT)                        # 加宽召回含 2 目标
    if strategy == "grep_symbol":
        return (6, HIT)                         # 已知符号名切片定位，最省
    raise ValueError(strategy)


def expB():
    print()
    print("=" * 72)
    print("[账 B] 仓库导航账：RAG over codebase——同一任务三种找法，把『全仓』变『相关切片』")
    print("=" * 72)
    rows = [
        ("全仓直读（长上下文暴力）", navigation("read_all"), "保底：一定命中，但给全部 18 文件买单"),
        ("仓库索引 top-6（语义检索）", navigation("index_top6"), "漏 1 目标：依赖常量语义近≠词面命中"),
        ("仓库索引 top-8（加宽召回）", navigation("index_top8"), "2/2 命中，仍读 44% 陪跑"),
        ("符号/grep 切片（词面精确）", navigation("grep_symbol"), "已知符号名直接定位，最省"),
    ]
    print(f'  {"策略":<28}{"读 token":>9}{"命中":>6}  说明')
    for name, (tok, hit), note in rows:
        print(f"  {name:<28}{tok:>7}%{hit:>4}/2  {note}")
    a, b, c, d = [navigation(k) for k in ("read_all", "index_top6", "index_top8", "grep_symbol")]
    assert (a, b, c, d) == ((100, 2), (26, 1), (44, 2), (6, 2))
    assert a[0] > c[0] > d[0]
    print(f"  → 三招都找到目标的成本：全读 {a[0]}% > 检索 {c[0]}% > 切片 {d[0]}%；检索漏项 {b[1]}/2 时"
          f"词面 grep 兜底 2/2=检索与词面互为备份（§9.11 『仓库索引』的真实分工：索引负责『哪个文件相关』，"
          f"grep 负责『哪一行精确』）")
    print("  → 断言：命中全 2/2（除 top-6 漏 1）· 成本 100 > 44 > 6 · 检索/词面互补")


# ================= 账 C：并行多分支账 =================
def expC():
    print()
    print("=" * 72)
    print("[账 C] 并行多分支账：并行探索多分支（§9.11 核心技术）——算力换墙钟")
    print("=" * 72)
    # 一个 bug 两个候选假设：A 改比较方向 / B 改测试期望；每假设 = 写测试1 + 实现1 + 跑1
    branch_steps = 3
    seq_steps = branch_steps * 2          # A 先失败 → B 再试
    par_steps = branch_steps              # A、B 同时跑，取通过的一条
    test_runs = 2                         # 两条分支各跑 1 次测试
    print(f"  · 候选假设 A（改比较方向）→ 写测试 1{'+'}实现 1{'+'}跑 1 = {branch_steps} 步，测试 1 次 → 通过")
    print(f"  · 候选假设 B（改测试期望）→ 写测试 1{'+'}实现 1{'+'}跑 1 = {branch_steps} 步，测试 1 次 → 不通过")
    print(f"  · 顺序执行：A {branch_steps} 步失败 → B {branch_steps} 步通过 = {seq_steps} 步墙钟 · 测试 {test_runs} 次")
    print(f"  · 并行执行：A‖B 同时跑 = {par_steps} 步墙钟 · 测试 {test_runs} 次 · 取通过分支")
    assert seq_steps == 6 and par_steps == 3 and test_runs == 2
    save = 1 - par_steps / seq_steps
    print(f"  → 墙钟省 {save*100:.0f}%（{seq_steps}→{par_steps} 步），测试运行次数不变（{test_runs} 次）"
          f"= 算力换墙钟：并行不增加验证开销，只翻倍算力（挂靠 04 账C：测试真值=价值函数，"
          f"多分支是 ToT 在代码域的形态）")
    print("  → 断言：并行 3 < 顺序 6 · 测试次数同 2")


# ================= 账 D：评测闸与安全台账 =================
def gate(patch_id):
    """可验证终结条件（§9.12 SWE-bench 思路）：过全套测试才接受，否则回退。"""
    results = {"p1": (0, 3), "p2": (3, 3), "p3": (2, 3)}
    p, t = results[patch_id]
    return (p, t, "接受" if p == t else "回退")


OPS = [
    ("read file", "只读", "自动"),
    ("grep 符号", "只读", "自动"),
    ("edit 测试文件", "改动", "自动"),
    ("edit 源码文件", "改动", "自动"),
    ("改生产配置", "高风险", "人工确认"),
    ("删除分支 / rm", "高风险", "人工确认"),
    ("发布 / push 生产", "高风险", "人工确认"),
]


def expD():
    print()
    print("=" * 72)
    print("[账 D] 评测闸与安全台账：可验证终结条件 + 最小权限分诊")
    print("=" * 72)
    print("  补丁评测闸（过全套测试才接受，否则回退不落盘）：")
    for pid in ("p1", "p2", "p3"):
        p, t, v = gate(pid)
        print(f"    · {pid} 测试 {p}/{t} → {v}")
    ac = sum(1 for pid in ("p1", "p2", "p3") if gate(pid)[2] == "接受")
    assert ac == 1
    print(f"  → 接受 1/3 · 回退 2/3（SWE-bench 思路：固定环境 + 可验证终结条件=测试通过 + 成本指标=测试运行/步数）")
    auto_n = sum(1 for _, _, lvl in OPS if lvl == "自动")
    man_n = sum(1 for _, _, lvl in OPS if lvl == "人工确认")
    print(f"  操作风险分诊（7 个）：自动 {auto_n} · 人工确认 {man_n}")
    for op, level, decision in OPS:
        print(f"    · {op:<18} 层级={level:<6} → {decision}")
    assert auto_n == 4 and man_n == 3
    print(f"  → 高风险 3/7 全拦截（改配置/删分支/发布）= 最小权限 + 人工确认高风险动作（§9.13 基线护栏，"
          f"挂靠 09-02 账 B 人工确认分诊 1/1：代码域的高危动作清单是 AGENT 判断的落地版）")
    print("  → 断言：补丁接受 1/3 · 危险操作 3/7 拦截")


def main():
    print("=" * 72)
    print("coding_agent_demo：09-Agent体系 · 05-Coding-Agent（测试回路·仓库导航·多分支·评测闸）")
    print("    关联：地图 §9.11（代码可验证=最佳落地）｜§9.12（SWE-bench 可验证终结）｜03 账D（代码即计划）")
    print("=" * 72)
    print("[0] 口径：代码任务/测试结果/报错文本/仓库文件/检索命中/路径成本/操作分级 = 作者按 §9.11 预置"
          "（仿真=非真实 Coding Agent，不 import 任何编码框架）；")
    print("    A-D 全部统计 = 本机确定性计算（纯 stdlib·零随机·零网络）")
    expA()
    expB()
    expC()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 测试回路 / B 仓库导航 / C 并行多分支 / D 评测闸与安全）")
    print("  A 看报错 2 次 < 盲改 4 次收敛 3/3——测试/编译输出=代码域的执行器真值（04 账A A==O 兑现）")
    print("  B 全读 100% > 索引 44% > grep 切片 6%，检索漏项由词面兜底=『全仓→相关切片』的分工")
    print("  C 顺序 6 步 vs 并行 3 步墙钟省 50%，测试次数不变=算力换墙钟（04 账C ToT 代码域）")
    print("  D 补丁过全套测试才接受（1/3 接受·2/3 回退）·高风险操作 3/7 人工拦截=最小权限")
    print("一句话：Coding Agent = 04 行动推理范式的最值钱落地——代码可验证，所以 Observation "
          "有执行器真值、评测有终结条件、RL 有燃料；会改仓库前先会改函数。")
    print("done · 一键复现：python code/notebooks/_tools/coding_agent_demo.py")


if __name__ == "__main__":
    main()
    W(f"[coding_agent_demo] wall-clock {perf_counter() - _t0:.3f} s")
