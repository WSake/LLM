# -*- coding: utf-8 -*-
"""threat_model_demo.py —— §12.0 威胁模型总览（12-安全 01 篇配套）

四账对应正文 📑 本节大纲四点：
账 A  五侧框架：已布防集合 → 覆盖缺口（只盯输出 = 4 面全裸）
账 B  攻击清单挂框架：应用形态 → 暴露哪几侧（5 场景 5/5 判定）
账 C  横切性：能力增量 → 新开的攻击面（逐档累加）
账 D  全章方法论：验收用例双臂校验（坏样本拦 + 好样本放，缺一臂判不通过）

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr。全部统计 = 本机确定性计算。

诚实边界（与正文「数字纪律」一致）：
- 五侧框架、各侧攻击名、合规名（个保法 / EU AI Act）= 地图 §12.0 要素事实，
  写作环境无外网、未在线复核。
- 本脚本的「应用形态 → 暴露面」「能力 → 新增面」映射 = 作者按
  「数据从哪进、动作从哪出」整理的**教学数**（判定规则写在 EXPOSE / ADDS 里，可逐行核对），
  不是任何真实系统的渗透测试结果，也不代表真实攻击成功率。
- 正文引用的 6→2→1、5/5→0/5、越狱率 0.62、打回率 60% 等数值锚点
  来自 09-07 / 11-07 / 14-07 已交付章节，本脚本**只引不重测**。
- 无任何真实模型调用、无真实攻击样本、无外部数据。
"""
from __future__ import annotations

import sys
import time

_WSTART = time.perf_counter()

SIDES = ("输入侧", "模型侧", "系统侧", "输出侧", "治理侧")

# 账 B：形态 -> 会暴露的侧。判定依据 = 数据从哪进、动作从哪出。
EXPOSE = {
    "纯聊天问答":         ("输入侧", "模型侧", "输出侧"),
    "RAG 客服":           ("输入侧", "模型侧", "系统侧", "输出侧", "治理侧"),
    "Coding Agent":       ("输入侧", "系统侧", "输出侧", "治理侧"),
    "GUI / 浏览器 Agent": ("输入侧", "系统侧", "输出侧", "治理侧"),
    "企业知识库(多租户)": ("输入侧", "系统侧", "输出侧", "治理侧"),
}

# 账 C：每加一个能力，新开一个（或一组）攻击面。
ADDS = (
    ("检索",     "系统侧", "文档可被人写 → RAG 投毒"),
    ("工具",     "系统侧", "动作真的发生 → 越权 / 工具输出注入"),
    ("多Agent",  "治理侧", "错误传导 → 检查点 / 责任边界"),
    ("多租户",   "模型侧", "隔离失效 → 记忆与数据泄漏"),
    ("行业落地", "治理侧", "合规红线 → 人审 / 审计 / 回滚"),
)

# 账 D：验收用例双臂。gaurd 为「该拒的拒了吗」，pass_ok 为「该放的放了吗」。
CASES = (
    # 名称, 坏样本被拦, 好样本放行, 期望是否通过
    ("只报拦截率(全拒)",        True,  False, False),
    ("只报可用性(全放)",        False, True,  False),
    ("坏样本拦+好样本放",       True,  True,  True),
    ("缓解前漏6/缓解后漏1",     True,  True,  True),
    ("回归后误拒升高",          True,  False, False),
)


def coverage_gap(deployed):
    """账 A：按固定五侧顺序返回未布防的侧（不依赖集合迭代顺序）。"""
    return [s for s in SIDES if s not in deployed]


def added_sides(capabilities):
    """账 C：按能力固定顺序返回新增面，顺序确定。"""
    return [(name, side) for name, side, _ in ADDS if name in capabilities]


def verdict(blocked, allowed):
    """账 D：双臂同时成立才算一份合格的验收用例。"""
    return blocked and allowed


def W(label):
    """墙钟只进 stderr，不污染 stdout 的确定性。"""
    print("[wall] %-28s %.4fs" % (label, time.perf_counter() - _WSTART), file=sys.stderr)


def main():
    p = f = 0

    def check(name, got, exp):
        nonlocal p, f
        hit = got == exp
        p += int(hit)
        f += int(not hit)
        print("  %-42s %s" % (name, "PASS" if hit else "FAIL(期望 %r 实得 %r)" % (exp, got)))

    print("12-01 §12.0 威胁模型总览 · 五侧框架（纯 stdlib / 零网络 / 零随机）")

    # ---------- 账 A：五侧覆盖缺口 ----------
    print("\n[账 A] 五侧框架 · 覆盖缺口判定（大纲 1）")
    check("只布防输出侧 → 缺 4 面", coverage_gap(["输出侧"]),
          ["输入侧", "模型侧", "系统侧", "治理侧"])
    check("输入+输出 → 缺 3 面", coverage_gap(["输入侧", "输出侧"]),
          ["模型侧", "系统侧", "治理侧"])
    check("五侧全布防 → 无缺口", coverage_gap(list(SIDES)), [])
    check("乱序输入不影响输出顺序", coverage_gap(["治理侧", "输出侧", "模型侧"]),
          ["输入侧", "系统侧"])
    W("账A")

    # ---------- 账 B：应用形态 → 暴露面（5 场景 5/5） ----------
    print("\n[账 B] 应用形态 → 暴露哪几侧（大纲 2 · 5 场景判定）")
    expected = {
        "纯聊天问答":         ("输入侧", "模型侧", "输出侧"),
        "RAG 客服":           ("输入侧", "模型侧", "系统侧", "输出侧", "治理侧"),
        "Coding Agent":       ("输入侧", "系统侧", "输出侧", "治理侧"),
        "GUI / 浏览器 Agent": ("输入侧", "系统侧", "输出侧", "治理侧"),
        "企业知识库(多租户)": ("输入侧", "系统侧", "输出侧", "治理侧"),
    }
    for i, (shape, exp) in enumerate(expected.items(), 1):
        check("[%d] %s" % (i, shape), EXPOSE[shape], exp)
    check("纯问答不含系统侧", "系统侧" in EXPOSE["纯聊天问答"], False)
    check("RAG 客服五侧全暴露", len(EXPOSE["RAG 客服"]), 5)
    check("所有形态都含输入侧", all("输入侧" in v for v in EXPOSE.values()), True)
    W("账B")

    # ---------- 账 C：能力增量 → 新开的面 ----------
    print("\n[账 C] 为什么安全是横切目录（大纲 3 · 能力逐档累加）")
    ladder = [[], ["检索"], ["检索", "工具"],
              ["检索", "工具", "多Agent", "多租户", "行业落地"]]
    counts = [len(added_sides(c)) for c in ladder]
    check("能力 0→1→2→5 档 新增面数", counts, [0, 1, 2, 5])
    check("只加检索 → 开系统侧", added_sides(["检索"]), [("检索", "系统侧")])
    check("加行业落地 → 开治理侧",
          [s for _, s in added_sides(["行业落地"])], ["治理侧"])
    check("能力列表乱序仍按固定序输出",
          [n for n, _ in added_sides(["行业落地", "检索"])],
          ["检索", "行业落地"])
    W("账C")

    # ---------- 账 D：验收用例双臂 ----------
    print("\n[账 D] 攻击→防御→验收用例 · 双臂校验（大纲 4）")
    for i, (name, blocked, allowed, exp) in enumerate(CASES, 1):
        check("[%d] %s" % (i, name), verdict(blocked, allowed), exp)
    ok = sum(1 for _, b, a, _ in CASES if verdict(b, a))
    check("合格用例数（5 中 2）", ok, 2)
    check("全拒被判不通过（准确率≠可用性）", verdict(True, False), False)
    W("账D")

    print("\nPASS %d  FAIL %d" % (p, f))
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())
