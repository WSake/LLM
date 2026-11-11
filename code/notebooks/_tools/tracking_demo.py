# -*- coding: utf-8 -*-
"""15-07 其他方向跟踪 —— 五方向「成熟度 × 主干锚点 × 学习动作」跟踪表。

账 A  成熟度档位：一行五方向（已标配/快速增长/在途/立法驱动/早期）
账 B  主干锚点都能在主干章节对得上（不映射到虚标题）
账 C  学习动作判定：由成熟度决定下一步动作
"""
import sys
import time

_WSTART = time.perf_counter()

# (方向, 成熟度档, 主干锚点, 学习动作)
TRACKS = (
    ("搜索增强生成",     "已标配",   "05-推理与部署|11-05 RAG评测",        "看论文+RAG demo"),
    ("代码执行/沙箱推理", "在途",    "02-16 GRPO与RLVR|12-06|11-06 Agent评测", "写沙箱 RL 小实验"),
    ("端侧 Agent",       "快速增长", "15-05 SLM与端侧|14-应用工程",        "跟端侧 SDK demo"),
    ("模型身份与安全",    "立法驱动", "12-安全全章|12-11 红队方法",          "跟法规+看标准"),
    ("长时任务与 Agent 云", "早期",  "02-20 推理模型|13-02 分布式|13-07 可观测", "等基建标准"),
)

MATURITY_ORDER = ("已标配", "快速增长", "在途", "立法驱动", "早期")

# 主干目录白名单（规矩：锚点第一个词必须是这些目录里的存在标题）
ANCHOR_ROOTS = (
    "05-推理与部署", "05-推理", "02-", "12-", "11-", "15-", "14-", "13-",
)


def maturity_rank(m):
    try:
        return MATURITY_ORDER.index(m)
    except ValueError:
        return -1


def anchor_ok(x):
    return any(x.startswith(r) for r in ANCHOR_ROOTS)


def learn_action(m):
    return {
        "已标配":   "看论文+RAG demo",
        "快速增长": "跟端侧 SDK demo",
        "在途":     "写沙箱 RL 小实验",
        "立法驱动": "跟法规+看标准",
        "早期":     "等基建标准",
    }[m]


def run():
    ok, total = 0, 0

    def check(name, got, exp):
        nonlocal ok, total
        total += 1
        if got == exp:
            ok += 1
            print("  %-44s PASS" % name)
        else:
            print("  %-44s FAIL(期望 %r 实得 %r)" % (name, exp, got))

    print("\n[账 A] 成熟度档位")
    check("五方向一行齐", len(TRACKS), 5)
    check("档位全在合法五档", all(maturity_rank(t[1]) >= 0 for t in TRACKS), True)
    check("已标配=搜索增强",
          next(t[0] for t in TRACKS if t[1] == "已标配"), "搜索增强生成")
    check("早期=长时任务 Agent 云",
          next(t[0] for t in TRACKS if t[1] == "早期"), "长时任务与 Agent 云")

    print("\n[账 B] 主干锚点可查")
    check("锚点不指向虚标题", all(anchor_ok(t[2].split("|")[0]) for t in TRACKS), True)
    check("代码沙箱锚 02-16/12-06/11-06",
          any("02-16" in t[2] and "12-06" in t[2] and "11-06" in t[2]
              for t in TRACKS), True)
    check("长时任务锚 02-20/13-02/13-07",
          any("02-20" in t[2] and "13-02" in t[2] and "13-07" in t[2]
              for t in TRACKS), True)
    check("端侧 Agent 锚 15-05",
          any(t[1] == "快速增长" and "15-05" in t[2] for t in TRACKS), True)

    print("\n[账 C] 学习动作判定")
    right = sum(1 for _, m, _, act in TRACKS if act == learn_action(m))
    check("成熟度→动作一一映射", right, len(TRACKS))

    print("\nPASS %d  FAIL %d" % (ok, total - ok))
    return ok, total


def W(label):
    print("[wall] %-24s %.4fs" % (label, time.perf_counter() - _WSTART),
          file=sys.stderr)


def main():
    W("tracking_demo")
    ok, total = run()
    return 1 if total - ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
