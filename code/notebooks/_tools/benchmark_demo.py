# -*- coding: utf-8 -*-
"""benchmark_demo.py —— §11.1 Benchmark 体系四账探针（11-评测与可观测 02 篇配套）

教"benchmark 的生命周期"：会看纸面数字、知道指标怎么被刷爆（里程碑 016 中文 benchmark 快跑）。

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（一句 W() 负责计时与落 stderr）。全部统计=本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- 账 A 的 七类基准清单 / 生命周期三态（新生·饱和·退役）/ 防污染标记 = 作者按知识地图 §11.1
  与 2025-2026 公开榜单常识整理的**要素事实（本机非实测，写作环境无外网未在线复核）**；
- 账 B 的 泄漏污染公式 / 二项分布 / best-of-k 数组 = **作者的确定性算例**（数字自洽，表示"这三根
  放大器怎么把纸面数字推高一层"的机制，不是某个真实模型被刷的实测）；
- 账 C 的 A/B 双模型×四尺分数表 与 中文 8B vs 英文 70B 表 = **作者预置仿真**（演示"榜单等权/任务加权
  决策翻转"与"语言不对齐结论相反"，非真实模型跑分）；
- 账 D 的 中文快评台 = **职业生涯中文样题 10 道 + 词面规则答案表当"模型"**，判分=本机确定性计算；
  其「测试题禁入 RAG/记忆」等防污染四查 = 知识地图 §11.1 解读基准 4 常识的落地清单（要素事实非本机实测）；
  **真实完整的中文 benchmark 快跑需 lm-eval-harness/OpenCompass + 模型 + 语料 = 非本机实测**（里程碑 016
  在本篇只到"确定性中文快评台 + 四查清单"这层，学术界与工程界做法一并在文章正文展开）。

四账 = §11.1：七类基准×生命周期（A）→ 三根刷分放大器（B）→ 榜单分≠业务分（C）→ 中文快跑+防污染（D）。
"""
from __future__ import annotations
import sys
import time

_WSTART = time.perf_counter()


def W(msg: str) -> None:
    """墙钟只进 stderr。"""
    print(f"[wall] {time.perf_counter() - _WSTART:.3f}s  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 账 A · 尺子谱系账：§11.1 七类基准 × 生命周期三态
# ---------------------------------------------------------------------------
# (类别, 基准名, 生命周期, 一句话标注)
# 生命周期三态：新生（榜单还在爬升/不断换新题）· 饱和（已被卷到天花板）· 退役（旧题被刷爆退居二线）
BENCHMARKS = [
    ("通用知识", "MMLU", "饱和", "2021 出生，2024 后多模型 ~90 封顶=经典饱和案例"),
    ("通用知识", "MMLU-Pro", "新生", "2024 加大难度+防泄漏，接 MMLU 的班"),
    ("通用知识", "HellaSwag", "饱和", "2020 词面常识，2024 后已挤爆"),
    ("通用知识", "WinoGrande", "饱和", "2019 指代消解，经典饱和题"),
    ("通用知识", "ARC-C", "退役", "2018 挑战集，已从新榜单撤下"),
    ("通用知识", "BBH", "新生", "2022 BIG-Bench Hard 推理难度精选"),
    ("推理数学", "GSM8K", "饱和", "2021 小学应用题，2024 后 ~95 到顶"),
    ("推理数学", "MATH", "饱和", "2021 竞赛数学，被 AIME 等新题超越"),
    ("推理数学", "AIME", "新生", "2024 后推理主战场=真竞赛题"),
    ("推理数学", "GPQA", "新生", "2023 研究生科学问答「钻石级」"),
    ("推理数学", "Humanity's Last Exam", "新生", "2025 最难封顶题，1.2 万题防泄漏"),
    ("代码", "HumanEval", "退役", "2021 鸡尾酒题 pass@1 90+ 已退役"),
    ("代码", "MBPP", "退役", "2021 初级编程，饱和后退役"),
    ("代码", "LiveCodeBench", "新生", "持续更新的比赛题=防污染金标准"),
    ("代码", "SWE-bench", "新生", "2024 真实仓库 issue 修复任务"),
    ("指令对话", "IFEval", "新生", "可验证的指令规则遵循"),
    ("指令对话", "MT-Bench", "饱和", "2023 多轮聊天人工/LLM 打分"),
    ("指令对话", "Arena-Hard", "新生", "2024 高质量问题对战"),
    ("指令对话", "LMArena", "新生", "全人工对战 Elo 榜"),
    ("长上下文", "Needle-in-a-Haystack", "饱和", "2023 大海捞针，通过率已趋同"),
    ("长上下文", "RULER", "新生", "2024 结构化合成长文本测试"),
    ("长上下文", "LongBench", "新生", "2023 十四任务长上下文"),
    ("长上下文", "LongICLBench", "新生", "2024 超长 in-context learning 测试"),
    ("长上下文", "LOFT", "新生", "2024 长上下文真实世界检索/理解/推理"),
    ("多模态", "MMMU", "新生", "2023 大学多学科多模态理解"),
    ("多模态", "MMBench", "稳定", "中文视觉综合基准"),
    ("多模态", "OCRBench", "新生", "OCR 专项测试"),
    ("多模态", "ChartQA", "稳定", "图表问答实测"),
    ("多模态", "VideoMME", "新生", "视频-文本融合理解"),
    ("中文", "C-Eval", "新生", "2023 中文知识考题，分学科可控"),
    ("中文", "CMMLU", "新生", "2023 中文知识考题，大陆专属"),
    ("中文", "SuperCLUE", "稳定", "中文综合能力榜单"),
    ("中文", "AlignBench", "新生", "2023 中文对齐评测"),
]

# 防污染标记：更新快的更可信（知识地图解读基准 4 常识 ①）
ANTI_LEAK = {
    "MMLU-Pro", "Humanity's Last Exam", "LiveCodeBench",
    "SWE-bench", "Arena-Hard", "RULER", "LOFT",
}


def account_a() -> str:
    out = []
    out.append("== A · 尺子谱系账：七类基准 × 生命周期三态 ==")
    classes = []
    current = None
    for cat, name, life, note in BENCHMARKS:
        if cat != current:
            current = cat
            classes.append([cat, []])
        classes[-1][1].append(name)
    class_cnt = {c: len(n) for c, n in classes}
    per_class_ge2 = all(v >= 2 for v in class_cnt.values())
    out.append(f"七类基准：{len(classes)} 类；每类≥2 断言 {'PASS' if per_class_ge2 else 'FAIL'}")
    for c, names in classes:
        out.append(f"  [{c}] {len(names)} 个：{', '.join(names)}")
    life_cnt: dict[str, int] = {}
    for _cat, _n, life, _note in BENCHMARKS:
        life_cnt[life] = life_cnt.get(life, 0) + 1
    life_keys = sorted(life_cnt)
    out.append(f"生命周期三态分布：{'; '.join(f'{k}={life_cnt[k]}' for k in life_keys)} · 每态≥2 断言 "
               f"{'PASS' if min(life_cnt.values()) >= 2 else 'FAIL'}")
    saturated = [n for _c, n, l, _t in BENCHMARKS if l == "饱和"]
    retired = [n for _c, n, l, _t in BENCHMARKS if l == "退役"]
    anti = [n for _c, n, _l, _t in BENCHMARKS if n in ANTI_LEAK]
    out.append(f"饱和代表：{', '.join(sorted(saturated))}")
    out.append(f"退役代表：{', '.join(sorted(retired))}")
    out.append(f"防污染标记（更新快更可信）{len(anti)} 个：{', '.join(sorted(anti))}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 账 B · 读数账：三根放大器把纸面数字刷高一层（会看图，更要会拆图）
# ---------------------------------------------------------------------------
def _binom_ge(n: int, k: int, p: float = 0.5) -> float:
    """二项分布 P(X>=k)，n 个小样本独立事件正确率 p。math.comb 精确整数运算。"""
    total = 1 << n
    return sum(comb := __import__("math").comb(n, i) for i in range(k, n + 1)) / total


def account_b() -> str:
    out = []
    out.append("== B · 读数账：三根刷分放大器 ==")
    # B1 泄漏污染：100 题基准，20 题训练泄漏，泄漏题 100% 答对，其余按真实水平 0.60
    N, L, c = 100, 20, 0.60
    leaked = L * 1.0
    clean = (N - L) * c
    reported = (leaked + clean) / N
    truth = c
    out.append(f"B1 泄漏污染：N={N} 题基准，污染覆盖 L={L} 题（训练见过→评测 100% 答对），"
               f"其余 {N - L} 题按真实水平 c={c:.2f}")
    out.append(f"    榜单得分 = ({L}×1.00 + {N - L}×{c:.2f})/{N} = {reported:.3f}；真实水平 = {truth:.3f}"
               f"；虚增 +{reported - truth:.3f}（{int(round((reported - truth) * 100))} pp）")
    # B2 小样本方差：20 题 p=0.5，考到 ≥60%（≥12 题）的概率=掷硬币
    n2, k2 = 20, 12
    p2 = _binom_ge(n2, k2)
    out.append(f"B2 小样本方差：仅 {n2} 题、两选一 p=0.5，靠猜考到 ≥{k2}/{n2}（60%+）的概率 = "
               f"{p2:.4f}（≈1/4）→ 小样本榜单是掷硬币，不能当水平差")
    # B3 best-of-k：固定分数数组，挑最好一次发布 vs 平均
    scores = [0.52, 0.50, 0.48, 0.55, 0.51]
    avg = sum(scores) / len(scores)
    best = max(scores)
    out.append(f"B3 best-of-k：同一模型 {len(scores)} 次评测得分 {scores}；平均={avg:.4f}，挑最好={best:.3f}"
               f" → 只报最好一次虚高 {best - avg:+.3f}（榜单小字里藏着『取最优』）")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 账 C · 相关性账：榜单分 ≠ 业务分；榜单口径一换，结论能翻车
# ---------------------------------------------------------------------------
# (模型, MMLU, GSM8K, HumanEval, 业务任务); 榜单用前三把尺，业务加权把「业务」放大 4 倍
MODELS = [
    ("模型 A", 0.72, 0.80, 0.55, 0.50),
    ("模型 B", 0.70, 0.60, 0.70, 0.85),
]
LEADERBOARD_KEYS = ["MMLU", "GSM8K", "HumanEval"]
WEIGHT_BIZ = 4


def account_c() -> str:
    out = []
    out.append("== C · 相关性账：榜单分 ≠ 业务分 ==")
    out.append("模型四尺分数 (MMLU/GSM8K/HumanEval/业务任务)：")
    for m, a, b, c, d in MODELS:
        out.append(f"    {m}: {a:.2f}/{b:.2f}/{c:.2f}/{d:.2f}")
    # 等权三榜单
    lb = {m[0]: (m[1] + m[2] + m[3]) / 3 for m in MODELS}
    # 任务加权：业务×4，其余×1
    wb = {m[0]: (m[1] + m[2] + m[3] + WEIGHT_BIZ * m[4]) / (3 + WEIGHT_BIZ) for m in MODELS}
    lb_win = max(lb, key=lb.get)
    wb_win = max(wb, key=wb.get)
    out.append(f"等权三榜单平均：{', '.join(f'{k}={v:.3f}' for k, v in lb.items())} → {lb_win} 赢")
    out.append(f"业务加权(×{WEIGHT_BIZ})平均：{', '.join(f'{k}={v:.3f}' for k, v in wb.items())} → {wb_win} 赢")
    out.append(f"决策翻转：{'PASS（榜单口径一换，结论反了）' if lb_win != wb_win else 'FAIL'}")
    # 语言不对齐：中文 8B vs 英文 70B，英文榜单 vs 中文榜单结论相反
    en_mmlu = {"中文 8B": 0.62, "英文 70B": 0.71}
    zh_ceval = {"中文 8B": 0.80, "英文 70B": 0.45}
    en_win = max(en_mmlu, key=en_mmlu.get)
    zh_win = max(zh_ceval, key=zh_ceval.get)
    out.append(f"语言不对齐：英文 MMLU {en_mmlu} → {en_win} 赢；中文 C-Eval {zh_ceval} → {zh_win} 赢"
               f" → 结论相反 {'PASS' if en_win != zh_win else 'FAIL'}（中文 8B 靠中文语料，英文 70B 靠规模）")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 账 D · 中文快跑账（里程碑 016 · 中文 benchmark 快跑）：确定性中文样题快评台 + 防污染四查
# ---------------------------------------------------------------------------
# 中文样题 10 道：5 常识 / 3 数学 / 2 代码。快评台分三段：
#   「模型」= 关键词规则表（词面触发 → 生成答案串，作者预置仿真）
#   「判分器」= 期望关键字是否落在生成的答案串里（本机确定性计算）
#   「报告」= 总分 + 分科 + 快筛结论
QUIZ = [
    ("常识", "中国的首都是哪座城市？", "首都", "北京"),
    ("常识", "太阳从哪个方向升起？", "升起", "东"),
    ("常识", "一年有多少个月？", "年有", "12"),
    ("常识", "水的化学式是什么？", "化学式", "H2O"),
    ("常识", "中华人民共和国的国庆日是几月几号？", "国庆日", "10月1日"),
    ("数学", "17 + 25 = ？", "+ 25", "42"),
    ("数学", "100 - 37 = ？", "- 37", "63"),
    ("数学", "12 × 8 = ？", "× 8", "96"),
    ("代码", "Python 里 list 最大值是多少？", "最大值", "5"),
    ("代码", "判断整数 n 是否偶数的表达式？", "偶数", "n % 2 == 0"),
]
# 快评台 v1「模型」：关键词→答案（答对表）；写死即 10/10
RULE_OK = {q: a for _c, q, _k, a in QUIZ}
# 快评台 v2「模型」：规则表忘了一条别名（"n % 2 == 0" 漏了空格）→ 该题生成答案不含期望串
RULE_BUGGY = dict(RULE_OK)
RULE_BUGGY["判断整数 n 是否偶数的表达式？"] = "n%2==0"


def run_demo_model(question: str, rule: dict) -> str:
    """关键词规则『模型』：题里有某关键词就吐该答案串（纯确定性，替代真实 LLM 推理）。"""
    return rule.get(question, "")


def account_d() -> str:
    out = []
    by_class = ["常识", "数学", "代码"]

    def run(rule: dict, label: str) -> tuple[int, list]:
        hit = [(c, q, a, a in run_demo_model(q, rule)) for c, q, _k, a in QUIZ]
        return sum(1 for _c, _q, _a, ok in hit if ok), hit

    out.append("== D · 中文快跑账：中文样题快评台 + 防污染四查（里程碑 016） ==")
    n1, hit1 = run(RULE_OK, "v1 答对表")
    out.append(f"快评台 v1（答对表全配齐）：{n1}/{len(QUIZ)} 命中 → "
               f"{'10/10=能跑通的中文能力层快筛基线' if n1 == len(QUIZ) else 'FAIL'}")
    for c, q, a, ok in hit1:
        out.append(f"    [{'✓' if ok else '✗'}]（{c}）{q} → 生成『{RULE_OK[q]}』期望『{a}』"
                   f"命中={ok}")
    n2, hit2 = run(RULE_BUGGY, "v2 漏一条别名")
    out.append(f"快评台 v2（规则表漏了『n%2==0』的空格）：{n2}/{len(QUIZ)} 命中 → 判分器抓 1 道 ✗：")
    for c, q, a, ok in hit2:
        if not ok:
            out.append(f"    [✗]（{c}）{q} → 生成『{RULE_BUGGY[q]}』期望『{a}』不命中 → "
                       f"词面判分=既能打分也能漏（规则是双刃）")
    cls_stat = {c: sum(1 for _c2, _q, _a, ok in hit1 if _c2 == c and ok) for c in by_class}
    cls_tot = {c: sum(1 for _c2, _q, _k, _a in QUIZ if _c2 == c) for c in by_class}
    out.append("v1 分科：" + ", ".join(f"{c} {cls_stat[c]}/{cls_tot[c]}"
                                       for c in by_class))
    # 防污染四查清单（知识地图解读基准 4 常识的落地形态）
    LEAK_CHECK = [
        ("测试题禁入语料", "评测集只用于评测，不 tune 进训练语料/模型记忆/检索库"),
        ("版本锁定", "同版本评测框架(lm-eval-harness)+同题目快照，换版本要重跑基线"),
        ("同语料子集", "固定采样 seed + 固定题目子集，两次比较跑同一批题"),
        ("更新快的优先", "选 LiveCodeBench 式持续换题的基准；中文 C-Eval/RULER 等新题优先"),
    ]
    out.append(f"防污染四查清单（{len(LEAK_CHECK)}/4 全过=可以对外报数）：")
    for i, (name, why) in enumerate(LEAK_CHECK, 1):
        out.append(f"    {i}. {name}——{why} → PASS")
    out.append(f"四查断言 {len(LEAK_CHECK)}/{len(LEAK_CHECK)} PASS：快评台 + 四查才是『报数』的完整闭环")
    return "\n".join(out) + "\n"


def main() -> None:
    W("start")
    plan = [
        account_a,
        account_b,
        account_c,
        account_d,
    ]
    print("\n".join(f() for f in plan), end="")
    W("done")


if __name__ == "__main__":
    main()
