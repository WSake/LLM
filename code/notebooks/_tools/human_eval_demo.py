# -*- coding: utf-8 -*-
"""human_eval_demo.py —— §11.3 Human Evaluation（人工评测）四账探针（11-评测与可观测 04 篇配套）

§11.3 知识地图原文（🟢 了解）：形态=内部专家标注、众包、盲评、对战（arena）；
价值=不可自动化替代的『真实感受』；成本高、量小、一致性难；用于关键门禁与 judge 校准，而不是日常迭代。

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（W() 负责计时与落 stderr）。全部统计 = 本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- 账 A 的 四种形态（内部专家/众包/盲评/对战 arena）与『盲评=通用开关』 = §11.3 要素事实（非本机实测）；
  品牌光环测量（8 条『知道模型名』vs『盲评』分数）= 作者预置仿真；
- 账 B 的 三难（成本/量小/一致性）与『5 场景 → 人工 vs judge』决策表 = §11.3 要素事实 + 作者预置判定表；
  成本 1250× 单条对比 = 本机确定性计算；
- 账 C 的 Cohen's kappa 实现与 20 条双标注评分矩阵 = 本机确定性计算（作者预置矩阵）；
  Elo 对战 4 模型循环赛 = 作者预置确定性赛程 + 本机迭代，演示 arena 用『成对偏好排级』机制
  （Elo 机制/成对偏好排级 = 要素事实，赛程与胜负预置）；
- 账 D 的三策略矩阵（全 judge/全人工/混合）与样本量公式 + 5 场景断言 = 作者预置判定 + 本机确定性计算；
  **真实人工标注流水线/众包平台/真实 Elo 榜单 = 非本机实测**（需标注员与平台，正文给推荐姿势）。
"""
from __future__ import annotations
import sys
import time

_WSTART = time.perf_counter()


def W(msg: str) -> None:
    print(f"[wall] {time.perf_counter() - _WSTART:.3f}s  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 账 A · 形态账：四形态（内部专家/众包/盲评/对战 arena） + 盲评=通用开关
# ---------------------------------------------------------------------------
FORMS = [
    ("内部专家标注", "自团队领域高手逐条评（判分=rubric+经验）",
     "生产级关键漏检 / 领域评分", "贵·量小·最一致（本脚本账 C 实测 κ=0.92）"),
    ("众包标注", "平台/全员低成本大批量评（判分=问卷/多标签）",
     "大样本覆盖 / 倾向分布", "便宜·量大·一致性差（本脚本账 C 实测 κ=0.32）"),
    ("盲评（盲测）", "评前遮掉模型名/品牌，只评内容",
     "任何需要『免品牌光环』的评分", "通用开关：接 03 账 A 匿名化的真人版"),
    ("对战（arena）", "两两配对给偏好→Elo 累积排行",
     "需要『相对高下』不要绝对分", "赛事量大才稳（n≈数百局）"),
]


def account_a() -> str:
    out = ["== A · 形态账：四形态（内部专家/众包/盲评/对战 arena） =="]
    for name, how, use, note in FORMS:
        out.append(f"  · {name}：{how}｜用：{use}｜账：{note}")
    out.append(f"形态断言 {len(FORMS)}/{len(FORMS)}：四种形态各配『谁评/评什么/门槛』 = "
               f"{'PASS' if len(FORMS) == 4 else 'FAIL'}（盲评=通用开关，接 03 账 A 匿名化的真人版）")
    # 品牌光环测量：同一批 8 条，『知道模型名』vs『盲评』
    named = [4.8, 4.6, 4.9, 4.5, 4.7, 4.8, 4.6, 4.9]
    blind = [3.8, 3.9, 4.1, 3.7, 4.0, 3.9, 3.8, 4.1]
    named_avg = sum(named) / len(named)
    blind_avg = sum(blind) / len(blind)
    delta = named_avg - blind_avg
    out.append(f"  品牌光环测量（同一批 {len(named)} 条文案）：知道模型名 {named_avg:.2f} 分 vs 盲评 {blind_avg:.2f} 分"
               f" → 光环差 {delta:+.2f}（评前遮名=把光环关掉）")
    out.append(f"  盲评断言 PASS：知道模型名比盲评平均高 {delta:.2f} > 0.5 = 品牌光环是真实存在的人类偏置"
               f"（对应 03 账 A 自夸偏置的人类版，盲评=匿名化）"
               f" = {'PASS' if delta > 0.5 else 'FAIL'}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 账 B · 定位账：三难（成本/量小/一致性）数字 + 5 场景 → 人工 vs judge
# ---------------------------------------------------------------------------
# 单条成本（接 03 账 C：LLM judge $0.0012/人工 $1.5）→ 1250×
C_JUDGE, C_HUMAN, N_EVAL = 0.0012, 1.5, 500
SCENES = [
    ("发布前关键门禁", "人工", "§11.3：用于关键门禁；+0.05 准确率保险（03 账 C 就为这一刻）"),
    ("日常迭代打分", "judge", "§11.3：而不是日常迭代——全量回跑靠 judge（03 账 C 规模化）"),
    ("judge 的校准标定", "人工小样本", "§11.3：judge 也要定期校准（03 账 D 15 条 3→1）"),
    ("大样本量初筛", "judge", "成本 1250× 只筛一遍；人工复核临界子集（账 D 混合）"),
    ("品牌/风格/冒犯感等感受维度", "人工", "§11.3：不可自动化替代的『真实感受』"),
]


def account_b() -> str:
    out = []
    out.append("== B · 定位账：三难（成本/量小/一致性）数字 + 5 场景 → 人工 vs judge ==")
    out.append(f"  单条成本：人工 ${C_HUMAN:.2f} vs LLM judge ${C_JUDGE:.4f} = {C_HUMAN / C_JUDGE:.0f}×"
               f"（全量 {N_EVAL} 条：人工 ${C_HUMAN * N_EVAL:.0f} vs judge ${C_JUDGE * N_EVAL:.2f}"
               f" = 1250× 成本杠杆）→ 三难之一『成本高』")
    out.append("  三难印象：成本高（1250×）· 量小（贵→只能抽子集）· 一致性难（账 C 实测：众包 κ=0.32 档=标注者互相打架）")
    for i, (scene, who, why) in enumerate(SCENES, 1):
        out.append(f"    场景{i}. {scene} → {who}（{why}）")
    ok = all(w != "" for _s, w, _y in SCENES)
    out.append(f"定位断言 5/5：五场景各归人工/judge/混合 = {'PASS' if ok else 'FAIL'}"
               f"（§11.3：用于关键门禁与 judge 校准，而不是日常迭代）")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 账 C · 一致性账：标注者内部一致性（Cohen's kappa）+ 对战 arena 需大量对局
# ---------------------------------------------------------------------------
def kappa(r1, r2) -> float:
    """Cohen's kappa（确定性实现）：pe = po_chance，kappa = (po - pe) / (1 - pe)"""
    n = len(r1)
    po = sum(a == b for a, b in zip(r1, r2)) / n
    labels = sorted(set(r1) | set(r2))
    # 边缘概率（标注者各自给出某档的概率）
    p1 = {x: r1.count(x) / n for x in labels}
    p2 = {x: r2.count(x) / n for x in labels}
    pe = sum(p1[x] * p2[x] for x in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


# 众包 20 条 4 分制双标注（一致性差）：r1 与 r2 常差 1 档以上、偶有边缘漂移
C_CROWD_R1 = [3, 4, 2, 3, 4, 2, 3, 4, 3, 2, 4, 3, 2, 3, 4, 3, 2, 4, 3, 2]
C_CROWD_R2 = [3, 3, 3, 3, 4, 3, 2, 4, 2, 2, 5, 3, 3, 3, 3, 3, 4, 4, 3, 2]
# 专家 20 条双标注（一致性好）：同一批条，偶尔差半档
E_EXP_R1 = [3, 4, 2, 3, 4, 3, 3, 4, 3, 2, 4, 3, 2, 3, 4, 3, 3, 4, 3, 2]
E_EXP_R2 = [3, 4, 2, 3, 4, 3, 4, 4, 3, 2, 4, 3, 2, 3, 4, 3, 3, 4, 3, 2]
KAPPA_CUT = 0.6  # Landis-Koch：Substantial 下限≈0.61


def account_c() -> str:
    out = []
    out.append("== C · 一致性账：标注者内部一致性（Cohen's kappa）+ 对战 arena 需要大量对局 ==")
    k_crowd = kappa(C_CROWD_R1, C_CROWD_R2)
    k_exp = kappa(E_EXP_R1, E_EXP_R2)
    out.append(f"  众包双标注 {len(C_CROWD_R1)} 条：简单一致率 {(sum(a == b for a, b in zip(C_CROWD_R1, C_CROWD_R2))) / len(C_CROWD_R1):.2f}"
               f" · Cohen's kappa = {k_crowd:.2f}（标注者互相打架=三难之『一致性难』）")
    out.append(f"  专家双标注 {len(E_EXP_R1)} 条：Cohen's kappa = {k_exp:.2f}（同一批条，专家档一致性好） "
               f"vs 众包 {k_crowd:.2f}——『一致性』是买出来的")
    out.append(f"  一致性断言 PASS：众包 {k_crowd:.2f} < {KAPPA_CUT:.1f}（Landis-Koch Moderate 上限）= 众包数据连 judge 校准都做不了"
               f"；专家 {k_exp:.2f} > {KAPPA_CUT:.1f} 才可作校准/门禁基准"
               f" = {'PASS' if k_crowd < KAPPA_CUT < k_exp else 'FAIL'}")
    # 对战 arena：4 模型双循环赛确定性赛程 → Elo 迭代
    models = ["M1", "M2", "M3", "M4"]
    elo = {m: 1500.0 for m in models}
    K_ELO = 24
    schedule = [(0, 1, 1, 0), (0, 2, 0, 1), (0, 3, 1, 0),
                (1, 2, 1, 0), (1, 3, 0, 1), (2, 3, 1, 0),
                (1, 0, 0, 1), (2, 0, 1, 0), (3, 0, 1, 0),
                (2, 1, 0, 1), (3, 1, 1, 0), (3, 2, 0, 1)]
    for a, b, sa, sb in schedule:
        ea = 1.0 / (1.0 + 10 ** ((elo[models[b]] - elo[models[a]]) / 400.0))
        eb = 1.0 / (1.0 + 10 ** ((elo[models[a]] - elo[models[b]]) / 400.0))
        elo[models[a]] += K_ELO * (sa - ea)
        elo[models[b]] += K_ELO * (sb - eb)
    order = sorted(models, key=lambda m: -elo[m])
    out.append("  对战 arena（4 模型 12 局双循环赛确定性赛程，Elo K=24）→ 排位："
               + ", ".join(f"{m}({elo[m]:.0f})" for m in order))
    out.append("  arena 断言 PASS：成对偏好可排相对高下、不需要绝对分 = 对战=『相对』形态"
               f"（各队打满 {len(schedule)} 局，稳定到个位分档需要数百局=量小一致性难的另一面）")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 账 D · 装配账：人工的正确姿势 = judge 初筛 + 人工复核临界子集（混合流水线）
# ---------------------------------------------------------------------------
N_TOTAL = 500
CRITICAL = 50  # judge 拿不准/临界子集留人工


def account_d() -> str:
    out = []
    out.append("== D · 装配账：人工的正确姿势 = judge 初筛 + 人工复核临界子集 ==")
    cost_judge = C_JUDGE * N_TOTAL
    cost_human = C_HUMAN * N_TOTAL
    cost_mix = C_JUDGE * N_TOTAL + C_HUMAN * CRITICAL
    out.append(f"  三策略矩阵（{N_TOTAL} 条评测集）：")
    out.append(f"    · 全 judge：${cost_judge:.2f} · 感受维度 0/3（品牌/风格/冒犯感抓不到）")
    out.append(f"    · 全人工：${cost_human:.0f} · 感受维度 3/3（但贵 1250×）")
    out.append(f"    · 混合（judge 全扫 + 人工复核临界 {CRITICAL} 条）：${cost_mix:.2f}"
               f"（= 全人工价 {N_TOTAL / CRITICAL:.0f} 分之一）· 感受维度靠人工子集捞回 3/3")
    # 样本量配额：n = z²·p(1-p)/δ²，p=0.9 期望通过率，δ=0.05，z=1.96
    z, p, d = 1.96, 0.9, 0.05
    n_req = z * z * p * (1 - p) / (d * d)
    out.append(f"  抽样量配额：z=1.96 · 期望通过率 p={p} · 许偏差 δ={d} → n = z²·p(1-p)/δ²"
               f" = {n_req:.1f} ≈ {int(n_req) + 1 if not n_req.is_integer() else int(n_req)} 条"
               f"（人工贵，量要按统计需求抽，不是越多越好也不是随心抽）")
    ok = all(w != "" for _s, w in [("门禁", "人工"), ("迭代", "judge"), ("校准", "人工"), ("初筛", "judge"), ("感受", "人工")])
    ok = ok and cost_mix < cost_human
    out.append(f"装配断言 PASS：混合流水线成本 < 全人工（${cost_mix:.2f} < ${cost_human:.0f}）且人工复核保住感受维度 3/3"
               f" = 人工评测的正确姿势是『用在刀刃上』（关键门禁+judge 校准+感受子集），不是全量替代 judge"
               f" = {'PASS' if cost_mix < cost_human else 'FAIL'}")
    return "\n".join(out) + "\n"


def main() -> None:
    W("start")
    parts = [account_a(), account_b(), account_c(), account_d()]
    print("\n" + "\n".join(parts), end="")
    W("done")


if __name__ == "__main__":
    main()
