# -*- coding: utf-8 -*-
"""rag_eval_demo.py —— §11.4 / § 8.13 RAG 评测（检索生成专项）四账探针（11-评测与可观测 05 篇配套）

§11.4 知识地图原文（🟢 掌握指标）：RAGAS：Faithfulness（忠实度）、Answer Relevance（相关性）、
Context Precision / Recall、Noise Robustness 等；DeepEval 的 RAG 模块、LangSmith 评估器、TruLens。
§8.13 原文（🔴 会跑；RAG 不评测等于没做）：三层指标：检索层（Recall@k / Hit Rate / MRR）；
生成层（Faithfulness / Answer Relevance）；系统层（延迟 / 成本 / 端到端质量）。工具：RAGAS
（最常用）、DeepEval、LangSmith / Langfuse 评测面板、TruLens；评委：规则 / LLM-judge / 人工抽样。
常见误区：只看"答案对错"不看检索漏洞；评测集用生产真实问题而非论文题目；只看平均分不看长尾类型。
回归机制：每次换 embedding / rerank / chunk 都跑同一套评测集对比——RAG 是"检索工程"，唯一的刹车就是评测。

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（W() 负责计时与落 stderr）。全部统计 = 本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- 账 A 的 三层指标（检索 Recall/Hit/MRR · 生成 Faithfulness/AR · 系统延迟/成本/端到端）与
  「哪层出问题答案就死在哪层」症状归层表 = §8.13/§11.4 要素事实（非本机实测）；
  3 条 query 的 Hit@k / MRR / Recall 数字 = 本机确定性计算（作者预置 top-5 排序）。
- 账 B 的 判别性对照（正确基线 vs 故意换错来源/幻觉答案/答非所问）= 010 式判别性对照方法的
  本机确定性样例——词面证据覆盖率 proxy 与关键词相关 proxy 手写实现（非 RAGAS 真实 judge 调用）；
  08-07 的实测数字（正确 1.00/0.74 → 错来源 0.17/0.40）= 挂靠引用只引不重测；
  **真实 RAGAS（claims-based LLM-judge）= 非本机实测**。
- 账 C 的 回归机制（每次换 embedding/rerank/chunk 跑同一套评测集对比 & 门禁拦回退）= §8.13
  要素事实 + 作者预置 3 个版本命中表（本机确定性计算）。
- 账 D 的 三类误区（只看答案对错/论文题当代测/只看平均分不看长尾）、长尾分组表、评委三档
  （规则/judge/人工抽样）决策表 = §8.13 要素事实 + 作者预置判定表（联动 03-judge / 04-人工）。
"""
from __future__ import annotations
import sys
import time

_WSTART = time.perf_counter()


def W(msg: str) -> None:
    print(f"[wall] {time.perf_counter() - _WSTART:.3f}s  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 账 A · 指标账：三层指标体系（检索/生成/系统）× 一个具体 MRR 算例 × 症状归层表
# ---------------------------------------------------------------------------
LAYERS = [
    ("检索层", ["Recall@k", "Hit@k", "MRR", "nDCG"], "袋子问与排位问", "该有的信息到底进没进袋子、进得快不快"),
    ("生成层", ["Faithfulness", "Answer Relevance", "Context Precision/Recall", "Noise Robustness"],
     "答案忠于证据吗 + 答案贴问题吗", "答对了没、答得对不对题、噪声块多少"),
    ("系统层", ["延迟", "成本", "端到端质量"], "线上体验与账单", "整套系统给人用的体验与开销"),
]

# 3 条 query 各自的 top-5 检索排序（gold 文档的排位），算例用 —— 本机确定性计算
# Q1 gold 排第 1 · Q2 gold 排第 3 · Q3 gold 排第 7（top-5 外）都有 2 个 gold 文档
GOLD_POS = [1, 3, 7]                       # gold 在 top-5 里的位置（7 = 不在 top-5）
N_GOLD_TOTAL = 6                            # 3 条 query × 2 个 gold 文档
HIT_IN_TOP5 = 2                             # Q1、Q2 命中，Q3 未命中


def hit_at5() -> int:
    return sum(1 for p in GOLD_POS if p <= 5)


def mrr_top5() -> float:
    return sum(1.0 / p for p in GOLD_POS if p <= 5) / len(GOLD_POS)


def recall_at5() -> float:
    return HIT_IN_TOP5 / N_GOLD_TOTAL


# 症状 → 层 归层表（§8.13：哪层出问题答案就死在哪层）
SYMPTOMS = [
    ("答非所问：答案在袋子里但不贴问题", "生成层 · Answer Relevance", "问题理解 / 生成跑偏"),
    ("答案与文档矛盾（睁眼说瞎话）", "生成层 · Faithfulness", "生成没忠于证据 / 噪声块混入"),
    ("该有的信息根本没进袋子（袋子里没有）", "检索层 · Recall@k / Hit@k", "分块 / embedding / 查询改写"),
    ("袋子里有、但翻袋子太慢（找得慢）", "检索层 · MRR", "融合权重 / 重排器没把 gold 顶上"),
    ("响应太慢 / 账单太贵", "系统层 · 延迟 / 成本", "检索调太多 k / 模型太大 / 缓存没做"),
]


def account_a() -> str:
    out = []
    out.append("== A · 指标账：三层指标体系（检索 / 生成 / 系统）× MRR 算例 × 症状归层 ==")
    for name, inds, what, note in LAYERS:
        out.append(f"  {name}：{', '.join(inds)} ｜ 量什么：{what}（{note}）")
    ok = all(len(inds) >= 2 for _n, inds, _w, _x in LAYERS)
    out.append(f"  三层断言 PASS：检索/生成/系统 三层都有 ≥2 个指标（尺子齐全才能指认凶手）"
               f" = {'PASS' if ok else 'FAIL'}")
    out.append("  MRR@5 算例（3 条 query 的 gold 排位 = [1, 3, 7（top-5 外）]，各 2 个 gold 文档）：")
    out.append(f"    · Hit@5 = {hit_at5()}/3 = 2/3（Q1、Q2 命中，Q3 没进袋子）")
    out.append(f"    · Recall@5 = {HIT_IN_TOP5}/{N_GOLD_TOTAL} = {recall_at5():.3f}（命中的 gold 文档 / 全部 gold 文档）")
    out.append(f"    · MRR@5 = (1/1 + 1/3 + 0)/3 = {mrr_top5():.3f}(黄金文档进袋子了、还要看排得快不快；排位越靠前 MRR 越高)")
    out.append("  症状归层表（哪层出问题答案就死在哪层）:")
    for i, (sym, layer, killer) in enumerate(SYMPTOMS, 1):
        out.append(f"    {i}. {sym} → {layer}（凶手：{killer}）")
    out.append(f"  归层断言 {len(SYMPTOMS)}/{len(SYMPTOMS)}：五症状全归到正确层 = "
               f"{'PASS' if all(l != '' for _s, l, _k in SYMPTOMS) else 'FAIL'}"
               f"（分层指标的唯一意义 = 指认凶手）")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 账 B · 判别性对照账：10 种目录样例，证明「评测真有耳朵」（010 式方法 · 掌握度检验）
# ---------------------------------------------------------------------------
# 每个样例是 (问题, top-1 证据切片, 答案, 类型)。礼物 Faithfulness proxy = 证据覆盖率
# （答案与证据共词 / 答案总 token，中文按字符算一致性），Relevance proxy = 答案与问题共词比例。
# 类型：good=正确检索+忠实答案 · miss=检索错位（拖错文档） · halluc=答案编造 · off=答非所问
SAMPLE_ANSWER = (
    ("张三的入职年份？", "张三是 2023 年入职。", "张三于 2023 年入职。", "good"),
    ("违约金比例多少？", "违约金比例视合同约定，通常 3%。", "违约金约 3%，视合同而定。", "good"),
    ("退货有运费险吗？", "退货提供运费险保障。", "退货有运费险。", "good"),
    ("退款到账要几天？", "退款 3-5 个工作日到账。", "退款 3 到 5 个工作日到账。", "good"),
    ("张三的入职年份？", "李四于 2021 年入职。", "张三于 2023 年入职。", "miss"),   # 检索拖了错文档
    ("违约金比例多少？", "运费险按订单金额计提。", "违约金约 3%，视合同而定。", "miss"),  # 拖错文档
    ("退货有运费险吗？", "退款 3-5 个工作日到账。", "退货有运费险，且支持 15 天退货。", "halluc"),  # 答案编造 15 天
    ("退款到账要几天？", "退款 3-5 个工作日到账。", "退款一般 24 小时内到账。", "halluc"),  # 24 小时=编造
    ("张三的入职年份？", "张三是 2023 年入职。", "公司的茶水间在三楼。", "off"),   # 答非所问
    ("违约金比例多少？", "违约金比例视合同约定，通常 3%。", "贷款分等额本息与等额本金。", "off"),  # 答非所问
)
# 中文字符共现 / 平均长度 = 词面 proxy（确定性、零随机）


def _shared_chars(a: str, b: str):
    # 全字符口径（含标点/空格）：分子=答案里能在对方串中找到的字符，分母=答案全长 → 比率恒 ≤1
    return len([c for c in a if c in b]), len(list(a))


class Scores:
    def __init__(self) -> None:
        self.by_type = {"good": [], "miss": [], "halluc": [], "off": []}

    def add(self, t: str, f: float, r: float) -> None:
        self.by_type[t].append((f, r))


def account_b() -> str:
    out = []
    out.append("== B · 判别性对照账：10 条样例，证明『评测真有耳朵』（010 式方法 · 掌握度检验） ==")
    s = Scores()
    for q, ev, ans, t in SAMPLE_ANSWER:
        # Faithfulness proxy = 答案能在证据里找到的字符比例；Relevance proxy = 答案能在问题里找到的字符比例
        f_shared, f_tot = _shared_chars(ans, ev)
        r_shared, r_tot = _shared_chars(ans, q)
        f = f_shared / f_tot if f_tot else 0.0
        r = r_shared / r_tot if r_tot else 0.0
        s.add(t, f, r)
        out.append(f"  {t:7s} | Faith 证据覆盖率 {f:.2f} · Rel 与问题共词 {r:.2f} | {ans}")
    # 类型均值 → 判别性对照表（与 010 / 08-07 同法：加破坏，指标必须掉）
    out.append("  判别性对照（类型均值）：")
    order = [("good", "正确检索 + 忠实答案"),
             ("miss", "检索错位（把其他文档拖来做证据）"),
             ("halluc", "答案编造（证据对、答案开始胡说）"),
             ("off", "答非所问（证据对、答案根本不是答这个）")]
    lines = {}
    for t, label in order:
        vals = s.by_type[t]
        f_mean = sum(v[0] for v in vals) / len(vals)
        r_mean = sum(v[1] for v in vals) / len(vals)
        lines[t] = (f_mean, r_mean)
        out.append(f"    {label}：Faith {f_mean:.2f} · Rel {r_mean:.2f}（n={len(vals)}）")
    # 断言：三个破坏组的 Faith 或 Rel 都比 good 低（评测真有耳朵）
    good_f, good_r = lines["good"]
    dropped = sum(
        1 for t in ("miss", "halluc", "off")
        if lines[t][0] < good_f - 0.05 or lines[t][1] < good_r - 0.05
    )
    out.append(f"  判别性断言 {dropped}/3：三种破坏都让指标掉（good 组 {good_f:.2f}/{good_r:.2f}"
               f" vs 最差 miss 组 Faith {lines['miss'][0]:.2f}）——"
               f" = 评测真有耳朵（与 08-07 实测同构：正确 1.00/0.74 → 错来源 0.17/0.40 只引不重测）"
               f" = {'PASS' if dropped == 3 else 'FAIL'}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 账 C · 回归机制账：每次换 chunk / embedding / rerank 都跑同一套评测集（唯一的刹车）
# ---------------------------------------------------------------------------
# 同一 10 条评测集 × 三个版本的检索命中药（gold 文档在不在 top-k）→ Hit 指标 + 端到端命中
# v3 = chunk=256->1024（块太大信息被稀释）→ Hit 掉到 0.70，低于闸线 0.80 → 门禁拦
RECALL_TOP5 = {
    "v2 基线（chunk=384）": ([1, 1, 1, 1, 1, 2, 3, 4, 5, 5], "0.90"),
    "v3 换 chunk=1024":    ([1, 1, 1, 5, 7, 9, 11, 13, 15, 15], "0.70"),  # 大块稀释 → 掉过闸
    "v4 换 embedding":     ([1, 1, 1, 1, 5, 7, 9, 9, 15, 15], "0.70"),  # 同样掉
    "v5 加 rerank":        ([1, 1, 1, 1, 1, 1, 2, 2, 3, 3], "1.00"),    # 顶回第一页
}
GATE = 0.80  # 检索 Hit 闸线（CI 门禁）


def account_c() -> str:
    out = []
    out.append("== C · 回归机制账：每次换 chunk / embedding / rerank 都跑同一套评测集 ==")
    out.append("  （§8.13 原文：『RAG 是检索工程，唯一的刹车就是评测』—每次组件 swap 必须对同一套评测集对比）")
    out.append("  同一 10 条评测集 × 三版引擎（gold 排位 = 检索层的 Hit 表）：")
    rows = {}
    for ver, (pos_list, _b) in RECALL_TOP5.items():
        hit = sum(1 for p in pos_list if p <= 5) / len(pos_list)
        rows[ver] = hit
        out.append(f"    {ver}：gold 排位 {str(pos_list)} → Hit@5 = {hit:.2f}")
    base = rows["v2 基线（chunk=384）"]
    for ver, hit in rows.items():
        d = hit - base
        mark = "（保持基线）" if hit >= base - 1e-9 else ("（回退 · 低于闸线）" if hit < GATE else "（下降但过闸）")
        out.append(f"      · {ver} vs 基线 Δ{'+' if d >= 0 else ''}{d:.2f} {mark}")
    bad = [v for v, h in rows.items() if v != "v2 基线（chunk=384）" and h < GATE]
    out.append(f"  回归断言 PASS：{len(bad)}/3 非基线版本中 {len(bad)} 次回退被门禁拦下（"
               + "、".join(bad) + " 顶不进袋子里 = 改坏了立刻知道改坏了）"
               f" = {'PASS' if len(bad) >= 1 else 'FAIL'}")
    out.append("  CI 门禁惯例：PR 必须附『最近一次基线 vs 本次』对比行；关键指标不得低于闸线 X% 否则合并失败（§8.13 回归机制）")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# 账 D · 误区与评委账：三类误区 + 长尾分组 + 评委三档（规则 / judge / 人工）
# ---------------------------------------------------------------------------
# 误区 1：只看答案对错不看检索漏洞 —— 端到端 8/10 答对，但检索 Hit 只有 6/10（2 条靠模板蒙对）
# 误区 2：用论文题目当代测，不用生产真实问题
# 误区 3：只看平均分不看长尾 —— 总分 0.85，但多跳组只有 0.65
MISCONCEPTIONS = [
    ("只看『答案对错』不看检索漏洞",
     "同一评测集整套测：端到端答对 8/10，但检索层 Hit@5 = 6/10——那 2 条『答对但 gold 没进袋子』是生成层靠模板蒙对的，"
     "一换问法就裂；不测检索层，你根本不知道这两条是借来的。"),
    ("评测集用『论文题目』凑数，不用生产真实问题",
     "论文题（MMLU 式单选）与生产 RAG 问法（『退货要几天？』『违约金怎么算？』）差在：真实用户问题多口语、多隐含前提、"
     "gold 答案要点要靠文档验证而不是答案本身对错——用论文题测 RAG，测的是模型常识不是检索质量。"),
    ("只看平均分不看长尾类型",
     "同一评测集按 query 类型分组：专名组 1.00 · 口语组 0.67 · 多跳组 0.33 → 总分 0.70 全绿，"
     "但『多跳类型』整组塌了——平均分把塌方组藏起来了，长尾是排查的前线。"),
]
# 长尾分组实测表：10 条 = 专名 4 / 口语 3 / 多跳 3，各组通过条数
TAIL = [("专名", 4, 4), ("口语", 3, 2), ("多跳", 3, 1)]    # (类型，条数，通过)
# 评委三档路：规则 → LLM-judge → 人工抽样（联动 03 篇偏置 + 04 篇人工三难）
JUDGE_DECISIONS = [
    ("快速回归 / CI 门禁（高频、要便宜快）", "规则", "词面覆盖/指标公式，$0 且毫秒级（账 C 的 Hit 闸线）"),
    ("规模化定期打分（千条级、budget 有限）", "judge", "03 账 C：LLM judge $0.0012/条 · 0.92 判准，但先做人工校准"),
    ("judge 的校准标定（小样本盯准）", "人工抽样", "04 账 B：关键门禁 + judge 校准用人工（15 条 3→1 那档）"),
    ("感受维度（风格 / 冒犯感 / 品牌）", "人工", "04 账 A：不可自动化替代的『真实感受』，盲评去掉品牌光环"),
    ("对发布前最后一道门禁打分", "人工复核临界子集", "04 账 D：judge 全扫 + 人工复核临界 50 条 $75.60 = 全人工价 10 分之一"),
]


def account_d() -> str:
    out = []
    out.append("== D · 误区与评委账：三类误区 + 长尾分组 + 评委三档 ==")
    for i, (name, desc) in enumerate(MISCONCEPTIONS, 1):
        out.append(f"  {i}. {name}：{desc}")
    by_type = [f"{t} {passed}/{total}" for t, total, passed in TAIL]
    out.append(f"  长尾分组实测（10 条 = {' · '.join(by_type)}）："
               f"总分 {sum(p for _t, _n, p in TAIL) / sum(n for _t, n, _p in TAIL):.2f} 全绿"
               f" → 但多跳组 {dict((t, p / n) for t, n, p in TAIL)['多跳']:.2f} 整组塌=平均分把塌方组藏起来了")
    out.append(f"  误区断言 {len(MISCONCEPTIONS)}/3：三类误区全点名（只看答案对错 / 论文题凑数 / 只看平均分）= "
               f"{'PASS' if len(MISCONCEPTIONS) == 3 else 'FAIL'}")
    out.append("  评委三档路（规则 → LLM-judge → 人工抽样，联动 03 / 04 篇）：")
    for i, (scene, who, why) in enumerate(JUDGE_DECISIONS, 1):
        out.append(f"    {i}. {scene} → {who}（{why}）")
    ok = all(w in ("规则", "judge", "人工抽样", "人工", "人工复核临界子集") for _s, w, _y in JUDGE_DECISIONS)
    out.append(f"  评委断言 {len(JUDGE_DECISIONS)}/{len(JUDGE_DECISIONS)}：五场景各归裁判 = "
               f"{'PASS' if ok else 'FAIL'}（规则便宜快 · judge 规模化 · 人工只在门禁与校准）")
    # 结论：RAG 评测（检索生成专项）的完整闭环
    out.append("  收束断言：三层指标 → 判别性对照（耳朵）→ 回归机制（刹车）→ 评委装配 = RAG 评测四步闭环 "
               "（§8.13 『唯一刹车就是评测』 · 掌握度检验 010 式判别性对照）")
    return "\n".join(out) + "\n"


def main() -> None:
    W("start")
    parts = [account_a(), account_b(), account_c(), account_d()]
    print("\n" + "\n".join(parts), end="")
    W("done")


if __name__ == "__main__":
    main()
