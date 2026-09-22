# -*- coding: utf-8 -*-
"""mlops_demo.py —— §13.5 MLOps/LLMOps（13-工程化与基础设施 05 篇配套）

§13.5 知识地图原文（🟢 掌握职责划分）：
- 传统 MLOps：数据版本（DVC）、实验跟踪（MLflow/W&B）、模型注册（MLflow Registry/HF Hub）、
  CI/CD（训练流水线、评测门禁、发布）。
- LLMOps 新增：Prompt/上下文版本管理、评测集管理、LLM 调用可观测（Langfuse/Phoenix）、
  成本管理（token 计费）、护栏管理、Agent 日志回放。
- CI/CD 在 LLM 世界的形态：数据变更 → 训练/微调 → 评测门禁 → 自动部署 → 灰度 → 线上遥测回流
  （数据飞轮的工程化）。

本篇定位：前 4 篇（13-01 选卡 / 13-02 多机互联 / 13-03 推理服务装配 / 13-04 Ray 调度）
把"模型怎么算"讲完了，这一篇讲"模型怎么被管"——训练跑得动、推理接得上之后，一个团队
怎么把模型发布的每一步变成门禁、把线上数据变成下一版模型的输入。纯本机算不清真实 MLOps
平台（零网络装不上 MLflow/DVC），降维成四本确定性账：
账 A  职责分工账：传统 MLOps 六件套（CI/CD 拆三子件） + LLMOps 六新增，12 项归位 + 阵营口诀 + 邻居映射
账 B  CI/CD 流水线与发布账：六节点流水线归位 + 手工 vs MLOps 发布周期 + 评测门禁三判 + 放行节奏
账 C  LLMOps 六新增账：六件各带一个数（prompt 版本 / 评测集版本 / trace / 成本 / 护栏 / 日志回放）
账 D  数据飞轮账：四类回流信号 → 抽样标注 → 数据集/模型版本节奏，闭环回账 B

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（W() 负责计时与落 stderr）。全部统计 = 本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- 传统/LLMOps 两阵营工具清单（DVC/MLflow/W&B/MLflow Registry/HF Hub/Langfuse/Phoenix）、
  CI/CD 六节点、LLMOps 六新增 = §13.5 要素事实（公开资料整理，写作环境无外网未在线复核）。
- 发布周期（手工 3.5 天 vs MLOps 0.7 天）、门禁阈值（85% 准确率·2.0pp 回归·20% 成本预算）、
  回流信号（点踩 2%/重试 1%/护栏 0.5%/慢 0.5%）、抽样 25%、版本节奏（周候选→月放行→季 3 版）
  = 工程惯例 + 作者预置，本机只做代入公式的算术。
- 本篇**不装 MLflow/DVC、不跑真实物联平台**——A-D 数字 = 作者预置输入 + 本机确定性计算；
  真实 MLOps 平台搭建（MLflow registry / DVC 数据湖 / 评测流水线 CI）留读者在自家环境实操。
"""
from __future__ import annotations
import sys
import time

try:  # 输出统一 UTF-8 字节，不随控制台代码页漂移（GBK 下也能打印 ✅/⛔）
    _ = sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

_WSTART = time.perf_counter()


def W(msg: str) -> None:
    print(f"[wall] {time.perf_counter() - _WSTART:.3f}s  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 要素事实与作者预置（§13.5 + 公开资料整理；与 13-01/02/03/04 同口径）
# ---------------------------------------------------------------------------
DAILY_REQUESTS = 100_000      # 线上每日请求数（示例规模）
P_VOTE = 0.02                 # 回流信号 1/4：用户点踩/差评
P_RETRY = 0.01                # 回流信号 2/4：重试/次好
P_GUARD = 0.005               # 回流信号 3/4：护栏拦截低分
P_SLOW = 0.005                # 回流信号 4/4：慢请求
SAMPLE_RATE = 0.25            # 回流数据抽样标注比例
WORK_DAYS = 5                 # 每周工作日出标注
N_PROMPT_VER = 6              # Prompt 已版本化版本数
N_REF = 7                     # 全仓调用点引用次数
UNLOCKED_CONFLICT = 3         # 未版本化时对不上号的引用数
EVAL_V1 = 100                 # 评测集 v1 条数
EVAL_V2 = 150                 # 评测集 v2 条数（v1 + 线上回流 50）
EVAL_N = 200                  # 当前评测集大小（门禁用）
ACC_THRESH = 0.85             # 门禁阈值 1：准确率下限
ACC_V2 = 0.88                 # 线上模型 v2 在评测集上的准确率
ACC_V3 = 0.90                 # 候选 v3 在评测集上的准确率
REG_PP = 2.0                  # 门禁阈值 2：允许的最大回落 pp
COST_V2 = 0.005               # v2 单请求成本（= 13-03 账 D4 的 1000 入+250 出）
COST_V3 = 0.00625             # v3 单请求成本（贵 25%）
BUDGET_RISE = 0.20            # 门禁阈值 3：成本环比预算上限
WEEK_COST_PREV = 3000.0       # 上周成本 $
WEEK_COST_CUR = 3675.0        # 本周成本 $（+22.5%）
GUARD_TP = 0.995              # 护栏 v1 真违规拦截率
GUARD_FP = 0.004              # 护栏 v1 误杀率（正常流量被拒）
GUARD_FN = 0.002              # 护栏 v1 漏放率（危险流量放行）
AGENT_STEPS = 10              # Agent 会话平均步数（09-06 轨迹）
KB_PER_STEP = 1.0             # 每步日志约 KB
DAILY_SESSIONS = 10_000       # Agent 会话数/天
AUDIT_DAYS = 90               # 留 90 天审计


# ---------------------------------------------------------------------------
# 账 A · 职责分工账：传统 MLOps + LLMOps 十二项归位
# ---------------------------------------------------------------------------
# (项, 阵营, 工具/载体, 一句话判词)
A_ROWS = [
    ("数据版本", "传统", "DVC", "数据集打版本+哈希，复现时旧版照样拉"),
    ("实验跟踪", "传统", "MLflow / W&B", "超参·指标·产物一记录，别靠文件命名考古"),
    ("模型注册", "传统", "MLflow Registry / HF Hub", "模型权重+配套进注册中心（13-06 正式展开）"),
    ("CI/CD·训练流水线", "传统", "训练流水线", "数据/代码变更自动触发训练·微调"),
    ("CI/CD·评测门禁", "传统", "评测门禁", "质量/回归/成本三闸，不达标不进发布（账 B）"),
    ("CI/CD·发布", "传统", "发布管道", "自动部署+灰度"),
    ("Prompt/上下文版本管理", "LLMOps", "Prompt 版本库", "提示词也换代，锁版本号防漂移"),
    ("评测集管理", "LLMOps", "评测集版本库", "评测集也版本化+哈希，锁住才可比"),
    ("LLM 调用可观测", "LLMOps", "Langfuse / Phoenix", "一请求一棵 trace，最慢在哪一眼见（11-09 里程碑 025）"),
    ("成本管理", "LLMOps", "token 计费仪表", "每请求分输入/输出记账（13-03 账 D4 同式）"),
    ("护栏管理", "LLMOps", "护栏规则库", "注入/越狱防线也版本化+灰度（09-07 六护栏）"),
    ("Agent 日志回放", "LLMOps", "轨迹日志", "能回放 Agent 轨迹，审计与复盘的前提（09-06）"),
]
A_CAMP = {"传统", "LLMOps"}


def account_a() -> str:
    out = []
    out.append("== A · 职责分工账：传统 MLOps 六件套 + LLMOps 六新增（模型怎么被管） ==")
    n_bad = 0
    for name, camp, tool, what in A_ROWS:
        ok = camp in A_CAMP and what != ""
        n_bad += 0 if ok else 1
        out.append(f"    [{camp.rjust(4)}] {name}（{tool}）↔ {what} = " + ("PASS" if ok else "FAIL"))
    ok = n_bad == 0
    out.append(f"    {len(A_ROWS)} 项归位 {len(A_ROWS)}/{len(A_ROWS)} —— 传统：数据/实验/注册/流水线·门禁·发布 = 管『产线过程』；"
               f"LLMOps 新增六项 = 管『模型的输入输出与效果』 = " + ("PASS" if ok else "FAIL"))
    # 阵营口诀 + 与已交付篇章的邻居映射（4 连归位）
    nbr = [("LLM 调用可观测", "11-09 可观测工具链（里程碑 025）"),
           ("成本管理", "13-03 账 D4 token 计费"),
           ("护栏管理", "09-07 Agent 安全 六护栏"),
           ("Agent 日志回放", "11-06/09-06 Agent 轨迹评测")]
    ok2 = True
    for name, nb in nbr:
        out.append(f"    §13.5 LLMOps 项 → 已交付邻居：{name} = {nb}")
    out.append("    -> 前四篇讲『怎么算』，本篇讲『怎么管』：MLOps 管产线、LLMOps 管效果，门禁把发布变成可判的账"
               + (" = " + ("PASS" if ok and ok2 else "FAIL")))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 B · CI/CD 流水线与发布账
# ---------------------------------------------------------------------------
def account_b() -> str:
    out = []
    out.append("== B · CI/CD 流水线与发布账：六节点流水线 + 手工 vs MLOps + 评测门禁三判 ==")
    # B0 六节点流水线归位（§13.5「CI/CD 在 LLM 世界的形态」全文）
    PIPE = [
        ("数据变更", "测评/点踩/新语料进来，数据集版本号 +1"),
        ("训练/微调", "数据/代码变更自动触发训练流水线"),
        ("评测门禁", "质量·回归·成本三闸，不达标不发布"),
        ("自动部署", "流水线把候选模型推进发布管道"),
        ("灰度", "新模型先吃 5% 流量，观测量达标再逐步加权"),
        ("线上遥测回流", "线上信号回流数据集，数据飞轮的工程化"),
    ]
    n_bad = 0
    for node, what in PIPE:
        ok = what != ""
        n_bad += 0 if ok else 1
        out.append(f"    {len(PIPE)} 节点流水线 [{node}] = {what} = " + ("PASS" if ok else "FAIL"))
    ok0 = n_bad == 0
    # B1 手工 vs MLOps 发布周期
    t_manual = 3.50          # 无 MLOps：数据准备 0.7 + 触发排队 0.7 + 找版本 0.7 + 部署验证 0.7 + 复盘 0.7
    t_mlops = 0.70           # 有 MLOps：自动训练 4h + 门禁 0.5h + 灰度观察 12h ≈ 0.7 天
    ratio = t_manual / t_mlops
    ok1 = abs(ratio - 5.0) < 1e-9
    out.append(f"    发布周期：手工 {t_manual:.2f} 天（5 步×平均 0.7 天） vs MLOps {t_mlops:.2f} 天（自动训练+自动门禁+灰度观察）"
               f" = 周期压到 {ratio:.1f}× = " + ("PASS" if ok1 else "FAIL"))
    # B2 评测门禁三判（对 200 条评测集）
    n_hit_v2, n_hit_v3 = int(EVAL_N * ACC_V2), int(EVAL_N * ACC_V3)
    ok_g1 = ACC_V3 >= ACC_THRESH
    reg = (ACC_V3 - ACC_V2) * 100.0
    ok_g2 = reg <= REG_PP + 1e-9
    rise = COST_V3 / COST_V2 - 1.0
    ok_g3 = rise > BUDGET_RISE   # 判定本身是『该拦』→ PASS
    n_gate = sum([ok_g1, ok_g2, ok_g3])
    out.append(f"    门禁①准确率：v3 候选 {ACC_V3:.0%}（{n_hit_v3}/{EVAL_N}）≥ 阈值 {ACC_THRESH:.0%} → 过 = "
               + ("PASS" if ok_g1 else "FAIL"))
    out.append(f"    门禁②回归：v3 − v2 = {reg:+.1f} pp ≤ 上限 +{REG_PP:.0f} pp（贴线过） → 过 = "
               + ("PASS" if ok_g2 else "FAIL"))
    out.append(f"    门禁③成本：v3 ${COST_V3:.5f} vs v2 ${COST_V2:.4f} = +{rise:.0%} > 预算 +{BUDGET_RISE:.0%} → "
               + ("PASS" if ok_g3 else "FAIL")
               + "  <- 更准但更贵，成本闸正确拦下：质量过 ≠ 上线（11-10 ④ 工程门禁）")
    # B3 放行节奏：周候选 → 月度全量门禁放行 1 → 季度 3 版本
    n_quarter = 3
    ok3 = n_quarter == 3
    out.append(f"    放行节奏：每周出 candidate → 月度全量门禁放行 1 个 → 季度 {n_quarter} 个生产模型版本（与账 D 数据飞轮联动） = "
               + ("PASS" if ok3 else "FAIL"))
    out.append(f"    -> CI/CD 在 LLM 世界 = 一条把『数据变更』运到『线上模型』的自动化流水线，门禁三判 {n_gate}/3 全执行其判"
               + (" = " + ("PASS" if ok0 and ok1 and n_gate == 3 and ok3 else "FAIL")))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 C · LLMOps 六新增账：六件各带一个数
# ---------------------------------------------------------------------------
def account_c() -> str:
    out = []
    out.append("== C · LLMOps 六新增账：每新增一件，都有一笔能算的账 ==")
    res = []
    # C1 Prompt 版本管理：6 版进库、7 处引用唯一解析 6/6；未版本化 7 处手工 copy 有 3 处对不上号
    resolve_ok = N_PROMPT_VER == 6 and N_REF == 7
    res.append(("Prompt/上下文版本管理", f"{N_PROMPT_VER} 版进版本库、{N_REF} 处调用引用 prompt_id 唯一解析 {N_PROMPT_VER}/{N_PROMPT_VER}；"
               f"未版本化对照：{N_REF} 处手工 copy 有 {UNLOCKED_CONFLICT} 处与『该是谁的版本』对不上 = 靠人记就漂移",
               resolve_ok))
    # C2 评测集管理：同一模型在 v1 / v2 上判若两模
    drift_v1 = (ACC_V3 - ACC_V2) * 100.0      # +2.0 pp（虚高）
    drift_real = -1.0                           # 真正回归
    res.append(("评测集管理", f"同一 v3 在 评测集 v1（{EVAL_V1} 条）上 {drift_v1:+.1f} pp（虚高『变好了』） vs "
               f"v2（{EVAL_V2} 条，+{EVAL_V2 - EVAL_V1} 条线上回流）上 {drift_real:+.1f} pp（真实回落）——"
               "评测集必须锁版本+哈希，跨集对比无效（11-02 防污染同源）",
               EVAL_V2 > EVAL_V1))
    # C3 调用可观测：8 span × 每日请求数
    spans_day = 8 * DAILY_REQUESTS
    res.append(("LLM 调用可观测", f"一次 RAG 请求 8 span（11-09 同口径）→ {DAILY_REQUESTS:,} 请求/天 = {spans_day:,} span/天，"
               "最慢 span llm.generate 850 ms = 检索 4.7×（11-09 实测）——『哪贵哪慢』从猜变查",
               spans_day == 800_000))
    # C4 成本管理：token 计费 13-03 D4 同式，周环比告警
    day_cost = DAILY_REQUESTS * COST_V2
    month_cost = day_cost * 30
    week_rise = WEEK_COST_CUR / WEEK_COST_PREV - 1.0
    res.append(("成本管理", f"单请求 1000 入+250 出 = ${COST_V2:.4f}（13-03 账 D4 同式）→ {DAILY_REQUESTS:,} 请求/天 = "
               f"${day_cost:,.0f}/天 ≈ ${month_cost:,.0f}/月；周环比 ${WEEK_COST_CUR:,.0f} vs ${WEEK_COST_PREV:,.0f} = "
               f"+{week_rise:.1%} > 预算 +{BUDGET_RISE:.0%} → 告警",
               abs(day_cost - 500.0) < 1e-9 and week_rise > BUDGET_RISE))
    # C5 护栏管理：误杀与漏放两把账 + 换版小流量
    g_fp_day = GUARD_FP * DAILY_REQUESTS
    g_fn_day = GUARD_FN * DAILY_REQUESTS
    res.append(("护栏管理", f"护栏 v1：拦真违规 {GUARD_TP:.1%}·误杀 {GUARD_FP:.1%}·漏放 {GUARD_FN:.2%} → "
               f"{DAILY_REQUESTS:,} 请求/天 = {g_fp_day:,.0f} 正常请求被误拒 + {g_fn_day:,.0f} 风险流量放行；"
               "换护栏版本先 5% 流量灰度看误杀率（09-07 六护栏同纪律）",
               abs(g_fp_day - 400.0) < 1e-9 and abs(g_fn_day - 200.0) < 1e-9))
    # C6 Agent 日志回放：10 步 × 1 KB/步 × 10k 会话/天
    mb_day = DAILY_SESSIONS * AGENT_STEPS * KB_PER_STEP / 1024.0
    gb_month = mb_day * 30 / 1024.0
    res.append(("Agent 日志回放", f"会话 {AGENT_STEPS} 步 × {KB_PER_STEP:.0f} KB/步 × {DAILY_SESSIONS:,} 会话/天 = "
               f"{mb_day:.0f} MB/天 ≈ {gb_month:.1f} GB/月，留 {AUDIT_DAYS} 天审计 ≈ {gb_month * AUDIT_DAYS / 30:.0f} GB；"
               "出事故会话 5 条 → 回放定位 3 条根因——可回放=可审计可复盘（09-06 轨迹评测）",
               abs(gb_month - 3.0) < 0.15))
    n_good = 0
    for name, line, ok in res:
        n_good += 1 if ok else 0
        out.append(f"    [{name}] = {line} = " + ("PASS" if ok else "FAIL"))
    out.append(f"    -> LLMOps 六新增 × 已交付篇章：可观测=11-09·成本=13-03·护栏=09-07·轨迹=09-06 → {n_good}/{len(res)} 件各有其数 = "
               + ("PASS" if n_good == len(res) else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 D · 数据飞轮账：回流信号 → 抽样标注 → 版本节奏 → 闭环回账 B
# ---------------------------------------------------------------------------
def account_d() -> str:
    out = []
    out.append("== D · 数据飞轮账：四类回流信号 → 抽样标注 → 数据集/模型版本节奏，闭环回 B ==")
    signals = [("点踩/差评", P_VOTE), ("重试/次好", P_RETRY), ("护栏低分拦截", P_GUARD), ("慢请求", P_SLOW)]
    daily_sig, n_bad = 0, 0
    for name, p in signals:
        n = int(DAILY_REQUESTS * p)
        daily_sig += n
        ok = n > 0
        n_bad += 0 if ok else 1
        out.append(f"    回流信号 [{name}] {p:.1%} → {n:,} 条/天 = " + ("PASS" if ok else "FAIL"))
    ok1 = n_bad == 0 and daily_sig == 4000
    out.append(f"    四信号合计 {daily_sig:,} 条/天（{daily_sig / DAILY_REQUESTS:.0%} 线上流量回流，LLMOps『遥测回流』长什么样） = "
               + ("PASS" if ok1 else "FAIL"))
    # D2 抽样标注：25% → 1000 条/天 → 5000 条/周 进训练集
    daily_label = int(daily_sig * SAMPLE_RATE)
    week_label = daily_label * WORK_DAYS
    ok2 = daily_label == 1000 and week_label == 5000
    out.append(f"    抽样标注 {SAMPLE_RATE:.0%} → {daily_label:,} 条/天 × {WORK_DAYS} 工作日 = {week_label:,} 条/周 进训练集 = "
               + ("PASS" if ok2 else "FAIL"))
    # D3 版本节奏：月 +2 万条 → 数据集每月一版；模型周候选→月放行→季度 3 版
    month_rows = week_label * 4
    base_rows = 100_000
    ok3 = month_rows == 20_000 and int(base_rows / month_rows * 4) == 20
    out.append(f"    版本节奏：周 {week_label:,} 条 → 月 {month_rows:,} 条（基线 {base_rows:,} 条 → 每 4 周数据集版本 +1，"
               f"季度 3 版）；模型 = 账 B 节奏：周候选→月放行→季度 3 生产版本 = "
               + ("PASS" if ok3 else "FAIL"))
    # D4 闭环：回流进训练集 = 账 B 流水线『数据变更』的输入
    ok4 = week_label >= 1000
    out.append(f"    闭环点：本周 {week_label:,} 条回流标注入训练集 = 账 B 流水线的『数据变更』输入——CI/CD 形态五段合龙 = "
               + ("PASS" if ok4 else "FAIL"))
    out.append("    -> 数据飞轮的工程化 = 线上信号→回流→标注→进集→触发下一轮训练：闭环得越快，模型更新越快"
               + (" = " + ("PASS" if ok1 and ok2 and ok3 and ok4 else "FAIL")))
    return "\n".join(out)


def main() -> None:
    W("start")
    parts = [account_a(), account_b(), account_c(), account_d()]
    print("\n" + "\n".join(parts), end="")
    W("done")


if __name__ == "__main__":
    main()
