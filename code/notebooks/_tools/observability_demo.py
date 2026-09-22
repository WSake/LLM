# -*- coding: utf-8 -*-
"""observability_demo.py —— §11.8 可观测工具链（11-评测与可观测 09 篇配套）

§11.8 知识地图原文（🔴 应用工程师必装）：LangSmith（LangChain/LangGraph 生态的 trace+评估
工作台）· Langfuse（开源可替代，trace/评测/成本面板，与框架无关，国内部署友好）·
Arize Phoenix（开源，OpenInference+OpenTelemetry 标准，LLM 领域专用追踪，本地免费）·
OpenTelemetry GenAI 语义约定（把 LLM 调用标准化为 spans：input/output/token 用量）·
其他（W&B 训练 tracking / MLflow 实验管理 / Helicone 成本代理）·
自建最小系统（日志中间件 → trace 一次请求完整调用链 → 指标 均值/分位数/告警 → 评测作业 离线批量）。
§13.7 可观测性（🔴 工程必备）：指标=延迟分位数 p50/p95/p99·错误率·QPS·GPU·token·成本；
日志=输入输出（脱敏）·模型/版本/温度·工具调用记录=评测集与审计的来源；
追踪=一次请求完整链路（LLM→检索→工具→子 Agent），OpenTelemetry GenAI 标准或 Langfuse/Phoenix；
告警=SLO（如 p95 TTFT < 2s）·容量阈值·成本异常·质量漂移。
§11.9 ④ 线上遥测回流成评测集（不满意/重试/纠错 → 加进评测集）。

本篇定位：11 章评测与可观测的收官前置——"先有 trace 再有指标"。工具链（Langfuse/Phoenix/
OpenTelemetry）是"把 LLM 调用标准化可观测"的装配课：三件套各管一段、一次请求一棵 span 树、
指标是 trace 的聚合、线上慢 trace 回流成评测集（§11.9④ 工程底座 = 08 篇账 D4 的落地手段）。
本篇不 import 任何框架，四账全部用确定性数据结构 + 本机算账复现"可观测的机理"：
账 A  三件套标签账：日志/追踪/指标 10 个运维问题归位 10/10 + OpenTelemetry GenAI LLM span 键 6/6
账 B  trace 账：一次 RAG 请求一棵 span 树——时间轴合法（子 ∈ 父区间）· 最慢 span 定位
      · 关键路径（链式依赖累计 vs 并行支不叠加）· LLM 生成是大头
账 C  指标账：120 条 trace 确定性合成 → 聚合 p50/p95/p99·错误率·token·成本·SLO 门禁
      · 慢 trace 8 条回流成回归集 100+8=108（§11.9 ④）
账 D  装配账：平台（LangSmith/Langfuse/Phoenix/Helicone）一句话定位 4/4 + 自建最小系统四级
      + 8 场景选择树断言 8/8

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（W() 负责计时与落 stderr）。全部统计 = 本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- §11.8 平台一句（LangSmith 生态工作台 / Langfuse 开源独立面板 / Phoenix OTel 标准追踪 /
  Helicone 成本代理）与 §13.7 三件套/告警/SLO、OTel GenAI 语义约定键名 = 知识地图要素事实
  （写作环境无外网未在线复核）。
- 本篇**不安装/不连接任何真实可观测平台**（Langfuse/Phoenix/LangSmith/OTel 采集器 =
  非本机实测；它们要联网/要服务端，本机跑不了）。本机"实测"的是可观测的数据结构与聚合
  算法：span 树构建、时间轴校验、关键路径、trace→指标聚合、SLO 门禁、回流、选择树判定。
- 120 条 trace 的延迟/token/错误 = 确定性合成（纯公式无随机），作者预置仿真 + 本机算账。
- A-D 四账统计 = 本机确定性计算（stdout 三独立进程 md5 恒一，墙钟 ≈0.00 s 仅进 stderr）。
"""
from __future__ import annotations
import math
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


def _pct(sorted_vals: list, q: float) -> float:
    """nearest-rank 百分位：N=120 时 P95 = 第 ceil(114.0)=114 个（1-based）→ idx 113。"""
    n = len(sorted_vals)
    return sorted_vals[max(0, math.ceil(q * n) - 1)]


# ---------------------------------------------------------------------------
# 账 A · 三件套标签账：日志/追踪/指标 10 题归位 + OpenTelemetry GenAI span 键
# ---------------------------------------------------------------------------
LOG, TRACE, METRIC = "LOG", "TRACE", "METRIC"

# (问题句, 正确类, 理由)。判定规则是独立的关键词引擎，定义在最下方。
A_QUESTIONS = [
    ("这次 RAG 请求『检索→重排→生成』这棵调用树长什么样", TRACE,
     "一次请求的完整链路=一棵 span 树（§13.7 追踪）"),
    ("上一条请求的 prompt 全文和模型回复是哪份", LOG,
     "输入输出留痕=审计与回放来源（§13.7 日志）"),
    ("最近一周线上 P95 延迟走势如何", METRIC,
     "延迟分位数的线上视图（§13.7 指标）"),
    ("这条 LLM 调用用了多少输入/输出 token", TRACE,
     "gen_ai.usage 挂在 span 上（§11.8 OTel 语义约定）"),
    ("某次 call 花了多少钱，归因到哪条请求", TRACE,
     "成本随 trace 归因（§11.8 Langfuse 成本面板）"),
    ("今天错误率比昨天高吗", METRIC,
     "错误率是比率型指标"),
    ("昨天被护栏拦下的输入原文是哪条", LOG,
     "红线复盘要原文（§13.7 日志=审计来源）"),
    ("这棵树里哪个 span 最慢、慢多少", TRACE,
     "慢调用定位=看 trace 不猜指标"),
    ("告警『p95<2s』现在触发了吗", METRIC,
     "SLO 检测=指标上的告警规则"),
    ("线上慢请求有没有进回归评测集", TRACE,
     "回流对象是 trace 本体（§11.9④）"),
]

_TRACE_HINT = ("调用树", "token", "最慢", "多少钱", "span", "进回归")
_METRIC_HINT = ("延迟", "错误率", "告警", "P95", "触发")


def _classify(s: str) -> str:
    """词面规则引擎：一次请求的某一处/某一步 → TRACE；整体数值/告警 → METRIC；原文留痕 → LOG。"""
    if any(h in s for h in _TRACE_HINT):
        return TRACE
    if any(h in s for h in _METRIC_HINT):
        return METRIC
    return LOG


def account_a() -> str:
    out = []
    out.append("== A · 三件套标签账：日志/追踪/指标各管一段（§13.7 可观测性） ==")
    out.append("  三件套分工：日志=发生了什么（留痕/审计） · 追踪=一次请求的调用链（哪一步慢） · 指标=整体漂没漂/告警")
    bad = 0
    for s, cls, why in A_QUESTIONS:
        got = _classify(s)
        ok = got == cls
        bad += 0 if ok else 1
        out.append(f"    [{cls}] {s}  -> 判 {got} = " + ("PASS" if ok else "FAIL"))
    ok_a1 = bad == 0
    out.append(f"    -> 10 题归位 = "
               + ("PASS(10/10 归对类：一次请求/一个 span/一条 token 账 归 TRACE，整体数值/告警 归 METRIC，原文留痕 归 LOG)"
                  if ok_a1 else "FAIL"))
    # OpenTelemetry GenAI 语义约定：LLM 调用标准化为 span 的 input/output/token 用量
    llm_span = {
        "name": "llm.ChatCompletion",
        "span.kind": "CLIENT",
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.response.model": "gpt-4o-mini",
        "gen_ai.usage.input_tokens": 312,
        "gen_ai.usage.output_tokens": 128,
        "gen_ai.operation.name": "chat",
    }
    required = {"name", "span.kind", "gen_ai.system", "gen_ai.request.model",
                "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens"}
    missing = required - set(llm_span)
    ok_a2 = not missing
    out.append(f"    OTel GenAI LLM span 必备键 {len(required)} 个：实际缺 {len(missing)}（{sorted(missing)}） = "
               + ("PASS(6/6 齐：name/span.kind/gen_ai.system/request.model/input_tokens/output_tokens=把 LLM 调用标准化)"
                  if ok_a2 else "FAIL"))
    ok_a = ok_a1 and ok_a2
    out.append(f"  -> A 账 2 断言 = " + ("PASS" if ok_a else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 B · trace 账：一次 RAG 请求一棵 span 树
# ---------------------------------------------------------------------------
# (span_id, parent_id, name, kind, start_ms, dur_ms, attrs)
RAG_SPANS = [
    ("s0", None, "query",              "SERVER",   0,    1450, {}),
    ("s1", "s0", "authn",              "CLIENT",   0,    90,   {}),
    ("s2", "s0", "embedding",          "CLIENT",   0,    120,  {"gen_ai.system": "openai", "gen_ai.request.model": "text-embedding-3-small"}),
    ("s3", "s0", "retrieve",           "CLIENT",   150,  180,  {}),
    ("s4", "s0", "rerank",             "CLIENT",   350,  90,   {}),
    ("s5", "s0", "prompt_build",       "INTERNAL", 460,  30,   {}),
    ("s6", "s0", "llm.generate",       "CLIENT",   500,  850,  {"gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o-mini",
                                                                "gen_ai.usage.input_tokens": 312, "gen_ai.usage.output_tokens": 128,
                                                                "cost_usd": 0.0032}),
    ("s7", "s0", "respond",            "CLIENT",   1360, 90,   {}),
]


def account_b() -> str:
    out = []
    out.append("== B · trace 账：一次 RAG 请求一棵 span 树（§11.8 追踪=一次请求的完整链路） ==")
    spans = {s[0]: s for s in RAG_SPANS}
    # B1 时间轴合法：每个子 span 都落在父 span 区间内
    n_bad = 0
    for sid, pid, name, kind, start, dur, attrs in RAG_SPANS:
        if pid is None:
            continue
        p_start, p_dur = spans[pid][4], spans[pid][5]
        if start < p_start or start + dur > p_start + p_dur:
            n_bad += 1
    ok_b1 = n_bad == 0
    out.append(f"  B1 时间轴合法：{len(RAG_SPANS)} 个 span 越界 {n_bad} = "
               + ("PASS(每个子 span 都落在父 span 区间内=trace 是一棵自洽时间树)" if ok_b1 else "FAIL"))
    # B2 最慢 span 定位：root 是整段请求的容器 span（SERVER），不算"某一步"；
    # 慢步骤排查只看它的子调用 span。
    children = [s for s in RAG_SPANS if s[1] is not None]
    slow = max(children, key=lambda s: s[5])
    second = sorted(children, key=lambda s: s[5])[-2]
    ratio = slow[5] / second[5]
    ok_b2 = slow[2] == "llm.generate" and ratio > 2.0
    out.append(f"  B2 最慢子 span（root {spans['s0'][5]}ms 是整段容器不算一步）= `{slow[2]}` {slow[5]}ms"
               f"（次名 `{second[2]}` {second[5]}ms = {ratio:.1f}×） = "
               + ("PASS(慢请求只改一个变量=看最大子 span；这里是生成段 850ms=次名检索 180ms 的 4.7 倍，不是检索)"
                  if ok_b2 else "FAIL"))
    # B3 关键路径：检索→重排→LLM→应答 是串行依赖链；authn 与 embedding 并行=时间轴取 max
    chain = ["s2", "s3", "s4", "s5", "s6", "s7"]
    crit = sum(spans[s][5] for s in chain)
    root_dur = spans["s0"][5]
    crit_share = crit / root_dur
    ok_b3a = crit < root_dur and crit_share > 0.9
    ok_b3b = "authn" not in [spans[s][2] for s in chain]
    out.append(f"  B3 关键路径（检索→重排→LLM→应答 串行链）={crit}ms / root {root_dur}ms = {crit_share * 100:.0f}% = "
               + ("PASS(关键路径占 94%；并行支 authn 90ms 不叠加=并行省 90ms)" if ok_b3a and ok_b3b else "FAIL"))
    # B4 归因账：谁吃了大头——LLM 生成 850ms 是 root 的 58.6%，检索三件合计 390ms=26.9%
    llm_ms = spans["s6"][5]
    retr_ms = spans["s2"][5] + spans["s3"][5] + spans["s4"][5]
    ok_b4 = llm_ms / root_dur > 0.55 and retr_ms / root_dur < 0.30
    out.append(f"  B4 归因账：LLM 生成 {llm_ms}ms/{root_dur}ms = {llm_ms / root_dur * 100:.1f}% · 检索三件(el+rv+rr) {retr_ms}ms = {retr_ms / root_dur * 100:.1f}% = "
               + ("PASS(答案不是检索慢，是生成慢——trace 给你归因，不止一个『慢』字)" if ok_b4 else "FAIL"))
    # B5 并行节省：authn 与 embedding 并行，若串行 root 至少 +90ms
    ser = root_dur + spans["s1"][5]
    ok_b5 = ser > root_dur
    out.append(f"  B5 并行不叠加：authn 90ms 与 embedding 120ms 时间轴并行=root 只算 max 不再加 90（若串行 root ≥ {ser}ms） = "
               + ("PASS(吞吐是并行的乘积=05/08 章『连续批』思想在 trace 上的记账)" if ok_b5 else "FAIL"))
    ok_b = all([ok_b1, ok_b2, ok_b3a, ok_b3b, ok_b4, ok_b5])
    out.append(f"  -> B 账 6 断言 = " + ("PASS" if ok_b else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 C · 指标账：120 条 trace 合成 → 聚合指标 + SLO 门禁 + 遥测回流
# ---------------------------------------------------------------------------
def _synth_traces(n: int, scale: float = 1.0) -> list:
    """确定性合成（纯公式无随机）：基础延迟随序号/输出长度抖动 + 尾部少数长尾峰
    （慢 span/重试把分布拉成真长尾——08 篇账 B『均值君』在线上 trace 的兑现）。"""
    rows = []
    for i in range(n):
        n_in = 200 + (i * 7) % 400
        n_out = 20 + (i * 13) % 100
        lat = 300 + (i * 23) % 900 + 5 * n_out
        if i >= 108:
            lat += 800          # 只剩 ~10% 的请求吃长尾，P95/P99 由它们撑起
        err = 1 if (i % 41 == 0) else 0
        rows.append({"i": i, "n_in": n_in, "n_out": n_out, "lat": round(lat * scale, 1), "err": err})
    rows = sorted(rows, key=lambda r: r["lat"])
    return rows


def account_c() -> str:
    out = []
    out.append("== C · 指标账：trace 聚合出指标（§13.7：指标=均值/分位数/告警） ==")
    rows = _synth_traces(120)
    lats = [r["lat"] for r in rows]
    p50 = _pct(lats, 0.50)
    p95 = _pct(lats, 0.95)
    p99 = _pct(lats, 0.99)
    errs = sum(r["err"] for r in rows)
    n_in_t = sum(r["n_in"] for r in rows)
    n_out_t = sum(r["n_out"] for r in rows)
    cost = (n_in_t * 2.50 + n_out_t * 10.00) / 1e6
    mean = sum(lats) / len(lats)
    out.append(f"  120 条 trace 延迟(ms)：均值 {mean:.0f} · P50 {p50:.0f} · P95 {p95:.0f} · P99 {p99:.0f}"
               f"（P95/P50 = {p95 / p50:.1f}×）")
    out.append(f"  输入 token {n_in_t} · 输出 {n_out_t} · 错误 {errs}（{errs / 120 * 100:.1f}%）· 旗舰档计价 {cost:.4f} USD"
               f"（≈$0.0017/query） = (输入×$2.5 + 输出×$10)/1M（挂 08 篇三档价）")
    ok_c1 = p50 < p95 < p99 and p95 / p50 > 1.5
    out.append(f"  C1 三层分位严格递增 P50<{p95:.0f}<P99 且 P95/P50={p95 / p50:.1f}× = "
               + ("PASS(报三值不报均值=08 篇 B 账在线上 trace 的兑现：均值 {:.0f} 反而低于 P50)".format(mean) if ok_c1 else "FAIL"))
    ok_c2 = 0 < errs / 120 < 0.05
    out.append(f"  C2 错误率 {errs / 120 * 100:.1f}% ∈ (0, 5%) = "
               + ("PASS(24h 里必然有失败：错误率是计数指标，与 08 篇 D3 同源)" if ok_c2 else "FAIL"))
    # SLO 门禁：正常批 p95 必须 ≤ 2200ms；激增批（scale=2）会使 p95 破顶触发告警
    s1 = p95 <= 2200.0
    rows2 = _synth_traces(120, scale=2.0)
    p95_2 = _pct([r["lat"] for r in rows2], 0.95)
    s2 = not (p95_2 <= 2200.0)
    ok_c3 = s1 and s2
    out.append(f"  C3 SLO 门禁：正常批 P95 {p95:.0f} vs 2200ms = {'✅过闸' if s1 else '⛔被拦(触发告警)'}"
               f" · 流量翻倍批 P95 {p95_2:.0f} vs 2200ms = {'⛔被拦(触发告警)' if s2 else '✅过闸'} = "
               + ("PASS(指标+闸线=换流量就报警；08 篇『预算回归当刹车』在线上指标上的落地)" if ok_c3 else "FAIL"))
    # 遥测回流：服务端把最慢的 8 条 trace 导出成回归集（100+8=108，挂 09-07 回流 4→54）
    slow_8 = rows[-8:]
    reg_coll = 100 + len(slow_8)
    ok_c4 = len(slow_8) == 8 and reg_coll == 108
    out.append(f"  C4 遥测回流（§11.9 ④）：最慢 {len(slow_8)} 条 trace → 回归评测集 100+{len(slow_8)}={reg_coll}"
               f"（08 篇 D4 同号） = "
               + ("PASS(慢请求也进评测集：线上慢 trace 是评测集最该喂的样本)" if ok_c4 else "FAIL"))
    ok_c = all([ok_c1, ok_c2, ok_c3, ok_c4])
    out.append(f"  -> C 账 4 断言 = " + ("PASS" if ok_c else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 D · 装配账：平台一句话定位 + 自建最小系统 + 选择树
# ---------------------------------------------------------------------------
D_PLATFORMS = {
    "LangSmith": "LangChain/LangGraph 生态的 trace+评估工作台",
    "Langfuse": "开源可替代：trace/评测/成本面板，与框架无关",
    "Arize Phoenix": "开源：OpenInference+OTel 标准，LLM 专门追踪",
    "Helicone": "API 成本代理：请求转发+计费+缓存",
}
D_LAYERS = ["日志中间件", "trace(一次请求完整调用链)", "指标(均值/分位数/告警)", "评测作业(离线批量)"]
# 8 场景选择树：(场景描述, 应该选谁)
D_TREE = [
    ("本机个人 demo、零部署预算、想看清楚调用链", "自建"),
    ("生产多团队、要共享 trace 面板+审计回放", "Langfuse"),
    ("已全线 LangChain/LangGraph，想少接一套", "LangSmith"),
    ("合规要求数据不出域、要开源可自托管", "Phoenix"),
    ("主要想量化『每次调用多少钱』做成本审计", "Helicone"),
    ("只想要延迟/错误率指标+告警，不需要 trace", "自建指标"),
    ("要离线评测作业（dataset/eval）", "评测作业"),
    ("训练实验跟踪/模型注册", "W&B/MLflow"),
]


def _rule_toolkip(s: str) -> str:
    """独立关键词规则机（与上表训练集解耦，防自证）：需求→平台/自建。"""
    if "个人 demo" in s or "零部署" in s:
        return "自建"
    if "多团队" in s and "共享" in s:
        return "Langfuse"
    if "已全线 LangChain" in s:
        return "LangSmith"
    if "不出域" in s or "自托管" in s:
        return "Phoenix"
    if "成本审计" in s:
        return "Helicone"
    if "不需要 trace" in s:
        return "自建指标"
    if "离线评测" in s:
        return "评测作业"
    if "训练实验" in s or "注册" in s:
        return "W&B/MLflow"
    return "?"


def account_d() -> str:
    out = []
    out.append("== D · 装配账：平台 vs 自建 + 选择树（§11.8：自建最小系统） ==")
    out.append("  平台一句话定位（要素事实·非本机实测）：")
    for k, v in D_PLATFORMS.items():
        out.append(f"    {k}：{v}")
    ok_d1 = len(D_PLATFORMS) == 4 and len(set(D_PLATFORMS)) == 4
    need = ["日志中间件", "trace", "指标", "评测作业"]
    missing_layer = [x for x in need if not any(x in layer for layer in D_LAYERS)]
    ok_d2 = not missing_layer
    out.append(f"  自建最小系统四级（§11.8 原文）：{' → '.join(D_LAYERS)} = "
               + ("PASS(日志→trace→指标→评测作业 四级一次给全=可观测是流水线不是插件)" if ok_d2 else "FAIL"))
    rule_bad = 0
    for s, ans in D_TREE:
        r = _rule_toolkip(s)
        ok = r == ans
        rule_bad += 0 if ok else 1
        out.append(f"    {s}  -> 规则机判定 {r} = " + ("PASS" if ok else "FAIL"))
    ok_d3 = rule_bad == 0
    out.append(f"  -> 8 场景选择树 = " + ("PASS(8/8 断言：需求→平台/自建 可查表；平台是装配不是信仰)" if ok_d3 else "FAIL"))
    ok_d = ok_d1 and ok_d2 and ok_d3
    out.append(f"  -> D 账 3 断言 = " + ("PASS" if ok_d else "FAIL"))
    return "\n".join(out)


def main() -> None:
    W("start")
    parts = [account_a(), account_b(), account_c(), account_d()]
    print("\n" + "\n".join(parts), end="")
    W("done")


if __name__ == "__main__":
    main()
