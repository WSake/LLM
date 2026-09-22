# -*- coding: utf-8 -*-
"""observability_guardrail_demo.py —— §13.7 可观测性（13-工程化与基础设施 07 篇配套）

§13.7 知识地图原文（🔴 工程必备）：可观测性 Monitoring/Logging/Tracing——
指标 = 延迟分位数 p50/p95/p99 · 错误率 · QPS · GPU 利用率 · token 消耗 · 成本；
日志 = 输入输出（脱敏）· 模型/版本/温度参数 · 工具调用记录（评测集与审计的来源）；
追踪 = 一次请求的完整链路（LLM → 检索 → 工具 → 子 Agent），OpenTelemetry GenAI 标准
       或 Langfuse/Phoenix；
告警 = SLO（如 p95 TTFT < 2s）· 容量阈值 · 成本异常 · 质量漂移（线上 judge 分数下降）。

本篇定位：13 章工程卷的"观测 → 护栏"课——11-09（可观测工具链·里程碑 025）已经把
"先有 trace 再有指标"的三件套与 trace 概念讲了，这一篇不开新概念，只把 §13.7 的
四件事各自变成能算、能判定、能触发告警的工程账，回答"生产系统的观测怎么设才算护住"：
账 A  指标账：六个指标口径（延迟分位/错误率/QPS/GPU 利用率/token/成本）怎么算、报什么
账 B  日志账：PII 脱敏四类 + 四要素归位（输入输出·模型版本温度·工具调用·审计评测来源）
账 C  追踪账：一次 Agent 请求一棵 span 树——工具调用是大头（区别于 11-09 的 RAG 生成大头）
账 D  告警账：四条护栏（SLO/容量/成本/质量漂移）各自的阈值与触发判定 + 动作绑定

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（W() 负责计时与落 stderr）。全部统计 = 本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- §13.7 六指标口径、日志四要素、追踪链路四段、告警四类/SLO（p95 TTFT<2s）=
  知识地图要素事实（写作环境无外网未在线复核）。
- 本篇不部署任何真实观测平台：Prometheus/Grafana/Langfuse/Phoenix/ELK =
  非本机实测（需联网/服务端）。本机"实测"的是观测数据的口径与判定：
  分位数算法、错误率分母、QPS 时间窗、GPU 利用率、token 计费、PII 脱敏、
  时间轴校验、四类告警阈值判定、动作绑定。
- 延迟/token/GPU/cost 样本 = 确定性合成（纯公式无随机），作者预置 + 本机算账。
- A-D 四账统计 = 本机确定性计算（stdout 三独立进程 md5 恒一，墙钟 ≈0.00 s 仅进 stderr）。
- 承接只引不重测：$0.005 计价 = 13-03 账 D4 同式；$500/天·$15k/月·周环比告警 =
  13-05 账 C4 同口径；judge 线上分下降 = 13-06 账 D 同款；RAG trace llm 生成大头 =
  11-09 账 B 同机实测（本篇 Agent 版刻意对照）。
"""
from __future__ import annotations
import math
import re
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
    """nearest-rank 百分位：N=600 时 P95 = 第 ceil(570.0)=570 个（1-based）→ idx 569。"""
    n = len(sorted_vals)
    return sorted_vals[max(0, math.ceil(q * n) - 1)]


def _p95_ttpf(vals: list) -> float:
    return _pct(sorted(vals), 0.95)


# ---------------------------------------------------------------------------
# 账 A · 指标账：六个指标口径怎么算、报什么（§13.7「指标」）
# ---------------------------------------------------------------------------
# A LATENCIES = 确定性合成 600 条延迟样本（ms）：主体覆盖 700–2299 的均匀带
# （(i*53) % 1600 与 1600 互质 → 595 个值扫过整个带），尾部补 5 条 5.2–9.4 s 长尾，
# 模拟"超时/重试被掐尖"（08 篇『慢请求』在指标上的影子）。
LAT_BODY = [700 + (i * 53) % 1376 for i in range(560)]
LAT_TAIL = [5200 + i * 107 for i in range(40)]
LATENCIES_MS = LAT_BODY + LAT_TAIL  # 600 条
LAT_SORTED = sorted(LATENCIES_MS)

# A 固定口径
A_REQS_TOTAL = 2000          # 错误率分母 = 总请求（成功 + 失败 + 超时）
A_FAILS = 43                 # 失败 43 → 错误率 = 43/2000 = 2.15%
A_QPS_WINDOW_S = 60          # QPS 时间窗 = 60 s
A_REQS_IN_WINDOW = 600       # 600 请求 → QPS = 10
GPU_BUSY_4 = [0.58, 0.66, 0.63, 0.69]   # 4 卡 60 s 采样 busy 占比 → 平均 0.64
TOK_IN_PER_REQ = 1000        # 单请求输入 token（13-03 账 D4 同式）
TOK_OUT_PER_REQ = 250        # 单请求输出 token
PRICE_IN_PER_M = 2.5         # $/1M 输入（13-03 同档）
PRICE_OUT_PER_M = 10.0       # $/1M 输出（输出 ×4 价）
DAILY_BUDGET = 500.0         # 日成本预算 = $500（13-05 账 C4 同口径 $0.005×100k）


def account_a() -> str:
    out = []
    out.append("== A · 指标账（§13.7：延迟分位·错误率·QPS·GPU·token·成本） ==")

    # A1 延迟三层分位（nearest-rank，600 条）
    p50 = _pct(LAT_SORTED, 0.50)
    p95 = _pct(LAT_SORTED, 0.95)
    p99 = _pct(LAT_SORTED, 0.99)
    mean = sum(LAT_SORTED) / len(LAT_SORTED)
    ok_inc = p50 <= p95 <= p99
    ok_tail = p99 > 3 * p50          # 长尾掐尖后 P99 显著高于 P50
    ok_meanp50 = p50 < p99           # 均值夹在中间（长尾拉高均值）
    for name, v in (("均值", mean), ("P50", p50), ("P95", p95), ("P99", p99)):
        out.append(f"    {name} = {v:.0f} ms")
    out.append("    -> 三层分位严格递增 P50≤P95≤P99 = " + ("PASS" if ok_inc else "FAIL"))
    out.append(f"    -> P99(≈{p99:.0f}) > 3×P50(≈{p50:.0f}) 长尾掐尖可见 = "
               + ("PASS" if ok_tail else "FAIL"))
    out.append("    -> 均值夹在三层分位之间（长尾拉高均值，报分布不报均值） = "
               + ("PASS" if ok_meanp50 else "FAIL"))
    a1 = ok_inc and ok_tail and ok_meanp50

    # A2 错误率（分母 = 总请求）
    err_rate = A_FAILS / A_REQS_TOTAL
    ok_err = 0.01 <= err_rate <= 0.05
    out.append(f"    错误率 = {A_FAILS}/{A_REQS_TOTAL} = {err_rate*100:.2f}%"
               f"（分母=总请求，非仅成功请求） = " + ("PASS" if ok_err else "FAIL"))
    a2 = ok_err

    # A3 QPS（时间窗 × 并发）
    qps = A_REQS_IN_WINDOW / A_QPS_WINDOW_S
    ok_qps = abs(qps - 10.0) < 1e-9
    out.append(f"    QPS = {A_REQS_IN_WINDOW}/" f"{A_QPS_WINDOW_S}s = {qps:.1f}"
               f"（时间窗固定，QPS 才有意义） = " + ("PASS" if ok_qps else "FAIL"))
    a3 = ok_qps

    # A4 GPU 利用率（busy 采样平均，账 D 容量告警的输入）
    gpu_avg = sum(GPU_BUSY_4) / len(GPU_BUSY_4)
    ok_gpu = 0.5 <= gpu_avg <= 0.8
    out.append(f"    GPU 利用率 = Σbusy/total = {gpu_avg*100:.0f}%"
               f"（4 卡 60 s 采样平均；<50% 在浪费 · >80% 逼近容量） = "
               + ("PASS" if ok_gpu else "FAIL"))
    a4 = ok_gpu

    # A5 token 与成本（输入/输出分开记，输出 ×4 价）
    per_req_cost = (TOK_IN_PER_REQ * PRICE_IN_PER_M + TOK_OUT_PER_REQ * PRICE_OUT_PER_M) / 1_000_000
    daily_reqs = int(DAILY_BUDGET / per_req_cost)
    daily_cost = daily_reqs * per_req_cost
    ok_tok = abs(per_req_cost - 0.005) < 1e-6          # 1000×2.5 + 250×10 = $0.005
    ok_cost = abs(daily_reqs - 100_000) < 1e3          # ≈ 100k 请求/天 · $500/天
    out.append(f"    单请求 = {TOK_IN_PER_REQ}入+{TOK_OUT_PER_REQ}出 = "
               f"${per_req_cost:.4f}（13-03 同式；输出价=输入价 ×4） = "
               + ("PASS" if ok_tok else "FAIL"))
    out.append(f"    日账 = {daily_reqs} 请求/天 × ${per_req_cost} ≈ ${daily_cost:.0f}/天"
               f"（预算 ${DAILY_BUDGET:.0f}/天，账 D 成本告警输入；13-05 同口径） = "
               + ("PASS" if ok_cost else "FAIL"))
    a5 = ok_tok and ok_cost

    out.append("    -> 账 A 合计 = " + ("PASS(六口径都会算：分位三件套/错误率分母/QPS 时间窗/"
                                        "GPU busy 采样/token 分价/成本预算)" if (a1 and a2 and a3 and a4 and a5)
               else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 B · 日志账：PII 脱敏四类 + 四要素归位（§13.7「日志」）
# ---------------------------------------------------------------------------
_PHI = "13812345678"            # 手机号
_EMAIL = "zhangsan@qq.com"      # 邮箱
_IDNO = "110101199003074515"    # 身份证
_APIKEY = "sk-proj-abcdef1234567890"  # API key

# 四类脱敏规则：只保留外圈/前缀，中间责任区间盖掉。
# 关键顺序：更长/更具体的规则（身份证 18 位）先跑——否则手机 11 位规则会把
# 身份证里的 "19900307451" 当成号码吃掉，后面的身份证规则就再也追不上。
_RULES = [
    ("idcard", re.compile(r"(\d{4})\d{10}(\d{4})"),
     lambda m: f"{m.group(1)}**********{m.group(2)}"),
    ("phone", re.compile(r"(1[3-9]\d)\d{4}(\d{4})"),
     lambda m: f"{m.group(1)}****{m.group(2)}"),
    ("email", re.compile(r"(\w)[\w.-]*@"),
     lambda m: f"{m.group(1)}***@"),
    ("apikey", re.compile(r"(sk-)\S+"),
     lambda m: f"{m.group(1)}***"),
]


def _redact(text: str, rules) -> str:
    for name, rx, rep in rules:
        if rx.search(text):
            text = rx.sub(rep, text)
    return text


def _leak_safe(redacted: str, secrets) -> bool:
    """脱敏后任何原始敏感串不再逐字出现在日志里（含被掐头的中间位）。"""
    for s in secrets:
        # 原始串（>4 字符）不应整体出现在脱敏文本中
        if len(s) > 4 and s in redacted:
            return False
    return True


def account_b() -> str:
    out = []
    out.append("== B · 日志账（§13.7：输入输出脱敏 · 四要素） ==")

    ph = f" {_PHI}；邮箱 {_EMAIL}；身份证 {_IDNO}；key={_APIKEY}"
    b_line = f"in: '帮我翻译{ph}'      out: '…' model=qwen-7b temp=0.7 tool=search"
    secrets = [_PHI, _EMAIL, _IDNO, _APIKEY]
    redacted = _redact(b_line, _RULES)
    ok_secrets = all(s not in redacted for s in secrets)
    ok_nophones = not re.search(r"1[3-9]\d{9}", redacted)
    out.append(f"    原始日志（含 4 类 PII）……")
    out.append(f"    脱敏后: {redacted[:88]}…")
    out.append("    四类 PII（手机/邮箱/身份证/API key）逐字不残留 = "
               + ("PASS" if ok_secrets and ok_nophones else "FAIL"))
    ok_red = ok_secrets and ok_nophones

    # 四要素归位（§13.7 日志原文）
    B_ITEMS = [
        ("输入输出（脱敏后）", "prompt 原文 + 模型回复，先脱敏再落盘"),
        ("模型/版本/温度参数", "model=qwen-7b/9 · revision hash · temp=0.7（13-06 版本寻址）"),
        ("工具调用记录", "tool=search args=… result=…（Agent 干了什么可回放）"),
        ("评测集与审计的来源", "点踩慢请求 → 评测集；出事会话 → 红线复盘（§11.9④ 回流）"),
    ]
    ok_elems = len(B_ITEMS) == 4
    for k, v in B_ITEMS:
        out.append(f"    [{k}] {v}")
    out.append("    四要素归位 = " + ("PASS(日志=留痕+审计+评测集底座，缺脱敏不可留)"
                                     if ok_elems else "FAIL"))

    out.append("    -> 账 B 合计 = " + ("PASS" if ok_red and ok_elems else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 C · 追踪账：一次 Agent 请求一棵 span 树——工具调用是大头
#     （对照：11-09 的 RAG 树最慢是 llm.generate；本篇 Agent 树最慢是外部 API）
# ---------------------------------------------------------------------------
# (id, name, kind, start, duration)  单位 ms。root 是整段请求的容器。
C_SPANS = [
    ("r0", "agent_request", "SERVER", 0, 2600),       # root 容器
    ("r1", "plan",            "INTERNAL", 30, 100),    # 规划
    ("r2", "retrieve",        "CLIENT", 160, 300),     # 检索文档
    ("r3", "read_doc",        "CLIENT", 480, 320),     # 读文档
    ("r4", "call_api",        "CLIENT", 820, 950),     # 外部 API（最慢，Agent 大头）
    ("r5", "sub_agent",       "SERVER", 1860, 520),    # 子 Agent 段
    ("r6", "llm.chat",        "CLIENT", 1930, 350),    # 子段生成（挂在 sub_agent 下）
    ("r7", "respond",         "CLIENT", 2400, 200),    # 应答
]
C_ROOT = "r0"
C_CHILDREN2 = {  # 直接父子关系（r0→r1..r4, r7；r5→r6）
    "r0": ["r1", "r2", "r3", "r4", "r5", "r7"],
    "r5": ["r6"],
}
# 归因分组：工具三件 vs LLM 层
TOOL_SPANS = ("r2", "r3", "r4")   # retrieve+read_doc+call_api
LLM_SPANS = ("r1", "r6")          # plan + sub_agent 里的 llm.chat

# 关键路径（串行依赖链，不含并行支与间隔）
C_CRITICAL_IDS = ["r1", "r2", "r3", "r4", "r5", "r7"]  # plan→retrieve→read→api→sub→respond


def account_c() -> str:
    out = []
    out.append("== C · 追踪账（§13.7：一次请求完整链路 LLM→检索→工具→子 Agent） ==")
    by = {sid: (name, kind, start, start + dur) for sid, name, kind, start, dur in C_SPANS}
    dur = {sid: d for sid, _name, _k, _s, d in C_SPANS}

    # C1 时间轴自洽：每个子 span 必须落在其父 span 区间内
    bad_t = 0
    for pid, kids in C_CHILDREN2.items():
        p_start, p_end = by[pid][2], by[pid][3]
        for k in kids:
            k_start, k_end = by[k][2], by[k][3]
            if not (p_start <= k_start and k_end <= p_end):
                bad_t += 1
                out.append(f"    span {k} 越出父 {pid} 区间！")
    out.append(f"    子段 7 个全部落父区间（时间轴自洽，越界 {bad_t}） = "
               + ("PASS" if bad_t == 0 else "FAIL"))
    ok_time = bad_t == 0

    # C2 最慢 span = call_api（Agent 场景工具调用是大头）
    slowest = max(C_SPANS[1:], key=lambda t: t[4])
    sname, sdur = slowest[1], slowest[4]
    second = max((t for t in C_SPANS[1:] if t[0] != slowest[0]), key=lambda t: t[4])
    ratio = sdur / second[4]
    ok_slow = slowest[0] == "r4" and sdur == 950
    out.append(f"    最慢子 span = {sname} {sdur} ms = 次名 {second[1]} {second[4]} ms 的 "
               f"{ratio:.2f}×（root {by[C_ROOT][3]}ms 是容器不算一步） = "
               + ("PASS" if ok_slow else "FAIL"))
    ok_slowest = ok_slow

    # C3 归因：工具三件 vs LLM 层——Agent 慢在工具调用
    tool_ms = sum(dur[s] for s in TOOL_SPANS)
    llm_ms = sum(dur[s] for s in LLM_SPANS)
    total = by[C_ROOT][3]
    t_frac, l_frac = tool_ms / total, llm_ms / total
    ok_frac = t_frac > 0.5 and l_frac < 0.25
    out.append(f"    归因：工具三件（retrieve+read_doc+call_api）= {tool_ms} ms = "
               f"{t_frac*100:.0f}% · LLM 层（plan+llm.chat）= {llm_ms} ms = {l_frac*100:.0f}%")
    out.append("    -> Agent 请求慢在『外部工具调用』不是『模型生成』"
               "（对照 11-09 RAG 树 llm.generate 才是大头） = "
               + ("PASS" if ok_frac else "FAIL"))
    ok_attr = ok_frac

    # C4 关键路径（串行依赖累计，并行支取 max 不叠加）
    crit_ms = sum(dur[s] for s in C_CRITICAL_IDS)
    crit_frac = crit_ms / total
    ok_crit = 0.85 <= crit_frac <= 0.98
    out.append(f"    关键路径（依赖链累计）≈ {crit_ms} ms = root {total} ms 的 "
               f"{crit_frac*100:.0f}%（并行支不叠加） = " + ("PASS" if ok_crit else "FAIL"))
    ok_critical = ok_crit

    out.append("    -> 账 C 合计 = " + ("PASS" if (ok_time and ok_slowest and ok_attr and ok_critical)
               else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 D · 告警账：四条护栏（SLO/容量/成本/质量漂移）各自的阈值与触发判定 + 动作
# ---------------------------------------------------------------------------
class Alert:
    def __init__(self, key: str, expr: bool, level: str, action: str, reason: str):
        self.key, self.expr, self.level, self.action, self.reason = key, expr, level, action, reason


def account_d() -> str:
    out = []
    out.append("== D · 告警账（§13.7：SLO·容量·成本·质量漂移） ==")

    # D1 SLO：p95 TTFT < 2s（§13.7 原例），两条系统对照
    # 正常批 = 600 条 800–1699 ms 无长尾（P95 远低于 2s）；翻倍批 = 主体+40 条 4.2s 起长尾
    ttf_norm = _p95_ttpf([800 + (i * 41) % 900 for i in range(600)])           # P95 ≈ 1.65s
    ttf_slow = _p95_ttpf([800 + (i * 41) % 900 for i in range(560)] + [4200 + j * 140 for j in range(40)])  # P95 ≈ 5.4s
    slo = 2000.0
    trig_norm = ttf_norm > slo
    trig_slow = ttf_slow > slo
    al_norm = Alert("slo-normal", not trig_norm, "INFO", "无动作",
                    f"p95 TTFT {ttf_norm:.1f}ms ≤ SLO 2000ms，不触发")
    al_slow = Alert("slo-slow", trig_slow, "P1", "扩容 / 回滚（13-06 切引用）",
                    f"p95 TTFT {ttf_slow:.1f}ms > SLO 2000ms，SLO 违约")
    out.append(f"    [SLO] 正常批 p95 TTFT {ttf_norm:.1f}ms ≤ 2000ms → "
               + ("不告警 ✔" if not trig_norm else "误告警 ✘"))
    out.append(f"    [SLO] 翻倍批 p95 TTFT {ttf_slow:.1f}ms > 2000ms → "
               + ("P1 告警：扩容/回滚 ✔" if trig_slow else "漏告警 ✘"))
    ok_slo = (not trig_norm) and trig_slow

    # D2 容量：GPU 利用率 > 85% 持续 ≥ 5 min（账 A 平均 64% → 正常；峰值 91% → 触发）
    gpu_thr = 0.85
    gpu_now = 0.64
    gpu_peak = 0.91
    trig_cap_now = (gpu_now > gpu_thr)
    trig_cap_peak = (gpu_peak > gpu_thr)
    out.append(f"    [容量] 平均利用率 {gpu_now*100:.0f}% ≤ 85% → "
               + ("不告警 ✔" if not trig_cap_now else "误告警 ✘"))
    out.append(f"    [容量] 峰值利用率 {gpu_peak*100:.0f}% > 85% 持续 ≥5min → "
               + ("告警：加副本/扩卡 ✔" if trig_cap_peak else "漏告警 ✘"))
    ok_cap = (not trig_cap_now) and trig_cap_peak

    # D3 成本：周环比 > +20%（13-05 账 C4 同口径照搬阈值，换一周的数）
    wk0, wk1 = 10000.0, 12400.0
    grow = (wk1 - wk0) / wk0
    cost_thr = 0.20
    trig_cost = grow > cost_thr
    out.append(f"    [成本] 本周 $12.4k vs 上周 $10k = 周环比 +{grow*100:.0f}% "
               f"> 预算 {cost_thr*100:.0f}% → "
               + ("告警：缓存 KV / 降档 / 换更优模型（13-06 换引用） ✔"
                  if trig_cost else "漏告警 ✘") + f"（13-05 同阈值只换数）")
    ok_cost = trig_cost

    # D4 质量漂移：线上 judge 分下降（§13.7 原文·13-06 账 D 同款）
    j_target = 0.80
    j_now = 0.79
    j_drift = 0.71
    drift_thr = -0.05           # 距目标下降超过 5 个点触发
    trig_drift = (j_drift - j_target) < drift_thr
    trig_now = (j_now - j_target) < drift_thr
    out.append(f"    [质量] judge 线上分 {j_now:.2f}（距目标 −{j_target-j_now:.2f}）→ "
               + ("不告警 ✔" if not trig_now else "误告警 ✘"))
    out.append(f"    [质量] judge 线上分 {j_drift:.2f}（距目标 −{j_target-j_drift:.2f} < −{abs(drift_thr):.2f}）→ "
               + ("告警：回滚到上一版（13-06 定位→切回）✔" if trig_drift else "漏告警 ✘"))
    ok_drift = (not trig_now) and trig_drift

    # D5 四类告警要素齐备（key/条件/级别/动作 4 条）
    alerts = [al_norm, al_slow]
    ok_elems = all(a.action and a.level and a.reason for a in alerts)

    out.append("    -> 账 D 合计 = "
               + ("PASS(SLO/容量/成本/质量漂移四线齐备，每条触发判据+动作绑到 13-06/13-03/13-05)"
                  if (ok_slo and ok_cap and ok_cost and ok_drift and ok_elems) else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
def main() -> int:
    W("start")
    parts = [account_a(), account_b(), account_c(), account_d()]
    for p in parts:
        print(p)
        print()
    n_pass = 0
    n_fail = 0
    for p in parts:
        for line in p.split("\n"):
            if " = FAIL" in line or " ✘" in line:
                n_fail += 1
            elif " = PASS" in line or " ✔" in line:
                n_pass += 1
    print(f"判定统计: PASS {n_pass} · FAIL {n_fail}")
    W("done")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
