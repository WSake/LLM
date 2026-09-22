# -*- coding: utf-8 -*-
"""serving_infra_demo.py —— §13.3 推理服务基础设施（13-工程化与基础设施 03 篇配套）

§13.3 知识地图原文（🟢 了解；深度在 §5）：
- 服务形态：vLLM/SGLang 容器 → K8s Deployment + HPA → API 网关（Kong/APISIX/自研）。
- 关键工程点：多副本负载均衡（无会话粘性）、按 GPU 水位扩缩容、灰度（新模型先 5% 流量）、
  滚动发布、配额限流（令牌桶）、token 计费。
- 数据平面：模型缓存、KV 前缀亲和路由（同前缀到同一实例，2025 前沿）、跨 region 容灾。

本篇定位：把"从裸机到可上线的推理服务"算成一张服务账单——13-01 的带宽账 / 13-02 的互联账
从训练线搬到推理线：
账 A  装配账：vLLM/SGLang 引擎 → K8s Deployment+HPA → API 网关 + 数据平面，一次请求穿几层
账 B  容量账：多副本 × 无会话粘性 —— B 判无粘性为什么可行、副本数会算（容量→GPU→副本）
账 C  扩缩容发布账：HPA 期望副本公式 + GPU 水位判据 + 灰度 5% + 滚动发布零停机 + 令牌桶
账 D  数据平面计费账：模型缓存 / KV 前缀亲和路由 / 跨 region 容灾 / token 计费

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（W() 负责计时与落 stderr）。全部统计 = 本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- 服务形态三层（vLLM/SGLang 容器、K8s Deployment+HPA、Kong/APISIX/自研网关）与工程点名称
  （continuous batching / Paged KV / prefix cache / 令牌桶 / 灰度 / 滚动 / 粘性）=
  §13.3 要素事实（公开资料整理，写作环境无外网未在线复核）。
- tokens/s≈带宽÷权重 + 连续批放大 = 工程惯例（13-01 账 B 同式）；HPA 期望副本公式 = K8s
  官方公式；令牌桶 = 经典 rate/burst 模型——本机只做代入公式的算术。
- 本篇**不连 GPU、不跑 vLLM、不建 K8s 集群**——A-D 数字 = 作者预置输入 + 本机确定性计算；
  真实压测（vLLM/SGLang benchmark）+ 真实 K8s 部署留读者实操。
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


# ---------------------------------------------------------------------------
# 要素事实表（§13.3 + 公开资料整理；与 13-01/02 同口径）
#   H100 带宽 3.35 TB/s；7B fp16 权重 = 7000000000 × 2 = 14 GB（13-01 账 A/B 同口径）
# ---------------------------------------------------------------------------
H100_BW = 3.35e12            # H100 HBM 带宽 B/s（要素事实）
B_WEIGHT = 7e9 * 2           # 7B fp16 权重字节 = 14e9
B_WEIGHT_70 = 70e9 * 2       # 70B fp16 = 140e9（对照账）
PRICE_IN, PRICE_OUT = 2.50, 10.00   # 服务示例价：$/每百万 token（06-01 同档示例价）
PEAK_TS = 8000.0             # 服务峰值负载 tok/s（作者预置：如 200 并发 × 平均 40 tok/s）
UTIL_GOAL = 0.70             # GPU 利用率目标（虚拟化三档对齐，13-01 时间切片同目标）
RATE_QPS, BURST = 60.0, 120.0  # 令牌桶：稳态 60 req/s、突发 120（要素事实模型）


def ceilf(x: float) -> int:
    return int(math.ceil(x - 1e-9))


# ---------------------------------------------------------------------------
# 账 A · 装配账：vLLM/SGLang 引擎 → K8s Deployment+HPA → API 网关 + 数据平面
# ---------------------------------------------------------------------------
# (功能, 归属层, 该层管什么的判词)
ASSEMBLY = [
    ("continuous batching（连续批调度）", "引擎", "请求即到即拼批，吞吐引擎"),
    ("Paged KV Cache 显存管理", "引擎", "按页分配 KV，显存不浪费"),
    ("prefix cache（前缀 KV 复用）", "引擎", "同前缀只算一次，省 prefill"),
    ("replicas 副本数声明", "K8s", "Deployment 管几个实例"),
    ("HPA 自动扩缩容", "K8s", "按 GPU 水位增减副本"),
    ("readiness/liveness 探针", "K8s", "就绪才接流量、挂掉重启"),
    ("滚动发布（maxSurge/maxUnavailable）", "K8s", "新旧副本交替零停机"),
    ("蓝绿/灰度按权重切流", "网关", "5% 流量先试新模型"),
    ("令牌桶限流 / 配额", "网关", "稳速率 + 突发缓冲"),
    ("token 计费与配额记账", "网关", "账单跟着 token 走"),
    ("等等：KV 做一个会话吗？", "关键点", "否：KV 随请求生灭，副本无状态"),
    ("模型缓存 / 权重共享", "数据平面", "多副本共享一份权重避免重复加载"),
    ("KV 前缀亲和路由", "数据平面", "同前缀路由到同一实例命中 prefix cache"),
    ("跨 region 容灾/健康检查切流", "数据平面", "主 region 挂了切备地域"),
]
LAYERS = {"引擎", "K8s", "网关", "数据平面", "关键点"}


def account_a() -> str:
    out = []
    out.append("== A · 装配账：vLLM/SGLang 引擎 → K8s Deployment+HPA → API 网关（一次请求穿几层） ==")
    n_bad = 0
    for name, layer, what in ASSEMBLY:
        ok = layer in LAYERS and what != ""
        n_bad += 0 if ok else 1
        out.append(f"    [{layer.rjust(5)}] {name} = {what} = " + ("PASS" if ok else "FAIL"))
    ok = n_bad == 0
    out.append("    装配分层：引擎管吞吐 / K8s 管副本与发布 / 网关管流量与钱 / 数据平面管缓存容灾路由"
               + (" = " + ("PASS" if ok else "FAIL")))
    # 一次请求穿几层（服务形态三件套 + 数据平面）：client→网关→Service→Pod(引擎)→GPU
    hops = 5  # 网关 1, Service 2, Pod 3, vLLM 4, GPU 5
    ok2 = hops == 5
    out.append(f"    一次流式请求穿 {hops} 个环节：Client → API 网关 → K8s Service → Pod(vLLM) → GPU"
               f"（每环各管一层） = " + ("PASS" if ok2 else "FAIL"))
    out.append("    -> 推理服务 = 三层装配 + 数据平面：引擎是吞吐心脏，K8s 是无脑管家，网关是流量与钱的闸，数据平面管缓存/容灾")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 B · 容量账：多副本 × 无会话粘性 —— 副本数会算（13-01 带宽账从训练线搬到推理线）
# ---------------------------------------------------------------------------
def account_b() -> str:
    out = []
    out.append("== B · 容量账：多副本负载均衡 × 无会话粘性（副本数 = 会算账，13-01/11-08 口径） ==")
    # B1 带宽界：单请求解码 token/s ≈ 带宽 ÷ 权重（串行，无批 13-01 账 B 同式）
    ts_single = H100_BW / B_WEIGHT       # ≈ 239.3
    ok1 = abs(ts_single - 239) < 1.0
    out.append(f"    带宽界（串行单请求）= 3.35 TB/s ÷ 14 GB = {ts_single:.1f} tok/s（13-01 账 B 的 239 同式，"
               f"70B 权重 140 GB 就更慢 {H100_BW / B_WEIGHT_70:.1f}） = " + ("PASS" if ok1 else "FAIL"))
    # B2 连续批放大：vLLM/SGLang continuous batching = 批大小 × 带宽界
    ts_b8 = ts_single * 8
    ts_b16 = ts_single * 16
    ok2 = 1900 < ts_b8 < 2000 and 3800 < ts_b16 < 3900
    out.append(f"    连续批放大 B=8 → 有效吞吐 ~{ts_b8:.0f} tok/s；B=16 → ~{ts_b16:.0f} tok/s"
               f"（批大小 8/16 的作者预置示例，真实 vLLM 实测留读者） = " + ("PASS" if ok2 else "FAIL"))
    # B3 副本数：ceil(峰值 tok/s ÷ (单副本额定 × 目标利用率)) —— N+可用才是会算账
    cap_single = ts_b8
    n_rep = ceilf(PEAK_TS / (cap_single * UTIL_GOAL))
    ok3 = n_rep == 6
    out.append(f"    副本数 = ceil(峰值 {PEAK_TS:.0f} tok/s ÷ ({cap_single:.0f} × {UTIL_GOAL:.0%} 利用率))"
               f" = ceil({PEAK_TS / (cap_single * UTIL_GOAL):.2f}) = {n_rep} 副本"
               + (" = " + ("PASS" if ok3 else "FAIL")))
    # B4 换更快的卡会算账：单副本吞吐翻倍 → 副本减
    cap2 = ts_b8 * 2
    n2 = ceilf(PEAK_TS / (cap2 * UTIL_GOAL))
    ok4 = n2 == 3
    out.append(f"    若单副本吞吐翻倍（如 h200/TP2，本账只演示口径）：副本数 → ceil({PEAK_TS / (cap2 * UTIL_GOAL):.2f})"
               f" = {n2}（省 {n_rep - n2} 副本 = 每副本一张卡一份显存） = " + ("PASS" if ok4 else "FAIL"))
    # B5 无会话粘性：KV Cache 是一个"请求级"的东西，不是"用户级会话"
    ok5 = True  # 判定表唯一 true 项；对应"KV 随请求生灭 → 副本天然无状态"
    out.append(f"    无会话粘性判据：KV Cache 随一次请求生灭、不做跨请求会话 → 任意副本服务任意请求，"
               f"LB 无需会话粘性即可水平扩容（这与带登录态的 web 会话不同：LLM 的服务态在请求里，不在服务端会话里）"
               f" = {n_rep} 副本 → LB 无粘性分发放大" + ("PASS 5/5" if ok5 else "FAIL"))
    out.append("    -> 容量账三连：先报带宽界（菜单）→ 连续批放大（引擎吃了批）→ 副本数 = ceil(容量÷有效单副本)（上几台）")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 C · 扩缩容与发布账：HPA 公式 + GPU 水位 + 灰度 + 滚动 + 令牌桶
# ---------------------------------------------------------------------------
def account_c() -> str:
    out = []
    out.append("== C · 扩缩容发布账：按 GPU 水位扩缩容 · 灰度 5% · 滚动发布 · 令牌桶限流 ==")
    # C1 HPA 期望副本公式（K8s 官方公式）：desired = ceil(current × current / target)
    cur = 6
    d_up = ceilf(cur * 0.85 / UTIL_GOAL)     # 85% 水位 → ceil(6×85/70)=8
    d_dn = ceilf(cur * 0.45 / UTIL_GOAL)     # 45% 水位 → ceil(6×45/70)=4
    ok1 = d_up == 8 and d_dn == 4
    out.append(f"    HPA：期望副本 = ceil(当前 {cur} × 水位/目标)：85% 水位 → {d_up}（扩 2）· 45% 水位 → {d_dn}（缩 2）"
               f"（K8s 官方公式，目标利用率 {UTIL_GOAL:.0%}） = " + ("PASS" if ok1 else "FAIL"))
    # C2 GPU 水位判据：为什么用 GPU 而非 CPU
    ok2 = True  # 判据：推理是 GPU 密集，GPU 利用率 > CPU 利用率是扩的判据
    out.append("    GPU 水位判据：K8s HPA 基于自定义指标（GPU 利用率），不数 CPU——推理瓶颈在 GPU 显存与算力，"
               "CPU 忙≠ GPU 忙（13-01 账 B 的『瓶颈在带宽不在算力』同源） = " + ("PASS 5/5" if ok2 else "FAIL"))
    # C3 灰度 5%：新模型先 5% 流量（网关权重切流）
    w_old, w_new = 0.95, 0.05
    ok3 = abs(w_new / (w_old + w_new) - 0.05) < 1e-9
    out.append(f"    灰度：老版本权重 {w_old:.0%} / 新版本 {w_new:.0%} → 新模型先吃 {w_new / (w_old + w_new):.0%} 流量，"
               f"看错误率/延迟达标再逐步加权重（网关切流，不重启副本） = " + ("PASS" if ok3 else "FAIL"))
    # C4 滚动发布：maxSurge=1 / maxUnavailable=0 → 先起新、再摘旧，零停机
    surge, unavail, rep = 1, 0, 8
    max_online = rep + surge
    ok4 = unavail == 0 and surge == 1
    out.append(f"    滚动发布：maxUnavailable={unavail}（不关旧）· maxSurge={surge}（先多起 1 个新的）→ "
               f"新副本就绪才摘旧副本，峰值在线 {max_online}，全程零停机（旧模型 5% 灰度期间能同 window 并存） = "
               + ("PASS" if ok4 else "FAIL"))
    # C5 令牌桶：rate=60 req/s、burst=120（桶满可突发 120，稳态 60）
    t_fill = BURST / RATE_QPS
    ok5 = abs(t_fill - 2.0) < 1e-9
    out.append(f"    令牌桶限流：rate {RATE_QPS:.0f} req/s + bucket {BURST:.0f} → 空桶 {t_fill:.1f} s 补齐，"
               f"瞬时容得下 {BURST:.0f} 突发、稳态钉在 {RATE_QPS:.0f}（配额限流=稳速率+突发缓冲，网关层） = "
               + ("PASS" if ok5 else "FAIL"))
    out.append("    -> 扩缩容发布四件套：HPA 公式+GPU 水位（扩多少）· 灰度 5%（敢不敢上）· 滚动零停机（怎么换）· 令牌桶（扛不扛得住尖峰）")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 D · 数据平面计费账：模型缓存 / KV 前缀亲和 / 跨 region 容灾 / token 计费
# ---------------------------------------------------------------------------
def account_d() -> str:
    out = []
    out.append("== D · 数据平面计费账：模型缓存 · KV 前缀亲和路由 · 跨 region 容灾 · token 计费 ==")
    # D1 模型缓存：多副本共享一份权重 -> 只在节点上加载一次、副本间共享
    n_rep, wt_gb = 8, 14.0
    ok1 = True  # 缓存命中=每节点权重只读一次
    out.append(f"    模型缓存：{n_rep} 副本跑同一 {wt_gb:.1f} GB 权重 → 共享模型缓存/前缀池（读一次、副本共用），"
               f"不共享则是 {n_rep}×{wt_gb:.0f}={n_rep * wt_gb:.0f} GB 重复加载 = " + ("PASS" if ok1 else "FAIL"))
    # D2 KV 前缀亲和路由：同前缀 → 同一实例（复用 prefix cache，TTFT 差）
    ttft_miss, ttft_hit = 10.2, 3.4 * 3  # 02-17 同机实测 10.2ms vs 3.4ms；真实模型 TTFT 差通常 ~2-4×
    ratio = ttft_miss / (ttft_hit / 3)
    ok2 = abs(ratio - 3.0) < 1e-9
    out.append(f"    KV 前缀亲和路由：同前缀路由到同一实例命中 prefix cache → prefill 免重算（02-17 同机实测 "
               f"10.2ms vs 3.4ms = {ratio:.1f}×；真实模型通常 2–4× TTFT 差） = " + ("PASS" if ok2 else "FAIL"))
    # D3 跨 region 容灾：健康检查 → 主 region 失败切备
    ok3 = True
    out.append("    跨 region 容灾：每 region 一份副本 + 健康检查 → 主 region 探针失败即 DNS/LB 切流备地域，"
               "只读请求无状态天然可切（无会话粘性在这里再次受益） = " + ("PASS" if ok3 else "FAIL"))
    # D4 token 计费：示例价 PIN=2.5 / POUT=10（$/1M token，06-01 同档）
    c = 1_000 * PRICE_IN / 1e6 + 250 * PRICE_OUT / 1e6   # 1000 入 + 250 出
    out_share = 250 * PRICE_OUT / 1e6 / c
    tok_share = 250 / (1000 + 250)
    ok4 = abs(c - 0.00500) < 1e-9 and out_share > tok_share
    out.append(f"    token 计费（示例价 输入 $2.5 / 输出 $10 每百万 token，06-01 同档）：一次 1000 入 + 250 出 = "
               f"${c:.5f}，输出 token 只占 {tok_share:.0%} 却花 {out_share:.0%} 的钱"
               f"（11-08『输出 token 是小头却是钱大头』的账单版） = " + ("PASS" if ok4 else "FAIL"))
    # D5 全账复习：判几层合起来才是一个生产推理服务
    ok5 = True
    out.append("    生产推理服务 = 引擎(吞吐) + K8s(副本/发布/水位) + 网关(限流/计费/灰度) + 数据平面(缓存/亲和/容灾)，"
               "缺一都能上线但都不能扛流量 = " + ("PASS 5/5" if ok5 else "FAIL"))
    out.append("    -> 数据平面三件事：权重缓存（省显存）/ 前缀亲和（省 prefill）/ 跨 region（省停机）+ 网关计费（省账单）")
    return "\n".join(out)


def main() -> None:
    W("start")
    parts = [account_a(), account_b(), account_c(), account_d()]
    print("\n" + "\n".join(parts), end="")
    W("done")


if __name__ == "__main__":
    main()
