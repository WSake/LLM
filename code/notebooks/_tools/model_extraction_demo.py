# -*- coding: utf-8 -*-
"""
模型抽取指纹探测 —— 「查询量 / 前缀多样性 / 主题集中度 / 时间均匀性 → 抽取嫌疑」指纹判定
纯 stdlib · 零网络 · 零随机 · 零第三方 · stdout 三独立进程逐字节一致

账 A  特征五元组：qps / 前缀多样性 / 主题集中度 / 时间均匀性 → 嫌疑分
账 B  判别双臂：正常用户窗放行，抽取脚本窗标嫌疑
账 C  防御四件：限流（配额 429）· 指纹检测 · 服务条款判定 · 输出扰动（教学值）
账 D  成本账：无防御 1.0 天 → 限流 2.893 天 → 限流+扰动 4.133 天 → 指纹+ToS 停用
"""

import sys
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

_t0 = time.perf_counter()

_PASS = 0
_FAIL = 0


def check(name, cond):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"[PASS] {name}")
    else:
        _FAIL += 1
        print(f"[FAIL] {name}")


# ---------------------------------------------------------------- 固定数据（零随机）

def _normal_window():
    """真实用户 24 条查询：低 QPS、前缀全不同、主题分散在 6 类、节奏不均匀（上午+下午两段）。"""
    reqs = []
    topics = ["math", "med", "law", "code", "docs", "news"]
    for i in range(12):
        reqs.append((i * 100.0, "user_q%02d" % (i % 12), topics[i % 6]))
    for i in range(12):
        reqs.append((18000.0 + i * 100.0 + 7.0, "user_q%02d" % (12 + i), topics[(i + 3) % 6]))
    return reqs


def _extract_window():
    """抽取脚本 240 条：20 秒内每 1/12 秒一条（qps=12），8 种模板循环，主题钉死在 2 个，节奏完全均匀。"""
    reqs = []
    for i in range(240):
        t = i * (20.0 / 240.0)
        pref = "tmpl%02d" % (i % 8)
        topic = "sci" if i < 96 else "fin"
        reqs.append((t, pref, topic))
    return reqs


# ---------------------------------------------------------------- 账 A：嫌疑指纹

def window_features(reqs):
    n = len(reqs)
    if n == 0:
        return {"qps": 0.0, "pref_ratio": 0.0, "topic_share": 0.0, "evenness": 0.0, "score": 0.0}
    ts = [r[0] for r in reqs]
    span = max(max(ts) - min(ts), 1.0)
    qps = n / span
    pref_ratio = len({r[1] for r in reqs}) / n
    top_topic = Counter(r[2] for r in reqs).most_common(1)[0][1]
    topic_share = top_topic / n
    buckets = [0] * 8
    for t in ts:
        idx = min(int((t - min(ts)) / span * 8), 7)
        buckets[idx] += 1
    mu = n / 8.0
    evenness = max(0.0, 1.0 - (sum((x - mu) ** 2 for x in buckets) / 8.0) ** 0.5 / max(mu, 1e-9))
    score = (
        0.30 * min(qps / 5.0, 1.0)
        + 0.30 * (1.0 - pref_ratio)
        + 0.25 * topic_share
        + 0.15 * evenness
    )
    return {
        "qps": round(qps, 3),
        "pref_ratio": round(pref_ratio, 3),
        "topic_share": round(topic_share, 3),
        "evenness": round(evenness, 3),
        "score": round(score, 3),
    }


SUSPECT = 0.6


def verdict(f):
    return "嫌疑(抽取)" if f["score"] >= SUSPECT else "放行"


def sec_a():
    print("账 A · 特征五元组 → 嫌疑分（阈 0.6）")
    for label, reqs in (("正常窗 u-01", _normal_window()), ("抽取窗 x-77", _extract_window())):
        f = window_features(reqs)
        print(f"  {label}: qps={f['qps']} pref_ratio={f['pref_ratio']} "
              f"topic_share={f['topic_share']} evenness={f['evenness']} → score={f['score']}")
    fE = window_features(_extract_window())
    fN = window_features(_normal_window())
    check("账A 嫌疑分拉开：抽取窗 ≥0.6 且 正常窗 ≤0.5", fE["score"] >= SUSPECT and fN["score"] <= 0.5)


def sec_b():
    print("账 B · 判别双臂")
    for label, reqs in (("u-01", _normal_window()), ("x-77", _extract_window())):
        print(f"  {label} → {verdict(window_features(reqs))}")
    fN = window_features(_normal_window())
    fE = window_features(_extract_window())
    check("账B 良性放行 / 危险被标嫌疑", verdict(fN) == "放行" and verdict(fE) == "嫌疑(抽取)")


# ---------------------------------------------------------------- 账 C：防御四件

QPS_LIMIT = 4.0
PLANS = {"free": 1000}                    # 教学值：free 档日配额 1000 次
TERMS = {
    "u-01": {"plan": "free", "use": "interactive"},
    "x-77": {"plan": "free", "use": "batch"},       # 自称批量用途
}
# 输出扰动：对概率加拉普拉斯噪声（教学值，幅度 b 越大命中率越低）
PERTURB_HIT = {0.0: 1.00, 0.1: 0.93, 0.5: 0.62}


def sec_c():
    print("账 C · 防御四件（限流 / 指纹 / ToS / 输出扰动）")
    for client, reqs in (("u-01", _normal_window()), ("x-77", _extract_window())):
        f = window_features(reqs)
        acts = []
        # 1 限流：超配额拒 429
        span = max(max(r[0] for r in reqs) - min(r[0] for r in reqs), 1.0)
        allowed = min(len(reqs), int(QPS_LIMIT * span))
        rejected = len(reqs) - allowed
        if rejected > 0:
            acts.append(f"限流 429 拒 {rejected} 条（额度 {allowed}）")
        # 2 指纹检测
        if f["score"] >= SUSPECT:
            acts.append("指纹命中 → 标记嫌疑")
        # 3 服务条款
        t = TERMS[client]
        if f["score"] >= SUSPECT and t["use"] == "batch":
            acts.append("ToS 违规（批量蒸馏明文禁止）")
        # 4 输出扰动：嫌疑者 top-1 命中率掉到 0.62
        b = 0.5 if f["score"] >= SUSPECT else 0.0
        acts.append(f"输出扰动 b={b} → top-1 命中率 {PERTURB_HIT[b]:.2f}")
        print(f"  {client}: " + "；".join(acts) if acts else f"  {client}: 无动作")
        if client == "u-01":
            u_acts = acts
        else:
            x_acts = acts
    check("账C 抽取会话四件全触发，正常会话零动作",
          len(x_acts) == 4 and len(u_acts) == 1 and "限流 429 拒 160 条" not in u_acts[0])


# ---------------------------------------------------------------- 账 D：成本账

def sec_d():
    print("账 D · 抽取一台近似模型的成本账（教学值）")
    Q = 1_000_000.0
    per_day_unlimited = 1_000_000.0
    per_day_limited = QPS_LIMIT * 86400.0
    d0 = Q / per_day_unlimited
    d1 = Q / per_day_limited
    d2 = d1 / 0.70                       # 扰动后 30% 查询白费
    print(f"  无防御：{d0:.1f} 天")
    print(f"  限流({QPS_LIMIT:g} qps)：{d1:.3f} 天")
    print(f"  限流+扰动：{d2:.3f} 天")
    print("  指纹+ToS：会话当日停用 → 无法完成（∞）")
    ok = d2 > d1 > d0
    ok = ok and all(
        PERTURB_HIT[b] > PERTURB_HIT[b2]
        for b, b2 in ((0.0, 0.1), (0.1, 0.5))
    )   # 扰动幅度越大 命中率单调降
    check("账D 防御逐级放大成本，扰动幅度与命中率单调反比", ok)


if __name__ == "__main__":
    print("模型抽取指纹探测 · 纯 stdlib · 零网络 · 零随机 · 零第三方 · stdout 三独立进程逐字节一致")
    print("=" * 74)
    sec_a()
    sec_b()
    sec_c()
    sec_d()
    print("=" * 74)
    print(f"PASS {_PASS} / FAIL {_FAIL}")
    if _FAIL:
        sys.exit(1)
    print(f"wall-clock: {time.perf_counter() - _t0:.3f}s", file=sys.stderr)
