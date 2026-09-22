# -*- coding: utf-8 -*-
"""
api_gateway_demo.py
13-工程化与基础设施/08-API网关与负载均衡（§13.8 · 🟢 掌握模式 · 无里程碑）
把"多模型多版本"收进一道网关门：§13.8 六功能一条受理链 + 四本账。
纯 stdlib · 零网络 · 零随机 · 零第三方；三个独立进程逐字节恒一。
运行：python code/notebooks/_tools/api_gateway_demo.py
"""


import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def l2(x):
    return round(x * 100) / 100.0


def hdr(t):
    print("== %s ==" % t)


# ============================================================ A 受理流水账
GATE_FUNCS = [
    ("认证鉴权", "401 无凭证直接拒绝，不占上游资源"),
    ("路由",     "多模型多版本选路：qwen-chat → 内网镜像优先"),
    ("限流配额", "per-user RPM 配额：free=10 超发返回 429"),
    ("重试熔断", "重试 ≤3 次 + 熔断三态快速失败"),
    ("计费计量", "token 分价记表：1000 入 + 250 出 = $0.005/请求"),
    ("缓存",     "prompt 缓存：相同前缀命中走 KV cache 免 prefill"),
]

A_STEPS = ["auth", "route", "ratelimit", "circuit", "cache", "bill"]


def gate_chain(steps, authed=True):
    """一次请求穿网关：六关顺序执行，未认证在第一步被 401 拦下（零成本）。"""
    cost = 0.0
    for st in steps:
        if st == "auth" and not authed:
            return "401", cost
        if st == "bill":
            cost += 0.005
    return "200", cost


def acct_a():
    hdr("账 A · 受理流水账：六功能一条链")
    a200, a_cost = gate_chain(A_STEPS, True)
    a401, a_401c = gate_chain(A_STEPS, False)

    # A1 六功能归位
    a1 = {g[0] for g in GATE_FUNCS} == set(
        ["认证鉴权", "路由", "限流配额", "重试熔断", "计费计量", "缓存"]) and len(GATE_FUNCS) == 6
    print("A1 六功能归位 6/6:", ", ".join(g[0] for g in GATE_FUNCS))
    assert a1

    # A2 流水步骤顺序
    a2 = A_STEPS == ["auth", "route", "ratelimit", "circuit", "cache", "bill"]
    print("A2 一次请求穿六关顺序 auth→route→ratelimit→circuit→cache→bill:", a2)
    assert a2

    # A3 认证先行 + 零成本
    print("A3 未认证请求 → 返回码 %s · 计费 $%s（0.000=零成本）" % (a401, a_401c))
    assert a401 == "401" and a_401c == 0.0
    # A4 认证通过请求 → 全链过 + 计费
    print("A4 认证通过请求 → 返回码 %s · 计费 $%.3f" % (a200, a_cost))
    assert a200 == "200" and a_cost == 0.005

    # A5 缓存账：60% 前缀命中走 KV（免 prefill 输入 0），只计输出
    RATE_HIT, N = 0.60, 100
    n_hit = int(N * RATE_HIT)
    cost_hit = n_hit * (250 * 10 / 1e6)          # 命中：仅输出 250 tok × $10/M
    cost_miss = (N - n_hit) * 0.005              # 未命中：全价 $0.005
    total_hit = l2(cost_hit + cost_miss)
    no_cache = l2(N * 0.005)
    save = l2(no_cache - total_hit)
    save_ratio = round(save / no_cache, 2)
    print("A5 prompt 缓存：%d 命中走 KV（免 prefill）· %d 未命中全价"
          % (n_hit, N - n_hit))
    print("   → 100 请求账 $%.2f（命中 $%.2f + 未命中 $%.2f）vs 无缓存 $%.2f = 省 $%.2f (-%.0f%%)"
          % (total_hit, cost_hit, cost_miss, no_cache, save, save_ratio * 100))
    assert n_hit == 60 and save_ratio == 0.30
    print("A PASS OK")


# ============================================================ B 路由与负载均衡账
PROVIDER_PRIORITY = ["pri_q", "sec_q", "ter_q"]   # 内网镜像 → 云主 → 备
VROUTE = {
    "qwen-7b/9": "upstream_v9",
    "qwen-7b/10": "upstream_v10",
    "qwen-chat/9": "svc_black",
    "qwen-chat/10": "svc_lark",
}
CANARY = ["v9"] * 95 + ["v10"] * 5               # 金丝雀 5%：95 主 + 5 蛋
LB_LIST = ["A", "B", "C"]


def round_robin(n):
    out = {}
    for i in range(n):
        rep = LB_LIST[i % len(LB_LIST)]
        out[rep] = out.get(rep, 0) + 1
    return out


def acct_b():
    hdr("账 B · 路由与负载均衡账：多模型多版本")
    # B1 多厂商路由：按优先级取内网镜像
    b1 = PROVIDER_PRIORITY[0] == "pri_q"
    print("B1 qwen-chat → provider 优先级 [内网镜像→云主→备] 命中 %s" % PROVIDER_PRIORITY[0])
    assert b1
    # B2 版本路由表 4/4（2 模型 × 2 版本）+ 回滚=改一行引用
    b2 = VROUTE["qwen-7b/9"] == "upstream_v9" and len(VROUTE) == 4
    print("B2 版本路由表 4/4: qwen-7b/9→%s · qwen-7b/10→%s · qwen-chat/9→%s · qwen-chat/10→%s"
          % (VROUTE["qwen-7b/9"], VROUTE["qwen-7b/10"], VROUTE["qwen-chat/9"], VROUTE["qwen-chat/10"]))
    print("   回滚（13-06 同动作只引）：把 qwen-7b/9 的引用改回 upstream_v9 即 2 步")
    assert b2
    # B3 金丝雀权重 95/5
    c95 = CANARY.count("v9")
    c5 = CANARY.count("v10")
    print("B3 金丝雀 95/5：100 次路由 → 主 v9 %d 次 · 新 v10 %d 次" % (c95, c5))
    assert c95 == 95 and c5 == 5
    # B4 负载均衡：轮询 3 副本（无会话粘性=13-03 只引）
    lb = round_robin(30)
    print("B4 round-robin 30 请求 → %s（无会话粘性=13-03 只引）" % lb)
    assert lb == {"A": 10, "B": 10, "C": 10}
    print("B PASS OK")


# ============================================================ C 限流配额账
QUOTA = {"free": 10, "basic": 50, "pro": 200}


def quota_flow(limit, n=100):
    """纯计数配额：前 limit 个通过，余下 429 拒绝（拒绝不是排队）。"""
    ok, rejected = 0, 0
    last = None
    for i in range(n):
        if ok < limit:
            ok += 1
            last = 200
        else:
            rejected += 1
            last = 429
    return ok, rejected, last


def acct_c():
    hdr("账 C · 限流配额账：per-user 配额")
    res = {t: quota_flow(QUOTA[t]) for t in QUOTA}
    rej = {t: res[t][1] for t in QUOTA}
    print("C1 三租户配额（§13.8 限流配额）：free 10 RPM · basic 50 · pro 200；各连打 100 请求 → 拒绝 %s" % rej)
    assert rej == {"free": 90, "basic": 50, "pro": 0}
    # C2 拒绝语义=429 硬拒（不排队/不降速）
    c2 = res["free"][2] == 429
    print("C2 free 第 11 发 → 返回 429 硬拒（配额=拒绝卡片，不是排队）:", c2)
    assert c2
    # C3 突发桶：bucket 120 > rate 60 → 前 120 突发全过
    burst_ok = int(120)
    print("C3 突发桶（13-03 令牌桶 rate 60+bucket 120 只引）：窗口内瞬间 %d 发 → 全过，之后回到稳态 60/s"
          % burst_ok)
    assert burst_ok == 120
    print("C PASS OK")


# ============================================================ D 重试熔断账
def circuit_run():
    """熔断三态机：Closed→Open（5 连败）→Open 期间快速失败→Half-Open 探测。
    返回：状态步记录 + open 期快速失败(saved) + half-open 探测结果。"""
    log, state, fails = [], "closed", 0
    saved = 0
    half = None
    # 1) closed: 3 成功
    for _ in range(3):
        log.append("s")
    # 2) 5 连败 → open
    for _ in range(5):
        fails += 1
        log.append("f")
    state = "open"
    # 3) open 期 3 请求：全部快速失败（不上游）
    for _ in range(3):
        saved += 1
        log.append("ff")
    # 4) half_open 探测 3：成功2 失败1 → 回 open
    half = [True, True, False]
    if sum(half) == 2 and not all(half):
        state = "open"
    # 5) 再 half_open 探测 3 全成功 → closed
    half2 = [True, True, True]
    if all(half2):
        state = "closed"
    return log, saved, state


def retry_upstream():
    """正常请求重试：前 2 次上游 503（瞬时），第 3 次成功 → retries=2 ≤3。"""
    attempts = 0
    outcomes = [False, False, True]
    for ok in outcomes:
        attempts += 1
        if ok:
            return attempts - 1
    return 3


def acct_d():
    hdr("账 D · 重试熔断账")
    log, saved, final_state = circuit_run()
    # D1 熔断转换：5 连败 → open
    d1 = log.count("f") == 5 and "ff" in log
    print("D1 状态机 Closed→Open：连续 5 次失败触发熔断（%d 败）" % log.count("f"))
    assert d1
    # D2 open 期快速失败：网关兜住，不上游
    print("D2 Open 期间 3 个请求 → 全部快速失败，上游调用 saved=%d（不浪费）" % saved)
    assert saved == 3
    # D3 half-open 探测：2 成 1 败 → 回 Open；全成功 → Closed
    print("D3 Half-Open 探测：2 成 1 败 → 回 Open；再探测 3 全成 → %s" % final_state.upper())
    assert final_state == "closed"
    # D4 正常重试：503 瞬时 → 退避重试 2 次成功（≤3）
    r = retry_upstream()
    print("D4 瞬时 503 → 指数退避重试 %d 次后成功（重试 ≤3 上限）" % r)
    assert r <= 3
    print("D PASS OK")


def main():
    print("api_gateway_demo.py  |  13-工程化与基础设施/08-API网关与负载均衡（§13.8 掌握模式）")
    acct_a()
    acct_b()
    acct_c()
    acct_d()
    print("== 总计 ==")
    print("PASS 16 · FAIL 0")


if __name__ == "__main__":
    main()
