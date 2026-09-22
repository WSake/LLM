# -*- coding: utf-8 -*-
"""
ha_design_demo.py
13-工程化与基础设施/09-应用高可用设计（§13.9 · 🟢 了解 · 无专属里程碑）
灾难是"活下来"的打法：四类灾难 × 五件设计 = 高可用设计。
纯 stdlib · 零网络 · 零随机 · 零第三方；三个独立进程逐字节恒一。
运行：python code/notebooks/_tools/ha_design_demo.py
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


# ============================================================ A 灾难识别账
HAZARDS = {
    "api_failure":  "API 厂商故障：第三方接口 5xx / 超时 / 限流频发",
    "gpu_loss":     "GPU 损坏：掉卡 / ECC 报错 / 可用算力减半",
    "ctx_overflow": "上下文溢出：输入超 max_len 被截尾 / KV 显存不足",
    "cost_blowout": "成本爆炸：账单超预算 / 思考档放大 / 输出 token 超额",
}
SIGNALS = [
    ("连续 5xx / 网关挂起",            "api_failure"),
    ("第三方时延膨胀 10× 不回落",      "api_failure"),
    ("显卡 ECC 报错 + 掉卡",           "gpu_loss"),
    ("重启实例后可用显存减半",         "gpu_loss"),
    ("长输入被 max_len 截尾",          "ctx_overflow"),
    ("KV 缓存分配失败 OOM",            "ctx_overflow"),
    ("账单周环比 +24%",                "cost_blowout"),
    ("思考档输出 token 翻倍超预算",    "cost_blowout"),
]
REMEDY = {
    "api_failure":  "多供应商回退（OpenAI→自托管→规则兜底）",
    "gpu_loss":     "降级换实例 + 原地重路由到健康卡",
    "ctx_overflow": "限额（单用户 token 上限）+ 分段 / 摘要",
    "cost_blowout": "预算闸（告警→停思考档→熔断）+ 缓存省 prefill",
}
DESIGNS = [   # §13.9 设计五件套
    ("多供应商回退", "OpenAI → 自托管 → 规则"),
    ("降级",         "强模型 → 轻模型 → 规则"),
    ("限额",         "单用户 token 上限"),
    ("超时与重试退避", "重试 ≤3 · 指数退避"),
    ("缓存",         "prompt 缓存命中走 KV 省 prefill"),
]


def classify(sig):
    for s, cls in SIGNALS:
        if s == sig:
            return cls
    return "unknown"


def acct_a():
    hdr("账 A · 灾难识别账：四类灾难 × 信号 × 处置")
    # A1 四类灾难归位
    a1 = set(HAZARDS) == set(["api_failure", "gpu_loss", "ctx_overflow", "cost_blowout"])
    print("A1 四类灾难（§13.9）：%s" % ", ".join(sorted(HAZARDS)))
    assert a1 and len(HAZARDS) == 4
    # A2 信号→灾难 8/8
    ok = 0
    for sig, cls in SIGNALS:
        if classify(sig) == cls:
            ok += 1
    print("A2 8 条信号归类（5xx→厂商 / ECC→GPU / 截尾→溢出 / 超预算→成本）: %d/8 命中" % ok)
    assert ok == 8
    # A3 处置映射 4/4（§13.9 五件设计落在四类灾难上）
    a3 = (
        "多供应商回退" in REMEDY["api_failure"]
        and "降级" in REMEDY["gpu_loss"]
        and "限额" in REMEDY["ctx_overflow"]
        and "预算闸" in REMEDY["cost_blowout"]
    )
    print("A3 四类灾难各挂一件设计：回退→厂商 / 降级→GPU / 限额→溢出 / 预算闸→成本 :", a3)
    assert a3
    # A4 五件设计归位 5/5
    a4 = len(DESIGNS) == 5 and set(n for n, _ in DESIGNS) == set(
        ["多供应商回退", "降级", "限额", "超时与重试退避", "缓存"])
    print("A4 设计五件套（§13.9）：%s" % " · ".join(n for n, _ in DESIGNS))
    assert a4
    print("A PASS OK")


# ============================================================ B 回退与降级账
PROVIDER_CHAIN = ["openai", "selfhosted", "rules"]   # §13.9 多供应商回退
OPENAI_FAIL = {1, 4, 6}                             # 确定性故障注入：index 集
SELFHOST_FAIL = {6}                                 # 双故障点：openai+selfhosted 同时挂


def failover_run(n=10):
    """确定性故障注入：依次尝试供应商链，记录每条请求最终由谁兑付。"""
    served_by = []
    for i in range(n):
        for p in PROVIDER_CHAIN:
            if p == "openai" and i in OPENAI_FAIL:
                continue
            if p == "selfhosted" and i in SELFHOST_FAIL:
                continue
            served_by.append(p)
            break
    return served_by


# 降级梯：正常运维省预算时不降可用性（§13.9 降级：强→轻→规则）
PRICE_STRONG, PRICE_LIGHT, PRICE_RULES = 0.0042, 0.00037, 0.0   # 11-08 单价只引


def acct_b():
    hdr("账 B · 回退与降级账：一条请求怎么活下来")
    # B1 供应商故障 → 回退链兑付
    sb = failover_run(10)
    n_openai = sb.count("openai")
    n_self = sb.count("selfhosted")
    n_rules = sb.count("rules")
    single = 10 - len(OPENAI_FAIL)
    print("B1 故障注入：openai 挂 3 次（其中 1 次 selfhosted 也挂）→ 10 请求全部兑付"
          "（openai %d · selfhosted %d · rules %d），单点 openai 只剩 %d/10"
          % (n_openai, n_self, n_rules, single))
    assert len(sb) == 10 and n_rules == 1 and single == 7
    # B2 降级梯成本：100 请求全走强 vs 按触发器分派三档
    cost_all_strong = l2(100 * PRICE_STRONG)
    cost_ladder = l2(
        75 * PRICE_STRONG + 20 * PRICE_LIGHT + 5 * PRICE_RULES)
    save = l2(cost_all_strong - cost_ladder)
    save_ratio = round(save / cost_all_strong, 2)
    print("B2 降级梯 75 强+20 轻+5 规则：全强 $%.2f → 分派 $%.2f = 省 $%.2f (-%.0f%%)"
          % (cost_all_strong, cost_ladder, save, save_ratio * 100))
    # 质量：三档加权（强 0.98 · 轻 0.92 · 规则 0.60）
    q_all = round(0.75 * 0.98 + 0.20 * 0.92 + 0.05 * 0.60, 3)
    print("   质量加权 %.3f ≥ 0.90（降级省预算，可用性不塌）: %s" % (q_all, q_all >= 0.90))
    assert cost_all_strong == 0.42 and save_ratio == 0.24 and q_all >= 0.90
    print("B PASS OK")


# ============================================================ C 限额与溢出账
TOKEN_DAY_LIMIT = 100000                       # §13.9 限额：单用户 token 上限
USERS = {"alice": 87000, "bob": 96000, "carol": 102000}


def quota_verdict(used):
    if used < 0.90 * TOKEN_DAY_LIMIT:
        return "ok"
    if used < TOKEN_DAY_LIMIT:
        return "warn"          # ≥90%：预警降速（停思考档/排队）
    return "block"             # ≥100%：拒绝（限额=硬边界，拒绝不是排队）


CTX_STRATS = [   # 长输入 20k 对 8k 窗口：三档策略（上下文溢出不=一次失败）
    ("截断尾部", 0.33, 0.0),
    ("分段/摘要", 0.67, 1.0),
    ("换 128k 长窗", 1.00, 8.0),
]


def acct_c():
    hdr("账 C · 限额与溢出账：把爆挡在账上")
    # C1 单用户 token 上限三档判定
    res = {u: quota_verdict(v) for u, v in USERS.items()}
    print("C1 单用户日上限 %d token：alice 87k→%s · bob 96k→%s · carol 102k→%s"
          % (TOKEN_DAY_LIMIT, res["alice"], res["bob"], res["carol"]))
    assert res == {"alice": "ok", "bob": "warn", "carol": "block"}
    # C2 上下文溢出三档策略：召回↑成本↑（接 13-08 缓存省 prefill 收尾）
    rc = [c for _, c, _ in CTX_STRATS]
    pc = [p for _, _, p in CTX_STRATS]
    print("C2 20k 长输入对 8k 窗口（溢出）：截断召回 %.2f 不花成本 → 摘要 %.2f → 换长窗 %.2f 成本 ×%.1f"
          % (rc[0], rc[1], rc[2], pc[2]))
    assert rc == [0.33, 0.67, 1.00] and pc[2] == 8.0
    print("C PASS OK")


# ============================================================ D 成本爆炸防波堤
DAILY = [40, 45, 50, 55, 60, 65, 70, 75, 80, 90]   # 无护栏 10 日实际支出（思考档逐日升）
DAY_BUDGET = 50.0
LINES = [        # 预算闸四线：水位 × 动作（13-07 告警四线同构只引）
    (0.80, "day  1 告警：邮件+降速"),
    (0.90, "day  2 停思考档"),
    (1.00, "day  3 降级轻模型"),
    (1.20, "day  5 熔断非关键流量"),
]


def guard_days():
    trig = []
    for ratio, label in LINES:
        threshold = DAY_BUDGET * ratio
        for i, d in enumerate(DAILY):
            if d >= threshold:
                trig.append(i + 1)
                break
    return trig


def acct_d():
    hdr("账 D · 成本爆炸防波堤账：预算闸四线")
    # D1 无护栏 vs 有护栏累计
    sum_no = sum(DAILY)
    sum_guard = sum(DAILY[:3]) + 30 + 0 + 30 * 5   # day1-3 原样 · day4 降级 30 · day5 熔断 0 · day6-10 30/天
    save_d = sum_no - sum_guard
    print("D1 无护栏 10 日 $%d → 预算闸后 $%d = 省 $%d (-%.0f%%)"
          % (sum_no, sum_guard, save_d, round(save_d / sum_no * 100)))
    # D2 四线各自触发日（§13.9 成本爆炸 + §13.5 周环比 +20% 同家族）
    trig = guard_days()
    print("D2 预算闸四线触发日：80%%→天%d · 90%%→天%d · 100%%→天%d · 120%%→天%d"
          % (trig[0], trig[1], trig[2], trig[3]))
    print("   动作=告警 / 停思考档 / 降级轻模型 / 熔断（13-07 告警线动作同构只引）")
    assert sum_no == 630 and sum_guard == 315 and save_d == 315
    assert trig == [1, 2, 3, 5]
    print("D PASS OK")


def main():
    print("ha_design_demo.py  |  13-工程化与基础设施/09-应用高可用设计（§13.9 了解）")
    acct_a()
    acct_b()
    acct_c()
    acct_d()
    print("== 总计 ==")
    print("PASS 10 · FAIL 0")


if __name__ == "__main__":
    main()
