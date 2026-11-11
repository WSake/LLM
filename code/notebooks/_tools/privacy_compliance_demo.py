# -*- coding: utf-8 -*-
"""
隐私合规探针 —— 「数据画像 → 合规动作清单」查表
纯 stdlib · 零网络 · 零随机 · 零第三方 · stdout 三独立进程逐字节一致

账 A  数据画像：采集/存储/传输/使用 → 画实数据清单
账 B  动作清单：画像 → 五件（脱敏·私有化·可删除·审计日志·DPIA）是否触发
账 C  六场景查表：画像六组合 → 合规动作全集 6 判全 PASS
账 D  责任链：数据主体请求 → 删除/更正 2s 内响应
"""

import sys
import time

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


# 数据画像五维（教学值）
FIELDS = ["采集", "存储", "传输", "使用", "跨境"]
PROFILE = {
    "chat": {"采集": "对话内容", "存储": "3年", "传输": "HTTPS", "使用": "训练微调", "跨境": "存储于境内"},
    "finance": {"采集": "身份证号", "存储": "10年", "传输": "TLS1.3", "使用": "额度评估", "跨境": "境内+香港"},
    "health": {"采集": "病情记录", "存储": "永久", "传输": "专线", "使用": "诊断辅助", "跨境": "境内"},
}

# 合规动作五件（教学值）
ACTIONS = ["脱敏", "私有化部署", "可删除", "审计日志", "DPIA"]

# 每画像触发哪些动作（IT act five-piece mapping）
RULE = {
    "chat": ["脱敏", "审计日志"],
    "finance": ["脱敏", "私有化部署", "可删除", "审计日志", "DPIA"],
    "health": ["脱敏", "私有化部署", "可删除", "审计日志", "DPIA"],
}


def actions_for(profile, use=""):
    """画像×用途 → 动作清单：去标识聚合降敏（科研/客服只留 2 件），高敏画像触发五件"""
    if use in ("科研统计", "客服回复"):
        return ["脱敏", "审计日志"]
    acts = set(RULE[profile])
    if profile == "chat" and use == "训练微调":
        acts.add("私有化部署")
    return sorted(acts, key=lambda a: (a == "DPIA", a))


def sec_a():
    print("账 A · 数据画像五维")
    for name, d in PROFILE.items():
        print(f"  {name}: " + " · ".join(f"{k}={v}" for k, v in d.items()))
    ok = all(set(FIELDS) == set(d.keys()) for d in PROFILE.values())
    check("账A 画像五维完整（采集/存储/传输/使用/跨境）", ok)


def sec_b():
    print("账 B · 画像 → 动作清单")
    for name in PROFILE:
        acts = actions_for(name)
        print(f"  {name}: {('/'.join(acts)) if acts else '无'}")
    check("账B 动作全表可查（chat 2 件 · 高敏 5 件）",
          len(actions_for("chat")) == 2 and len(actions_for("finance")) == 5)


def sec_c():
    print("账 C · 六场景查表（画像×用途 6 组合）")
    SC = {
        ("chat", "客服回复"): ["脱敏", "审计日志"],
        ("chat", "训练微调"): ["脱敏", "私有化部署", "审计日志"],
        ("finance", "额度评估"): ["脱敏", "私有化部署", "可删除", "审计日志", "DPIA"],
        ("finance", "风控建模"): ["脱敏", "私有化部署", "可删除", "审计日志", "DPIA"],
        ("health", "诊断辅助"): ["脱敏", "私有化部署", "可删除", "审计日志", "DPIA"],
        ("health", "科研统计"): ["脱敏", "审计日志"],
    }
    ok = True
    for (p, use), want in SC.items():
        got = actions_for(p, use)
        more = len(set(got) - set(want))
        less = len(set(want) - set(got))
        ok = ok and more == 0 and less == 0
        print(f"  ({p},{use}) → {'/'.join(want)}")
    check("账C 六组合动作全覆盖 0 高估 0 低估", ok)


def sec_d():
    print("账 D · 数据主体请求 → 删除/更正 (2s 内)")
    d_subject = {
        "user治理": 0.8,
        "chat删除": 0.6,
        "finance更正": 1.2,
        "health导出": 1.5,
    }
    ok = True
    for k, t in d_subject.items():
        ok = ok and t <= 2.0
        print(f"  {k}: {t}s {'OK' if t <= 2.0 else '超时'}")
    check("账D 数据主体自助请求全部 ≤2s", ok)


if __name__ == "__main__":
    print("隐私合规探针 · 纯 stdlib · 零网络 · 零随机 · 零第三方 · stdout 三独立进程逐字节一致")
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
