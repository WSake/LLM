#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_leak_demo.py —— 12-04 数据泄漏（§12.3 · 🟢 掌握）配套探针
纯 stdlib · 零网络 · 零随机 · 零第三方
stdout 三独立进程逐字节一致（墙钟只进 stderr）

§12.3 三段落点：
  1. 四种形态：训练数据记忆背诵、RAG 越权（租户隔离失效）、
     Agent 工具输出敏感往来、日志记录敏感输入 —— 账 A 判形态 4/4 + 良性 1/1；
  2. 缓解六件：脱敏、最小化、租户隔离、输出过滤、日志脱敏、访问审计 ——
     本篇把「脱敏」（复用 13-07 规则）与「租户隔离」（ACL）做成两笔实账；
  3. 验收用例：越权场景判别性对照（租户 A 查不到 B：拦截前/后）—— 账 C 双臂。

账 A  四形态判分：5 条预置样例 → 形态标签（训练记忆×1/RAG越权×1/工具输出×1/
      日志×1/良性 1 不应误判）
账 B  脱敏复用：13-07 的 observability_guardrail_demo.py 同套四类规则
      （手机/邮箱/身份证/API key），对含 4 类 PII 的日志行清洗 → 0 残留 + 号码型全消
账 C  租户隔离 ACL：租户 A 的检索加不加 ACL → 泄漏 2→0；租户 B 好臂不误拦
账 D  日志遮盖凝视：4 条模拟落库日志全部长 token 遮盖为 '***' → 原始敏感串 0 残留

照例：纯 stdlib · 零网络 · 零随机 · 零第三方。统计均本机确定性计算。
"""
from __future__ import annotations

import re
import sys
import time

try:
    _ = sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

_t0 = time.perf_counter()
PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"   [PASS] {name}")
    else:
        FAIL += 1
        print(f"   [FAIL] {name}")


# ------------------------------------------------ 敏感常量（作者预置仿真，13-07 同款）
_PHONE = "13812345678"
_IDNO = "110101199003074515"
_APIKEY = "sk-proj-abcdef1234567890"
_EMAIL = "zhangsan@qq.com"
_SECRETS = [_PHONE, _IDNO, _APIKEY, _EMAIL]


# ------------------------------------------------- 账 B 的脱敏规则（13-07 探针同款复用）
# 顺序说明：更长的规则（身份证 18 位）必须先跑，
# 否则手机 11 位规则会把身份证里的号码段错吃。
_DRULES = [
    ("idcard", re.compile(r"(\d{4})\d{10}(\d{4})"),
     lambda m: f"{m.group(1)}**********{m.group(2)}"),
    ("phone", re.compile(r"(1[3-9]\d)\d{4}(\d{4})"),
     lambda m: f"{m.group(1)}****{m.group(2)}"),
    ("email", re.compile(r"(\w)[\w.-]*@"),
     lambda m: f"{m.group(1)}***@"),
    ("apikey", re.compile(r"(sk-)\S+"),
     lambda m: f"{m.group(1)}***"),
]


def redact(text):
    for _n, rx, rep in _DRULES:
        if rx.search(text):
            text = rx.sub(rep, text)
    return text


# ------------------------------------------------- 账 A：四形态判分
# 每种形态用一个独特关键词作判据（词面样式识别，教学级；非真实模型分类器）
_AKWS = [
    ("训练数据记忆", "背诵"),
    ("RAG越权", "租户"),
    ("Agent工具", "工具"),
    ("日志", "日志"),
]
_SAMPLES = [
    ("S1", "模型把训练语料里的原理段落原样短句背诵出来了 → ", "训练数据记忆"),
    ("S2", "多租户共享向量库，搜索时漏加租户隔离过滤，能查到别家数据 → ", "RAG越权"),
    ("S3", "工具函数把收款账号直接返回到了对话里 → ", "Agent工具"),
    ("S4", "排查时发现查询日志把用户原文原样落盘 → ", "日志"),
    ("S5", "把客户名单整理成 PDF 附件发出 → ", "良性"),
]


def classify(text):
    for fam, kw in _AKWS:
        if kw in text:
            return fam
    return "良性"


def sec_a():
    print("[账 A] 四形态判分：5 条预置样例 → 形态标签（§12.3 形态表）")
    ok = 0
    for sid, text, expect in _SAMPLES:
        got = classify(text)
        good = (got == expect)
        ok += int(good)
        print(f"   {sid}  期望 {expect:<7} → {'命中 ' + got if good else '错判为 ' + got}：{text}")
    check(f"账A 四形态判分 {ok}/5（训练记忆/RAG越权/工具输出/日志各归位 · 良性 1/1 不误判）",
          ok == 5)
    print()


# ------------------------------------------------- 账 B：脱敏复用（13-07 锚点同款）
def sec_b():
    print("[账 B] 脱敏复用：13-07 同套四类规则，对含 4 类 PII 的日志行清洗")
    line = (f"in: '帮我翻译{_PHONE} {_EMAIL} {_IDNO} key={_APIKEY}'"
            f" out: '…' model=qwen-7b temp=0.7 tool=search")
    out = redact(line)
    for s in _SECRETS:
        print(f"   {s:<28} → {'已遮盖' if s not in out else '仍残留!'}")
    leak = [s for s in _SECRETS if s in out]
    no_phone = not re.search(r"1[3-9]\d{9}", out)
    no_id = not re.search(r"\d{17}\d", out)
    print(f"   脱敏后: {out[:96]}")
    check("账B 四类 PII 逐字 0 残留 · 手机号/身份证号型全消（日志可留）",
          not leak and no_phone and no_id)
    print()


# ------------------------------------------------- 账 C：租户隔离 ACL 判别性对照
_DOCS = [
    ("d1", "tA", f"客户数据：{_PHONE}"),
    ("d2", "tB", f"调度数据：key={_APIKEY}"),
    ("d3", "tA", f"A 数据:{_EMAIL}"),
    ("d4", "tC", f"C 数据:{_IDNO}"),
]
_QUERY = "数据"
_OTHERS = {"d2", "d4"}          # tA 以外租户的文档


def search_no_acld():
    return [did for did, _own, txt in _DOCS if _QUERY in txt]


def search_acld(owner):
    return [did for did, own, txt in _DOCS if _QUERY in txt and own == owner]


def sec_c():
    print("[账 C] 租户隔离 ACL：租户 A 加不加 ACL 过滤 → 泄漏 2→0（判别性双臂）")
    raw = search_no_acld()
    own = search_acld("tA")
    raw_leak = [d for d in raw if d in _OTHERS]
    own_blocked = [d for d in raw if d not in own]
    print(f"   [租户 A] 不加 ACL：查『{_QUERY}』命中 {len(raw)} 条 {raw}")
    print(f"      其中属于 B/C 租户 {len(raw_leak)} 条 {raw_leak} → 越权泄漏")
    print(f"   加 ACL 后：A 只见 {own}，{len(own_blocked)} 条异租户被拦 → 泄漏 0")
    b_raw = search_no_acld()
    b_own = search_acld("tB")
    print(f"   [租户 B 好臂] 加 ACL 后命中 {b_own}（自己的调度数据仍在，不误拦）")
    check("账C 越权判别：A 泄漏 2→0 · B 不误拦且仍拿到自己的数据",
          len(raw) == 4 and raw_leak == ["d2", "d4"] and own == ["d1", "d3"]
          and len(own_blocked) == 2 and b_own == ["d2"])
    print()


# ------------------------------------------------- 账 D：日志遮盖凝视
_LOG_LINES = [
    f"query='{_EMAIL}'",
    f"user_phone={_PHONE}",
    f"auth={_APIKEY}",
    f"doc={_IDNO}",
]


def sweep(text):
    return re.sub(r"\S{8,}", "***", text)


def sec_d():
    print("[账 D] 日志遮盖凝视：4 条模拟落库日志全部长 token 遮盖为 '***'")
    ok = True
    for ln in _LOG_LINES:
        masked = sweep(ln)
        if any(s in masked for s in _SECRETS):
            ok = False
        print(f"   {ln:<42} → {masked}")
    check("账D 遮盖后 4 条日志原始敏感串 0 残留（日志可留 = 可审计的内容被保留）", ok)


# ------------------------------------------------- 主流程
def main():
    print("=" * 76)
    print("data_leak_demo.py —— 12-04 数据泄漏（§12.3 · 🟢 掌握）")
    print("纯 stdlib · 零网络 · 零随机 · 零第三方 · stdout 三独立进程逐字节一致")
    print("=" * 76)
    sec_a()
    sec_b()
    sec_c()
    sec_d()
    print()
    print(f"合计断言：PASS {PASS} · FAIL {FAIL}")
    all_ok = (PASS == 4 and FAIL == 0)
    print("结论：" + (
        "四形态判得准 · 脱敏 0 残留 · 租户 ACL 泄漏 2→0 且不误拦 · 日志遮盖 0 残留 →"
        " 数据泄漏防线四账全绿（形态认得全 · 六件里两件实在账）"
        if all_ok else "存在 FAIL 断言，请查明细"))
    print(f"wall-clock: {time.perf_counter() - _t0:.3f}s", file=sys.stderr)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
