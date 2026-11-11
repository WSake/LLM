#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""12-05 幻觉缓解 · 工程解法（§12.4）——三招缓解的判别性双臂对照（纯 stdlib 确定性探针）

与 11-07 分工：11-07 交付**检测三法**（事实核对/NLI/采样一致），本篇交付**拦截机制**——
在"检测到低可靠"之后，系统**怎么处置**。四账：
  账 A  幻觉三层判分（知识幻觉 / 逻辑幻觉 / 引用幻觉 · 5 样本 5/5）
  账 B  「未命中即拒答」中断判定（依据水位闸线 · 挂 14-03 锚点）
  账 C  三招组合拳 → 三闸拦截（RAG 引用 → 检索校验 → 自我一致投票）
  账 D  高风险任务人工兜底（安全事件>0 不上生产 · 挂 09-07 锚点）

输出确定性；stdout 三进程字节全同。
"""
import sys

# ---------------------------------------------------------------------------
# 账 A 数据：幻觉三层（§12.4）。每条 = (实际层, 判据特征, 期望层)
# ---------------------------------------------------------------------------
HALLU = [
    ("知识幻觉", "事故报告 H-1139", "知识幻觉"),
    ("知识幻觉", "2024 年 Q4 财报", "知识幻觉"),
    ("逻辑幻觉", "因为 A>B 跳步到 C>D", "逻辑幻觉"),
    ("引用幻觉", "《ACM 学报》2017 论文", "引用幻觉"),
    ("引用幻觉", "《经济学人》2023-04 期", "引用幻觉"),
]


def classify(feature):
    """按特征词判幻觉层：论文/期 → 引用层；『跳步』→ 逻辑层；其余 → 知识层。"""
    if "论文" in feature or "期" in feature:
        return "引用幻觉"
    if "跳步" in feature:
        return "逻辑幻觉"
    return "知识幻觉"


# ---------------------------------------------------------------------------
# 账 B 依据水位：片段是否带『知识库证据』标记
# ---------------------------------------------------------------------------
KB_MARK = "我的知识库提到"


def has_kb(text):
    return KB_MARK in text


# ---------------------------------------------------------------------------
# 账 C 三闸
# ---------------------------------------------------------------------------
def has_source_id(text):
    return "KB-" in text            # 闸 1：RAG 引用带来源 ID


def number_ok(text):
    return "_NUM_OK" in text        # 闸 2：实体-数值命中原文


def agree(text):
    return "吻合" in text           # 单次采样：内容与依据一致则为 "吻合"


def vote(snippets):
    """自我一致投票（SelfCheckGPT 口径）：多次采样的吻合票数 > 半数才放行。"""
    agrees = sum(1 for s in snippets if agree(s))
    return agrees * 2 > len(snippets)


def pipeline(snippets):
    """三闸合成：任一审不放行 → 拒答。"""
    if not any(has_source_id(s) for s in snippets):
        return False
    if not any(number_ok(s) for s in snippets):
        return False
    if not vote(snippets):
        return False
    return True


# ---------------------------------------------------------------------------
# 主账
# ---------------------------------------------------------------------------
def main():
    results = []

    def check(name, cond):
        results.append((name, cond))

    # 账 A：幻觉三层判分 5/5
    a_ok = sum(1 for _act, feat, exp in HALLU if classify(feat) == exp)
    check(f"账A 幻觉三层判分 {a_ok}/5（知识/逻辑/引用）", a_ok == 5)

    # 账 B：未命中即拒答
    hit = any(has_kb(t) for t in ["我的知识库提到：故宫午门 8:30-17:00", "夏季 17:00 闭馆"])
    miss = not any(has_kb(t) for t in ["故宫是明朝建的", "坐地铁可到"])
    check(f"账B 未命中即拒答：命中放行 {hit} / 未命中拒答 {miss}", hit and miss)

    # 账 C：三闸拦截（合法 / 换数 / 无源 / 低一致）
    legal = pipeline(["KB-108: 实体-数值 _NUM_OK 吻合 100 亿", "KB-109: 实体-数值 _NUM_OK 吻合 100 亿"])
    swap = pipeline(["KB-108: 实体-数值 3 亿 违和", "KB-109: 实体-数值 3 亿 违和"])       # 闸2
    nosrc = pipeline(["无检索结果", "无检索结果"])                                        # 闸1
    loose = pipeline(["KB-108: 实体-数值 _NUM_OK 吻合", "KB-109: 实体-数值 _NUM_OK 矛盾"])   # 闸3
    check(f"账C 三闸拦截：合法 {legal} / 换数拒 {not swap} / 无源拒 {not nosrc} / 低一致拒 {not loose}",
          legal and not swap and not nosrc and not loose)

    # 账 D：安全事件>0 不上生产 → 人工兜底
    def auto_gate(risk):
        return "人工兜底" if risk["事件数"] > 0 else "上生产"

    d1 = auto_gate({"事件数": 0})
    d2 = auto_gate({"事件数": 2})
    check(f"账D 安全事件>0 闸线：低风险 {d1} / 高风险 {d2}", d1 == "上生产" and d2 == "人工兜底")

    ok = sum(1 for _, c in results if c)
    fail = len(results) - ok
    for name, cond in results:
        print(f"{'PASS' if cond else 'FAIL'} | {name}")
    print(f"RESULT: PASS {ok} · FAIL {fail}")
    sys.exit(0 if (ok == 4 and fail == 0) else 1)


if __name__ == "__main__":
    main()
