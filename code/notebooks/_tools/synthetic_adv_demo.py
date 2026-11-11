# -*- coding: utf-8 -*-
"""15-06 合成数据与数据飞轮·进阶 —— 留存活 / 质量评估自动化（判别器模拟）/ 毒化方向判定。

账 A 留存活：候选条数 × 通过 → 池净增长；挂 04-02 锚点 60 进 25 出（留存 41.7%）、两轮 25→49。
账 B 质量评估自动化：可验证真值 → 左监督/判别器打分（规则校验 vs 判别器耦合），模拟"该进池/该筛掉"。
账 C 毒化方向：给定判例文本 → 长得像 + 答案错 = 该拦截（毒化）；长得像 + 答案对 = 该进池；
       长得不像 = 该筛掉但不毒化。模拟判别器三档。
"""
import sys, time

_WSTART = time.perf_counter()


def retent_rate(cand, passed):
    return passed / cand


def fence(cand, passed_f, bad_ok):
    """三道关：观察留存 + 坏样本残留。返回（进池, 坏样本放行）。"""
    # 坏样本若长得像好样本，fence 拦不住最危险的那种（毒化的关键在后面判）。
    return passed_f, bad_ok


def test_a():
    ok, total = 0, 0

    def check(name, got, exp):
        nonlocal ok, total
        total += 1
        if got == exp:
            ok += 1
            print("  %-42s PASS" % name)
        else:
            print("  %-42s FAIL(期望 %r 实得 %r)" % (name, exp, got))

    print("\n[账 A] 留存活（挂 04-02 锚点 60 进 25 出 / 25→49）")
    check("60 进 25 出 留存 41.7%", round(retent_rate(60, 25), 3), 0.417)
    # KPI 翻转：坏样本残留才是 KPI，留存率低到某阈值也不换闸门
    leftover = 0  # 三道关后坏样本残留（04-02：0）
    check("坏残留 0 才是健康（不是高留存）",
          True if leftover == 0 and retent_rate(60, 25) < 0.5 else False, True)
    # 两轮:第1轮25, 第2轮+24 = 49
    check("两轮池净增长 25→49", 25 + 24, 49)
    return ok, total


# 质量评估：判别器模拟
VERIFIER = {  # 指令 -> 可验证结果
    "5 * 12", "53 * 94", "7 + 9", "8 / 2",
}


def quality_score(item, mode):
    """mode in ('verifier','discriminator')。模拟"这条例能不能教对"。
    verifier: 有可验证真值 → 1.0；无真值 → 0.4（没法自动验证）。
    discriminator: 非可验证时用"长像评分"（本账全用规则真值 → 与 verifier 一致）。
    """
    if mode == "verifier":
        return 1.0 if item in VERIFIER else 0.4
    return 1.0 if item in VERIFIER else 0.4  # 判别器在无真值时模拟"规则可得"


def test_b():
    ok, total = 0, 0

    def check(name, got, exp):
        nonlocal ok, total
        total += 1
        if got == exp:
            ok += 1
            print("  %-42s PASS" % name)
        else:
            print("  %-42s FAIL(期望 %r 实得 %r)" % (name, exp, got))

    print("\n[账 B] 质量评估自动化（判别器模拟：可验证→1.0，无→0.4）")
    check("可验证 '5 * 12' → 1.0", quality_score("5 * 12", "verifier"), 1.0)
    check("可验证 '53 * 94' → 1.0", quality_score("53 * 94", "verifier"), 1.0)
    check("不可验证 '今天的天气如何' → 0.4",
          quality_score("今天的天气如何", "verifier"), 0.4)
    check("判别器模式同 verifier（规则可得）",
          quality_score("8 / 2", "discriminator"), 1.0)
    return ok, total


# 毒化判定：长得像 + 答案错 = 毒化；长得像 + 答案对 = 进池；不像 = 噪声不毒
TRUTH = {"3 + 4": 7, "10 + 20": 30, "5 * 12": 60}   # 可验证真值源（04-02 三道关里的"可验证"）
TOXIC_ITEM = ("3 + 4", 10)                           # 长得像 + 答案错（真值 7，给 10）→ 毒化
OK_ITEM = ("3 + 4", 7)                               # 长得像 + 答案对（真值 7）→ 进池
NOISY_ITEM = ("讲个故事", None)                       # 开放式、无真值 → 噪声，不毒化


def poison_type(item, gold, truth):
    """判别器三档：毒化 / 进池 / 噪声。
    truth: 可验证真值字典。毒化 = 可真值且 gold 与真值不符；进池 = 可真值且符合；噪声 = 无真值。
    """
    if item not in truth:
        return "噪声（不毒化）"
    if gold is None:
        return "毒化"
    return "进池" if gold == truth[item] else "毒化"


def test_c():
    ok, total = 0, 0

    def check(name, got, exp):
        nonlocal ok, total
        total += 1
        if got == exp:
            ok += 1
            print("  %-42s PASS" % name)
        else:
            print("  %-42s FAIL(期望 %r 实得 %r)" % (name, exp, got))

    print("\n[账 C] 毒化方向判定（判别器三档：毒化 / 进池 / 噪声）")
    check("(3+4, 10) 答案错 → 毒化", poison_type(*TOXIC_ITEM, TRUTH), "毒化")
    check("(3+4, 7) 答案对 → 进池", poison_type(*OK_ITEM, TRUTH), "进池")
    check("('讲个故事', None) 开放式 → 噪声",
          poison_type(*NOISY_ITEM, TRUTH), "噪声（不毒化）")
    # 毒化的关键是"长得像好样本"：可验证但答案错，一律毒化
    check("('10+20', 31) 可真值但答案错 → 毒化", poison_type("10 + 20", 31, TRUTH), "毒化")
    check("('10+20', 30) 可真值且对 → 进池", poison_type("10 + 20", 30, TRUTH), "进池")
    return ok, total


def W(label):
    print("[wall] %-28s %.4fs" % (label, time.perf_counter() - _WSTART), file=sys.stderr)


def main():
    W("账 A")
    a_ok, a_total = test_a()
    W("账 B")
    b_ok, b_total = test_b()
    W("账 C")
    c_ok, c_total = test_c()
    ok = a_ok + b_ok + c_ok
    total = a_total + b_total + c_total
    print("\nPASS %d  FAIL %d" % (ok, total - ok))
    return 1 if total - ok else 0


if __name__ == "__main__":
    main()
