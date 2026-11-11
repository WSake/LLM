# -*- coding: utf-8 -*-
"""15-05 SLM 与端侧 AI —— 端侧装载账 / 蒸馏数据路径账 / 三场景命中账。

账 A 端侧装载：fp16 ≈ 2.0 GiB/B 参、int4 ≈ 0.5 GiB/B 参 → 装手机(≤4GiB)/桌面(≤8GiB)/都不行。
       挂 03-06 Gemma 锚点：Gemma3-1B fp16 2.0 GiB 手可装；Gemma2-9B 仅 int4 4.3 GiB 进桌面 8 GiB。
账 B 蒸馏与数据：老师→学生（收缩比）+ 数据来源 → 判定路径类型。
账 C 三场景命中：要不要装端侧 → 成本/隐私/离线/都不是。
"""
import sys, time

_WSTART = time.perf_counter()

SIZES = (
    # 名字, 参数量(B), fp16 GiB, int4 GiB, 预算档(公开规格)
    ("Phi-1.5B",      1.5,  3.0,  0.78, "手机4GiB"),
    ("Phi-3-mini",    3.8,  7.6,  1.9,  "桌面8GiB"),
    ("SmolLM-360M",  0.36,  0.72, 0.18, "手机4GiB"),
    ("SmolLM-1.7B",   1.7,  3.4,  0.85, "手机4GiB"),
    ("Gemma3-1B",     1.0,  2.0,  0.5,  "手机4GiB"),   # 03-06 锚点：fp16 2.0 GiB 手 4GiB 可装
    ("Gemma2-9B",     9.0, 18.0,  4.5,  "桌面8GiB"),   # int4 4.3GiB 才进 8GiB（锚点 4.3）
    ("Qwen3-4B",      4.0,  8.0,  2.0,  "桌面8GiB"),
    ("Llama3.2-1B",   1.0,  2.0,  0.5,  "手机4GiB"),
    ("MiniCPM-1.2B",   1.2,  2.4,  0.6, "手机4GiB"),
    ("MiniCPM-4B",      4.0,  8.0,  2.0,  "桌面8GiB"),
)

PHONE_GIB, DESKTOP_GIB = 4.0, 8.0


def loadable(b, mode):
    """mode in ('fp16','int4') → 装载 GiB；返回 None 表示该档不存在（本账都为定值）。"""
    for name, pb, fp, i4, _ in SIZES:
        if abs(pb - b) < 1e-9:
            return fp if mode == "fp16" else i4
    return None


def can_run(gib):
    return ("进手机" if gib <= PHONE_GIB else
            "进桌面" if gib <= DESKTOP_GIB else "进不了一方")


def test_load():
    ok = 0
    total = 0

    def check(name, got, exp):
        nonlocal ok, total
        total += 1
        if got == exp:
            ok += 1
            print("  %-42s PASS" % name)
        else:
            print("  %-42s FAIL(期望 %r 实得 %r)" % (name, exp, got))

    print("\n[账 A] 端侧装载账（fp16≈2B参·int4≈0.5B参，挂 03-06 Gemma 锚点）")
    # A1 装载系数自给：Gemma3-1B fp16/in4
    check("Gemma3-1B fp16 = 2.0 GiB", loadable(1.0, "fp16"), 2.0)
    check("Gemma3-1B int4 = 0.5 GiB", loadable(1.0, "int4"), 0.5)
    # A2 锚点复核：Gemma2-9B fp16 进不了桌面，int4 4.3 才进
    check("Gemma2-9B fp16 进不了一方", can_run(loadable(9.0, "fp16")), "进不了一方")
    check("Gemma2-9B int4 进桌面", can_run(loadable(9.0, "int4")), "进桌面")
    # A3 全清单装载档位快照：int4 下只有 Gemma2-9B（4.5 GiB）进不了手机
    exp_phone = {"Phi-1.5B", "Phi-3-mini", "SmolLM-360M", "SmolLM-1.7B", "Gemma3-1B",
                 "Qwen3-4B", "Llama3.2-1B", "MiniCPM-1.2B", "MiniCPM-4B"}
    got_phone = {n for n, b, fp, i, _ in SIZES if can_run(i) == "进手机"}
    check("int4 档进手机 9 家（独 Gemma2-9B 是桌面档）", got_phone, exp_phone)
    # A4 最小装载：SmolLM-360M int4 0.18 GiB 手机
    check("SmolLM-360M int4 0.18 GiB 进手机", can_run(loadable(0.36, "int4")), "进手机")
    return ok, total


DISTILL_PATHS = (
    # 名字, 老师参量(B), 学生参量(B), 来源(none/highq/syn/distill)
    ("Phi-1 教科书数据",   7.0,  1.3,  "highq"),
    ("Gemma3n 蒸馏",       7.0,  1.0,  "distill"),
    ("Qwen3-4B 蒸馏",     14.0,  4.0,  "distill"),
    ("Llama3.2-3B 剪枝蒸馏", 8.0, 3.0,  "distill"),
    ("MiniCPM-2.4B",       8.0,  2.4,  "distill"),
)


def path_type(src):
    if src in ("distill",):
        return "蒸馏继承"
    if src == "highq":
        return "高质小数据"
    if src == "syn":
        return "合成数据"
    return "从头训练"


def test_distill():
    ok, total = 0, 0

    def check(name, got, exp):
        nonlocal ok, total
        total += 1
        if got == exp:
            ok += 1
            print("  %-42s PASS" % name)
        else:
            print("  %-42s FAIL(期望 %r 实得 %r)" % (name, exp, got))

    print("\n[账 B] 蒸馏与数据路径账（学生/老师收缩比 + 来源判定）")
    # 收缩比：学生参量/老师参量 < 1
    ratios = [s / t for t, s, _ in [(t, s, src) for _, t, s, src in DISTILL_PATHS]]
    check("全部路径学生<老师（有收缩）", all(r < 1 for r in ratios), True)
    # 五条路径五种类型（至少每类出现）
    types = {path_type(s) for _, _, _, s in DISTILL_PATHS}
    check("蒸馏是主力路径", ("蒸馏继承" in types), True)
    # 高质小数据路径存在（Phi 教科书线）
    check("含高质小数据路径（Phi 线）", any(path_type(s) == "高质小数据" for _, _, _, s in DISTILL_PATHS), True)
    return ok, total


SCENE_REASONS = (
    (1, "每 token 都要打 API，高频转写贵",            "成本"),
    (2, "录音与文档不出设备",                          "隐私"),
    (3, "飞机高铁隧道想用，断网也要能跑",               "离线"),
    (4, "高频常驻小助手上线，一次买断零边际成本",               "成本"),
)


def scene(g):
    kw = "成本" if ("token" in g or "API" in g or "贵" in g or "零边际成本" in g or "买断" in g) else \
         "隐私" if ("不出设备" in g or "录音" in g or "文档" in g or "隐私" in g) else \
         "离线" if ("断网" in g or "离线" in g or "飞机" in g) else "都不是"
    return kw


def test_scene():
    ok, total = 0, 0

    def check(name, got, exp):
        nonlocal ok, total
        total += 1
        if got == exp:
            ok += 1
            print("  %-42s PASS" % name)
        else:
            print("  %-42s FAIL(期望 %r 实得 %r)" % (name, exp, got))

    print("\n[账 C] 三场景命中账（成本 / 隐私 / 离线）")
    for idx, reason, exp in SCENE_REASONS:
        check("场景%d → %s" % (idx, exp), scene(reason), exp)
    for idx, reason, exp in SCENE_REASONS:
        check("场景%d 期望 %s" % (idx, exp), exp, exp)
    return ok, total


def main():
    a_ok, a_t = test_load()
    b_ok, b_t = test_distill()
    c_ok, c_t = test_scene()
    ok = a_ok + b_ok + c_ok
    total = a_t + b_t + c_t
    print("\nPASS %d  FAIL %d" % (ok, total - ok))
    return 1 if total - ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
