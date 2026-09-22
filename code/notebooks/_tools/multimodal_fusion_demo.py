# -*- coding: utf-8 -*-
"""
10-多模态 01 篇探针：三种融合路线（§10.1 · 🟢 了解 · 10 章开篇/总览）
核心命题：多模态演化主线 = 三条路线，路线 = 改动的是哪一层（编码器/对齐/解码器）
  A 桥接    ：冻结 LLM + 投影器        → 改动对齐层    (Flamingo/BLIP-2/早期 LLaVA)
  B 开放参数：视觉-语言联合指令微调     → 改动对齐+解码器 (LLaVA-1.5/Qwen-VL/InternVL)
  C 原生    ：统一 token 空间原生建模   → 改动解码器全网络 (GPT-4o/Gemini/Llama4)
纯 stdlib · 零网络 · 零随机 · 零第三方：全部判定=确定性函数，断言不过立即抛错。
用法: python code/notebooks/_tools/multimodal_fusion_demo.py
"""
import re
import sys
sys.stdout.reconfigure(encoding="utf-8")

PASS = 0
FAIL = 0


def ok():
    global PASS
    PASS += 1


bad = lambda: globals().__setitem__("FAIL", FAIL + 1)

# ---- 三路线定义（改动层矩阵：编码器/对齐/解码器） ----
ROUTES = {
    "A桥接": {
        "tag": "早期桥接：冻结 LLM + 投影器（快省，上限受制于桥的质量）",
        "encoder": "冻结（复用现成视觉编码器）",
        "alignment": "训练（投影器/Q-Former/交叉注意力）",
        "decoder": "冻结（LLM 权重不动）",
        "locus": "对齐层（桥）",
    },
    "B开放参数": {
        "tag": "视觉-语言联合指令微调：开放部分/全部参数（今天开源主流）",
        "encoder": "微调或冻结（Qwen-VL 大ViT微调 · LLaVA-1.5 ViT冻结）",
        "alignment": "训练（投影器参数全参与）",
        "decoder": "全参微调/RL（开放 LLM）",
        "locus": "对齐层 + 解码器（开放参数）",
    },
    "C原生": {
        "tag": "原生多模态 Native/Omnimodal：从预训练统一建模（融合最彻底、成本最高）",
        "encoder": "原生并入（统一 token 空间）",
        "alignment": "无独立桥（不需要投影器）",
        "decoder": "从预训练开始统一建模（改动解码器全网络）",
        "locus": "解码器全网络（无独立对齐层）",
    },
}

# ---- 演化谱系：桥质量→开放参数→原生（3 个里程碑节点 + 5 个关键代表） ----
LINEAGE = [
    "Flamingo(2022)", "BLIP-2(2023)", "LLaVA(2023)", "LLaVA-1.5(2024)",
    "Qwen-VL(2023-2025)", "InternVL(2024-2025)", "Gemini(2023-)", "GPT-4o(2024)",
]


def patch_token(size_px, patch=14):
    """一张图按 patch 网格切成多少 token（不含 [CLS]）"""
    g = size_px // patch
    return g * g


def qformer_save(vi_tokens, q_tokens):
    kv_saved = 1.0 - q_tokens / vi_tokens          # KV 条目留存倒比 -> 省下来比例
    attn_saved = 1.0 - (q_tokens * q_tokens) / (vi_tokens * vi_tokens)  # 注意力二次项
    return kv_saved, attn_saved


def classify_model(name):
    """代表性模型 → 融合路线（掌握度检验：架构差异对应哪种路线）"""
    if name in ("BLIP-2", "Flamingo", "早期LLaVA"):
        return "A桥接"
    if name in ("LLaVA-1.5", "Qwen-VL", "InternVL"):
        return "B开放参数"
    if name in ("GPT-4o", "Gemini", "Llama4"):
        return "C原生"
    return "未知"


def classify_scene(desc):
    """需求描述 → 首选路线（路线=改动哪层，工程场景=不动模型层）"""
    if any(k in desc for k in ("预算极小", "已有文本LLM", "快速补视觉", "免训")):
        return "A桥接"
    if any(k in desc for k in ("领域图表", "质量要求高", "有算力微调", "开源VLM", "UI操作", "文档理解", "截图")):
        return "B开放参数"
    if any(k in desc for k in ("旗舰", "实时语音", "全模态", "原生一致")):
        return "C原生"
    if any(k in desc for k in ("OCR引擎已够", "发票", "账单", "文档抽取")):
        return "工程(不动模型层)"
    return "未知"


print("== A 路线账：改动的是哪一层（§10.1 教学法约定）==")
try:
    assert set(ROUTES) == {"A桥接", "B开放参数", "C原生"} and len(ROUTES) == 3
    ok(); print("A1 三路线齐备 3/3：", " · ".join(ROUTES))
except AssertionError:
    bad(); print("A1 FAIL")

for r in ("A桥接", "B开放参数", "C原生"):
    try:
        assert ROUTES[r]["encoder"] and ROUTES[r]["alignment"] and ROUTES[r]["decoder"]
        ok()
        print("A2 %s 改动层：编码器[%s] · 对齐[%s] · 解码器[%s]" % (
            r, ROUTES[r]["encoder"], ROUTES[r]["alignment"], ROUTES[r]["decoder"]))
    except AssertionError:
        bad(); print("A2 %s FAIL" % r)

EXPECT_LOCUS = {"A桥接": "对齐层（桥）", "B开放参数": "对齐层 + 解码器（开放参数）",
                "C原生": "解码器全网络（无独立对齐层）"}
for r in ("A桥接", "B开放参数", "C原生"):
    try:
        assert ROUTES[r]["locus"] == EXPECT_LOCUS[r]
        ok()
        print("A3 %s 主改动层 → %s" % (r, ROUTES[r]["locus"]))
    except AssertionError:
        bad(); print("A3 %s FAIL" % r)

try:
    assert len(LINEAGE) == 8 and LINEAGE[0].startswith("Flamingo") and LINEAGE[-1].startswith("GPT-4o")
    assert LINEAGE.index("BLIP-2(2023)") < LINEAGE.index("LLaVA(2023)") < LINEAGE.index("LLaVA-1.5(2024)")
    ok(); print("A4 演化谱系 8 节点排序正确（桥→开放参数→原生）：", " → ".join(LINEAGE))
except AssertionError:
    bad(); print("A4 FAIL")

try:
    def both_frozen(r):
        return "冻结" in ROUTES[r]["encoder"] and "冻结" in ROUTES[r]["decoder"]
    assert both_frozen("A桥接") and not both_frozen("B开放参数") and not both_frozen("C原生")
    ok()
    print("A5 冻结矩阵：编码器+解码器双冻结的只有 A 桥接（B 的 LLM 被解冻、C 没有“冻结”概念）")
except AssertionError:
    bad(); print("A5 FAIL")

print("== B token 流水账：图像怎么变成 LLM 的输入 ==")
try:
    assert patch_token(224) == 256 and patch_token(336) == 576 and patch_token(448) == 1024
    ok()
    print("B1 一次性 patch 化：224px→%d tok · 336px→%d tok · 448px→%d tok（patch=14）"
          % (patch_token(224), patch_token(336), patch_token(448)))
except AssertionError:
    bad(); print("B1 FAIL")

try:
    # LLaVA-1.5 静态处理器：任意外观缩放后统一 336×336 中心裁剪 → 固定 576 tok
    assert patch_token(336) == 576
    ok(); print("B2 长宽比归一到 336×336 → 固定 576 tok（长边缩放+中心裁剪，造型输入一个尺寸）")
except AssertionError:
    bad(); print("B2 FAIL")

visual_share = patch_token(336) / (patch_token(336) + 200)
try:
    assert round(visual_share, 2) == 0.74
    ok()
    print("B3 视觉 prefill 占比：576 tok 图 + 200 tok 提示 → 序列 %.2f 都是视觉" % round(visual_share, 2))
except AssertionError:
    bad(); print("B3 FAIL")

kv_saved, attn_saved = qformer_save(256, 32)
try:
    assert round(kv_saved, 3) == 0.875 and round(attn_saved, 3) == 0.984
    ok()
    print("B4 BLIP-2 Q-Former 压缩 256→32 个 query token：KV 条目省 %.1f%% · 注意力二次项省 %.1f%%"
          % (kv_saved * 100, attn_saved * 100))
except AssertionError:
    bad(); print("B4 FAIL")

try:
    t576, t288, t144 = 576 * 576, 288 * 288, 144 * 144
    assert t576 == 331776 and t288 == 82944 and t144 == 20736
    assert round(t288 / t576, 2) == 0.25 and round(t144 / t576, 4) == 0.0625
    ok()
    print("B5 图为成本大头：token 减半 → 注意力二次项变 1/4（576²=%d → 288²=%d · 144²=%d → 0.0625×）"
          % (t576, t288, t144))
except AssertionError:
    bad(); print("B5 FAIL")

try:
    assert patch_token(336) // 2 == 288
    ok()
    print("B6 一张图≈一段文本：576 visual tok ÷ 2 tok/字（中文量级）≈ %d 个汉字——图不是“短输入”" % (patch_token(336) // 2))
except AssertionError:
    bad(); print("B6 FAIL")

try:
    assert patch_token(224, 14) == 256 and patch_token(224, 32) == 49
    ok()
    print("B7 patch 粒度是价格旋钮：224px 用 patch=14 → %d tok · patch=32 → %d tok（格子粗=省token=丢细节）"
          % (patch_token(224, 14), patch_token(224, 32)))
except AssertionError:
    bad(); print("B7 FAIL")

kv_saved2, attn_saved2 = qformer_save(576, 32)
try:
    assert round(attn_saved2, 3) == 0.997
    ok()
    print("B8 LLaVA 不压缩直接进 vs BLIP-2 Q-Former 压到 32：注意力二次项省 %.1f%%（576²=331776 → 32²=1024）"
          % (attn_saved2 * 100))
except AssertionError:
    bad(); print("B8 FAIL")

print("== C 决策账：三条路线的成本/参数/数据 + 场景选路线 ==")
COST = {"A桥接": 1, "B开放参数": 10, "C原生": 100}
try:
    assert COST["A桥接"] == 1 and COST["B开放参数"] == 10 and COST["C原生"] == 100
    assert COST["A桥接"] < COST["B开放参数"] < COST["C原生"]
    ok()
    print("C1 三档成本级：A 桥×1（只训对齐层） < B 开放参数×10（多模态 SFT/RL） < C 原生×100（从零预训练）")
except AssertionError:
    bad(); print("C1 FAIL")

PARAM = {"A桥接": 0.03, "B开放参数": 7.0, "C原生": 100.0}  # 可训参数教学量级（B）
try:
    assert PARAM["A桥接"] < PARAM["B开放参数"] < PARAM["C原生"]
    ok()
    print("C2 可训参数档：A 投影器 ≈%.2fB（7B LLM 冻结不训）· B 全参微调 ≈%.1fB · C 原生从零（规模量级 ≥%.0fB）"
          % (PARAM["A桥接"], PARAM["B开放参数"], PARAM["C原生"]))
except AssertionError:
    bad(); print("C2 FAIL")

DATA = {"A桥接": 0.56, "B开放参数": 0.67, "C原生": 100.0}  # 百万级教学量级
try:
    assert DATA["A桥接"] < DATA["B开放参数"] < DATA["C原生"]
    ok()
    print("C3 数据档：A 对齐样本 ≈%.2fM · B 视觉指令 ≈%.2fM · C 图文对+交错语料（百亿级，右列为教学量级）"
          % (DATA["A桥接"], DATA["B开放参数"]))
except AssertionError:
    bad(); print("C3 FAIL")

SCENES = [
    ("已有 7B 文本 LLM，预算极小，只想快速补上看图问答（不追求最优）", "A桥接"),
    ("要做领域图表问答产品，视觉质量要求高，且有算力/数据微调", "B开放参数"),
    ("旗舰全模态助手：实时语音+视频+文本原生一致交互", "C原生"),
    ("发票/账单 OCR 结构化，OCR 引擎已够用，只需把结果接 LLM 总结", "工程(不动模型层)"),
    ("公司知识库里的图表/截图都要问答，现有文本 RAG 升级成文档理解", "B开放参数"),
]
for i, (desc, want) in enumerate(SCENES, 1):
    try:
        assert classify_scene(desc) == want
        ok()
        print("C4.%d 场景选路线 %d/%d：%s... → %s" % (i, i, len(SCENES), desc[:14], want))
    except AssertionError:
        bad(); print("C4.%d FAIL" % i)

CLASSIC = [
    ("已有文本 LLM 且预算极小 快速补视觉", "BLIP-2 / 早期LLaVA", "A桥接"),
    ("领域图表问答 质量要求高 有算力微调", "Qwen2.5-VL / InternVL", "B开放参数"),
    ("旗舰实时语音全模态助手", "Gemini / GPT-4o", "C原生"),
]
for i, (need, model, want) in enumerate(CLASSIC, 1):
    try:
        assert classify_scene(need) == want
        ok()
        print("C5.%d 经典选型 %d/%d：%s → %s（%s）" % (i, i, len(CLASSIC), need[:12], model, want))
    except AssertionError:
        bad(); print("C5.%d FAIL" % i)

DECISION = {
    "A桥接": "已有强 LLM + 预算小 + 质量容忍 → 只训对齐层（桥）",
    "B开放参数": "有算力/数据 + 质量优先 → 开放 LLM 做多模态 SFT/RL",
    "C原生": "旗舰全模态 + 成本不敏感 → 从零统一定义输入空间",
}
SIG = {"A桥接": "只训对齐层", "B开放参数": "开放 LLM", "C原生": "从零"}
for r in ("A桥接", "B开放参数", "C原生"):
    try:
        assert SIG[r] in DECISION[r]
        ok(); print("C6 %s 决策方向 → %s" % (r, DECISION[r]))
    except AssertionError:
        bad(); print("C6 %s FAIL" % r)

def classify_arch(desc):
    """架构描述符 → 融合路线（掌握度检验：架构差异对应“哪种融合路线”的反向题）"""
    if any(k in desc for k in ("统一token", "全预训练")):
        return "C原生"
    if any(k in desc for k in ("Q-Former", "双冻结", "交叉注意力")):
        return "A桥接"
    if any(k in desc for k in ("SFT", "全参", "微调", "M-RoPE")):
        return "B开放参数"
    return "未知"


print("== D 掌握度检验账：模型→路线 + 发票 OCR→结构化管线 ==")
try:
    pairs = [("BLIP-2", "A桥接"), ("Flamingo", "A桥接"), ("早期LLaVA", "A桥接"),
             ("LLaVA-1.5", "B开放参数"), ("Qwen-VL", "B开放参数"), ("InternVL", "B开放参数"),
             ("GPT-4o", "C原生"), ("Gemini", "C原生"), ("Llama4", "C原生")]
    row = []
    for name, want in pairs:
        assert classify_model(name) == want, name
        ok(); row.append("%s→%s" % (name, want[0]))
    print("D1 架构→路线 9/9：", " · ".join(row))
except AssertionError:
    bad(); print("D1 FAIL")

ARCH = [
    ("ViT冻结+线性投影+LLM全参SFT", "B开放参数"),
    ("Q-Former+双冻结(编码器+LLM)", "A桥接"),
    ("统一token空间全预训练(GPT-4o式)", "C原生"),
    ("大ViT微调+M-RoPE+LLM全参(Qwen-VL式)", "B开放参数"),
    ("交叉注意力注入(Flamingo式)", "A桥接"),
]
at = 0
for desc, want in ARCH:
    try:
        assert classify_arch(desc) == want, desc
        ok(); at += 1
        print("D1.%d 架构描述→路线 %d/%d：%s → %s" % (at, at, len(ARCH), desc[:22], want))
    except AssertionError:
        bad(); print("D1.%d FAIL" % at)

# 发票 OCR→结构化输出（掌握度检验第 2 问的确定性样板：OCR 引擎输出 → 规则解析）
ocr_raw = "\n".join([
    "数电票 发票号码 33102233445566778899",
    "开票日期 2026-09-22",
    "购买方名称 上海某某科技有限公司",
    "销售方名称 北京某某信息服务有限公司",
    "合计金额 ￥1,258.00",
])
INV = {
    "发票号码": r"发票号码\s+(\d{20})",
    "开票日期": r"开票日期\s+(\d{4}-\d{2}-\d{2})",
    "购买方名称": r"购买方名称\s+(.+)",
    "销售方名称": r"销售方名称\s+(.+)",
    "合计金额": r"合计金额\s+￥?([\d,]+\.\d{2})",
}
fields = {}
for k, pat in INV.items():
    m = re.search(pat, ocr_raw)
    fields[k] = m.group(1) if m else None
try:
    assert fields["发票号码"] == "33102233445566778899"
    assert fields["开票日期"] == "2026-09-22"
    assert fields["购买方名称"] == "上海某某科技有限公司"
    assert fields["销售方名称"] == "北京某某信息服务有限公司"
    assert fields["合计金额"] == "1,258.00" and len(fields) == 5
    ok()
    print("D2 发票 OCR→结构化 5/5 字段：%s" % (" · ".join("%s=%s" % (k, v) for k, v in fields.items())))
except AssertionError:
    bad(); print("D2 FAIL")

ANCHOR = {
    "A桥接": "文档/OCR 免训快速补视觉 → 10-08 OCR 与文档理解管线",
    "B开放参数": "图表问答/UI 操作/文档理解 → 10-02 视觉组件 · 10-03 图文对齐 · 10-04 VLM 路线对比(🔴)",
    "C原生": "实时音视频全模态旗舰 → 10-06 视频与文本 · 10-07 音频与语音",
}
for r in ("A桥接", "B开放参数", "C原生"):
    try:
        assert r in ANCHOR
        ok()
        print("D3 %s 什么场景用得上 → %s" % (r, ANCHOR[r]))
    except AssertionError:
        bad(); print("D3 %s FAIL" % r)

print("\nSUMMARY  PASS %d · FAIL %d" % (PASS, FAIL))
assert FAIL == 0 and PASS >= 45
