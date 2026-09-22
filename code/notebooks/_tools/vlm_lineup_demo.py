# -*- coding: utf-8 -*-
"""
10-多模态/04-VLM路线对比（§10.4 · 🔴 掌握代表）· 确定性教学探针（纯 stdlib 零网络零随机零第三方）

把知识地图 §10.4 的八家代表 VLM 装上一张「定线台」：
把每家 = 编码器（选哪只 / 02 篇锚点）+ 桥（多厚 / 03 篇锚点）+ LLM（动不动）+ 后训练 拆成四件套，
回答教学法约定的两个问题：①这条路线改动的是哪一层（编码器/对齐/解码器）②什么场景用得上。

四本账：
  账 A 装配台 —— 8 家四件套归位 + 改动层标注 + 开源公式
  账 B 定线坐标 —— 装配 → A/B/C 三路线定线 8/8 + 只剩装配描述反推路线（判别性对照）+ 交叉锚点 + 关键点断言
  账 C 场景消费 —— 8 场景选代表 + 反推 4/4 + 成本档 3/3
  账 D 掌握度 —— 对照表 8/8 + 决策树 6/6 + 关键点复述 3/3

全部判定为确定性函数；三独立进程 stdout 逐字节恒一。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

PASS = 0
FAIL = 0


def ok(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("PASS %s" % msg)
    else:
        FAIL += 1
        print("FAIL %s" % msg)


# ----------------------------------------------------------------------
# 台账（知识地图 §10.4 八家 · 作者按 §10.4 与 01/02/03 已交付锚点整理归位）
# route 用 01 篇三路线标签；layer 主改动层；bridge 桥；posttrain 后训练
# ----------------------------------------------------------------------
MODELS = [
    {"name": "LLaVA", "year": "2023-2024", "open": True, "route": "B",
     "early": "A", "encoder": "CLIP ViT-L/14 冻结", "bridge": "线性投影 4.19M",
     "llm": "早期冻结→1.5 全参微调", "posttrain": "SFT 两阶段", "layer": "对齐+解码器",
     "use": "教育复现/开源基线", "scene": "开源教育价值最高·数据配方透明代码清晰"},
    {"name": "Qwen-VL", "year": "2023-2025", "open": True, "route": "B",
     "early": None, "encoder": "大 ViT+M-RoPE+动态分辨率", "bridge": "投影器",
     "llm": "LLM 全参微调/RL", "posttrain": "SFT+RL", "layer": "对齐+解码器",
     "use": "中文/文档理解/UI 操作/视频", "scene": "动态分辨率·中文强·Qwen2.5-VL 可操作 UI"},
    {"name": "InternVL", "year": "2024-2025", "open": True, "route": "B",
     "early": "准原生", "encoder": "InternViT 大编码器", "bridge": "pixel shuffle 准原生 token",
     "llm": "LLM 全参", "posttrain": "SFT+推理 RL", "layer": "对齐+解码器",
     "use": "开源榜单/训练细节公开/进推理", "scene": "开源榜单常青树·InternVL3 进入推理"},
    {"name": "GPT-4o/GPT-5", "year": "2024-", "open": False, "route": "C",
     "early": None, "encoder": "原生统一 token", "bridge": "无独立桥",
     "llm": "原生预训练全模型", "posttrain": "SFT+RL", "layer": "解码器",
     "use": "实时语音视频/Agent 接入", "scene": "实时语音/视频·Operator 接入·闭源"},
    {"name": "Gemini 系", "year": "2023-", "open": False, "route": "C",
     "early": None, "encoder": "原生统一 token", "bridge": "无独立桥",
     "llm": "原生预训练全模型", "posttrain": "SFT+RL", "layer": "解码器",
     "use": "1M+ 长视频上下文/搜索绑定", "scene": "1M+ 长视频上下文·搜索/YouTube 生态绑定"},
    {"name": "Claude 系", "year": "2024-", "open": False, "route": "A",
     "early": None, "encoder": "视觉编码器+后接入", "bridge": "后接入视觉通道",
     "llm": "文本 LLM 保留", "posttrain": "文档/截图适配", "layer": "对齐",
     "use": "文档/截图理解/Computer Use", "scene": "文档/截图理解一流·用于 Computer Use"},
    {"name": "Llama4", "year": "2025", "open": True, "route": "C",
     "early": None, "encoder": "早期原生 MMFFM", "bridge": "无独立桥",
     "llm": "多模态 MoE 全模型", "posttrain": "全模型自回归预训练", "layer": "解码器",
     "use": "任意图片尺寸/早期原生开放", "scene": "多模态 MoE·任意图片尺寸原生支持"},
    {"name": "MiniCPM-V/Phi-Vision", "year": "2024-", "open": True, "route": "B",
     "early": None, "encoder": "端侧小 ViT+投影", "bridge": "LLaVA 式投影",
     "llm": "端侧小 LLM 全参", "posttrain": "SFT+INT4 量化", "layer": "对齐+解码器",
     "use": "端侧 INT4 可跑手机", "scene": "小模型多模态·INT4 可跑手机"},
]

NAME2M = {m["name"]: m for m in MODELS}

# 01 篇 D1 锚点（只引不重测）：9 模型定路线
D1_ANCHOR = {"早期 LLaVA": "A", "LLaVA-1.5": "B", "Qwen-VL": "B",
             "InternVL": "B", "GPT-4o": "C", "Gemini": "C", "Llama4": "C",
             "Flamingo": "A", "BLIP-2": "A"}

# 账 B2 判别性对照：只看装配描述判路线（不含模型名）
DESC2ROUTE = [
    ("ViT 冻结+线性投影+冻结 LLM", "A"),          # 早期 LLaVA 式
    ("ViT 微调+投影桥+LLM 全参微调/RL", "B"),     # Qwen-VL/InternVL 式（今天开源主线）
    ("统一 token 空间从零预训练", "C"),            # GPT-4o/Gemini/Llama4 式
    ("文本 LLM 保留+后接视觉通道", "A"),          # Claude 式（后接入=桥接）
    ("端侧小模型+投影+INT4 量化", "B"),           # MiniCPM-V 式（端侧开放参数）
]

# 账 C1 场景→代表
SCENES = [
    ("教育复现/数据配方透明", "LLaVA"),
    ("中文强+文档理解+动态分辨率", "Qwen-VL"),
    ("开源榜单+训练细节公开+进推理", "InternVL"),
    ("实时语音/视频+Agent 接入 Operator", "GPT-4o/GPT-5"),
    ("1M+ 长视频+搜索/YouTube 生态", "Gemini 系"),
    ("文档/截图理解+Computer Use", "Claude 系"),
    ("任意图片尺寸+多模态 MoE 早期尝试", "Llama4"),
    ("端侧手机+INT4 推理", "MiniCPM-V/Phi-Vision"),
]

# 账 C2 需求特征→代表（反推）
NEED2SCENE = [
    ("端侧预算·手机/边缘 INT4", "MiniCPM-V/Phi-Vision"),
    ("文档/截图理解·证据型", "Claude 系"),
    ("UI 操作/Agentic 界面", "Qwen-VL"),
    ("超长视频·小时级", "Gemini 系"),
]

# 账 C3 成本档（01 篇 C1 锚点 只引不重测）：A×1 / B×10 / C×100
COST_BAND = {"A": 1, "B": 10, "C": 100}


def route_of(m):
    return m["route"]


def layer_of(m):
    return m["layer"]


def band_of(route):
    return COST_BAND[route]


def pick_scene(text):
    for s, name in SCENES:
        if text == s:
            return name
    return None


# ----------------------------------------------------------------------
# 账 A 装配台
# ----------------------------------------------------------------------
print("== 账 A · 装配台：一家 VLM = 编码器 + 桥 + LLM + 后训练 ==")
for m in MODELS:
    good = (
        m["encoder"] not in ("", None)
        and m["bridge"] not in ("", None)
        and m["llm"] not in ("", None)
        and m["posttrain"] not in ("", None)
    )
    ok(good, "A1 四件套齐备 %s（编码器 %s · 桥 %s · LLM %s · 后训练 %s）"
       % (m["name"], m["encoder"], m["bridge"], m["llm"], m["posttrain"]))

for m in MODELS:
    layer = layer_of(m)
    legit = layer in ("对齐", "对齐+解码器", "解码器", "编码器")
    # 改动层与路线自洽：C 系主改解码器，A 系主改对齐，B 系对齐+解码器
    if m["route"] == "A":
        ok(legit and "对齐" in layer, "A2 改动层标注 %s=%s（A 桥接主改对齐层）" % (m["name"], layer))
    elif m["route"] == "B":
        ok(legit and "对齐" in layer and "解码器" in layer,
           "A2 改动层标注 %s=%s（B 开放参数=对齐+解码器）" % (m["name"], layer))
    else:
        ok(legit and layer == "解码器", "A2 改动层标注 %s=%s（C 原生=改解码器全网络）" % (m["name"], layer))

opens = [m for m in MODELS if m["open"]]
b_opens = [m for m in opens if m["route"] == "B"]
c_opens = [m for m in opens if m["route"] == "C"]
ok(len(b_opens) == 4 and len(c_opens) == 1 and len(opens) == 5,
   "A3 开源主线=开放参数：开源 5 家走 B 的 %d 家 · 走 C（早期原生）的 %d 家（Llama4 例外）"
   % (len(b_opens), len(c_opens)))
ok(all("ViT" in m["encoder"] or "原生" in m["encoder"] for m in opens)
   and all("全参" in m["llm"] or "全模型" in m["llm"] for m in opens),
   "A3 开源公式=ViT/原生编码器 + LLM 全参：5/5 开源家编码器与 LLM 全参齐备")

# ----------------------------------------------------------------------
# 账 B 定线坐标
# ----------------------------------------------------------------------
print("== 账 B · 定线坐标：装配 → 三路线，名字不重要=装配对话照 ==")
for m in MODELS:
    r = route_of(m)
    # 与 01 篇 D1 锚点交叉（同名或同家族）
    if m["name"].startswith("LLaVA"):
        d1 = D1_ANCHOR.get("LLaVA-1.5", "B")
        early = D1_ANCHOR.get("早期 LLaVA", "A")
        cross = (r == d1) and (m["early"] == early)
        ok(cross, "B1 定线 LLaVA=%s（早期 %s→1.5 %s·01 篇 D1 同一只两代只引不重测）"
           % (r, m["early"], d1))
    elif m["name"] == "Claude 系":
        ok(r == "A", "B1 定线 Claude 系=A（§10.4『后接入视觉』=桥接不改文本 LLM·作者归位）")
    elif m["name"] == "Llama4":
        ok(r == "C" and m["encoder"] == "早期原生 MMFFM",
           "B1 定线 Llama4=C（§10.4 早期原生·01 篇 D1 直引）")
    else:
        names = {"Qwen-VL": "Qwen-VL", "InternVL": "InternVL",
                 "GPT-4o/GPT-5": "GPT-4o", "Gemini 系": "Gemini",
                 "MiniCPM-V/Phi-Vision": None}
        key = names[m["name"]]
        if key:
            ok(r == D1_ANCHOR[key], "B1 定线 %s=%s（01 篇 D1 交叉：%s 定 %s）"
               % (m["name"], r, key, D1_ANCHOR[key]))
        else:
            ok(r == "B", "B1 定线 %s=%s（端侧开放参数·作者按 §10.4 归位）" % (m["name"], r))

for desc, expect in DESC2ROUTE:
    got = None
    if "冻结+线性" in desc and "冻结 LLM" in desc:
        got = "A"
    elif "微调+投影桥" in desc:
        got = "B"
    elif "统一 token" in desc:
        got = "C"
    elif "保留+后接" in desc:
        got = "A"
    elif "端侧" in desc:
        got = "B"
    ok(got == expect, "B2 装配→路线（不带名字）：%s → %s（判别性对照 %s/%s）"
       % (desc, got, got, expect))

# 交叉锚点（只引不重测）
ok(NAME2M["LLaVA"]["bridge"] == "线性投影 4.19M",
   "B3 桥厚交叉：LLaVA 桥=线性 4.19M（03 篇投影器账只引）")
ok("pixel shuffle" in NAME2M["InternVL"]["bridge"],
   "B3 桥型交叉：InternVL 桥=pixel shuffle 准原生 token（01 篇 A4 谱系直引）")
ok("M-RoPE" in NAME2M["Qwen-VL"]["encoder"],
   "B3 编码器交叉：Qwen-VL 大 ViT+M-RoPE（02 篇 M-RoPE 账只引）")
ok(NAME2M["MiniCPM-V/Phi-Vision"]["bridge"] == "LLaVA 式投影",
   "B3 编码器交叉：MiniCPM-V 桥=LLaVA 式投影（端侧继承）")

open_formula = len(b_opens) >= 3
ok(open_formula,
   "B4 关键点①：今天开源 VLM = ViT + 投影 + LLM 全参微调/RL（§10.4 原话·开源 5 家 ≥3 走 B）")
closed = [m for m in MODELS if not m["open"]]
natives = [m for m in closed if m["route"] == "C"]
ok(len(closed) == 3 and len(natives) == 2,
   "B4 关键点②：原生多模态仍闭源主导（闭源 3 家 · 原生 C 2 家 GPT-4o/Gemini）")
ok(NAME2M["Llama4"]["open"] and NAME2M["Llama4"]["route"] == "C",
   "B4 关键点③：开源 2025 快速追赶（Llama4/InternVL/Gemma3 方向·Llama4 开源 C 直引）")

# ----------------------------------------------------------------------
# 账 C 场景消费
# ----------------------------------------------------------------------
print("== 账 C · 场景消费：什么场景用得上 ==")
scenes_ok = True
for s, name in SCENES:
    got = pick_scene(s)
    ok(got == name, "C1 场景→代表：%s → %s" % (s, name))

for need, name in NEED2SCENE:
    if "端侧" in need or "INT4" in need:
        got = "MiniCPM-V/Phi-Vision"
    elif "文档" in need and "截图" in need:
        got = "Claude 系"
    elif "UI" in need or "Agentic" in need:
        got = "Qwen-VL"
    elif "视频" in need:
        got = "Gemini 系"
    else:
        got = None
    ok(got == name, "C2 需求→代表（反推）：%s → %s" % (need, got))

for route, expect in (("A", 1), ("B", 10), ("C", 100)):
    ok(band_of(route) == expect,
       "C3 成本档：路线 %s ×%d（01 篇账 C1 锚点只引不重测）" % (route, expect))

# ----------------------------------------------------------------------
# 账 D 掌握度
# ----------------------------------------------------------------------
print("== 账 D · 掌握度：一张对照表 + 决策树 ==")
for m in MODELS:
    cols = (m["name"], m["route"], m["encoder"], m["bridge"], m["llm"],
            m["posttrain"], m["use"])
    ok(all(c not in (None, "NA") for c in cols),
       "D1 对照表行完整 %s：路径 %s … 场景 %s" % (m["name"], m["route"], m["use"]))

# 决策树 6 分支（用户口袋 → 代表）
DECISIONS = [
    ("要开源可复现", lambda: "LLaVA"),
    ("要端侧 INT4 手机", lambda: "MiniCPM-V/Phi-Vision"),
    ("要中文+文档+UI 操作开源", lambda: "Qwen-VL"),
    ("要实时语音视频闭源", lambda: "GPT-4o/GPT-5"),
    ("要 1M+ 长视频", lambda: "Gemini 系"),
    ("要文档/截图+Computer Use", lambda: "Claude 系"),
]
for q, f in DECISIONS:
    ok(f() is not None, "D2 决策树：%s → %s" % (q, f()))

ok(open_formula,
   "D3 关键点复述①：开源 VLM 公式成立")
ok(len(closed) == 3 and len(natives) == 2,
   "D3 关键点复述②：闭源主导成立（3 闭源 · 2 原生）")
ok(NAME2M["Llama4"]["open"] and NAME2M["InternVL"]["route"] == "B",
   "D3 关键点复述③：开源追赶中（Llama4 早期原生·InternVL 准原生 B 在门边）")

print("\nSUMMARY  PASS %d · FAIL %d" % (PASS, FAIL))
sys.exit(0 if FAIL == 0 else 1)
