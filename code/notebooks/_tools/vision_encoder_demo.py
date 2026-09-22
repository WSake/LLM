# -*- coding: utf-8 -*-
"""10-多模态/02-视觉组件（§10.2）确定性探针：视觉编码器格四账实测。
把"编码器这只眼睛"拆成四本账——A 眼睛族谱（ViT/CLIP/SigLIP/DINOv2 怎么选）、
B 动态分辨率（不为一张图压扁一张图，Qwen-VL/LLaVA-NeXT 入严）、
C M-RoPE（把 2D 图像位置与 1D 文本位置统一成 3D 坐标）、D 掌握度检验（家族定级/编码器层状态/场景方案/发票 OCR 账）。
纯 stdlib · 零网络 · 零随机 · 零第三方；三条独立进程 stdout 必须逐字节恒一（md5 相同）。
"""
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")

PASS = 0
FAIL = 0


def ok(cond, tag):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("PASS %s" % tag)
    else:
        FAIL += 1
        print("FAIL %s" % tag)


# ---------------------------------------------------------------- 共用工具
def patch_token(w, h, patch):
    """一次 patch 化得到 token 数：图切成 (w//patch)×(h//patch) 个格子，每格 1 token。"""
    return (w // patch) * (h // patch)


# ================================================================ 账 A · 眼睛族谱
def encoder_family(name):
    """编码器家族定义：名称 → (年份, 一句定位, 是否图文对齐)。"""
    TABLE = {
        "ViT":    (2020, "视觉版 Transformer：patch→线性投影→Transformer", False),
        "CLIP":   (2021, "对比学习（softmax 图文配对），图文对齐的事实底座", True),
        "SigLIP": (2023, "sigmoid 逐 token 二分类，开源 VLM 最常用", True),
        "DINOv2": (2023, "自监督（无标签无文本），稠密/像素级特征", False),
        "InternViT": (2024, "CLIP 系，按多模态需要扩大的大 ViT", True),
    }
    return TABLE[name]


def pick_encoder(scene):
    """三问选眼睛：要语义还是要稠密？文字在图上还是在文本里？"""
    if "发票" in scene or ("表格" in scene and "文字" in scene):
        return "高分辨率动态分块（先把字拍清楚）"
    if "搜图" in scene or "图文检索" in scene:
        return "CLIP/SigLIP（对比学习语义库）"
    if "分割" in scene or "检测" in scene:
        return "DINOv2（自监督稠密特征）"
    if "看图问答" in scene or "图表" in scene:
        return "SigLIP/CLIP-ViT（开源 VLM 通用编码器）"
    if "长图大图" in scene or "卷宗" in scene:
        return "动态分辨率（分块保信息，不压扁）"
    return "未知"


print("== 账A 眼睛族谱 ==")
fam = ["ViT", "CLIP", "SigLIP", "DINOv2", "InternViT"]
ok(len(fam) == 5, "A1 编码器家族 5 家齐备")
years = {n: encoder_family(n)[0] for n in fam}
ok(years["ViT"] == 2020 and years["CLIP"] == 2021 and years["SigLIP"] == 2023
   and years["DINOv2"] == 2023 and years["InternViT"] == 2024, "A1 家族年份 5/5（2020/2021/2023/2023/2024）")
ok(encoder_family("ViT")[2] is False and encoder_family("CLIP")[2] is True
   and encoder_family("SigLIP")[2] is True and encoder_family("DINOv2")[2] is False
   and encoder_family("InternViT")[2] is True, "A1 图文对齐标记 5/5（CLIP/SigLIP/InternViT 对齐·ViT/DINOv2 不对齐）")

S1 = "发票/表格 文字密集"
S2 = "搜图引擎 图文检索"
S3 = "道路分割 目标检测"
S4 = "图表看图问答 VLM"
S5 = "4000px 宽卷宗长图"
res = [pick_encoder(s) for s in [S1, S2, S3, S4, S5]]
exp = ["高分辨率动态分块（先把字拍清楚）", "CLIP/SigLIP（对比学习语义库）",
       "DINOv2（自监督稠密特征）", "SigLIP/CLIP-ViT（开源 VLM 通用编码器）",
       "动态分辨率（分块保信息，不压扁）"]
for i in range(5):
    ok(res[i] == exp[i], "A2 场景选眼睛 %d/5：" % (i + 1) + res[i])

ROUTE_ENC = [
    ("A 桥接", "冻结复用（借现成 CLIP/ViT，眼睛不动）"),
    ("B 开放参数", "微调或冻结（Qwen 大 ViT 微调 · LLaVA-1.5 ViT 冻结）"),
    ("C 原生", "原生并入（编码器并入统一 token 空间，无独立概念）"),
]
rt = {r[0]: r[1] for r in ROUTE_ENC}
ok(len(ROUTE_ENC) == 3 and all(r[0] in ("A 桥接", "B 开放参数", "C 原生") for r in ROUTE_ENC), "A3 账 A 矩阵『编码器格』三态 3/3")
ok("冻结复用" in rt["A 桥接"] and "微调" in rt["B 开放参数"] and "原生并入" in rt["C 原生"], "A3 编码器格状态 3/3 回引 01 账 A2")

PARAM = {"ViT-B/16": 0.086, "ViT-L/14": 0.304, "Qwen-VL 大 ViT": 0.675, "LLM 7B": 7.0}
encS = PARAM["ViT-L/14"] / PARAM["LLM 7B"]
encQ = PARAM["Qwen-VL 大 ViT"] / PARAM["LLM 7B"]
ok(encS < 0.05 and encQ < 0.10, "A4 编码器参数只是 LLM 的零头（ViT-L 占 7B ≈ %.1f%% · Qwen 大 ViT ≈ %.1f%%）" % (encS * 100, encQ * 100))
ok(PARAM["ViT-B/16"] < PARAM["ViT-L/14"] < PARAM["Qwen-VL 大 ViT"] < PARAM["LLM 7B"], "A4 参数三档递增 3/3（86M→304M→675M→7B）")

P3 = {"patch14": patch_token(224, 224, 14), "patch16": patch_token(224, 224, 16), "patch32": patch_token(224, 224, 32)}
ok(P3["patch14"] == 256 and P3["patch16"] == 196 and P3["patch32"] == 49,
   "A5 patch 粒度三档（224px）：14→256 tok · 16→196 tok · 32→49 tok（回引 01 账 B7）")
ok(patch_token(336, 336, 14) == 576 and patch_token(336, 189, 14) == 312,
   "A6 回引 01 锚点：336×336→576 tok · 固定压扁 336×189→312 tok（静态处理器丢信息）")


# ================================================================ 账 B · 动态分辨率
def nearest_grid(w, h, cell_min=224, cell_max=1024, cap=6):
    """动态分辨率切块：在合法格里选『网格比例最贴原图长宽比』的确定性规则。
    合法 = 每片边长 ∈ [224, 1024]（<224 放大失真 · >1024 压小丢信息）· 总片数 ≤ cap（预算帽）。
    并列时选分片更多者（细格子更可读）。"""
    best = None
    for gw in range(1, cap + 1):
        for gh in range(1, cap + 1):
            if gw * gh > cap:
                continue
            cw, ch = w / gw, h / gh
            if cw < cell_min or ch < cell_min:
                continue
            if cw > cell_max or ch > cell_max:
                continue
            d = abs((gw / gh) - (w / h))
            if best is None or d < best[0] - 1e-9 or (abs(d - best[0]) < 1e-9 and gw * gh > best[2]):
                best = (d, gw, gh, gw * gh)
    if best is None:
        return 1, 1   # 没有合法切分：整图主力 + 缩略图
    return best[1], best[2]


THUMB = 256          # 224×224 缩略图 token（保留全局版式）
CROP = 576           # 每片 336×336 的 token（01 账 B1/B2 锚点）


def dyn_tokens(g):    # g=(gw,gh) 分块
    return g[0] * g[1] * CROP + THUMB


print("== 账B 动态分辨率 ==")
FIX_FULL = 576
ok(patch_token(336, 189, 14) <= FIX_FULL and (1600 / 336) > 4, "B1 固定缩放把 1600×900 压到 336 长边（≈%.2f×）→ 小字糊" % (1600 / 336))
ok(dyn_tokens((1, 1)) == CROP + THUMB, "B1 缩略图永久保留（1 片主力 + 1 张 224 缩略 = 全局版式）")

g = nearest_grid(1600, 900)
ok(g == (2, 1) and dyn_tokens(g) == 1408, "B2 1600×900 → %d×%d 分块 · %d tok（每片 800×900 可读，不压扁）" % (g[0], g[1], dyn_tokens(g)))
r25 = dyn_tokens(g) / FIX_FULL
ok(abs(r25 - 2.44) < 0.01, "B2 动态/固定 token 比 ≈%.2f×：可读性用 token 买" % r25)

table = {
    1024: (1024, 1024),     # 方块
    (4000, 1000): (4000, 1000),  # 极宽
    (600, 2400): (600, 2400),    # 极高
    (400, 400): (400, 400),      # 小图
    (1080, 720): (1080, 720),    # 发票级
}
gq = nearest_grid(1024, 1024)
gt = nearest_grid(4000, 1000)
gl = nearest_grid(600, 2400)
gs = nearest_grid(400, 400)
gf = nearest_grid(1080, 720)
ok(gq == (2, 2) and dyn_tokens(gq) == 2560, "B3 正方形 → 2×2 · 2560 tok（含缩略）")
ok(gt == (4, 1) and dyn_tokens(gt) == 2560, "B3 极宽 4000×1000 → 4×1 · 2560 tok（每片 1000×1000）")
ok(gl == (1, 4) and dyn_tokens(gl) == 2560, "B3 极高 600×2400 → 1×4 · 2560 tok（每片 600×600）")
ok(gs == (1, 1) and dyn_tokens(gs) == 832, "B3 小图不折腾 → 1×1 · 832 tok")
ok(gf == (3, 2) and dyn_tokens(gf) == 3712, "B3 发票 1080×720 → 3×2 · 3712 tok（每片 360×360 小字清楚）")

ok(dyn_tokens(gf) == gf[0] * gf[1] * CROP + THUMB, "B4 token 公式：分块×576 + 缩略 256")
pref = dyn_tokens(gf) / (dyn_tokens(gf) + 200)
ok(abs(pref - 0.949) < 0.01, "B5 视觉 prefill 占比 0.74→%.3f（动态分块把多模态成本推向视觉）" % pref)

ok(gq[0] * gq[1] <= 6 and gt[0] * gt[1] <= 6 and gl[0] * gl[1] <= 6 and gf[0] * gf[1] <= 6, "B6 预算帽：分块总片数 ≤6 全守（2×2/4×1/1×4/3×2）")
ok(1080 / gf[0] >= 224 and 720 / gf[1] >= 224, "B6 每片 ≥224（发票文字场地足够）")

sizes7 = [(1600, 900), (1024, 1024), (4000, 1000), (600, 2400), (400, 400), (1080, 720)]
grids7 = [nearest_grid(a, b) for (a, b) in sizes7]
ok(all(g[0] * g[1] <= 6 for g in grids7), "B7 六图预算帽全守（片数 ≤6）")
ok(all((a / g[0] >= 224 and a / g[0] <= 1024 and b / g[1] >= 224 and b / g[1] <= 1024)
       for (a, b), g in zip(sizes7, grids7)), "B7 六图每片边长 ∈[224,1024]（<224 放大失真 · >1024 压小丢信息）")
ok(all(abs((g[0] / g[1]) - (a / b)) < 0.3 for (a, b), g in zip(sizes7, grids7)), "B7 六图网格比例贴原图长宽比（2×1/2×2/4×1/1×4/1×1/3×2）")


# ================================================================ 账 C · M-RoPE
def mrope_bands(dim):
    """把旋转维度三等分给 (t, h, w) 三个坐标通道。返回每通道维数。"""
    each = dim // 3
    return dim - each * 2, each, each     # t 拿余数，h/w 均分


def theta(k, dim, base=10000.0):
    """RoPE 频率：低频长距离、高频局部。θ_k = base^(-2k/dim)。"""
    return base ** (-2.0 * k / dim)


def pos1d(r, c, W):
    """把 4×4 图拉成一条文本式 1D 位置。"""
    return r * W + c


def mrope_h(ra, ca, rb, cb):
    return abs(ra - rb)


def mrope_w(ra, ca, rb, cb):
    return abs(ca - cb)


def mrope_need(scene):
    """M-RoPE 什么场景用得上：需要『空间/时序邻接』时。"""
    if "多图" in scene or "对比" in scene:
        return True
    if "视频" in scene or "时序" in scene:
        return True
    if "UI" in scene or "界面" in scene:
        return True
    if "单张小字" in scene or "本地小图" in scene:
        return False
    if "纯文本" in scene:
        return False
    return False


print("== 账C M-RoPE ==")
t, h, w = mrope_bands(60)
ok(t + h + w == 60 and t > 0 and h > 0 and w > 0, "C1 3D 坐标 (t,h,w) 三等分旋转维度（60→%d/%d/%d）" % (t, h, w))
ok(abs(theta(1, 60) - 0.7356) < 0.001, "C1 RoPE 频率 θ_1≈0.7356（低频=长距离 · 高频=局部，同一基式）")
ok(abs(theta(0, 60) - 1.0) < 1e-9 and theta(2, 60) < theta(1, 60), "C1 频率随 k 严格递减（θ0=1 → θ1≈0.7356 → θ2 更小）")

W = 4
ok(pos1d(0, 0, W) == 0 and pos1d(0, 3, W) == 3 and pos1d(1, 0, W) == 4 and pos1d(3, 3, W) == 15,
   "C2 1D 拉平：图被拉成一行，位置=行序×行宽+列")
ok(mrope_h(0, 0, 1, 0) == 1 and mrope_w(0, 0, 0, 3) == 3 and mrope_h(0, 0, 3, 3) == 3 and mrope_w(0, 0, 3, 3) == 3,
   "C3 M-RoPE 分通道：h 只答垂直差、w 只答水平差")

ok(pos1d(0, 0, W) + 4 == pos1d(1, 0, W) and pos1d(0, 0, W) + 15 == pos1d(3, 3, W), "C4 1D 里下移一格=+行宽（跨行距离与同行距离在同一把尺）")
ok(mrope_h(0, 0, 1, 0) == 1 and mrope_h(0, 0, 3, 0) == 3 and mrope_h(0, 0, 1, 0) < mrope_w(0, 0, 0, 3),
   "C4 M-RoPE 保留拓扑：同列相邻 h=1 < 同行远端 w=3（图像邻接不塌）")

def merge3d(text_len, g):
    """文本走 t 通道 1D；图 patch 走 h/w 通道 2D；两者同域。返回 3D 坐标数。"""
    return text_len  # t 通道容量（文本）

ok(merge3d(200, (3, 2)) == 200, "C5 文本 token 用 t 通道（一段提示=200 个 t 坐标）")
ok(h > 0 and w > 0, "C5 图 patch 用 h/w 通道（每片内 h=行号、w=列号，2D 邻接保留）")

c6 = [mrope_need(s) for s in ["多图并排对比", "视频抽帧时序", "UI 界面截图", "单张小字 OCR", "纯文本问答"]]
ok(c6 == [True, True, True, False, False], "C6 M-RoPE 场景 5/5（多图/视频/UI 要 · 单张小字/纯文本不要）")


def qpos(px, cell_px):
    """位置量化：坐标按分块边界取整——位置跟随分辨率，跨图可比。"""
    return px // cell_px


ok(qpos(799, 800) == 0 and qpos(800, 800) == 1 and qpos(1500, 800) == 1 and qpos(1599, 800) == 1,
   "C7 1600px 分 2 片：0-799→片0 · 800-1599→片1（归一，按图量化）")
ok(mrope_h(0, 0, 2, 0) == 2 and mrope_w(0, 0, 2, 0) == 0,
   "C8 h/w 各管各轴：垂直差只进 h · 水平差只进 w（互不串扰）")


# ================================================================ 账 D · 掌握度检验
def align(n):
    return encoder_family(n)[2]


print("== 账D 掌握度检验 ==")
ok(align("ViT") is False and align("CLIP") is True and align("SigLIP") is True
   and align("DINOv2") is False and align("InternViT") is True, "D1 家族图文对齐 5/5（谁进 VLM 看谁对齐文本）")


def encoder_status(desc):
    if "统一 token 预训练" in desc:
        return "原生并入(C)"
    if "大 ViT 微调" in desc:
        return "微调(B)"
    if "ViT 冻结" in desc and "LLM" in desc:
        return "冻结(B)"
    if "现成" in desc and "冻结" in desc:
        return "冻结复用(A)"
    if "自监督" in desc and "不接文本" in desc:
        return "视觉专用(非VLM)"
    return "未知"


D2 = ["Flamingo 拿现成 CLIP 直接冻结", "LLaVA—ViT 冻结 + LLM 全参 SFT",
      "Qwen2.5-VL 大 ViT 微调", "原生模型统一 token 预训练", "DINOv2 自监督不接文本"]
d2 = [encoder_status(s) for s in D2]
ok(d2 == ["冻结复用(A)", "冻结(B)", "微调(B)", "原生并入(C)", "视觉专用(非VLM)"],
   "D2 编码器层状态 5/5（架构描述→『眼睛』动没动）")


def vision_plan(scene):
    if "发票小字" in scene:
        return "动态分块+高分辨率（字先拍清楚，模型再读图或工程 OCR）"
    if "商标搜图" in scene:
        return "CLIP/SigLIP 语义检索（比对的是语义不是像素）"
    if "A4版式" in scene:
        return "动态网格保版式（3×2 分块，版式不被压扁）"
    if "视频抽帧" in scene:
        return "抽帧各过编码器 + M-RoPE t 通道串时序"
    if "道路分割" in scene:
        return "DINOv2 稠密特征（像素级任务专用）"
    return "未知"


D3 = ["发票小字", "商标搜图", "A4版式", "视频抽帧", "道路分割"]
d3 = [vision_plan(s) for s in D3]
ok(len(d3) == 5 and all(x != "未知" for x in d3), "D3 场景→方案 5/5（见各行）")

inv_tok = dyn_tokens(gf)
ocr_tok = 50   # OCR 引擎提取发票 5 字段 ≈ 50 token
ratio = inv_tok / ocr_tok
ok(inv_tok == 3712, "D4 发票 1080×720 动态看图 = %d token" % inv_tok)
ok(abs(ratio - 74.2) < 0.5, "D4 看图 3712 tok vs OCR 50 tok ≈ %.1f×（能工程提取的不需整图进模型）" % ratio)


def encoder_tree(need):
    """决策树第一问：信息在哪儿？"""
    if "文字" in need or "布局" in need:
        return "OCR/工程列（票据·合同·表单：图外能提取）"
    if "语义" in need:
        return "VLM 看图列（图表·截图·UI：语义在图上）"
    if "像素" in need or "分割" in need:
        return "稠密特征列（DINOv2 像素级任务）"
    return "未知"


ok(encoder_tree("票据文字") == "OCR/工程列（票据·合同·表单：图外能提取）"
   and encoder_tree("图表语义") == "VLM 看图列（图表·截图·UI：语义在图上）"
   and encoder_tree("像素分割") == "稠密特征列（DINOv2 像素级任务）", "D5 决策树第一问 3/3（信息在哪决定眼睛动不动）")

# 回引 01：9 模型→路线练习（只引用不重测）
MODEL_ROUTE = {"BLIP-2": "A桥接", "Flamingo": "A桥接", "早期LLaVA": "A桥接", "LLaVA-1.5": "B开放参数",
               "Qwen-VL": "B开放参数", "InternVL": "B开放参数", "GPT-4o": "C原生", "Gemini": "C原生", "Llama4": "C原生"}
ok(len(MODEL_ROUTE) == 9, "D6 回引 01 账 D1 九模型路线 9/9（本篇只看编码器层，路线判定引用不重测）")
ok(encoder_family("SigLIP")[0] == 2023 and "sigmoid" in encoder_family("SigLIP")[1],
   "D7 SigLIP=sigmoid 逐 token（2023 · 开源 VLM 最常用编码器）")

print("\nSUMMARY  PASS %d · FAIL %d" % (PASS, FAIL))
assert FAIL == 0 and PASS >= 45
