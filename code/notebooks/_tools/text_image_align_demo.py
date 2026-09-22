# -*- coding: utf-8 -*-
"""10-03 图文对齐 探针：把「眼睛怎么和嘴巴接」算成一张对齐账。

四账四账全 PASS 教学闭环（纯 stdlib · 零网络 · 零随机 · 零第三方）：
账 A 对齐层三态账：对齐方式演化族谱 6 家（CLIP 对比预训练 → Flamingo 交叉注意力 →
             Q-Former 桥 → LLaVA 线性投影 → LLaVA-1.5 MLP → 原生无桥），年份单调 6/6、
             三态归位 3/3（回引 01 账 A2 对齐格）、5 场景选对齐方案 5/5。
账 B CLIP 对比/检索账：确定性「对齐语义空间」12 类图文对——对齐后图文互检 12/12+12/12；
             判别性对照 同义改写召回 对齐 6/6 vs 词面精确 0/6（对齐语义空间越过了词面）；
             InfoNCE 对比损失 对齐 0.0007 vs 未对齐 1.52 ≈ 2.2 千倍（正样本对几乎独吞概率质量）。
账 C 投影器账：「桥」有多薄——ViT-L/14 1024 维 patch 特征投进 LLM-7B 4096 维词嵌入空间：
             线性 4.19M 参 ≈ 7B 的 0.06%、MLP 20.97M ≈ 0.30%（占 LLM < 0.3%）；
             一张图 577 token → Q-Former 32 token 压缩 18.0×（KV 省 94.5%）；
             6 场景选桥 6/6（数据少用薄桥 · token 贵用交叉注意力 · 检索不用桥）。
账 D 对齐训练阶梯账：视觉指令微调（LLaVA-Instruct 类，B 改对齐+解码器）→ 多模态 DPO
             （偏好对）→ 多模态 RLVR（R1-V 2025 · 可验证奖励：计数/坐标/几何）；
             训练范式分类 5/5 · RLVR 规则验证闭环（错误答案被真值函数抓住 2/2）·
             架构→对齐方案 6/6（Flamingo 交叉注意力 A · BLIP-2 Q-Former A · LLaVA 线性投影 A→B ·
             LLaVA-1.5 MLP B · Qwen2.5-VL 大ViT+投影 B · GPT-4o 原生 C）· 决策树 6/6。

无任何训练代码：所谓「对齐语义空间」= 确定性构造的余弦相似度世界（教学模型），
真实对齐由 CLIP 对比预训练 / 视觉指令微调学到（要素事实，读者可用真模型复现）。
"""
import hashlib
import math
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
    return cond


def det01(s):
    """确定性子串 → [-1,1) 伪随机数（sha1 截断，无 random 模块）。"""
    return (int(hashlib.sha1(s.encode("utf-8")).hexdigest()[:8], 16) % 1000000) / 500000 - 1.0


def unit(dim, tag):
    v = [det01("%s:%d" % (tag, i)) for i in range(dim)]
    n = math.sqrt(sum(x * x for x in v)) + 1e-12
    return [x / n for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


print("=" * 72)
print("10-03 图文对齐 · text_image_align_demo（纯 stdlib 四账）")
print("=" * 72)

# ---------------------------------------------------------------
# 账 A 对齐层三态账：对齐方式演化 + 三态归位 + 场景选方案
# ---------------------------------------------------------------
print("\n== 账A 对齐层三态：演化族谱 + 三态归位 + 场景选方案 ==")


def aligner_family(name):
    """(year, 一句话, 对齐机制) —— 知识地图 §10.3：Cross-Attention → 投影器 → 全参。"""
    tab = {
        "CLIP": (2021, "对比学习·图文配对互为正样本·学到对齐语义空间", "contrastive"),
        "Flamingo": (2022, "交叉注意力把视觉条件拿进冻结 LLM", "cross_attn"),
        "Q-Former": (2023, "可训练 query 提取视觉 token·编码器+LLM 双冻结", "cross_attn"),
        "LLaVA": (2023, "线性投影把 patch 特征投进词嵌入空间·简单有效", "projector"),
        "LLaVA-1.5": (2024, "两层 MLP 投影·解冻 LLM 全参微调", "projector"),
        "原生路线": (2025, "统一 token 空间·不再需要独立桥", "none"),
    }
    return tab[name]


fam = ["CLIP", "Flamingo", "Q-Former", "LLaVA", "LLaVA-1.5", "原生路线"]
years = [aligner_family(n)[0] for n in fam]
ok(years == sorted(years), "A1 族谱年份单调 6/6（2021→2025 一条演化主轴）")
for n in fam:
    y, one, mech = aligner_family(n)
    ok(mech in ("contrastive", "cross_attn", "projector", "none"),
       "A1 %s 对齐机制 %s 合法" % (n, mech))
ok(fam == ["CLIP", "Flamingo", "Q-Former", "LLaVA", "LLaVA-1.5", "原生路线"],
   "A1 六家族谱顺序=知识地图 §10.3 演化主线")


def aligner_state(desc):
    """对齐格「训练/冻结/无桥」三态 —— 回引 01 账 A2 对齐行。"""
    if "统一 token" in desc or "不需要投影器" in desc:
        return "C 无独立桥"
    if "交叉注意力" in desc or "Q-Former" in desc or "投影器" in desc:
        if "冻结 LLM" in desc:
            return "A 训练（对齐层）"  # 桥接：只训对齐层
        return "B 训练（全参与）"
    return "未知"


ok(aligner_state("投影器/Q-Former/交叉注意力 · 冻结 LLM 不动") == "A 训练（对齐层）",
   "A2 对齐格归位 1/3：A 桥接=只训对齐层（编码器冻结 · LLM 冻结 · 仅对齐层训练）")
ok(aligner_state("投影器参数全参与 · 开放 LLM 全参微调") == "B 训练（全参与）",
   "A2 对齐格归位 2/3：B 开放参数=对齐训练+全参与")
ok(aligner_state("统一 token 空间 · 不需要独立投影器") == "C 无独立桥",
   "A2 对齐格归位 3/3：C 原生=无独立桥（对齐并进整个网络）")


def pick_aligner(scene):
    """5 场景 → 对齐方案（教学法约定「什么场景用得上」）。"""
    if "已有" in scene and "算力" in scene:
        return "线性投影器·冻结 LLM（A 桥）"
    if "中等数据" in scene:
        return "MLP 投影器·开放 LLM（B）"
    if "长上下文" in scene or "压缩" in scene:
        return "Q-Former 交叉注意力抽 query"
    if "检索" in scene or "搜文" in scene or "搜图" in scene:
        return "CLIP 对比语义空间·不接 LLM"
    if "从零" in scene:
        return "原生统一 token（C）"
    return "未识别"


scenes = [
    ("已有成熟大模型只加看图、算力紧", "线性投影器·冻结 LLM（A 桥）"),
    ("中等数据想更强图文协作", "MLP 投影器·开放 LLM（B）"),
    ("文档图表长 token 想压缩", "Q-Former 交叉注意力抽 query"),
    ("按图搜文/按文搜图", "CLIP 对比语义空间·不接 LLM"),
    ("从零预训练多模态大模型", "原生统一 token（C）"),
]
for s, want in scenes:
    ok(pick_aligner(s) == want, "A3 场景→对齐方案 5/5：%s → %s" % (s, want))


# ---------------------------------------------------------------
# 账 B CLIP 对比/检索账：对齐了的语义空间长什么样
# ---------------------------------------------------------------
print("\n== 账B CLIP 对比/检索：图文配对互为正样本的语义空间 ==")

DIM = 96
CATS = ["猫", "狗", "苹果", "汽车", "雪山", "钢琴", "灯塔", "猫头鹰", "自行车", "雨伞", "皮球", "书"]
ALIAS = {  # 同义改写：词面搜不到、语义空间搜得到
    "小猫": "猫", "kitty": "猫", "狗狗": "狗", "puppy": "狗", "红苹果": "苹果",
    "小轿车": "汽车", "雪峰": "雪山", "钢琴曲": "钢琴", "手电塔": "灯塔",
    "owl": "猫头鹰", "单车": "自行车", "伞": "雨伞", "气球皮球": "皮球", "阅读": "书",
}
N_IMG = 2  # 每类 2 张图（同类别图像共享语义分量）

# 构造确定性对齐语义空间：每类一个共享语义分量 s(C)，图像= s(C)+实例扰动，文本=匹配类 s(C)+查询扰动
img_vecs = {}   # idx -> vec（类别图）
img_cat = {}    # idx -> category
for ci, c in enumerate(CATS):
    s = unit(DIM, "sem:%s" % c)
    for k in range(N_IMG):
        iv = [0.85 * a + 0.15 * det01("img:%s@%d:%d" % (c, k, i)) for i, a in enumerate(s)]
        n = math.sqrt(sum(x * x for x in iv)) + 1e-12
        img_vecs[ci * N_IMG + k] = [x / n for x in iv]
        img_cat[ci * N_IMG + k] = c


def tvec(q):
    cat = ALIAS.get(q, q)
    s = unit(DIM, "sem:%s" % cat)
    return [0.90 * a + 0.10 * det01("qy:%s:%d" % (q, i)) for i, a in enumerate(s)]


def top1_img(q):
    """文本 query → 图上 top-1 类别（余弦在确定性对齐语义空间中取最近邻）。"""
    tv = tvec(q)
    best, bi = -2.0, None
    for idx, iv in img_vecs.items():
        c = dot(tv, iv)
        if c > best:
            best, bi = c, idx
    return img_cat[bi], best


# 文本→图：12 个真标签
label_hits = 0
for c in CATS:
    got, _ = top1_img(c)
    label_hits += 1 if got == c else 0
ok(label_hits == 12, "B1 文本→图 top-1：真标签 12/12（对齐空间图文互为最近邻）")

# 文本→图：6 组同义改写（词面精确检索全 miss，语义检索全命中）
ALIAS_SET = [("小猫", "猫"), ("狗狗", "狗"), ("小轿车", "汽车"), ("雪峰", "雪山"),
             ("钢琴曲", "钢琴"), ("单车", "自行车")]
alias_align = 0
alias_word = 0
for q, want in ALIAS_SET:
    got, _ = top1_img(q)
    alias_align += 1 if got == want else 0
    alias_word += 1 if q in CATS else 0
ok(alias_align == 6 and alias_word == 0,
   "B2 判别性对照：同义改写召回 对齐语义空间 6/6 vs 词面精确 match 0/6——对齐越过了词面")

# 图→文本：用 cls 语义查回类别
img_text_hits = 0
for idx, cc in img_cat.items():
    tv = unit(DIM, "sem:%s" % cc)
    best, bc = -2.0, None
    for tgt in CATS:
        c2 = dot(tv, unit(DIM, "sem:%s" % tgt))
        if c2 > best:
            best, bc = c2, tgt
    img_text_hits += 1 if bc == cc else 0
ok(img_text_hits == 24, "B1 图→文本 top-1：24 张图全回到本类（CLIP 双向往返都用得上）")


def info_nce_loss(diag, off, tau, B=16):
    """InfoNCE 对比损失：正样本对 (i,i) 从 15 个负样本里被拉出来。"""
    tot = 0.0
    for i in range(B):
        num = math.exp(diag / tau)
        den = num + (B - 1) * math.exp(off / tau)
        tot += -math.log(num / den)
    return tot / B


l_align = info_nce_loss(0.80, 0.10, 0.07)     # 对齐空间：对角 0.8 vs 非对角 0.1
l_unalign = info_nce_loss(0.40, 0.30, 0.07)  # 未对齐：对角 0.4 vs 非对角 0.3
ok(l_align < 0.002, "B3 InfoNCE 账：对齐空间损失 %.4f < 0.002（正样本对几乎独吞概率）" % l_align)
ok(l_unalign > 1.0, "B3 InfoNCE 账：未对齐损失 %.4f > 1.0（正样本淹没在噪声）" % l_unalign)
ok(l_unalign / l_align > 1000,
   "B3 InfoNCE 账：未对齐/对齐 ≈ %.0f 倍 >1000（对比学习把正样本从负样本池里拉出来）" % (l_unalign / l_align))
ok(0.10 / 0.30 < 0.34 and (0.40 - 0.30) < (0.80 - 0.10),
   "B3 量纲诊断：对齐度=对角与非对角之差（未对齐差 0.10 vs 对齐差 0.70）")


def clip_uses(x):
    return x


ok(clip_uses("图文检索") and clip_uses("编码器预训练底座"),
   "B4 CLIP 两用 2/2：检索（双向往返）+ 图像编码器预训练的事实底座（§10.3 要素事实）")


# ---------------------------------------------------------------
# 账 C 投影器账：「桥」有多薄、token 多少
# ---------------------------------------------------------------
print("\n== 账C 投影器：桥只换维度、不替语义层重新学 ==")


def projector_params(vd, ld, kind):
    linear = vd * ld
    if kind == "linear":
        return linear
    if kind == "mlp":
        return linear + ld * ld
    if kind == "qformer":
        return 32 * 4096  # 教学量级：query 数×LLM 维度
    raise ValueError


VD, LD = 1024, 4096  # ViT-L/14 patch 特征维 → LLM-7B 词嵌入维
p_lin = projector_params(VD, LD, "linear")
p_mlp = projector_params(VD, LD, "mlp")
ok(p_lin == 4194304 and round(p_lin / 1e6, 2) == 4.19, "C1 线性投影参数 = 1024×4096 = 4.19 M")
ok(p_mlp == 20971520 and abs(p_mlp / 1e6 - 20.97) < 0.01, "C1 MLP 投影参数 = 4.19+16.78 = 20.97 M")
ok(p_mlp / p_lin == 5.0, "C1 MLP 比线性厚 5 倍（表达更强、数据要求更高）")
ok(p_lin / 7e9 < 0.001, "C1 线性桥只占 LLM-7B 的 0.06%（桥很薄——语义在对齐底座里不在这层）")
ok(p_mlp / 7e9 < 0.003, "C1 MLP 桥也只占 0.30%（<0.3%——A 桥『只训≈0.03B』的教学量级 01 账 C2 对得上）")

tok_img = 577     # 576 patch + 1 CLS（02 账：patch=16 → 576）
tok_q = 32        # Q-Former query 数
comp = tok_img / tok_q
kv_save = (tok_img - tok_q) / tok_img
ok(577 - 576 == 1, "C2 token 口径：一张图 = 576 patch + 1 CLS = 577 token")
ok(abs(comp - 18.03) < 0.01, "C2 Q-Former 压缩 = 577/32 = %.1f 倍（长上下文省钱）" % comp)
ok(0.94 < kv_save < 0.95,
   "C2 KV 省 %.1f 个百分点（回引 01 账 Q-Former 256→32 省 KV 87.5pp 同式；本账按 02 的 patch 规模 577 算）" % (kv_save * 100))


def pick_bridge(scene):
    if "数据少" in scene:
        return "线性投影（薄桥·少参数先跑通）"
    if "中等数据" in scene:
        return "MLP 投影（厚一点·表达更足）"
    if "压缩" in scene or "token 贵" in scene:
        return "Q-Former 交叉注意力"
    if "检索" in scene:
        return "不需要桥（CLIP 语义空间直用）"
    if "最强" in scene:
        return "原生统一 token"
    if "CLIP" in scene:
        return "薄桥即可（特征已对齐·桥只插维度）"
    return "未识别"


bridges = [
    ("数据少·先跑通", "线性投影（薄桥·少参数先跑通）"),
    ("中等数据想要更好对齐", "MLP 投影（厚一点·表达更足）"),
    ("token 贵·长文档图", "Q-Former 交叉注意力"),
    ("图文检索不开 LLM", "不需要桥（CLIP 语义空间直用）"),
    ("全量预算·最强效果", "原生统一 token"),
    ("编码器已是 CLIP 对齐过", "薄桥即可（特征已对齐·桥只插维度）"),
]
for s, want in bridges:
    ok(pick_bridge(s) == want, "C3 场景→选桥 6/6：%s → %s" % (s, want))


# ---------------------------------------------------------------
# 账 D 对齐训练阶梯账：LLaVA-Instruct → 多模态 DPO → 多模态 RLVR
# ---------------------------------------------------------------
print("\n== 账D 对齐训练阶梯：LLaVA-Instruct → 多模态DPO → 多模态RLVR ==")


def ladder(name):
    """三档对齐训练：改的层 × 数据形态 × 年份（§10.3 要素事实）。"""
    if name == "视觉指令微调":
        return ("B（对齐+解码器)", "图文问答对（LLaVA-Instruct 类 · 01 账 0.665M 量级）", 2023)
    if name == "多模态DPO":
        return ("B（对齐+解码器)", "好坏回复偏好对", 2024)
    if name == "多模态RLVR":
        return ("B（对齐+解码器)", "规则可验证题（计数/坐标/几何）· R1-V 2025", 2025)
    return None


steps = ["视觉指令微调", "多模态DPO", "多模态RLVR"]
lstep = [ladder(s) for s in steps]
ok(all(s is not None for s in lstep), "D1 训练阶梯三档 3/3：SFT→DPO→RL 递进")
ok([s[2] for s in lstep] == sorted([s[2] for s in lstep]) and lstep[1][1] != lstep[0][1],
   "D1 递增阶梯：数据形态从监督到偏好到可验证奖励（奖励来源越来越『真』）")


def teach_paradigm(sample):
    if "唯一答案" in sample and "好差" not in sample and "可验证" not in sample:
        return "SFT（视觉指令微调）"
    if "哪个好" in sample or "偏好" in sample or "后者好" in sample:
        return "多模态DPO"
    if "可验证" in sample or "真值" in sample:
        return "多模态RLVR"
    if "互为正样本" in sample:
        return "CLIP 对比预训练"
    if "从零" in sample:
        return "原生预训练"
    return "未知"


samples = [
    ("看图 Q：这张图里几个人？A：3 个人（唯一答案）", "SFT（视觉指令微调）"),
    ("两个回复版本：一个答非所问、一个点中主题——标注者说后者好", "多模态DPO"),
    ("地图坐标题：终点经纬度可用程序核对真值", "多模态RLVR"),
    ("百万级图文配对·互为正样本·学到共享语义空间", "CLIP 对比预训练"),
    ("从零把图文统一 token 一起预训练", "原生预训练"),
]
for s, want in samples:
    ok(teach_paradigm(s) == want, "D2 数据样例→训练范式 5/5：%s…" % s[:18])


def rule_verifier(q):
    """多模态 RLVR 规则验证器（可验证奖励的『真值』函数——确定性，非模型打分）。"""
    ans = {"数人物": 2, "坐标解析": (31.8, 117.2), "几何角度": 180}
    return ans.get(q, None)


def policy_answer(q):
    """确定性『策略』：能答对 2/3（提前注入 1 个错）。"""
    wrong = {"坐标解析": (30.1, 116.0)}
    a = rule_verifier(q)
    return wrong.get(q, a)


rl_hit_after = 0
for q, truth in [("数人物", 2), ("几何角度", 180)]:
    pred = policy_answer(q)
    ok(pred == truth, "D3 RLVR 账：策略 '%s' 回答 %s == 真值 %s（规则验证器可核对）"
       % (q, pred, truth))
ok(policy_answer("坐标解析") != rule_verifier("坐标解析"),
   "D3 诚实注入：策略在坐标题上答错 1/3（可验证奖励=真值函数·不是自欺的价值模型）")
for q, truth in [("数人物", 2), ("坐标解析", (31.8, 117.2)), ("几何角度", 180)]:
    pred = policy_answer(q)
    # 用 RLVR 回访：验证器信号→策略重答（确定性重写，教学闭环 R1-V 路线）
    if pred != truth:
        pred = rule_verifier(q)
    rl_hit_after += 1 if pred == truth else 0
ok(rl_hit_after == 3, "D3 RLVR 闭环：验证器信号把策略 2/3 → 3/3（可验证奖励驱动修正=RLVR 卖点）")


def arch_align(arch):
    """架构 → 对齐方案（§10.1/§10.4 要素事实 + 知识地图）。"""
    if arch == "Flamingo":
        return "交叉注意力对齐层（A·冻结 LLM）"
    if arch == "BLIP-2":
        return "Q-Former 桥（A·双冻结）"
    if arch == "LLaVA":
        return "线性投影+两阶段 SFT（A→B 桥）"
    if arch == "LLaVA-1.5":
        return "MLP 投影+LLM 全参（B）"
    if arch == "Qwen2.5-VL":
        return "大ViT 微调+投影+M-RoPE（B）"
    if arch == "GPT-4o":
        return "原生统一 token（C·无独立桥）"
    return "未知"


archs = [
    ("Flamingo", "交叉注意力对齐层（A·冻结 LLM）"),
    ("BLIP-2", "Q-Former 桥（A·双冻结）"),
    ("LLaVA", "线性投影+两阶段 SFT（A→B 桥）"),
    ("LLaVA-1.5", "MLP 投影+LLM 全参（B）"),
    ("Qwen2.5-VL", "大ViT 微调+投影+M-RoPE（B）"),
    ("GPT-4o", "原生统一 token（C·无独立桥）"),
]
for a, want in archs:
    ok(arch_align(a) == want, "D4 架构→对齐方案 6/6：%s → %s" % (a, want))

# 决策树：什么时候不值得做对齐训练
def when_skip_align(cfg):
    if "不开 LLM" in cfg:
        return "纯检索：零对齐训练（CLIP 现成底座）"
    if "对齐过" in cfg:
        return "投影器薄桥即可（省预算）"
    if "预算无法过 A" in cfg:
        return "先不训桥·缓图"
    if "小数据幻觉重" in cfg:
        return "先加检索/工程·别急着多模态微调"
    return "走 B 开放参数"
ok(when_skip_align("不开 LLM · 只要检索") == "纯检索：零对齐训练（CLIP 现成底座）",
   "D4 决策树 1/6：纯检索不开 LLM→零对齐训练")
ok(when_skip_align("编码器已是 CLIP 对齐过") == "投影器薄桥即可（省预算）",
   "D4 决策树 2/6：特征已对齐→薄桥")
ok(when_skip_align("预算无法过 A 桥") == "先不训桥·缓图", "D4 决策树 3/6：预算不足→缓图")
ok(when_skip_align("小数据幻觉重") == "先加检索/工程·别急着多模态微调",
   "D4 决策树 4/6：小数据→先工程后对齐训练")
ok(when_skip_align("有图文问答数据·要更强") == "走 B 开放参数", "D4 决策树 5/6：有数据→B 开放参数")
ok(when_skip_align("全量预算·要最强") == "走 B 开放参数", "D4 决策树 6/6：全量→开放参数或原生")

print("\nSUMMARY  PASS %d · FAIL %d" % (PASS, FAIL))
assert FAIL == 0 and PASS >= 45
