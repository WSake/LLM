# -*- coding: utf-8 -*-
"""03-模型家族 · 11-八要素横向对比表（知识地图 §3.11 收束章）——十一个家族一张表 + 收敛度量化 + 选型决策。

[A] 全家族八要素横向表：把 01-10 各家族的代表模型按统一八要素模板（架构/Attention/FFN·MoE/
    训练/上下文/多模态/Reasoning）并排写成一张表——综合 §3.11 既有六列（Llama/Qwen/DeepSeek/
    Gemini/Claude/GLM）扩到全部家族。"横向"的意义=同字段并排，看家族差异与扩散路径。
    要素事实=非本机实测（源：知识地图 §3.1–§3.10 速写与各篇家族章，写作环境无外网未在线复核）。
[B] 派生账（本机真算）：复用 00 章同一组公式（count_params / kv_bytes_per_layer_token，
    Dense/GQA/MoE 结构级推导、MLA 走官方），先把四家基线逐位对账（与 00 章同式同函数），
    再给结构已知的家算 KV 母线、给闭源/未公开的家列官方面值或"未公开"。
[C] 复习链断言：八要素 → 02 核心原理章节映射自校（8/8 全非空），与 00 章同表。
[D] 收敛度量化（本脚本新计算）：§3.12"2026 收敛态"拆成 8 个特征（RoPE/SwiGLU/多阶段训练/
    thinking 模式/原生多模态/长上下文≥128k/MoE 或小 Dense/工具接口 Agent），对 14 个家族代表
    逐一布尔判分 → 收敛度 X/8 + 差分项。教学点："收敛度看同质化（照抄的部分），选型看差分
    （谁还留着独门）——选型不选最优，选匹配（§3.11 口号）。"
[E] 决策引擎：任务 → 家族规则表（开源自部署与闭源 API 双路径）+ 断言自校（同一输入永远同一推荐）。

全部为确定性算术/打印，无随机；wall-clock 只进 stderr；stdout run1==run2==run3 逐位一致。
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
t0 = time.perf_counter()

GiB = 2 ** 30


def logw(msg):
    """wall-clock 只进 stderr，不进入 stdout 逐位比对。"""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# ---- [A] 家族代表与八要素（要素事实=非本机实测：知识地图 §3.1–§3.10 + 各篇家族章）----
# 每格一句卖点速写，横向同字段并排；[ ] 内=该格总括。
FAMILIES = [
    # 文件名      模型代表        架构      Attention        FFN           训练            上下文      多模态        Reasoning
    ("01-GPT",    "GPT-3.5/4(o)", "decoder", "MHA→未公开", "MLP→未公开", "预训练+RLHF", "8k→128k",   "GPT-4o原生", "o系列开创thinking"),
    ("02-Llama",  "Llama-3.1-8B", "decoder", "GQA+RoPE",    "SwiGLU",     "预+SF+DPO/RL", "128k",     "Llama4系列有", "有思考(Llama4)"),
    ("03-Qwen",   "Qwen2.5-7B",   "decoder", "GQA+M-RoPE",  "SwiGLU",     "预+SF+RLVR",  "128k→1M",   "Qwen-VL强", "QwQ→Qwen3思考"),
    ("04-DeepSeek", "V3 (671B MoE)", "decoder", "MLA",   "细粒度MoE+共享", "FP8预+GRPO/RLVR", "128k", "R1-V",     "R1开源推理标杆"),
    ("05-Mistral", "Mixtral-8x7B", "decoder", "GQA+SWA",    "MoE 8×top2", "公开权重",    "32k",       "弱",          "一般"),
    ("06-Gemma",  "Gemma3-1B~27B","decoder", "交替局部/全局", "SwiGLU",    "Google数据(+蒸馏)", "128k", "Gemma3图像", "弱(通用向)"),
    ("07-Claude", "Claude系列",   "闭源",    "未公开",       "未公开",     "Constitutional+RLAIF", "200k~1M", "文档/截图强", "thinking切换(3.7)"),
    ("08-Gemini", "Gemini系列",   "原生多模态", "MoE(未公开)", "未公开",    "多模态语料直训", "1M~2M",    "原生全模态", "thinking模式"),
    ("09-GLM",    "GLM-4.5",      "decoder", "2D RoPE",     "GLU系",       "预训练+工具强化", "128k~200k", "GLM-4V",   "GLM-Z1思考(hmm)"),
    ("10a-Kimi",  "Kimi K2(1T级)", "decoder", "长上下文",   "MoE(1T级)",   "长ctx+Agent", "长上下文",  "K1.5多模态推理", "Agent激进"),
    ("10b-MiniMax", "M1(201B)",   "decoder", "MoE",        "MoE(201B)",   "多模态原生",   "长上下文",  "原生多模态",  "推理"),
    ("10c-Doubao", "Seed-Thinking","MoE(未公开)","未公开",  "未公开",      "RL采样税前置", "未公开",    "中文多模态", "Thinking(RL)"),
    ("10d-Hunyuan","Hunyuan",     "MoE(未公开)", "未公开",  "MoE",        "预训练+长上下文", "128k",    "混元多模态", "有推理"),
    ("10e-InternLM", "InternLM",  "decoder", "GQA",        "SwiGLU/MoE",  "开源+InternVL", "128k",    "InternVL",  "一般"),
]

# ---- 结构配置（[B] 派生账输入；与 00 章同字段同名，函数直接复用）----
MODELS_OPEN = {
    "Llama-3.1-8B": dict(
        arch="decoder-only Dense", attn="GQA(8/32×128)+RoPE", ffn="SwiGLU",
        train="预训练+SFT+DPO/RL", inf="开源权重可自部署", ctx=131072,
        mm="暂无（Llama3.1 纯语言）", reason="弱（无 thinking 模式）",
        L=32, d=4096, qh=32, kvh=8, vocab=128256, ffn_dim=14336, moe=None, mla=None,
        official=8.03e9, note="开源骨架事实标准 §3.2"),
    "Qwen2.5-7B": dict(
        arch="decoder-only Dense", attn="GQA(4/28×128)+RoPE", ffn="SwiGLU",
        train="超长预训练+强化学习", inf="开源全栈（API+权重）", ctx=131072,
        mm="Qwen-VL 系列", reason="QwQ→Qwen3 思考模式",
        L=28, d=3584, qh=28, kvh=4, vocab=151936, ffn_dim=18944, moe=None, mla=None,
        official=7.61e9, note="中文全家桶 §3.3"),
    "Mixtral-8x7B": dict(
        arch="decoder-only MoE", attn="GQA(8/32×128)+SWA", ffn="MoE 8×SwiGLU·top2",
        train="公开权重（细节少）", inf="开源权重可自部署", ctx=32768,
        mm="弱", reason="一般",
        L=32, d=4096, qh=32, kvh=8, vocab=32000, ffn_dim=14336,
        moe=dict(n=8, act=2), mla=None,
        official=46.7e9, note="开源 MoE 先驱 §3.5"),
    "DeepSeek-V3": dict(
        arch="decoder-only 大MoE", attn="MLA（KV 压缩）", ffn="细粒度MoE+共享专家",
        train="FP8预训练+MTP+GRPO/RLVR", inf="开源权重可自部署·长思考", ctx=131072,
        mm="R1-V 视觉推理", reason="开源推理标杆（R1）",
        L=61, d=7168, qh=128, kvh=None, vocab=129280, ffn_dim=None,
        moe=dict(n=257, act=9, shared=1), mla=dict(rank=512, rope=64),
        official=671e9, official_active=37e9, note="MLA+极致MoE §3.4"),
}

# 结构已知但非 00 章基线（新行，同公式真算）
MODELS_EXTRA = {
    "Gemma2-9B": dict(
        arch="decoder-only Dense", attn="交替局部/全局(42 层)", ffn="SwiGLU",
        train="Google 数据（含蒸馏传闻）", inf="开源权重", ctx=131072,
        mm="Gemma3 起图像输入", reason="弱（通用任务向）",
        L=42, d=3584, qh=16, kvh=16, vocab=256000, ffn_dim=14336, moe=None, mla=None,
        official=9.2e9, note="端侧质量标杆 §3.6（9B 档，同 06 章量级）"),
    "GLM-4-9B(档)": dict(
        arch="decoder-only Dense", attn="2D RoPE", ffn="GLU 系",
        train="大规模预训练+工具强化", inf="开源权重+API", ctx=131072,
        mm="GLM-4V 系", reason="GLM-Z1 思考模式",
        L=32, d=3584, qh=28, kvh=4, vocab=150000, ffn_dim=18944, moe=None, mla=None,
        official=8.83e9, note="国内开源主力（9B 档代表）§3.9"),
}

ELEMENTS = [
    ("架构", "decoder-only / encoder-decoder / 原生多模态"),
    ("Attention", "MHA / GQA / MLA / SWA——谁在压缩 KV，谁在省显存"),
    ("FFN·MoE", "SwiGLU / 细粒度专家 / 共享专家——参数放哪，激活取多少"),
    ("训练方式", "预训练 / SFT / DPO / RLHF / GRPO·RLVR / FP8 / 蒸馏 / Constitutional"),
    ("推理方式", "闭源 API / 开源自部署 / 思考预算（thinking budget）"),
    ("上下文", "窗口大小 / 长上下文技术（窗内外推）"),
    ("多模态", "adapter 拼接 / 原生全模态 / 文档-截图能力"),
    ("Reasoning", "o 系列开创 / thinking 模式 / 纯 RL 涌现（R1-Zero）"),
]

REVIEW_CHAIN = {
    "架构": ["02-01-Transformer", "02-00-技术演化时间线"],
    "Attention": ["02-03-多头注意力-MHA-MQA-GQA", "02-04-MLA-多头潜注意力", "02-10-KV-Cache与显存账本"],
    "FFN·MoE": ["02-06-FFN与激活函数-ReLU-GELU-SwiGLU", "02-18-MoE混合专家"],
    "训练方式": ["02-13-SFT", "02-14-对齐-RLHF与PPO", "02-16-GRPO与RLVR", "02-22-蒸馏与CPT"],
    "推理方式": ["05-推理与部署", "02-20-推理模型与Test-time-Scaling"],
    "上下文": ["02-17-长上下文-四层技术路径"],
    "多模态": ["10-多模态"],
    "Reasoning": ["02-20-推理模型与Test-time-Scaling"],
}


def count_params(m):
    """与 00 章同函数（逐字节复制）；Dense/GQA/MoE 结构级推导，MLA 返回 None。"""
    if m["mla"] is not None:
        return None
    d, L, vocab = m["d"], m["L"], m["vocab"]
    hdim = d // m["qh"]
    kv_dim = d if m["kvh"] is None else m["kvh"] * hdim
    attn_layer = 2 * d * d + 2 * d * kv_dim
    ffn_dim = m["ffn_dim"]
    if m["moe"] is None:
        ffn_layer = 3 * d * ffn_dim
        ffn_act = ffn_layer
        n_full, n_act, ex = 1, 1, None
    else:
        expert_ffn = 3 * d * ffn_dim
        n_full, n_act = m["moe"]["n"], m["moe"]["act"]
        ffn_layer = n_full * expert_ffn + d * n_full
        ffn_act = n_act * expert_ffn + d * n_full
        ex = expert_ffn
    norm_layer = 2 * d
    emb = vocab * d
    per_layer = attn_layer + ffn_layer + norm_layer
    total = emb + L * per_layer
    active = emb + L * (attn_layer + ffn_act + norm_layer)
    return dict(total=total, active=active, emb=emb, per_layer=per_layer,
                attn=attn_layer, ffn_tot=ffn_layer, ffn_act=ffn_act, norm=norm_layer,
                exp_full=n_full, exp_act=n_act, exp_ffn=ex)


def kv_bytes_per_layer_token(m):
    """与 00 章同函数；MLA 缓存=压缩潜向量+RoPE 头。"""
    if m["mla"] is not None:
        return (m["mla"]["rank"] + m["mla"]["rope"]) * 2
    hdim = m["d"] // m["qh"]
    nkv = m["qh"] if m["kvh"] is None else m["kvh"]
    return 2 * nkv * hdim * 2


def fmt_g(b):
    return f"{b / 1e9:.3f}B"


# ---- [D] 收敛度：§3.12 "2026 收敛态" 的 8 特征 × 14 家族布尔（定性判分，非本机实测）----
# 特征顺序即收敛态出现顺序；T=占位（有=该家族具备此收敛特征）。
CONV_FEATS = [
    ("RoPE 系位置编码（或更优）", "rope"),
    ("SwiGLU 系中间激活",        "swiglu"),
    ("MoE 或小 Dense（≤27B 可单/双卡）", "moedor"),
    ("多阶段训练（预训练→SFT→RL/RLVR）", "multistage"),
    ("thinking 模式（思考预算可调）", "thinking"),
    ("原生多模态（非纯 adapter 拼）", "multimodal"),
    ("长上下文 ≥128k",            "longctx"),
    ("工具接口 / Agent（Function call·MCP）", "agent"),
]
# 家族代表 → 8 个布尔，序同 CONV_FEATS；基于知地图 §3.x 速写。
CONV_PROFILE = {
    "01-GPT":     [1, 1, 1, 1, 1, 1, 1, 1],   # GPT-4o/o 系列：收敛态最完整的定义者
    "02-Llama":   [1, 1, 0, 1, 1, 1, 1, 1],   # Llama4 思考+多模态+10M；8B 档非小 Dense 条款按全家
    "03-Qwen":    [1, 1, 0, 1, 1, 1, 1, 1],   # Qwen3 Thinking+VL+1M；7B 非 MoE 但家族有 large
    "04-DeepSeek": [1, 1, 1, 1, 1, 1, 1, 1],  # V3 MoE+R1 thinking+R1-V；8/8 与 GPT 同分但路径不同
    "05-Mistral": [1, 1, 1, 0, 0, 0, 0, 0],   # MoE 先驱但少多阶段/thinking/多模态/Agent
    "06-Gemma":   [1, 1, 1, 1, 0, 1, 1, 0],   # 小 Dense 端侧+Gemma3 图像 128k；无 thinking/Agent
    "07-Claude":  [1, 1, 0, 1, 1, 1, 1, 1],   # thinking 切换+MCP 发起者；闭源大模型（非小Dense）
    "08-Gemini":  [1, 1, 1, 1, 1, 1, 1, 1],   # MoE+原生多模态+1-2M+thinking+Agent；8/8
    "09-GLM":     [1, 1, 0, 1, 1, 1, 1, 1],   # Z1 thinking+4V+128k+工具；9B 档非 MoE
    "10a-Kimi":   [1, 1, 1, 1, 1, 1, 1, 1],   # 1T MoE+K1.5 多模态推理+Agent
    "10b-MiniMax": [1, 1, 1, 1, 1, 1, 1, 0],  # M1 201B MoE 原生多模态；Agent 未标
    "10c-Doubao": [1, 1, 1, 1, 1, 1, 1, 1],   # Seed-Thinking=RL thinking 样本；中文多模态生态
    "10d-Hunyuan":[1, 1, 1, 1, 1, 1, 1, 0],   # MoE+128k+推理多模态；Agent 未标
    "10e-InternLM": [1, 1, 1, 1, 0, 1, 1, 0], # InternVL 多模态 128k；无 thinking/Agent 强调
}
# 各家族"差分项"（独门/领先贡献，§3.12 口径）——选型看这里
DIFF_LINE = {
    "01-GPT": "范式开创（ICL/RLHF/thinking 都由你这定义）",
    "02-Llama": "开源骨架四件套（RoPE/GQA/SwiGLU/RMSNorm）被抄成事实标准",
    "03-Qwen": "中文词表经济（0.3-0.5 token/字）+ 中文全家桶",
    "04-DeepSeek": "MLA+极致 MoE+开源 RL-for-Reasoning（R1）",
    "05-Mistral": "SWA 滑窗 + 普及 MoE（欧洲示范）",
    "06-Gemma": "端侧质量标杆（int4 入场券 + 256k 词表驻留账）",
    "07-Claude": "对齐可产品化（Constitutional/RLAIF）+ MCP 标准",
    "08-Gemini": "上下文即工作内存（1M-2M）+ 原生多模态 token 空间",
    "09-GLM": "填空续写一张脸（Autoregressive Blank Infilling）+ 2D RoPE",
    "10a-Kimi": "1T 级 MoE 开源 + Mooncake KV + 长上下文 Agent",
    "10b-MiniMax": "大 MoE 全开源第 2 样本 + 多模态原生",
    "10c-Doubao": "应用铺量（中文生态最大之一）+ 采样税前置成训练税",
    "10d-Hunyuan": "开源 MoE + 长上下文（国产第二梯队）",
    "10e-InternLM": "社区最强开源 VLM（InternVL）+ OpenCompass 评测",
}


def expA():
    print("=" * 68)
    print("[A] 全家族八要素横向表（十一个家族 · 14 个代表模型并排；要素事实=非本机实测）")
    print("=" * 68)
    print(f"    {'族':<11}{'架构':<12}{'Attention':<16}{'FFN':<16}{'训练':<22}{'上下文':<10}{'多模态':<14}{'Reasoning'}")
    for fn, mod, arch, attn, ffn, train, ctx, mm, reason in FAMILIES:
        print(f"    {fn+'-'+mod:<22}{arch:<10}{attn:<14}{ffn:<14}{train:<20}{ctx:<8}{mm:<13}{reason}")
    print("    → 横向看差异：Attention 一列最乱（MHA/GQA/MLA/SWA/交替/未公开），FFN 一列收敛")
    print("      （SwiGLU 系或 MoE 呈主流），训练一列 2024 后都在向『多阶段+RLVR』收；")
    print("      闭源（07/08/10c）三行多数格子『未公开』=要素卡的有效边界。")
    print("      复习：八要素=00 章模板；每格挂 02 章复习点（见 [C]）。")


def expB():
    print()
    print("=" * 68)
    print("[B] 跨家族派生账（本机真算 · 确定性算术；官方参数量=非本机实测；闭源按面值/未公开处理）")
    print("=" * 68)
    print("    参数账（与 00 章同函数同式；La=Llama Qw=Qwen Mx=Mixtral DS=DeepSeek 四格为 00 基线对账）：")
    print(f"      {'模型':<16}{'推导总参':<10}{'active参':<10}{'官方':<8}{'偏差':<8}{'fp16 GiB':<9}{'结构'}")
    rows = []
    for name, m in {**MODELS_OPEN, **MODELS_EXTRA}.items():
        c = count_params(m)
        if c is None:
            tot, act = m["official"], m["official_active"]
            err, ratio = "——", f"(act/总 {m['official_active']/m['official']*100:.1f}%)"
            struct = "MLA(官方)"
        else:
            tot, act = c["total"], c["active"]
            err = f"{(c['total']-m['official'])/m['official']*100:+.1f}%"
            ratio = f"(act/总 {act/tot*100:.1f}%)"
            struct = "推导"
        wgt = tot * 2 / GiB
        rows.append((name, tot, act, wgt, err, ratio, struct, m))
        print(f"      {name:<16}{fmt_g(tot):<10}{fmt_g(act):<10}{m['official']/1e9:<8.2f}{err:<8}{wgt:<9.1f}{struct}")
        if c is not None:
            assert abs((c['total']-m['official'])/m['official']) < 0.15, f"{name} 偏差越 ±15% 走廊"
    print("    → 推导六格（La/Qw/Mx 00 基线对账 + Gemma2-9B/GLM-4 两格全落 ±15% 走廊；DeepSeek MLA 走官方；")

    # 闭源/未公开面值行（非本机实测，按公开口径列）
    print("    闭源 API / 面值行（官方公布参量则以公布为准，未公开标『未公开』）：")
    closed = [
        ("07-Claude 系列", "未公开", "未公开"),
        ("08-Gemini 系列", "未公开（MoE 体量）", "未公开"),
        ("10c-Doubao Seed", "未公开", "未公开"),
        ("10a-Kimi K2",  "1T 级 MoE", "长上下文"),
        ("10b-MiniMax M1", "201B MoE", "原生多模态"),
        ("10d-Hunyuan",  "未公开（MoE 体量）", "128k 长上下文"),
        ("10e-InternLM", "面值 20B/100B 级", "InternVL"),
    ]
    for name, prm, diff in closed:
        print(f"      {name:<22}{prm:<16}{diff}")
    print("    → 闭源 API 三家的『参数』本身就是未公开信息——选型它们的维度是能力/生态/成本，")
    print("      不是参数账；这是『推理方式=闭源 API』格带来的选型规则变化（接 07/08 章）。")

    # KV 母线（结构已知家真算；闭源未公开 → 不做推测）
    print("    KV 母线（fp16 · 结构已知家真算；K = 上下文刻度）")
    print(f"      {'模型':<16}{'KV/层/token':<13}{'KV/token':<11}{'1k':>6}{'32k':>7}{'64k':>7}{'128k':>7}")
    for name, m in {**MODELS_OPEN, **MODELS_EXTRA}.items():
        per_layer = kv_bytes_per_layer_token(m)
        per_token = per_layer * m["L"]
        row = f"      {name:<16}{per_layer:>9}B{per_token:>11}B"
        for k in (1024, 32768, 65536, 131072):
            row += f"{per_token*k/GiB:>7.1f}"
        print(row)
        if name == "DeepSeek-V3":
            assert abs(per_token * 131072 / GiB - 8.6) < 0.05   # 与 00/02-04 交叉验证
        if name in ("Llama-3.1-8B", "Qwen2.5-7B", "Mixtral-8x7B"):
            exp = {"Llama-3.1-8B": (16.0,), "Qwen2.5-7B": (7.0,), "Mixtral-8x7B": (16.0,)}[name]
            assert abs(per_token * 131072 / GiB - exp[0]) < 0.2   # 与 00 章 run1 同数
    print("    → GQA 档一致的两家（Llama 8×32、Mixtral 8×32）128k 同为 ≈16 GiB；Qwen 4×28 更扁省一半；")
    print("      MLA 只用普通家 1/2（8.6 vs 16.0）；Gemma 交替注意力不省 KV 驻留（打分省、驻留同）。")


def expC():
    print()
    print("=" * 68)
    print("[C] 复习链断言：八要素每一条都能挂到 02 核心原理（或多模态/推理出口）——断言 8/8 全非空")
    print("=" * 68)
    for k, refs in REVIEW_CHAIN.items():
        assert refs, f"{k} 复习链为空"
        print(f"    {k} → {', '.join(refs)}")
    print("    → 横向表不造新知识：A/B 两段的词（GQA/MLA/SwGLU/MoE/多阶段/thinking）全是 02 章")
    print("      学过的决策点；选型前先能看到『这张表背后的复习链』=把 02 收货变成可调用知识。")


def expD():
    print()
    print("=" * 68)
    print("[D] 收敛度量化（8 特征 × 14 家族布尔判分）——『收敛度』看同质化，『差分』选独门")
    print("=" * 68)
    print("    特征序：RoPE | SwiGLU | MoE或小Dense | 多阶段 | thinking | 原生多模态 | ≥128k | Agent")
    scores = []
    order = {key: i for i, key in enumerate([fn for fn, *_ in FAMILIES])}
    for fn, mod, *_ in FAMILIES:
        key = fn
        prof = CONV_PROFILE[key]
        assert len(prof) == len(CONV_FEATS), f"{key} 特征数错"
        s = sum(prof)
        scores.append((s, -order[key], key))   # 并列按家族文件序号排
        bits = "".join("■" if b else "·" for b in prof)
        print(f"    {key:<15}{s}/8  [{bits}]")
        print(f"       差分项：{DIFF_LINE[key]}")
    scores.sort(reverse=True)
    ranked = [(s, key) for s, _, key in scores]
    hi = scores[0][0]; lo = scores[-1][0]
    tops = [key for s, _, key in scores if s == hi]
    print(f"    → 14 家收敛度 {hi}-{lo}/8；最高 {hi}/8 并列 {len(tops)} 家（{', '.join(tops)}）；"
          f"最低 {ranked[-1][1]}（保留最多独门 → 选型差异化来源）。")
    assert hi == 8 and lo <= 3
    assert len(tops) == 5   # GPT/DeepSeek/Gemini/Kimi/Doubao 5 家 8/8
    assert ranked[-1][1] == "05-Mistral"        # Mistral 3/8 最低（独门最多）
    print("    → 教学点：收敛度看『照抄的部分』=要与不要都得会（RoPE/SwiGLU/多阶段是入门的共同语言）；")
    print("      『差分项』才是选的依据——选型不选最优，选匹配（§3.11 口号）：中文→Qwen/GLM；")
    print("      端侧→Gemma；超长上下文→Gemini/Kimi；对齐与 Agent→Claude；开源推理→DeepSeek。")


def expE():
    print()
    print("=" * 68)
    print("[E] 决策引擎（规则表 + 断言自校；task→top-1 + 一句话理由 · 开源/闭源双路径）")
    print("=" * 68)

    def decide(task):
        if task == "中文优先":
            return "Qwen（中文全家桶 0.3-0.5 token/字）·GLM/GPT-4o 备选", "公开：词表账（03-Qwen A 段）；闭源：生态"
        if task == "单卡私有部署（≤40 GiB）":
            return "Qwen2.5-7B / Gemma3-4B", "权重 13-14 GiB（Qwen）/ int4 端侧（Gemma 06 章装载账）"
        if task == "超长上下文（1M+ 工作内存）":
            return "Gemini 系列（闭源）", "原生 1M-2M+thinking；开源侧 Kimi K2 长上下文（API/权重）"
        if task == "对齐与 Agent 标准":
            return "Claude 系列（闭源）", "Constitutional/RLAIF + MCP 发起者（07 章）"
        if task == "开源最大推理性价比":
            return "DeepSeek-R1/V3", "开源 RL-for-Reasoning 标杆 + MLA KV 省 8.6GiB（04 章）"
        if task == "端侧离线（手机/边缘）":
            return "Gemma3-1B/4B（int4）", "≤4 GiB 档装载（06 章装载账）；Phi/SmolLM 备选"
        if task == "原生全模态（文本/图/音/视频）":
            return "Gemini 系列（闭源）", "统一 token 空间（08 章）；开源侧 InternVL/GLM-4V"
        if task == "研究/可复现标本":
            return "DeepSeek（论文+权重）· OLMo（数据全开）", "研究优先全开放（知识地图 §3.10 OLMo 注）"
        raise ValueError(task)

    for task in ["中文优先", "单卡私有部署（≤40 GiB）", "超长上下文（1M+ 工作内存）",
                 "对齐与 Agent 标准", "开源最大推理性价比", "端侧离线（手机/边缘）",
                 "原生全模态（文本/图/音/视频）", "研究/可复现标本"]:
        a, why = decide(task)
        b, _ = decide(task)
        assert a == b, f"{task} 决策不稳定"
        print(f"    {task}\n        → {a}\n        理由：{why}")
    print("    → 8 任务全断言通过（同一输入永远同一推荐）；选型=拿任务需求×差分项→同一结论，")
    print("      不是『哪个模型最强』而是『哪个家族匹配这个任务』——这就是 §3.11 的『选匹配』。")


def main():
    print("=" * 68)
    print("cross_family_demo：03-模型家族 11-八要素横向对比表（知识地图 §3.11 收束章·第 1 弹）")
    print("    十一个家族一张表 · 收敛度量化 · 选型选匹配 —— 收束 03 模型家族篇")
    print("=" * 68)
    print("[0] 口径：横向表要素=非本机实测（知识地图 §3.1-3.10 + 各篇家族章，无外网未在线复核）；")
    print("    派生账/收敛度/决策=确定性算术与规则（本机真算）；闭源家族参数未公开处如实标『未公开』。")
    expA()
    expB()
    expC()
    expD()
    expE()


if __name__ == "__main__":
    main()
    logw(f"[cross_family_demo] wall-clock {time.perf_counter()-t0:.1f} s")
    print()
    print("done · 一键复现：python code/scripts/cross_family_demo.py")
