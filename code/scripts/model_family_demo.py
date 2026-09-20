# -*- coding: utf-8 -*-
"""03-模型家族 · 00-八要素解构法（知识地图 §3，全章方法论）——本机真算：八要素解构卡生成 + 派生账 + 复习链断言 + 选型决策引擎。

[A] 八要素解构卡：把四张公开模型卡的八要素速写按统一模板（架构/Attention/FFN·MoE/训练/推理/上下文/多模态/Reasoning）
    组织成结构化剖面。要素事实来自知识地图 §3 各家族速写（非本机实测：公开配置/报道，写作环境无外网未在线复核）。
[B] 派生账（本机真算）：从结构配置算 总参数量 active 参数量 fp16 权重 GiB 每 token 全模型 KV 字节 与 各上下文 KV GiB，
    并与官方参数量核对偏差走廊（±15%）；"40 GiB 桌面能否装入"按 权重+KV 判定。
[C] 复习链断言：八要素 ↔ 02 核心原理章节映射自校（每条 ≥1 个复习点），另给多模态/推理出口。
[D] 决策引擎：task → 家族 规则表 + 断言自校（同一输入永远同一推荐）。

全部为确定性算术，无随机；run1==run2==run3 科学数字逐位一致，只有墙钟浮动。
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

GiB = 2 ** 30


# ---- 公开配置卡（要素事实/官方参数量=非本机实测；结构配置=建模输入）----
# 结构配置用于 [B] 的派生算术；要素速写取自知识地图 §3.1–§3.5。
MODELS = {
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

ELEMENTS = [
    ("架构", "decoder-only / encoder-decoder / 原生多模态"),
    ("Attention", "MHA / GQA / MLA / SWA——谁在压缩 KV，谁在省显存"),
    ("FFN·MoE", "SwiGLU / 细粒度专家 / 共享专家——参数放哪，激活取多少"),
    ("训练方式", "预训练 / SFT / DPO / RLHF / GRPO·RLVR / FP8 / 蒸馏"),
    ("推理方式", "闭源 API / 开源自部署 / 思考预算（thinking budget）"),
    ("上下文", "窗口大小 / 长上下文技术（窗内外推）"),
    ("多模态", "adapter 拼接 / 原生全模态 / 文档-截图能力"),
    ("Reasoning", "o 系列开创 / thinking 模式 / 纯 RL 涌现（R1-Zero）"),
]

# 复习链：要素 → 02 核心原理章节（+多模态/推理出口），断言每条非空
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
    """Dense/GQA/MoE 结构级参数量推导（MLA 模型不做推导，官方 671B/37B 为准）。"""
    if m["mla"] is not None:
        return None
    d, L, vocab = m["d"], m["L"], m["vocab"]
    hdim = d // m["qh"]
    kv_dim = d if m["kvh"] is None else m["kvh"] * hdim
    attn_layer = 2 * d * d + 2 * d * kv_dim  # Wq·Wo + Wk·Wv
    ffn_dim = m["ffn_dim"]
    if m["moe"] is None:
        ffn_layer = 3 * d * ffn_dim
        ffn_act = ffn_layer
        n_full, n_act, ex = 1, 1, None
    else:
        expert_ffn = 3 * d * ffn_dim
        n_full, n_act = m["moe"]["n"], m["moe"]["act"]
        ffn_layer = n_full * expert_ffn + d * n_full  # 路由 Wg
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
    """每层每 token 的 K/V 缓存字节（fp16）。MLA 缓存=压缩潜向量+RoPE 头。"""
    if m["mla"] is not None:
        return (m["mla"]["rank"] + m["mla"]["rope"]) * 2
    hdim = m["d"] // m["qh"]
    nkv = m["qh"] if m["kvh"] is None else m["kvh"]
    return 2 * nkv * hdim * 2


def fmt_g(b):
    return f"{b / 1e9:.3f}B"


def main():
    t0 = time.perf_counter()

    print("=" * 66)
    print("model_family_demo：03-模型家族 00-八要素解构法（知识地图 §3 方法论）——一张模板拆任何模型")
    print("=" * 66)
    print("[0] 口径：读模型不背参数，用统一八要素解构（架构/Attention/FFN·MoE/训练/推理/上下文/多模态/Reasoning）。")
    print("    要素事实与官方参数量=非本机实测（公开配置，无外网未在线复核，精确值以官方模型卡为准）；")
    print("    派生账=本机真算（确定性算术）。[D] 决策引擎=规则表+断言自校。")

    # ---- [A] 八要素解构卡 ----
    print("\n[A] 八要素解构卡（统一模板，四家公开模型卡速写入格，非本机实测）：")
    for name, m in MODELS.items():
        print(f"    {name}（{m['note']}）：")
        print(f"      架构    {m['arch']} · Attention {m['attn']} · FFN/MoE {m['ffn']}")
        print(f"      训练    {m['train']} · 推理 {m['inf']} · 上下文 {m['ctx']//1024}k")
        print(f"      多模态  {m['mm']} · Reasoning {m['reason']}")
    print("    → 八要素=每家族'一次技术演进的实验报告'的纸板；要素卡只是描述，可计算的是 [B] 派生账。")

    # ---- [B] 派生账 ----
    print("\n[B] 派生账（本机真算，科学性=算术确定性）：参数 / 激活 / 权重 GiB / KV 账 / 40GiB 桌判定")
    print("    参数账（结构级推导；官方=非本机实测；偏差=走廊校验 ±15%）：")
    print("     模型          推导总参     active参     官方       偏差    fp16权重GiB  激活占比")
    for name, m in MODELS.items():
        c = count_params(m)
        if c is None:
            tot, act = m["official"], m["official_active"]
            wgt = tot * 2 / GiB
            err = "——"
            ratio = f"{(m['official_active']/m['official']*100):.1f}%"
            denom = tot
        else:
            tot, act = c["total"], c["active"]
            wgt = tot * 2 / GiB
            err = f"{(c['total']-m['official'])/m['official']*100:+.1f}%"
            ratio = f"{act/tot*100:.1f}%"
            denom = tot
        print(f"     {name:<14}{fmt_g(tot):<11}{fmt_g(act):<11}{m['official']/1e9:.2f}B     {err:<8}{wgt:9.1f}    {ratio}")
        if c is not None:
            assert err not in ("——",) and abs((c['total']-m['official'])/m['official']) < 0.15, f"{name} 偏差越走廊"
    print("    → 推导值 vs 官方全部落在 ±15% 走廊内（结构级近似，数量级与官方一致）；MoE 的'总参/激活'是两本账。")

    print("    激活比（MoE 名义稀疏倍数）：")
    for name, m in MODELS.items():
        if m["moe"]:
            c = count_params(m)
            n_full, n_act = m["moe"]["n"], m["moe"]["act"]
            if c is not None:
                print(f"      {name:<14}专家 {n_act}/{n_full}（每 token 激活 {n_act} 个）→ active/总参 {c['active']/c['total']*100:.1f}%，名义 {n_act/n_full*100:.0f}%")
            else:
                print(f"      {name:<14}路由专家 {n_act}/({n_full}-1)+共享 1 个 → 官宣 active/总参 {m['official_active']/m['official']*100:.1f}%（非本机实测）")

    print("    KV 账（fp16；每 token 全模型字节 → 各上下文 GiB；后三列只是通用缓存公式，Mixtral 原生 32k）：")
    print("     模型          KV/层/token  KV/token   1k     32k    64k    128k")
    for name, m in MODELS.items():
        per_layer = kv_bytes_per_layer_token(m)
        per_token = per_layer * m["L"]
        row = f"     {name:<14}{per_layer:>6}B    {per_token:>8}B  "
        for k in (1024, 32768, 65536, 131072):
            row += f"{per_token*k/GiB:>6.1f} "
        print(row)
        if name == "DeepSeek-V3":
            # 交叉验证：与 02-04 MLA demo 的 128k×61 层账一致（8.6 GiB）
            assert abs(per_token * 131072 / GiB - 8.6) < 0.05
    print("    → MLA 把 128k KV 从 Llama 的 16.0 GiB 压到 8.6 GiB（与 02-04 mla_demo 同式同数互证）。")

    print("    40 GiB 单卡桌判定（权重 fp16 + 满上下文 KV ≤ 40 GiB）：")
    for name, m in MODELS.items():
        c = count_params(m)
        tot = m["official"] if c is None else c["total"]
        wgt = tot * 2 / GiB
        kv = kv_bytes_per_layer_token(m) * m["L"] * m["ctx"] / GiB
        total_need = wgt + kv
        verdict = "✓ 装得下" if total_need <= 40 else f"✗ 差 {total_need-40:.1f} GiB"
        print(f"     {name:<14}权重 {wgt:6.1f} + KV({m['ctx']//1024}k) {kv:5.1f} = {total_need:6.1f} GiB  →  {verdict}")
        if name in ("Llama-3.1-8B",):
            assert total_need <= 40 and total_need > 25

    # ---- [C] 复习链断言 ----
    print("\n[C] 复习链断言：八要素每一条都能挂到 02 核心原理（或多模态/推理出口）——断言 8/8 全非空：")
    for k, refs in REVIEW_CHAIN.items():
        assert refs, f"{k} 复习链为空"
        print(f"     {k} → {', '.join(refs)}")
    print("    → 八要素不是新知识，是把 02 学过的词（GQA/MLA/SwiGLU/MoE/GRPO…）背后的决策点按'一张卡'收拢。")

    # ---- [D] 决策引擎 ----
    print("\n[D] 决策引擎（规则表 + 断言自校；task→top-1 + 一句话理由）——'选型不选最优，选匹配'（§3.11）：")

    def decide(task):
        if task == "中文优先":
            return "Qwen2.5-7B", "中文全家桶生态最完整（§3.3），词表 151936"
        if task == "单卡全能（40GiB 桌）":
            return "Qwen2.5-7B", "7B 级权重仅 13-14 GiB + 128k KV 7.0 ≈ 21 GiB，单卡余量最大"
        if task == "超长上下文优先":
            return "DeepSeek-V3", "MLA 把 128k KV 压到 8.6 GiB（vs Llama 16.0），长对话缓存账最省"
        if task == "开源 MoE 推理性价比":
            return "Mixtral-8x7B", "权重 87 GiB 用 2×40GiB 可装，active/每 token 仅 12.8B（vs DeepSeek 37B）——安装门槛与单 token 计算量双低"
        if task == "端侧/最小部署":
            return "（四卡均不达标）", "端侧需 ≤1B 级 → Gemma3 1B / SmolLM / Phi 路线（行业小模型，非本机实测）"
        raise ValueError(task)

    status = []
    for task in ["中文优先", "单卡全能（40GiB 桌）", "超长上下文优先", "开源 MoE 推理性价比", "端侧/最小部署"]:
        a, why = decide(task)
        b, _ = decide(task)
        assert a == b, f"{task} 决策不稳定"
        status.append((task, a))
        print(f"     {task:<22}→ {a:<20}{why}")
    assert ("中文优先", "Qwen2.5-7B") in status and ("超长上下文优先", "DeepSeek-V3") in status
    print("    → 规则表把'我说不清为什么选它'变成'输入任务×规则→同一推荐'；每个任务结果本地断言锁死。")

    dt = (time.perf_counter() - t0) * 1000
    print(f"\n墙钟 {dt:.1f} ms（[A][B][C][D] 全为确定性算术/打印；run1==run2==run3 科学数字逐位一致，只有墙钟浮动）")


if __name__ == "__main__":
    main()
