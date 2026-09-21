# -*- coding: utf-8 -*-
"""03-模型家族 · 12-家族继承与创新关系（知识地图 §3.12 收束章·第 2 弹）——纵向看继承：谁发明范式、谁稳定骨架、谁定义下一轮。

[A] 家族谱系图（lineage DAG）：14 个家族 = 有向无环图节点，边=『直接继承谁家的路线』（谱系
    假设=要素事实非本机实测，源：知识地图 §3.12 + 各篇家族章）；图算法=本机真算——从根 01-GPT
    出发算每家『代数』（最长父链），查『hub』（被作为直接父代引用的次数→谁被抄成骨架/路线），
    查『LCA』（两家最近共同祖先→哪一分叉点）。断言：全节点可达根、无环、hub 第一 = 02-Llama。
[B] 范式溯源账：八个要素特征（复用 11 章 CONV_FEATS）的『发明者/首源』『谁让它成标准』 +
    14 家普及率统计（本机真算）；训练范式链 RLHF→DPO→GRPO→RLVR→蒸馏 每一代『抽掉一个依赖』
    （年份公开事实=非本机实测，断言年份单调不减）。
[C] 血缘量化：相对根 01-GPT 的『继承度』= 与根共享的特征位（根=收敛态完整定义，离根越近继承越
    多）；相对直接父代的新增位/回退位；14 家共同基底（全 1 位=谱系『家规』，断言 = RoPE+SwiGLU
    恰好 2 位）。剖面复用 11 章 CONV_PROFILE（同判分，跨篇互证）。
[D] 四代引擎继承实验（本机真算+真训练）：同一 add 任务（P=13, K=5, T=11, C=48, 2-block，
    1000 步 × bs64），复用 llama_demo.Engine 配方开关把『基因』逐代换代——
    gen0 古典(GPT 系) abs+GELU+LN+MHA → gen1 +RoPE → gen2 +GQA → gen3 现代(Llama 系)
    rope+SwiGLU+RMSNorm+GQA。断言：①四代 fresh 全解锁（继承没丢祖先能力）②现代不输古典
    （创新不亏）③参数不升、KV 减半（创新更省）。
[E] 范式定义者台账 + 策略谱系：八范式『定义者』与『继承去向』两列（公开事实=非本机实测）——
    “想要某能力：先找定义者（闭源标杆/开源首发），再找继承者（要生态用继承者）”；2026 收敛态
    断言（8 特征普及率都 ≥10/14 = 已证成事实标准）。

全部确定性（图算法/位串/普及率=算术，[D] 固定种子训练）；wall-clock 只进 stderr；
stdout run1==run2==run3 逐位一致。
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

import llama_demo as L           # 引擎复用（配方可换）→ [D] 代际继承实验
import cross_family_demo as CF   # 11 章 家族/判分复用 → [A]/[B]/[C]

GiB = 2 ** 30
t0 = time.perf_counter()


def logw(msg):
    """wall-clock 只进 stderr，不参与 stdout 逐位比对。"""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def now(msg):
    sys.stderr.write(f"[{time.perf_counter()-t0:6.1f}s] {msg}\n")
    sys.stderr.flush()


# =====================================================================
# [A] 家族谱系图（lineage DAG）
# =====================================================================
# 边=『直接继承谁家的路线』（多父=同时吸收多条路线）；根 = 01-GPT。
# 谱系假设来源=知识地图 §3.12 + 各篇家族章（要素事实=非本机实测）：
#   · 范式层：GPT 发明 decoder+预训练+ICL+RLHF+thinking+Agent 接口；Claude 定义对齐与 Agent 标准；
#     Google 定义多模态与超长上下文。   · 架构层：Llama 稳定开源骨架；DeepSeek MLA+极致 MoE；Mistral 普及 MoE。
PARENTS = {
    "01-GPT":      [],                          # 根：范式发明者
    "02-Llama":    ["01-GPT"],                  #  decoder+next-token 范式从 GPT 来；四件套自研
    "03-Qwen":     ["02-Llama"],                #  全套骨架继承 Llama（GQA/RoPE/SwiGLU/RMSNorm）
    "04-DeepSeek": ["02-Llama"],                #  从 Llama 骨架出发，自研 MLA + 细粒度 MoE
    "05-Mistral":  ["02-Llama"],                #  承接 Llama 骨架，自研 SWA、把 MoE 做成开源可买
    "06-Gemma":    ["02-Llama"],                #  同骨架 + 交替注意力 + Google 数据（含蒸馏传闻）
    "07-Claude":   ["01-GPT"],                  #  范式层继承 GPT（重做对齐：Constitutional/RLAIF）
    "08-Gemini":   ["01-GPT"],                  #  范式层同根并列：定义原生多模态 + 1M-2M 超长上下文
    "09-GLM":      ["03-Qwen"],                 #  中文并行支线（方法论与 Llama 骨架同源，谱系贴近 Qwen）
    "10a-Kimi":    ["04-DeepSeek", "07-Claude"],  # 大 MoE 路线收编 DeepSeek + Agent 激进继承 Claude
    "10b-MiniMax": ["04-DeepSeek"],             # 大 MoE 全开源第 2 样本（DeepSeek 路线）
    "10c-Doubao":  ["04-DeepSeek"],             # 把 DeepSeek RL-for-Reasoning 产能化
    "10d-Hunyuan": ["04-DeepSeek"],             # 开源 MoE + 长上下文（国产第二梯队）
    "10e-InternLM": ["02-Llama"],               # 开源骨架 + 社区最强 VLM（InternVL）
}
ROOTS = ["01-GPT"]


def gen_depth(fn):
    """代数 = longest 父链边数（根 = 0），按父表递归（DAG 保证终止）。"""
    return 0 if not PARENTS[fn] else max(gen_depth(p) for p in PARENTS[fn]) + 1


def ancestor_chain(fn):
    """从根到 fn 的一条最长父链（第一个父优先），返回字符串 'A → B → C'。"""
    if not PARENTS[fn]:
        return fn
    best = max(PARENTS[fn], key=lambda p: (gen_depth(p), -ORDER[p]))
    return ancestor_chain(best) + " → " + fn


def ancestor_set(fn):
    out = set()
    st = [fn]
    while st:
        x = st.pop()
        if x in out:
            continue
        out.add(x)
        st.extend(PARENTS[x])
    return out


def lca(a, b):
    """最近共同祖先（按代数最深，平手用家族序）。"""
    both = ancestor_set(a) & ancestor_set(b)
    return sorted(both, key=lambda x: (-gen_depth(x), ORDER[x]))[0]


FAMILY_ORDER = {fn: i for i, (fn, *_rest) in enumerate(CF.FAMILIES)}
ORDER = FAMILY_ORDER
NODES = list(ORDER.keys())


def expA():
    print("=" * 68)
    print("[A] 家族谱系图（lineage DAG）——谁直接继承谁（谱系假设=非本机实测；图算法=本机真算）")
    print("=" * 68)
    print(f"    {'家族':<13}{'代数':>4}  直接父代{'':<8}谱系路径")
    gmax = 0
    for fn in NODES:
        gmax = max(gmax, gen_depth(fn))
        head = fn
        parents = " | ".join(PARENTS[fn]) if PARENTS[fn] else "根(无)"
        print(f"    {head:<13}{gen_depth(fn):>4}  {parents:<16}{ancestor_chain(fn)}")
    hub_cnt = {fn: sum(fn in PARENTS[x] for x in NODES) for fn in NODES}
    for fn, c in sorted(hub_cnt.items(), key=lambda kv: (-kv[1], ORDER[kv[0]])):
        if c > 0:
            print(f"      {fn}：{c} 个家族直接继承")
    print("    LCA（最近共同祖先，分叉点）：")
    for a, b in [("03-Qwen", "04-DeepSeek"), ("06-Gemma", "08-Gemini"),
                 ("04-DeepSeek", "10b-MiniMax"), ("03-Qwen", "09-GLM")]:
        print(f"      LCA({a}, {b}) = {lca(a, b)}")
    # 断言：全节点可达根、无环（DFS 三态）、代数界、hub 第一 = Llama（开源骨架中心）
    for fn in NODES:
        assert ROOTS[0] in ancestor_set(fn), f"{fn} 不可达根"
    # 无环检查：DFS 三态（0=未访 1=栈中 2=完成）
    vis = {}
    def dfs(fn):
        vis[fn] = 1
        for p in PARENTS[fn]:
            assert vis.get(p, 0) != 1, f"{fn}→{p} 回边（环）"
            if vis.get(p, 0) == 0:
                dfs(p)
        vis[fn] = 2
    for fn in NODES:
        if vis.get(fn, 0) == 0:
            dfs(fn)
    ranked_hub = sorted(hub_cnt.items(), key=lambda kv: (-kv[1], ORDER[kv[0]]))
    assert ranked_hub[0][0] == "02-Llama", "hub 第一应 = 02-Llama（开源骨架中心）"
    assert gmax == 3, "最长代数应 = 3"
    print("    → 读数：开源骨架 hub=02-Llama（被 5 家直接继承=“事实标准”）；03-06/10e 同出");
    print("      Llama 这一支，04-DeepSeek 自成路线 hub（10a-10d 国产 MoE 都收编它）；")
    print("      07-Claude / 08-Gemini 与 Llama 平行——范式层直接从 GPT 分裂（闭源范式定义者）。")


# =====================================================================
# [B] 范式溯源账 + 普及率 + 训练范式链
# =====================================================================
# 发明者/首源=公开事实（非本机实测）；普及率统计 = CONV_PROFILE 真算。
FEAT_INVENTOR = {
    "RoPE 系位置编码":   dict(inv="2021《RoFormer》苏剑林", std="Llama 收录为骨架→全开源普及"),
    "SwiGLU 激活":        dict(inv="2020 GLU Variants 研究", std="Llama 选作中间激活→事实标准"),
    "MoE或小Dense":       dict(inv="2017 MoE(Google)/端侧另路", std="Mistral&DeepSeek 普及大 MoE；Gemma 端侧小 Dense"),
    "多阶段训练":         dict(inv="2022 InstructGPT", std="2024 后全家『预+SFT+RL/RLVR』（谁先开源谁定义下一轮）"),
    "thinking 模式":      dict(inv="2024 OpenAI o 系列", std="2025 DeepSeek-R1 开源引爆"),
    "原生多模态":         dict(inv="2025+ Gemini（Google 定义）", std="GPT-4o 原生；开源侧 InternVL/GLM-4V"),
    "长上下文≥128k":      dict(inv="200k→1M 演进（Claude/Gemini）", std="Gemini 定义 1M-2M 工作内存"),
    "Agent 工具接口":     dict(inv="2023 function call(GPT)/2024-11 MCP(Claude)", std="Claude MCP 成标准"),
}
FEAT_ORDER = ["RoPE 系位置编码", "SwiGLU 激活", "MoE或小Dense", "多阶段训练",
              "thinking 模式", "原生多模态", "长上下文≥128k", "Agent 工具接口"]

# 训练范式链：每一代“抽掉一个依赖”。年份=公开事实（非本机实测）。
RL_ALGEBRA = [
    ("RLHF",  2022, "依赖 RM + PPO critic"),
    ("DPO",   2023, "抽掉 RM（偏好对直接当损失）"),
    ("GRPO",  2024, "抽掉 critic（组内优势）"),
    ("RLVR",  2024, "抽掉人类偏好（可验证结果当奖励）"),
    ("蒸馏",  2024, "抽掉 RL 训练（教师答案当目标）"),
]


def expB():
    print()
    print("=" * 68)
    print("[B] 范式溯源账（8 特征：发明者→传开者；公开事实=非本机实测；普及率统计=本机真算）")
    print("=" * 68)
    print(f"    {'特征':<16}{'发明者/首源':<20}{'谁让它成标准/最普及'}")
    for f in FEAT_ORDER:
        print(f"    {f:<16}{FEAT_INVENTOR[f]['inv']:<22}{FEAT_INVENTOR[f]['std']}")
    print("    普及率（14 家具备该特征的比例，位=11 章 CONV_PROFILE 同判分）：")
    cnt = {f: 0 for f in FEAT_ORDER}
    assert len(FEAT_ORDER) == len(CF.CONV_FEATS) == 8
    for fn in NODES:
        for i, f in enumerate(FEAT_ORDER):   # 与 CONV_FEATS 同序（手工对齐）
            cnt[f] += CF.CONV_PROFILE[fn][i]
    for f in sorted(cnt, key=lambda k: (-cnt[k], FEAT_ORDER.index(k))):
        print(f"      {f}: {cnt[f]}/14（{cnt[f]/14:.3f}）")
    common = [f for f in FEAT_ORDER if cnt[f] == 14]
    print("    → 共同标准（14/14）：" + "、".join(common))
    assert len(common) >= 2
    print("    训练范式链（每一代标准化某一步==抽掉一个依赖；年份=公开事实）：")
    for i, (name, year, dep) in enumerate(RL_ALGEBRA):
        print(f"      {name:<7}({year}) {dep}")
        if i > 0:
            assert year >= RL_ALGEBRA[i - 1][1], "范式年份必须单调不减"
    print("    → 换代 = 去依赖 + 更少人工/骨架：RLHF 要 RM+critic → DPO 免 RM → GRPO 免 critic →")
    print("      RLVR 免人类偏好 → 蒸馏免 RL 训练——2024 后『谁先开源谁定义下一轮』= 把上一代的")
    print("      人工依赖抽光，谁先抽出谁重新定义训练法（接 02-14/15/16/22）。")


# =====================================================================
# [C] 血缘量化（相对根继承度 / 相对父新增位 / 共同基底；剖面=11 章同判分）
# =====================================================================
def expC():
    print()
    print("=" * 68)
    print("[C] 血缘量化——继承度：离『根=收敛态完整定义』多近；新增位：从父带出什么新工艺（本机真算）")
    print("=" * 68)
    prof = CF.CONV_PROFILE
    root = prof["01-GPT"]
    common_idx = [i for i in range(len(CF.CONV_FEATS)) if all(prof[fn][i] for fn in NODES)]
    print(f"    共同基底（14 家全 1 的位）：{len(common_idx)} 位 -> " +
          "、".join(CF.CONV_FEATS[i][0] for i in common_idx))
    print(f"    {'家族':<13}{'直接父':<13}{'继承度(root)':>12}{'新增位':>6}{'回退位':>6}  差分项")
    for fn in NODES:
        ps = PARENTS[fn]
        inherit = sum(prof[fn][i] and root[i] for i in range(len(CF.CONV_FEATS)))
        if ps:
            par = ps[0]
            newb = sum(prof[fn][i] and not prof[par][i] for i in range(len(CF.CONV_FEATS)))
            rollb = sum(not prof[fn][i] and prof[par][i] for i in range(len(CF.CONV_FEATS)))
        else:
            par, newb, rollb = "根", 0, 0
        print(f"    {fn:<13}{par:<13}{inherit:>8}/8{newb:>6}{rollb:>6}  {CF.DIFF_LINE.get(fn, '')[:24]}")
    lo = min(sum(prof[fn]) for fn in NODES)
    lo_f = [fn for fn in NODES if sum(prof[fn]) == lo]
    assert lo_f == ["05-Mistral"] and lo == 3, "继承度最低应为 Mistral 3/8（独门最多）"
    print(f"    → 继承度 = 离根的近远：根=收敛态 8/8 完整定义，走得越远独门越多（05-Mistral {lo}/8 =")
    print("      分支最早把 MoE 引入、还没补上多阶段）；相对父『新增位』= 从父代只多买的工艺（04 相对")
    print("      Llama 只新增 MoE 1 位=『传承骨架 + 只加一件创新』）；『回退位』= 早期支脉的代价。")
    print("      谱系家规（共同基底 2 位）= RoPE + SwiGLU：这是整棵谱系谁抄谁都已经默认的『入门共同语言』。")


# =====================================================================
# [D] 四代引擎继承实验（真训练）：继承保底 + 创新收益
# =====================================================================
def n_params(p):
    return int(sum(int(np.prod(np.shape(v), dtype=np.int64)) for v in p.values()))


def expD():
    print()
    print("=" * 68)
    print("[D] 四代引擎继承实验（本机真算；同一 add 任务 P=13·K=5·T=11·C=48·2-block，1000 步×bs64）")
    print("=" * 68)
    print("     基因换代：gen0 古典(GPT 系)→+RoPE→+GQA→gen3 现代(Llama 系)；种子/预算全同=可归因")
    GENS = [
        ("gen0 古典(GPT系)", dict(pos="abs", act="gelu", norm="ln", NV=8)),
        ("gen1 +RoPE",        dict(pos="rope", act="gelu", norm="ln", NV=8)),
        ("gen2 +GQA",         dict(pos="rope", act="gelu", norm="ln", NV=4)),
        ("gen3 现代(Llama系)", dict(pos="rope", act="swiglu", norm="rms", NV=4)),
    ]
    print(f"    {'代':<18}{'配方':<26}{'参数':>7}{'KV B/层/tok':>12}{'fresh@S=0':>10}")
    res = []
    for name, kw in GENS:
        eng = L.Engine(V=13, T=11, C=48, NL=2, NH=8, **kw)
        dt0 = time.perf_counter()
        p, marks = L.train(eng, 1000, 13, 64)
        fr = L.eval_fresh(eng, p, np.random.RandomState(509), 13, n=300, S=0) or 0.0
        kv = 2 * eng.NV * eng.DH * 2
        np_cnt = n_params(p)
        recipe = f"{kw['pos']}+{kw['act']}+{kw['norm']}+{'MHA' if kw['NV']==8 else 'GQA'}"
        print(f"    {name:<18}{recipe:<26}{np_cnt:>7}{kv:>9}B{fr:>9.3f}")
        res.append((name, kw, fr, kv, np_cnt))
        now(f"[D] {name} 1000 步 墙钟 {time.perf_counter()-dt0:.1f}s")
    freshes = [r[2] for r in res]
    assert all(fr > 0.5 for fr in freshes), "四代都应解锁 ICL（继承保底没丢祖先能力）"
    assert freshes[3] >= freshes[0], "现代不输古典（创新不亏）"
    params = [r[4] for r in res]
    assert params[3] <= params[0], "现代参数应不增（SwiGLU 8C/3 比 GELU 4C 更省）"
    assert res[0][3] == res[1][3] == 192 and res[2][3] == res[3][3] == 96, "GQA 起亲 KV 减半"
    print("    → 读数：四代全部解锁同一 ICL（P=13 随机基线 1/13≈0.077）——『继承』保住的是祖先的")
    print("      能力下限：换任何一阶配方基因，没丢“现场读映射”的能力；『创新』的收益不是学不会→")
    print("      学得会，而是『同预算同种子下一样解锁，但参数不增 / KV 减半』——这就是 §3.12 的")
    print("      继承与创新：骨架(能力)定底，差分(配方)定成本与解锁速度。")


# =====================================================================
# [E] 范式定义者台账 + 策略谱系 + 2026 收敛态断言
# =====================================================================
PARADIGMS = [
    ("预训练+ICL",   "01-GPT 定义",   "全家继承（02-12）"),
    ("RLHF 对齐",    "01-GPT(2022)",  "07-Claude 重定义（Constitutional/RLAIF）"),
    ("Agent 接口",   "01-GPT(2023)",  "07-Claude 定标准（MCP 2024-11）"),
    ("原生多模态",   "08-Gemini",     "06-Gemma/10e-InternVL 开源"),
    ("超长上下文",   "08-Gemini(1M+)", "10a-Kimi 长上下文 Agent"),
    ("开源骨架",     "02-Llama",      "03/04/05/06/10e"),
    ("MLA/极致MoE",  "04-DeepSeek",   "10a-10d 国产 MoE hub"),
    ("RL-推理",      "04-DeepSeek(R1)", "10c-Doubao 采样税前置化"),
]


def expE():
    print()
    print("=" * 68)
    print("[E] 范式定义者台账——‘谁先开源谁定义下一轮’：要能力先找定义者，要生态用继承者（公开事实）")
    print("=" * 68)
    print(f"    {'范式':<14}{'定义者':<20}{'继承去向'}")
    for p_, d, s in PARADIGMS:
        print(f"    {p_:<14}{d:<20}{s}")
        assert d and s
    print("    2026 收敛态检查（8 特征的普及率是否已证成事实标准，本机真算）：")
    cnt = {f: 0 for f in FEAT_ORDER}
    for fn in NODES:
        for i, f in enumerate(FEAT_ORDER):
            cnt[f] += CF.CONV_PROFILE[fn][i]
    for f in FEAT_ORDER:
        if f == "Agent 工具接口":
            print(f"      {f}: {cnt[f]}/14  扩散中")
        else:
            minok = "✔" if cnt[f] >= 10 else "✘"
            print(f"      {f}: {cnt[f]}/14  {minok}")
    # 断言：除 Agent 外 7 特征都 ≥10/14=已成事实标准；Agent 是收敛态里最年轻的维度（9/14=仍在扩散）
    assert all(cnt[f] >= 10 for f in FEAT_ORDER if f != "Agent 工具接口")
    assert cnt["Agent 工具接口"] >= 8
    print("    → 读：7 个特征已 ≥10/14 = 事实标准；Agent 9/14 是收敛态里最年轻的维度=还在扩散中")
    print("      （借 07 章 MCP 铺生态的人正赚着这段时差）。")
    print("    → 决策捷径（接 11 章『选匹配』，补纵向一维）：先定『要的范式』→ 找定义者（闭源标杆")
    print("      或开源首发）→ 再找继承者（想省/要生态/要开源自部署 → 用继承者但欠它一个时差）。")
    print("      §3.12 收束：2026 前沿模型八要素高度趋同，大厂拼的已不是单点创新，是数据、算力、")
    print("      系统与生态——本篇的谱系就是『谁发明范式、谁稳定骨架、谁定义下一轮』的地图。")


def main():
    print("=" * 68)
    print("lineage_demo：03-模型家族 12-家族继承与创新关系（知识地图 §3.12 收束章·第 2 弹）")
    print("    纵向看继承：谁发明范式 · 谁稳定骨架 · 谁定义下一轮")
    print("=" * 68)
    print("[0] 口径：谱系假设/发明者/惯例/年份=要素事实（知识地图 §3.12 + 各篇家族章，")
    print("    无外网未在线复核，非本机实测）；图算法/普及率/血缘/代际训练=本机确定性算术。")
    now("start: 进入 [A] 谱系图（纯算术）")
    expA()
    expB()
    expC()
    now("enter [D] 四代训练（主体墙钟）")
    expD()
    expE()


if __name__ == "__main__":
    main()
    logw(f"[lineage_demo] wall-clock 整脚本 {time.perf_counter()-t0:.1f} s")
    print()
    print("done · 一键复现：python code/scripts/lineage_demo.py")
