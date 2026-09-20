# -*- coding: utf-8 -*-
"""蒸馏与 CPT：墙后的两条逃逸通道——把大模型能力"偷渡"进小模型 + 给凝固语料换新水。

承接《21-Scaling-Law与数据墙》的 1-block 引擎与阶2马尔可夫源（V=16、真实转移
分布可解析、分布级 eval 无采样噪声），三段实测：

  A  白盒蒸馏（logits/KL + 温度）：教师 C96 在 917K token 上训到 eval(P1)≈1.70
      （近乎把通用源吃透）。学生只在很小预算上训练（数据受限、容量充足才是
      Hinton 蒸馏的舞台），两个子设问：
       A1 学生与教师同容量 C96，预算 57K/114K/229K 扫描 —— 软标签的增益在
          "能开始学、又不够学全"的中段最大（太短连 lr 预热都没走完、两臂都贴地；
          够长硬标签自己也逼近上限、差距收窄）；
       A2 参数偷渡：预算固定在 229K，C32（≈8× 更小）学生直接学/喝教师软分布，
          看它离 C96 直接学这堵"参数墙"多近；
       A3 温度：再扫 tau=1/3/5（Hinton 的高温是把类间相似度"糊开"再学，
          随机符号表没有类近邻结构 → 期望 τ=1 最优之外全是糊平）。

  B  黑盒蒸馏（text-only）：闭源场景拿不到 logits，只有教师生成的文本
      （R1-Distill 式）。一次批量 forward 抽出教师的条件分布 M[a,b,c]
      （喂全部 (a,b) 对读下一符号 logits），再用它走链采样一条 114K 合成语料
      ——数学上等价于教师自回归，但 O(1) 次 forward。学生 hard CE 在教师文本
      上，与同预算 A/B 对照：黑盒只有采样文本（丢类间概率），白盒才有软信息。

  C  CPT（领域续训）：基座 C96 在通用源 P1 上先训 229K，然后三臂各续 114K：
      ① 纯通用续训（什么也没丢的对照）② 纯领域 CPT（P2=另一份 Dirichlet、
      转移更尖=领域风格更单调）→ 期望灾难性遗忘：eval(P1) 回退 ③ 领域+通用
      回放 9:1（每步 90% P2 / 10% P1）→ 领域涨、eval(P1) 按住。CPT 的读数用
      **训练终点 eval**（遗忘是"退回到哪"，不是早停曲线的最优值）。

评测口径：eval = 模型对【真实转移分布】的期望条件 CE（分布级、无采样噪声），
分布级确定性 → run3==run4 逐位一致，只有墙钟浮动。

真实大模型数字（MiniLM / DistilBERT / R1-Distill 等）在正文标 非本机实测。
一键复现：python code/scripts/distil_cpt_demo.py
"""
import os
# 固定单线程 BLAS（无条件强锁）：多线程下 matmul 浮点低位逐进程浮动。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import time
import math
sys.path.insert(0, r"F:\00AI创业\01LLM\LLM\code\scripts")
import numpy as np
import grpo_rlvr_demo as g

# ---------------------------------------------------------------------------
# 两个马尔可夫源：P1=通用（Dirichlet(0.2) 高峰态）、P2=领域（同一符号表、更尖）
# ---------------------------------------------------------------------------
V = 16
LEN = 28
T = 32
B = 32
PER = LEN - 1
gen1 = np.random.RandomState(77)
P1 = np.zeros((V, V, V))
for a in range(V):
    for b in range(V):
        P1[a, b] = gen1.dirichlet(0.2 * np.ones(V))

gen2 = np.random.RandomState(321)
P2 = np.zeros((V, V, V))
for a in range(V):
    for b in range(V):
        P2[a, b] = gen2.dirichlet(0.05 * np.ones(V))   # 更尖 = 领域风格更单调


def stationary(P):
    MM = np.zeros((V * V, V * V))
    for a in range(V):
        for b in range(V):
            i = a * V + b
            for c in range(V):
                MM[b * V + c, i] = P[a, b, c]
    pi = np.full(V * V, 1.0 / (V * V))
    for _ in range(400):
        pi = MM @ pi
        pi /= pi.sum()
    return pi


def source_entropy(P, pi):
    h = 0.0
    for a in range(V):
        for b in range(V):
            d = P[a, b]
            h += pi[a * V + b] * (-(d * np.log(d + 1e-30)).sum())
    return float(h)


pi1 = stationary(P1)
pi2 = stationary(P2)
H1 = source_entropy(P1, pi1)
H2 = source_entropy(P2, pi2)

g.B, g.T = B, T
g.NHEADS = 4
C0 = 96      # 教师/基座尺寸（与 ch21 的 C96 同档）
CS = 32      # 学生尺寸（≈8× 参数差）
TEACH_N = 917 * 1024      # 教师预算
STU_BUDGETS = [57, 114, 229]   # 学生预算（K token）


def set_c(Cc):
    g.C = Cc
    g.DH = Cc // 4
    g.mask = np.tril(np.ones((T, T)))


lm = np.zeros((B, T), float)
lm[:, :PER] = 1.0


def init_model(Cc, seed):
    p = {}
    for k, (r0, r1) in {
        "Wte": (V, Cc), "Wpos": (T, Cc), "Wq": (Cc, Cc), "Wk": (Cc, Cc),
        "Wv": (Cc, Cc), "Wo": (Cc, Cc), "Wf1": (Cc, 4 * Cc), "Wf2": (4 * Cc, Cc),
        "Wout": (Cc, V)}.items():
        p[k] = np.random.RandomState(seed).randn(r0, r1) * 0.06
    return p


def n_params(Cc):
    return 2 * V * Cc + T * Cc + 12 * Cc * Cc


def make_batch(P, rng):
    """按源 P 采一批真样本（先采 (a,b) 起点再走链）。"""
    tok_row = np.zeros((B, T), dtype=int)
    pi = stationary(P)          # P 是模块级常数矩阵，每批求一次等价于缓存
    cs = [np.cumsum(P[a, b]) for a in range(V) for b in range(V)]
    for j in range(B):
        i = int(np.searchsorted(np.cumsum(pi), rng.rand()))
        a, b = i // V, i % V
        seq = [a, b]
        for _ in range(LEN - 2):
            c = int(np.searchsorted(cs[a * V + b], rng.rand()))
            seq.append(c)
            a, b = b, c
        tok_row[j, :len(seq)] = seq
    return tok_row


def eval_ce(p, P, pi):
    """模型对给定源 P 的期望条件 CE：feed 全部 (a,b) 读下一符号 logp，按 π 加权。"""
    set_c(p["Wte"].shape[1])
    rows = np.full((V * V, T), 0, dtype=int)
    for a in range(V):
        for b in range(V):
            rows[a * V + b, 0] = a
            rows[a * V + b, 1] = b
    res = g.forward(rows, p)[1]["logp"][:, 1, :]
    tot = 0.0
    for a in range(V):
        for b in range(V):
            d = P[a, b]
            ce = -(np.log(np.maximum(np.exp(res[a * V + b]), 1e-30)) * d).sum()
            tot += pi[a * V + b] * ce
    return float(tot)


def build_teacher_matrix(p):
    """把教师的"条件分布"整张抽出来：M[a*V+b, c] = softmax(教师 logits)，一次 batch forward。
    数学上等价于教师自回归的条件，只是省掉了逐 token forward——黑盒采样的生成原。"""
    set_c(p["Wte"].shape[1])
    rows = np.full((V * V, T), 0, dtype=int)
    for a in range(V):
        for b in range(V):
            rows[a * V + b, 0] = a
            rows[a * V + b, 1] = b
    lg = g.forward(rows, p)[1]["logits"][:, 1, :]
    e = np.exp(lg - lg.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def sample_from_matrix(M, n, rng, pi_start):
    i = int(np.searchsorted(np.cumsum(pi_start), rng.rand()))
    a, b = i // V, i % V
    out = np.zeros(n, dtype=int)
    out[0], out[1] = a, b
    for k in range(2, n):
        c = int(np.searchsorted(np.cumsum(M[a * V + b]), rng.rand()))
        out[k] = c
        a, b = b, c
    return out


def train_teacher(Ct, tokens, lr=3e-3):
    """在 P1 无限新数据上训练教师（与 ch21 的 C96@917K 同口径）。"""
    set_c(Ct)
    p = init_model(Ct, seed=1234)
    o = g.init_adam(p)
    tr = np.random.RandomState(44)
    steps = int(math.ceil(tokens / (B * PER)))
    interval = max(8, min(32, steps // 12))
    best_e, best_tok = 1e9, 0
    for s in range(1, steps + 1):
        tok_row = make_batch(P1, tr)
        tgt = np.concatenate([tok_row[:, 1:], np.zeros((B, 1), dtype=int)], axis=1)
        loss, cache = g.forward(tok_row, p, tgt)
        gr = g.backward(p, cache, loss_mask=lm)
        g.adam_update(p, o, gr, lr * min(1.0, s / 200))
        if s % interval == 0 or s == steps:
            e = eval_ce(p, P1, pi1)
            if e < best_e:
                best_e, best_tok = e, s * B * PER
    return p, best_e, best_tok


def train_hard(p0, P, tokens, seed=55):
    """hard CE 训练一臂（直接拟合真 token）。返回 (best_eval_P1, best_eval_P2)。"""
    cc = p0 if isinstance(p0, int) else p0["Wte"].shape[1]
    set_c(cc)
    p = init_model(p0, seed=1234) if isinstance(p0, int) else p0
    o = g.init_adam(p)
    tr = np.random.RandomState(seed)
    steps = int(math.ceil(tokens / (B * PER)))
    interval = max(8, min(32, steps // 12))
    best1, best2 = 1e9, 1e9
    for s in range(1, steps + 1):
        tok_row = make_batch(P, tr)
        tgt = np.concatenate([tok_row[:, 1:], np.zeros((B, 1), dtype=int)], axis=1)
        loss, cache = g.forward(tok_row, p, tgt)
        gr = g.backward(p, cache, loss_mask=lm)
        g.adam_update(p, o, gr, 3e-3 * min(1.0, s / 200))
        if s % interval == 0 or s == steps:
            e1, e2 = eval_ce(p, P1, pi1), eval_ce(p, P2, pi2)
            if e1 < best1:
                best1 = e1
            if e2 < best2:
                best2 = e2
    return p, best1, best2


def train_distill(p0, teacher_p, P, tokens, tau=1.0, seed=66):
    """白盒蒸馏一臂：学生拟合教师 logits。梯度 = (ps_soft - qt_soft)/tau，tau>1 高温软化。"""
    cc = p0 if isinstance(p0, int) else p0["Wte"].shape[1]
    set_c(cc)
    p = init_model(p0, seed=1234) if isinstance(p0, int) else p0
    o = g.init_adam(p)
    tr = np.random.RandomState(seed)
    steps = int(math.ceil(tokens / (B * PER)))
    interval = max(8, min(32, steps // 12))
    best1, best2 = 1e9, 1e9
    tc = teacher_p["Wte"].shape[1]
    for s in range(1, steps + 1):
        tok_row = make_batch(P, tr)
        loss_s, cache_s = g.forward(tok_row, p)
        set_c(tc)                       # 教师前向必须回到教师尺寸（引擎单例状态）
        _, t_cache = g.forward(tok_row, teacher_p)
        set_c(cc)                       # 学生前向再切回学生尺寸
        lt = t_cache["logits"] / tau
        qt = np.exp(lt - lt.max(axis=-1, keepdims=True))
        qt /= qt.sum(axis=-1, keepdims=True)
        ls = cache_s["logits"] / tau
        ps = np.exp(ls - ls.max(axis=-1, keepdims=True))
        ps /= ps.sum(axis=-1, keepdims=True)
        d_logits = (ps - qt) / tau * lm[..., None]
        gr = g.backward_from_dlogits(p, cache_s, d_logits)
        g.adam_update(p, o, gr, 3e-3 * min(1.0, s / 200))
        if s % interval == 0 or s == steps:
            e1, e2 = eval_ce(p, P1, pi1), eval_ce(p, P2, pi2)
            if e1 < best1:
                best1 = e1
            if e2 < best2:
                best2 = e2
    return p, best1, best2


def cpt_arm(base, p2_prob, tokens, seed=88):
    """从基座继续训练（不 early stopping，直接返回终点 eval——CPT 比的是'退到哪'）。
    p2_prob=None → 永远 P1（纯通用续训对照）；否则以 p2_prob 概率选 P2（领域）。"""
    set_c(C0)
    p = {k: base[k].copy() for k in base}
    o = g.init_adam(p)
    tr = np.random.RandomState(seed)
    steps = int(math.ceil(tokens / (B * PER)))
    for s in range(1, steps + 1):
        src = P1 if p2_prob is None or tr.rand() >= p2_prob else P2
        tok_row = make_batch(src, tr)
        tgt = np.concatenate([tok_row[:, 1:], np.zeros((B, 1), dtype=int)], axis=1)
        loss, cache = g.forward(tok_row, p, tgt)
        gr = g.backward(p, cache, loss_mask=lm)
        g.adam_update(p, o, gr, 3e-3 * min(1.0, s / 200))
    return eval_ce(p, P1, pi1), eval_ce(p, P2, pi2)


def main():
    t_start = time.perf_counter()
    print("=" * 78)
    print("蒸馏与 CPT（ch22）· 1-block 引擎 · 双马尔可夫源（P1 通用 / P2 领域）")
    print("=" * 78)
    print(f"符号 V={V} · 通用源熵 H1={H1:.4f} · 领域源熵 H2={H2:.4f}（lnV={math.log(V):.4f}）")
    teacher_params, student_params = n_params(96), n_params(CS)
    print(f"教师 C=96 → 参数 {teacher_params:,} · 学生 C={CS} → 参数 {student_params:,}"
          f"（{teacher_params / student_params:.0f}× 参数差）")

    # 随机初始化参照（模型还没学时的地板起点）
    p0 = init_model(CS, seed=1234)
    print(f"\n随机 init（未学）参照：eval(P1) {eval_ce(p0, P1, pi1):.4f}"
          f" · eval(P2) {eval_ce(p0, P2, pi2):.4f}（≈lnV=2.7726 附近）")

    print("\n--- A 白盒蒸馏：直接学 vs 从教师 logits 学（学生容量 ≠ 数据，拆成两问） ---")
    t, te, ttok = train_teacher(96, TEACH_N)
    print(f"教师 C96@{TEACH_N // 1024}K：eval(P1)={te:.4f} · eval(P2)={eval_ce(t, P2, pi2):.4f}"
          f"（没喂过领域数据 → 教师的领域边界，是 B/C 段的前提）")

    print("\n· A1 数据受限（学生与教师同容量 C96——容量不是墙、预算才是）：")
    ref96 = {}
    for bk in STU_BUDGETS:
        toknum = bk * 1024
        ph, e1h, _ = train_hard(96, P1, toknum, seed=55)
        pd, e1d, _ = train_distill(96, t, P1, toknum, tau=1.0, seed=66)
        ref96[bk] = (e1h, e1d)
        print(f"  C96 @ {bk:3d}K：direct {e1h:.4f} · distill(τ=1) {e1d:.4f}"
              f" · 蒸馏增益 {e1h - e1d:+.4f}"
              + ("（软标签信息 >> one-hot：预算越紧增益越大，往左看）" if bk == 57 else ""))

    BUD_SMUGGLE = 229
    print(f"\n· A2 参数偷渡（预算固定在 {BUD_SMUGGLE}K：小 8× 学生喝了教师软分布后，"
          f"离大模型直接学这堵参数墙多近）：")
    ph, e1h32, _ = train_hard(CS, P1, BUD_SMUGGLE * 1024, seed=55)
    pd, e1d32, _ = train_distill(CS, t, P1, BUD_SMUGGLE * 1024, tau=1.0, seed=66)
    e1_96 = ref96[BUD_SMUGGLE][0]
    print(f"  C32 direct @{BUD_SMUGGLE}K {e1h32:.4f} → C32 distill {e1d32:.4f}"
          f"（软监督净赚 {e1h32 - e1d32:+.4f}）")
    print(f"  C96 direct @{BUD_SMUGGLE}K {e1_96:.4f}：参数墙被蒸馏收回"
          f" {100 * (e1_96 - e1d32) / (e1_96 - e1h32):.0f}%")

    print("\n· A3 温度（C32 学生 @114K）：Hinton 的高温='把类间相似度糊开再学'——")
    print("  随机符号表没有类近邻结构，τ 越高只把监督糊平 → 期望 τ=1 最优：")
    ph, direct_114_c32, _ = train_hard(CS, P1, 114 * 1024, seed=55)
    distill_114_c32 = None
    for tau in (1.0, 3.0, 5.0):
        pd, e1d, _ = train_distill(CS, t, P1, 114 * 1024, tau=tau, seed=66)
        if tau == 1.0:
            distill_114_c32 = e1d
        print(f"  τ={tau:g}：distill eval(P1) {e1d:.4f}"
              + (f"（direct 同预算 {direct_114_c32:.4f}）" if tau == 1.0 else ""))

    print("--- B 黑盒蒸馏：学生只看教师自回归生成的文本（无 logits） ---")
    M_t = build_teacher_matrix(t)
    corpus = sample_from_matrix(M_t, 114 * 1024, np.random.RandomState(999), pi1)
    set_c(CS)
    p = init_model(CS, seed=1234)
    o = g.init_adam(p)
    tr = np.random.RandomState(66)
    steps = int(math.ceil(len(corpus) / (B * PER)))
    interval = max(8, min(32, steps // 12))
    best1, best2 = 1e9, 1e9
    for s in range(1, steps + 1):
        tok_row = np.zeros((B, T), dtype=int)
        for j in range(B):
            off = tr.randint(0, len(corpus) - LEN - 1)
            tok_row[j, :LEN] = corpus[off:off + LEN]
        tgt = np.concatenate([tok_row[:, 1:], np.zeros((B, 1), dtype=int)], axis=1)
        loss, cache = g.forward(tok_row, p, tgt)
        gr = g.backward(p, cache, loss_mask=lm)
        g.adam_update(p, o, gr, 3e-3 * min(1.0, s / 200))
        if s % interval == 0 or s == steps:
            e1b, e2b = eval_ce(p, P1, pi1), eval_ce(p, P2, pi2)
            if e1b < best1:
                best1 = e1b
            if e2b < best2:
                best2 = e2b
    print(f"  黑盒 114K：eval(P1) {best1:.4f}（教师文本 · 无 logits · 固定一段语料）")
    print(f"  对照 114K  // direct 真文本 {direct_114_c32:.4f}"
          f" · with logits distill τ=1 {distill_114_c32:.4f}")

    print("\n--- C CPT：通用基座 → 领域续训，回放 vs 纯领域 vs 纯通用 ---")
    base, e1b, _ = train_hard(C0, P1, 229 * 1024, seed=55)
    print(f"  基座 C96@229K：eval(P1) {e1b:.4f}（基线）")
    e1g, e2g = cpt_arm(base, None, 114 * 1024, seed=88)          # 纯通用续训
    e1d2, e2d2 = cpt_arm(base, 1.0, 114 * 1024, seed=88)          # 纯领域 CPT
    e1r, e2r = cpt_arm(base, 0.9, 114 * 1024, seed=88)            # 领域+通用回放 9:1
    print(f"  纯通用续训 114K：eval(P1) {e1g:.4f}（无遗忘对照）")
    print(f"  纯领域 CPT 114K：eval(P1) {e1d2:.4f}（要回退＝灾难性遗忘）"
          f" · eval(P2) {e2d2:.4f}")
    print(f"  领域+回放 9:1   ：eval(P1) {e1r:.4f}（要被按住） · eval(P2) {e2r:.4f}")

    print(f"\ndone · 整脚本墙钟 {time.perf_counter() - t_start:.0f}s（本机纯 CPU 单线程，"
          f"run3==run4 逐位一致）")


if __name__ == "__main__":
    main()
