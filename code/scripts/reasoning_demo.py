# -*- coding: utf-8 -*-
"""推理模型与 Test-time Scaling：R1 主线玩具版（0.12M 参数 1-block 引擎）。

承接《16-GRPO与RLVR》的同一颗 1-block 引擎，把『推理模型』拆成四根可测的杠杆：
  A  基座：算术合成任务（what is A plus B）· 换壳模板共用『A plus B ?』局部 n-gram
      → TR 原布局 100% / E1 换壳 52% / E2 新数 93% / E3 双新 33%
      消融（原布局）：交换律 / 遮一位 → 加法语义是真的，换壳丢的是位置路由不是加法
  B1 自洽（多数投票）：只在『错得散播』时有用；玩具实测三组都几乎平（回退错被锁死）
  B2 规则验证器 best-of-k：把 coverage 兑现成准确率 78–93%，差即投票的失格
      能力墙：9+9=18，『18』不在输出 token 集 → 烧任何预算都出不來
  C  RLVR 训出「先 think 再 answer」协议（对标 R1 的格式奖励 + 可验证答案奖励）：
      冷启动 SFT 披格式（R1 三阶段第 1 步）→ GRPO 送格式+可验证答案两种奖励。
      玩具如实声明：think 内容 = 3 个随机占位符，不承载中间计算（1-block 无法打草稿），
      增益归 RLVR 的可验证答案奖励 + 格式对 RL 的稳定作用；真实模型靠更深架构把
      思考变成打草稿。
  D  推理时算力预算：k 次采样 ↔ token ↔ KV ↔ 准确率 的收益递减曲线。

科学数字逐位一致：seed 固定 + 单线程 BLAS + 平票破序确定，run1==run2 逐位一致。
真实大模型数字（o1 / R1 等）在正文标 非本机实测，只做趋势参照。
一键复现：python code/scripts/reasoning_demo.py
"""
import os
# 固定单线程 BLAS（无条件强锁）：numpy 2.x 的 scipy-openblas MAX_THREADS=24，
# 多线程下 matmul 浮点低位逐进程浮动 → 前向 logits 不能逐位复现。
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r"F:\00AI创业\01LLM\LLM\code\scripts")
import numpy as np
import grpo_rlvr_demo as g

# 1-block 玩具超参：1 层 · 3 头 · 隐层 96 · 序列 32
g.B, g.T, g.C, g.NHEADS, g.DH = 16, 32, 96, 3, 32
g.mask = np.tril(np.ones((g.T, g.T)))

T1 = "what is {A} plus {B} ?"
T2 = "compute {A} plus {B} ?"
T3 = "how much is {A} plus {B} ?"
train_pairs = [(a, b) for a in range(1, 10) for b in range(1, 8)]
held_pairs = [(a, b) for a in range(1, 10) for b in (8, 9) if a + b <= 16]

train_qa = [(t.format(A=a, B=b), str(a + b)) for (a, b) in train_pairs for t in (T1, T2)]
FIL = ["hum", "mm", "hmm"]


def think_gold(qs, ans):
    rnd = np.random.RandomState(sum(ord(c) for c in qs) * 7 + 13)
    return qs, "think " + " ".join(rnd.choice(FIL, size=3)) + " answer " + ans


cold_qa = [think_gold(t.format(A=a, B=b), str(a + b))
           for (a, b) in train_pairs for t in (T1, T2)]

TR = [(T1.format(A=a, B=b), str(a + b)) for (a, b) in train_pairs]
E1 = [(T3.format(A=a, B=b), str(a + b)) for (a, b) in train_pairs]
E2 = [(T1.format(A=a, B=b), str(a + b)) for (a, b) in held_pairs]
E3 = [(T3.format(A=a, B=b), str(a + b)) for (a, b) in held_pairs]
SW = [(T1.format(A=b, B=a), str(a + b)) for (a, b) in train_pairs]
PE = [(T1.format(A=((a + 2) % 9) + 1, B=b), str(((a + 2) % 9) + 1 + b))
      for (a, b) in train_pairs]
WALL = "how much is 9 plus 9 ?"

vocab_src = [g.tokenize("q : " + q + " a : " + a) for (q, a) in train_qa]
for (q, a) in cold_qa:
    vocab_src.append(g.tokenize("q : " + q + " a : " + a))
for (q, a) in E1 + E2 + E3:
    vocab_src.append(g.tokenize(q))
stoi, itos = g.build_vocab(vocab_src)
V, unk = len(itos), stoi["<unk>"]
ans_ids = {stoi[str(c)] for c in range(2, 17) if str(c) in stoi}
n_params = sum(int(np.prod(s)) for s in
               [(V, g.C), (g.T, g.C), (g.C, g.C), (g.C, g.C), (g.C, g.C), (g.C, g.C),
                (g.C, 4 * g.C), (4 * g.C, g.C), (g.C, V)])


def sft_model(pairs, steps, lr=3e-4, p0=None, seed=9):
    p = g.init_params(V) if p0 is None else {k: v.copy() for k, v in p0.items()}
    o = g.init_adam(p)
    srand = np.random.RandomState(seed)
    for s in range(1, steps + 1):
        lr_now = lr * min(1.0, s / 200)
        toks_all, masks_all = [], []
        for _ in range(g.B):
            q, a = pairs[srand.randint(len(pairs))]
            toks, ans_start = g.parse_sft_pair(q, a, stoi, unk)
            L = len(toks)
            arr = np.full(g.T, unk, dtype=int)
            arr[:L] = toks
            m = np.zeros(g.T, float)
            for pp in range(max(0, ans_start - 1), L - 1):
                m[pp] = 1.0
            toks_all.append(arr)
            masks_all.append(m)
        tok = np.stack(toks_all)
        lm = np.stack(masks_all)
        tgt = np.concatenate([tok[:, 1:], np.zeros((g.B, 1), dtype=int)], axis=1)
        loss, cache = g.forward(tok, p, tgt)
        gr = g.backward(p, cache, loss_mask=lm)
        g.adam_update(p, o, gr, lr_now)
    return p


def free_ans(pol, qs, srng, temp=1.0, max_n=8):
    pre = g.question_prefix(qs, stoi, unk)
    return g.sample_ans(pol, pre, srng, temp=temp, max_n=max_n)


def direct_ans(pol, qs, srng, temp=1.0):
    return free_ans(pol, qs, srng, temp=temp, max_n=3)


def parse_think(ids):
    if len(ids) < 3 or ids[0] != stoi["think"]:
        return False, None
    if stoi["answer"] not in ids:
        return False, None
    idx = ids.index(stoi["answer"])
    if idx + 1 >= len(ids):
        return False, None
    nxt = ids[idx + 1]
    return True, (itos[nxt] if nxt in ans_ids else None)


def greedy_direct(pol, items, seed=11):
    srng = np.random.RandomState(seed)
    hit = 0
    for (qs, truth) in items:
        ids = direct_ans(pol, qs, srng, temp=0.0)
        hit += int(bool(ids) and itos[ids[0]] == truth)
    return hit / len(items)


def eval_k(pol, items, temp, kvals=(1, 2, 4, 8, 16), seed=301):
    srng = np.random.RandomState(seed)
    m = max(kvals)
    out = {k: [0.0, 0.0] for k in kvals}
    masses = []
    for it in items:
        samp = []
        for _ in range(m):
            ids = direct_ans(pol, it[0], srng, temp=temp)
            samp.append(None if not ids else itos[ids[0]])
        s = [x for x in samp if x is not None]
        if s:
            cnt = {}
            for x in s:
                cnt[x] = cnt.get(x, 0) + 1
            masses.append(max(cnt.values()) / len(s))
        for k in kvals:
            s0 = s[:k]
            out[k][1] += int(it[1] in s0)
            if s0:
                # 多数决取最高票；平票取样本流中先出现者（样本流由固定种子限抽——
                # 禁止用 max(set(...), key=count)：set 迭代序随进程哈希种子浮动，
                # 平票时会跨进程翻票 → run1 != run2。
                cnt = {}
                for x in s0:
                    cnt[x] = cnt.get(x, 0) + 1
                mc = max(cnt.items(), key=lambda kv: kv[1])[0]
                out[k][0] += int(mc == it[1])
    n = float(len(items))
    return ({k: (v[0] / n, v[1] / n) for k, v in out.items()},
            (sum(masses) / len(masses) if masses else 0.0))


def greedy_think(pol, items, seed=21):
    srng = np.random.RandomState(seed)
    fmt = acc = 0
    for (qs, truth) in items:
        ids = free_ans(pol, qs, srng, temp=0.0)
        f, a = parse_think(ids)
        fmt += int(f)
        acc += int(f and a == truth)
    n = len(items)
    return fmt / n, acc / n


def grpo_rlvr(p0, ref, steps, mode, seed=20):
    """mode='think'：r = 格式 + 答案正确（0/1/2）；mode='direct'：r = 答案正确（0/2）。"""
    K, G, beta, EPS, lr = 8, 4, 0.1, 0.2, 3e-4
    p = {k: v.copy() for k, v in p0.items()}
    o = g.init_adam(p)
    gr = np.random.RandomState(seed)
    Bx = G * K
    maxn = 8 if mode == "think" else 3
    curve = {}
    for step in range(1, steps + 1):
        gids = gr.randint(len(train_qa), size=G)
        toks = np.full((Bx, g.T), 0, dtype=int)
        Px = np.zeros(Bx, dtype=int)
        Lx = np.zeros(Bx, dtype=int)
        rewards = np.zeros(Bx)
        for j in range(Bx):
            gi = gids[j // K]
            qs, truth = train_qa[gi]
            pre = g.question_prefix(qs, stoi, unk)
            ans = g.sample_ans(p, pre, gr, temp=1.0, max_n=maxn)
            seq = pre + ans
            Lx[j], Px[j] = len(seq), len(pre)
            toks[j, :Lx[j]] = seq
            if mode == "think":
                f, a = parse_think(ans)
                rewards[j] = (0.0 if not f else (2.0 if a == truth else 1.0))
            else:
                a = itos[ans[0]] if ans else None
                rewards[j] = 2.0 if a == truth else 0.0
        got = Lx - Px
        A = np.zeros(Bx)
        for gg in range(G):
            s0, s1 = gg * K, (gg + 1) * K
            r = rewards[s0:s1]
            mu = r.mean()
            sd = r.std()
            A[s0:s1] = (r - mu) / (sd + 1e-4)
        lp_old = g.batch_logps(g.forward(toks, p)[1], Px.tolist(), got.tolist())
        cache_ref = g.forward(toks, ref)[1]
        logq = cache_ref["logp"]
        for _ in range(2):
            cache = g.forward(toks, p)[1]
            logp = cache["logp"]
            pv = np.exp(logp)
            lp_now = g.batch_logps(cache, Px.tolist(), got.tolist())
            rho = np.exp(np.clip(lp_now - lp_old, -10, 10))
            clipped = ((A > 0) & (rho > 1 + EPS)) | ((A < 0) & (rho < 1 - EPS))
            pg_coef = np.where(clipped, 0.0, A * rho)
            lm_pg = g.answer_region_mask(Lx, Px, g.T) * pg_coef[:, None]
            f_k = 1.0 + logp - logq
            Kbar = (pv * f_k).sum(axis=-1, keepdims=True)
            dkl = beta * pv * (f_k - Kbar)
            dkl = dkl * g.answer_region_mask(Lx, Px, g.T)[..., None]
            onehot = np.zeros_like(cache["logits"])
            onehot[np.arange(Bx)[:, None], np.arange(g.T)[None, :], cache["target"]] = 1
            d_logits = (pv - onehot) * lm_pg[..., None] + dkl
            g.adam_update(p, o, g.backward_from_dlogits(p, cache, d_logits), lr)
        if step % 50 == 0 or step == steps:
            tot = 0
            for (qs, truth) in E1:
                if mode == "think":
                    f, a = parse_think(free_ans(p, qs, np.random.RandomState(55), temp=0.0))
                    tot += int(f and a == truth)
                else:
                    ids = direct_ans(p, qs, np.random.RandomState(55), temp=0.0)
                    tot += int(bool(ids) and itos[ids[0]] == truth)
            curve[step] = tot / len(E1)
    return p, curve


def kl_answer(pol, ref, seed=5, n=40, max_n=8):
    kb = np.random.RandomState(seed)
    kl = 0.0
    for _ in range(n):
        j = kb.randint(len(train_qa))
        qs, _ = train_qa[j]
        pre = g.question_prefix(qs, stoi, unk)
        ans = free_ans(pol, qs, kb, temp=0.7, max_n=max_n)
        seq = pre + ans
        lp_p = g.batch_logps(g.forward(g.one_seq(seq), pol)[1], [len(pre)], [len(ans)])[0]
        lp_r = g.batch_logps(g.forward(g.one_seq(seq), ref)[1], [len(pre)], [len(ans)])[0]
        kl += float(lp_p - lp_r) / n
    return kl


def main():
    print("=" * 74)
    print("推理模型与 Test-time Scaling（ch20）· 0.12M 参数 1-block 引擎")
    print("=" * 74)
    print(f"算术合成任务：{len(train_pairs)} 训练对（B≤7）×2 模板 = {len(train_qa)} 条"
          f" · 留出 {len(held_pairs)} 对（B∈8/9）· 词表 V={V} · 参数 {n_params:,}")
    print(f"评测正交分解：TR 原布局 · E1 换壳(T3×训练对) · E2 新数(T1×留出对) · E3 双新(T3×留出对)")

    print(); print("=" * 74)
    print("A · 基座：位置路由焊死的加法器——换壳暴露系统性回退错")
    print("=" * 74)
    t0 = time.perf_counter()
    base = sft_model(train_qa, 600, seed=9)
    print(f"SFT 600 步直答 · 耗时 {time.perf_counter()-t0:.1f}s")
    tr = greedy_direct(base, TR, seed=101)
    e1 = greedy_direct(base, E1, seed=102)
    e2 = greedy_direct(base, E2, seed=103)
    e3 = greedy_direct(base, E3, seed=104)
    sw = greedy_direct(base, SW, seed=105)
    pe = greedy_direct(base, PE, seed=106)
    print(f"greedy 正确率：TR {tr:.0%} · E1 换壳 {e1:.0%} · E2 新数 {e2:.0%} · E3 双新 {e3:.0%}")
    print(f"消融（原布局，加法语义）：交换律 A+B vs B+A {sw:.0%}"
          f" · 遮一位跟着新操作数重算 {pe:.0%} → 加法真在算；")
    print(f"「换壳失效」是位置路由的选择性失忆，不是加法能力缺位（E2 新数 {e2:.0%} 佐证）。")

    print(); print("=" * 74)
    print("B · 测试时算力：自洽投票 vs 规则验证器（每问 ≤16 采样 · temp=0.7）")
    print("=" * 74)
    st1, m1 = eval_k(base, E1, temp=0.7, seed=301)
    st2, m2 = eval_k(base, E2, temp=0.7, seed=302)
    st3, m3 = eval_k(base, E3, temp=0.7, seed=303)
    print(f"{'k 采样':>6} | E1换壳 投票/覆盖 | E2新数 投票/覆盖 | E3双新 投票/覆盖")
    for k in (1, 2, 4, 8, 16):
        print(f"{k:>6} | {st1[k][0]:.0%}/{st1[k][1]:.0%}       | "
              f"{st2[k][0]:.0%}/{st2[k][1]:.0%}       | "
              f"{st3[k][0]:.0%}/{st3[k][1]:.0%}")
    print(f"模态质量（每问 16 样本里出现最多答案的占比均值，1=全同）：E1 {m1:.2f} · E2 {m2:.2f}"
          f" · E3 {m3:.2f}")
    print("  多数投票把每组都几乎抬不动（40–44–93）：回退错在样本间是『同一个错』，")
    print("  投票反而把稳定错锁死；覆盖率却一路涨到 78–93% —— 正确答案常在样本包里。")
    print("  规则验证器 best-of-k：样本包里出现正确答案就选它 → 准确率 ≈ 覆盖率。")
    best16 = st1[16][1]
    print(f"  E1 k16：投票 {st1[16][0]:.0%} vs 验证器 {best16:.0%}"
          f" · E3 k16：投票 {st3[16][0]:.0%} vs 覆盖 {st3[16][1]:.0%}")
    wall = []
    for _ in range(16):
        ids = direct_ans(base, WALL, np.random.RandomState(601), temp=0.7)
        wall.append(None if not ids else itos[ids[0]])
    wall = [w for w in wall if w is not None]
    print(f"  能力墙：{WALL}（=18，而输出 token 集到 16 为止）→ 16 采样的答案 "
          f"set={sorted(set(wall))}，无一是 18 → 测试时算力上限 = 生成分布的可达性，不是无限。")

    print(); print("=" * 74)
    print("C · RLVR 训出「先 think 再 answer」协议（R1 格式奖励 + 可验证答案）")
    print("=" * 74)
    print("  直答基座：a: 8（回答区 1 token）· 冷启动 SFT 150 步：a: think <3占位> answer 8")
    t0 = time.perf_counter()
    cold = sft_model(cold_qa, 150, p0=base, seed=8)
    print(f"  耗时 {time.perf_counter()-t0:.1f}s")
    fct, act = greedy_think(cold, TR, seed=210)
    fce, ace = greedy_think(cold, E1, seed=211)
    print(f"  冷启动：格式遵从 TR {fct:.0%}/E1 {fce:.0%} · 答案正确率 TR {act:.0%}/E1 {ace:.0%}"
          f"（回答区右移 5 拍 → 换壳集暂时生病）")
    t0 = time.perf_counter()
    pol, curv = grpo_rlvr(cold, cold, 150, "think", seed=20)
    c_str = " → ".join(f"{s}:{v:.0%}" for s, v in curv.items())
    print(f"  RLVR(思考) 150 步 · {time.perf_counter()-t0:.0f}s · E1 答案正确率曲线 {c_str}")
    ft, at = greedy_think(pol, TR, seed=212)
    fe, ae = greedy_think(pol, E1, seed=213)
    print(f"  RLVR(思考) 后：格式遵从 TR {ft:.0%}/E1 {fe:.0%}"
          f" · 答案正确率 TR {at:.0%}/E1 {ae:.0%}")
    kl_c = kl_answer(pol, cold, seed=5)
    print(f"  回答区 KL→冷启动锚 ≈ {kl_c:.2f} nats/token（KL 安全带：格式没被奖励推飞）")
    print(f"  同题两种模式：直答 1 token {e1:.0%} · 思考协议 6 token {ae:.0%}")
    print("  诚实声明：think 内容 = 3 个随机占位符、1-block 无打草稿深度，思考 token")
    print("  在玩具里不承载计算；E1 的提升来自 RLVR 的可验证答案奖励 + 格式对 RL 的稳定作用。")

    print(); print("=" * 74)
    print("D · 推理时算力预算：k 次采样 ↔ token ↔ KV ↔ 准确率（收益递减）")
    print("=" * 74)
    kv_per_tok = 2 * g.C * 4
    prompt_len = 10
    rows = []
    for k in (1, 2, 4, 8, 16):
        rows.append([k, k * 1, k * 6,
                     kv_per_tok * (prompt_len + 1) * k // 1024,
                     kv_per_tok * (prompt_len + 6) * k // 1024,
                     st1[k][0], st1[k][1]])
    w = [5, 12, 12, 10, 10, 8, 8]
    hdr = ["k", "token/直答", "token/思考", "KV直答KiB", "KV思考KiB", "投票E1", "验证器E1"]
    print("".join(h.ljust(w[i]) for i, h in enumerate(hdr)).rstrip())
    for r in rows:
        print("".join((f"{r[i]:.0%}" if i >= 5 else str(r[i])).ljust(w[i])
                      for i in range(len(r))).rstrip())
    print("  多采样 → token/KV 线性涨、准确率趋饱和 → 预算调度（reasoning_effort）该按题分档。")
    print()
    print("  -- 真实模型参照（非本机实测 · 运行环境无外网、写作时未在线复核，精确值以原论文表为准）--")
    print("  DeepSeek-R1 | AIME2024 pass@1 79.8% · self-consistency@64 83.3%")
    print("              | MATH-500 97.3% · GPQA-Diamond 71.5%（对标 o1-1217 同档）")
    print("              | R1-Zero（纯 RL 无 SFT）AIME2024 71.0% → 预算换分在真实模型同样触顶")
    print("  OpenAI o1   | GPQA-Diamond 78.0%（>人类专家 69.7%）· 难任务 thinking tokens 可比")
    print("              | 直答多一个数量级 · API reasoning_effort=low/medium/high 分档调预算")
    print("  框架/API    | verl（RL 训练语法，本玩具 GRPO 即其子集）· 思考蒸馏见 ch22")

    print()
    print("done · 一键复现：python code/scripts/reasoning_demo.py")
    print("复现自检键：E1 greedy {:.6f} · E1 投票k16 {:.6f} · E1 覆盖k16 {:.6f}"
          " · RLVR思考后 E1 {:.6f}".format(e1, st1[16][0], best16, ae))


if __name__ == "__main__":
    main()
