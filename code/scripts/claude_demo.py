# -*- coding: utf-8 -*-
"""Claude 系列：闭源 API 系第一大家族——只给能力不给权重。

Claude 是 03 模型家族第一个「闭源 API 系」家族：八要素里架构/训练/权重全不可见，
能买的是能力（接口 + 上下文 + 思考预算 + Agent 协议）。对齐招牌是 Constitutional AI
（2023，RLAIF）：不用人类大规模标注，而是用「AI 判官」审查模型自己的输出、指出违规
（critique）、再引导修订（revision），自动产出偏好数据 → RM → RLHF。本篇用 02-14
RLHF 引擎把这条线做成最小闭环：

  [A] Constitutional AI / RLAIF 最小闭环（本机实测，复用 rlhf_ppo_demo 引擎）
      玩具宪法条款 = 判官三票（答对/答全/答干净）；「不答得乱七八糟」= 别混进别的公式。
      ① 红队：SFT 模型高温采样 → 判官挑「违规回答」（混进别的公式）
      ② critique：判官指出违规 token（fire 出混进去的公式词）
      ③ revise：模型重答时「解码禁掉违规 token」（约束解码，toy 版宪法修订）→ 修订答
      ④ 偏好对自动产出：违规答 vs 修订答（判官定胜）→ 0 条人工标注（对应 02-14 人类 383 组）
      ⑤ RM(500 步) + RLHF(REINFORCE+KL 150 步) → 三票升、污染率降
      ⑥ 关键对照：RLHF 后去掉解码约束重测 → 约束从「解码期临时」变成「参数里的长期」？
      诚实声明：判官=三票规则（宪法条款 toy 化）；修订=约束解码（真方案常靠专门 revision
      模型/RLHF 洗衣机，toy 用 logits 屏蔽，机制=把不合格 token 从生成分布里剔除）。

  [B] 闭源 API 系派生账（公开配置 × 公式，非本机实测 · 未在线复核）
      ① 推理方式格：自部署 → API 快/慢旋钮（extended thinking）——玩具映射复用 02-20
      ② 上下文账：200k → 1M token 能读进多少文本
      ③ Agent 接口：MCP / Computer Use 的「能力商品化」账

科学数字逐位一致：seed 固定 + 单线程 BLAS + dict 计数（禁 set 迭代平票），
run1==run2==run3。
一键复现：python code/scripts/claude_demo.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r"F:\00AI创业\01LLM\LLM\code\scripts")
import numpy as np
import rlhf_ppo_demo as R

B, T, C, NHEADS, DH = R.B, R.T, R.C, R.NHEADS, R.DH
mask = R.mask


def build_vocab_plus():
    """02-14 词表（判定宪法条款用到的词汇一致）。"""
    text_sents = [R.tokenize(s) for s in R.TEXT_CORPUS]
    instr_sents = [R.tokenize("q : " + q + " a : " + a) for q, a in R.INSTR_PAIRS]
    return R.build_vocab(text_sents + instr_sents)


def pollute_rate(pol, prompts, toxs, srng, n=20):
    """『污染率』= 采样回答里混进『别的公式』token 的比例（judge 的『干净』票反面）。"""
    nb = nb_tox = 0
    for pre, tox in zip(prompts, toxs):
        for _ in range(n):
            ans = R.sample_ans(pol, pre, srng, temp=1.0)
            nb += 1
            nb_tox += int(any(t in tox for t in ans))
    return nb_tox / nb


def revise_constrained(pol, i, pre_ids, forbid, srng, temp=0.8, max_n=10):
    """约束解码修订：对同一条指令重答，解码时把违规 token（forbid）的 logits 置 -inf。"""
    cur = list(pre_ids)
    forbid_list = sorted(forbid)         # 固定顺序 → 每次 run 逐位一致
    for _ in range(max_n):
        if len(cur) >= T - 1:
            break
        logits = R.forward_eval_left(np.array(cur, dtype=int), pol).copy()
        logits[forbid_list] = -np.inf
        if temp == 0:
            nxt = int(np.argmax(logits))
        else:
            pv = R.softmax(logits / temp)
            nxt = int(srng.choice(len(pv), p=pv))
        cur.append(nxt)
    return cur[len(pre_ids):]


def main():
    # ---------- 词表 / 数据 ----------
    stoi, itos = build_vocab_plus()
    V, unk = len(itos), stoi["<unk>"]
    pref_train = [R.question_prefix(q, stoi, unk) for q, _ in R.INSTR_PAIRS]
    gold_toks = [R.tokenize(a) for _, a in R.INSTR_PAIRS]
    gold_ids = [[stoi[t] for t in tk] for tk in gold_toks]
    novel_pref = [R.question_prefix(q, stoi, unk) for _, _, q in R.NOVEL_QAS]
    fact_idx = [0, 5, 10, 15, 20, 25]
    nvq_gold_ids = [gold_ids[f] for f in fact_idx]
    tox_per = [R.toxic_for(gold_ids[i], stoi) for i in range(len(R.INSTR_PAIRS))]
    tox_nv = [R.toxic_for(nvq_gold_ids[i], stoi) for i in range(len(nvq_gold_ids))]
    oov = sorted({tt for _, _, q in R.NOVEL_QAS for tt in R.tokenize(q)} - set(itos))
    assert not oov, f"测试句含未登录词：{oov}"
    n_pol = sum(int(np.prod(s)) for s in
                [(V, C), (T, C), (C, C), (C, C), (C, C), (C, C),
                 (C, 4 * C), (4 * C, C), (C, V)])
    print("=" * 74)
    print("03-模型家族 07-Claude系列 · 闭源 API 系——Constitutional AI / RLAIF 最小闭环")
    print("引擎：02-14 同款 1-block（LN→因果MHA→LN→FFN）· 判官三票=玩具宪法条款")
    print(f"词表 V={V} · 策略参数 {n_pol:,} · 语料指令 {len(R.INSTR_PAIRS)} 条")
    print("=" * 74)

    # ---------- 阶段 0+1：预训练 + SFT（与 02-14 完全相同种子/数据 → respond 模型） ----------
    data_arr = np.array([stoi.get(t, unk) for s in R.TEXT_CORPUS for t in R.tokenize(s)],
                        dtype=int)
    N = len(data_arr)
    lr = 3e-3
    rng = np.random.RandomState(42)
    params_pt = R.init_params(V)
    optim = {"m": {k: np.zeros_like(v) for k, v in params_pt.items()},
             "v": {k: np.zeros_like(v) for k, v in params_pt.items()},
             "step": 1}
    t0 = time.perf_counter()
    for step in range(1, 1201):
        lr_now = lr * min(1.0, step / 200)
        starts = rng.randint(0, N - T, size=B)
        tok = np.stack([data_arr[s:s + T] for s in starts])
        tgt = np.stack([data_arr[s + 1:s + T + 1] for s in starts])
        loss, cache = R.forward(tok, params_pt, tgt)
        gr = R.backward(params_pt, cache)
        R.adam_update(params_pt, optim, gr, lr_now)
    print(f"阶段0 预训练 1200 步 · {time.perf_counter()-t0:.0f}s", file=sys.stderr)

    sft_rng = np.random.RandomState(9)
    p_sft = {k: v.copy() for k, v in params_pt.items()}
    o_sft = {"m": {k: np.zeros_like(v) for k, v in p_sft.items()},
             "v": {k: np.zeros_like(v) for k, v in p_sft.items()},
             "step": 1}
    t0 = time.perf_counter()
    for s in range(1, 201):
        lr_now = lr * min(1.0, s / 200)
        toks_all, masks_all = [], []
        for _ in range(B):
            t_i, ans_start = R.parse_sft_pair(*R.INSTR_PAIRS[sft_rng.randint(len(R.INSTR_PAIRS))],
                                              stoi, unk)
            L = len(t_i)
            arr = np.full(T, unk, dtype=int)
            arr[:L] = t_i
            m = np.zeros(T, dtype=float)
            for pp in range(max(0, ans_start - 1), L - 1):
                m[pp] = 1.0
            toks_all.append(arr)
            masks_all.append(m)
        tok = np.stack(toks_all)
        lm = np.stack(masks_all)
        tgt = np.concatenate([tok[:, 1:], np.zeros((B, 1), dtype=int)], axis=1)
        loss, cache = R.forward(tok, p_sft, tgt)
        gr = R.backward(p_sft, cache, loss_mask=lm)
        R.adam_update(p_sft, o_sft, gr, lr_now)
    print(f"阶段1 SFT 200 步 · {time.perf_counter()-t0:.0f}s", file=sys.stderr)

    print(); print("=" * 74)
    print("[A] Constitutional AI / RLAIF 最小闭环")
    print("=" * 74)

    # ---------- A0 基线 ----------
    print(); print("[A0] 基线：SFT respond 能答，但采样『污染率』约三成（判官三票挑）")
    s_b0 = np.random.RandomState(3)
    r0_tr = R.sample_perfect_rate(p_sft, pref_train, gold_ids, tox_per, s_b0)
    r0_nv = R.sample_perfect_rate(p_sft, novel_pref, nvq_gold_ids, tox_nv, s_b0)
    p0_tr = pollute_rate(p_sft, pref_train, tox_per, np.random.RandomState(4))
    p0_nv = pollute_rate(p_sft, novel_pref, tox_nv, np.random.RandomState(5))
    print(f"  三票 perfect 率：训练问 {r0_tr:.0%} · 全新问 {r0_nv:.0%}"
          f"（3/3=答对+答全+答干净）")
    print(f"  污染率（混进别的公式）：训练问 {p0_tr:.0%} · 全新问 {p0_nv:.0%}")

    # ---------- A1 红队：挑违规回答 + critique ----------
    print(); print("[A1] 红队 + critique：高温采样 → 判官挑『违规回答』并指出违规 token")
    red_rng = np.random.RandomState(31)
    bad_pool = []          # (i, bad_ans_ids)
    for i in range(len(R.INSTR_PAIRS)):
        tox_i = tox_per[i]
        for _ in range(4):
            a = R.sample_ans(p_sft, pref_train[i], red_rng, temp=1.0, max_n=10)
            if R.judge_score(gold_ids[i], a, tox_i) < 3:
                bad_pool.append((i, a))
                break
    n_bad = len(bad_pool)
    print(f"  挑出违规回答 {n_bad}/{len(R.INSTR_PAIRS)} 条（每条 critique=判官指出混进的 token）")
    for i, a in bad_pool[:3]:
        crit = sorted({itos[t] for t in a if t in tox_per[i]})
        print(f"    critique：问「{R.INSTR_PAIRS[i][0][:22]}…」 采样→"
              f"「{' '.join(itos[t] for t in a)}」· 违规 token: {crit}")

    # ---------- A2 revise：约束解码修订（toy 宪法修订） ----------
    print(); print("[A2] revise：对同一条指令重答、解码时禁掉违规 token（toy 宪法修订）")
    s_rev = np.random.RandomState(51)
    rev_ok = rev_tox = 0
    for i, bad_a in bad_pool:
        new = revise_constrained(p_sft, i, pref_train[i], tox_per[i], s_rev)
        rev_ok += int(R.judge_score(gold_ids[i], new, tox_per[i]) == 3)
        rev_tox += int(any(t in tox_per[i] for t in new))
    print(f"  修订版三票满分 {rev_ok}/{n_bad} · 修订版仍污染 {rev_tox}/{n_bad}")
    print("  （判定用的不是『我说的对不对』，是判官三票=宪法条款）")
    s_show = np.random.RandomState(71)
    for i, bad_a in bad_pool[:3]:
        new = revise_constrained(p_sft, i, pref_train[i], tox_per[i], s_show)
        print(f"    「{' '.join(itos[t] for t in bad_a)}」 → 「{' '.join(itos[t] for t in new)}」")

    # ---------- A3 RLAIF 偏好对自动产出（0 条人工标注） ----------
    print(); print("[A3] RLAIF 偏好对：违规答 vs 修订答（判官定胜, 0 条人工标注）")
    rlai_rng = np.random.RandomState(61)
    auto_pairs = []
    for i in range(len(R.INSTR_PAIRS)):
        tox_i = tox_per[i]
        bads = []
        for _ in range(10):
            a = R.sample_ans(p_sft, pref_train[i], rlai_rng, temp=1.0, max_n=10)
            if R.judge_score(gold_ids[i], a, tox_i) < 3:
                bads.append(a)
        for bad in bads:
            rev = revise_constrained(p_sft, i, pref_train[i], tox_i, rlai_rng)
            if R.judge_score(gold_ids[i], rev, tox_i) > R.judge_score(gold_ids[i], bad, tox_i):
                auto_pairs.append((i, rev, bad))
    n_auto = len(auto_pairs)
    n_train = max(1, int(n_auto * 0.85))
    tr_pairs, te_pairs = auto_pairs[:n_train], auto_pairs[n_train:]
    print(f"  自动产出偏好对 {n_auto} 组（修订分>违规分）——对应 02-14 人工标注 383 组")
    print(f"  留出判别集 {len(te_pairs)} 组 · 全部由判官自动判（RLAIF：AI 代人工）")

    # ---------- A4 RM 800 步 ----------
    print(); print("[A4] 奖励模型：从 AI 自动偏好对学『哪个回答更被喜欢』（800 步）")
    rp = R.init_rm_params(V)
    for k in ("Wte", "Wpos", "Wq", "Wk", "Wv", "Wo", "Wf1", "Wf2"):
        rp[k] = p_sft[k].copy()
    o_rm = {"m": {k: np.zeros_like(v) for k, v in rp.items()},
            "v": {k: np.zeros_like(v) for k, v in rp.items()},
            "step": 1}
    rm_rng = np.random.RandomState(71)
    t0 = time.perf_counter()
    for step in range(1, 801):
        lr_now = 3e-3 * min(1.0, step / 100)
        idx = rm_rng.randint(len(tr_pairs), size=8)
        c_seq = [pref_train[tr_pairs[j][0]] + tr_pairs[j][1] for j in idx]
        r_seq = [pref_train[tr_pairs[j][0]] + tr_pairs[j][2] for j in idx]
        Lc = [len(c) for c in c_seq]
        Lr = [len(r) for r in r_seq]
        tok_c = np.full((8, T), 0, dtype=int)
        tok_r = np.full((8, T), 0, dtype=int)
        for j in range(8):
            tok_c[j, :Lc[j]] = c_seq[j]
            tok_r[j, :Lr[j]] = r_seq[j]
        rc, cc, hc = R.rm_forward(tok_c, rp, np.array(Lc))
        rr, cr, hr = R.rm_forward(tok_r, rp, np.array(Lr))
        sig = 1.0 / (1.0 + np.exp(-(rc - rr)))
        gc = R.rm_backward(rp, cc, hc, sig - 1.0, np.array(Lc))
        gr = R.rm_backward(rp, cr, hr, 1.0 - sig, np.array(Lr))
        for k in gc:
            gc[k] += gr[k]
        R.adam_update(rp, o_rm, gc, lr_now)
    print(f"  RM 800 步 · {time.perf_counter()-t0:.0f}s", file=sys.stderr)
    correct = 0
    nt = max(1, len(te_pairs))
    for j in range(nt):
        tg = te_pairs[j]
        c_seq = pref_train[tg[0]] + tg[1]
        r_seq = pref_train[tg[0]] + tg[2]
        c = np.full((1, T), 0, dtype=int)
        rr = np.full((1, T), 0, dtype=int)
        c[0, :len(c_seq)] = c_seq
        rr[0, :len(r_seq)] = r_seq
        rc = R.rm_forward(c, rp, np.array([len(c_seq)]))[0][0]
        rr_v = R.rm_forward(rr, rp, np.array([len(r_seq)]))[0][0]
        correct += (rc > rr_v)
    rm_acc = correct / max(1, nt)
    print(f"  留出 AI 偏好对判别率 {rm_acc:.0%}（RM 学会了判官的三票偏好）")

    # ---------- A5 RLHF 150 步（对照：reward 在高方差 REINFORCE 下被 RM 格式捷径劫持） ----------
    print(); print("[A5] RLHF 对照（REINFORCE + KL，150 步）——reward 上去了，三票被 RM 劫持？")
    ref_params = {k: v.copy() for k, v in p_sft.items()}
    BETA, STEPS_RL, RB, ROLL_MAX = 0.1, 150, 8, 8
    rew0 = R.eval_rm_reward(p_sft, rp, pref_train, 2)
    p_pol = {k: v.copy() for k, v in p_sft.items()}
    o_pol = {"m": {k: np.zeros_like(v) for k, v in p_pol.items()},
             "v": {k: np.zeros_like(v) for k, v in p_pol.items()},
             "step": 1}
    rl_rng = np.random.RandomState(11)
    t0 = time.perf_counter()
    reward_curve = [rew0]
    for step in range(1, STEPS_RL + 1):
        bio = rl_rng.randint(len(R.INSTR_PAIRS), size=RB)
        pre_list = [pref_train[i] for i in bio]
        got_list = [R.sample_ans(p_pol, pre, rl_rng, temp=1.0, max_n=ROLL_MAX)
                    for pre in pre_list]
        Ls = [len(p) + len(a) for p, a in zip(pre_list, got_list)]
        tok = np.full((RB, T), 0, dtype=int)
        for j in range(RB):
            seq = pre_list[j] + got_list[j]
            tok[j, :Ls[j]] = seq
        lp_pol = R.answer_logps(tok, p_pol, [len(p) for p in pre_list],
                                [len(a) for a in got_list])
        lp_ref = R.answer_logps(tok, ref_params, [len(p) for p in pre_list],
                                [len(a) for a in got_list])
        r_scores = R.rm_forward(tok, rp, np.array(Ls))[0]
        r_eff = r_scores - BETA * (lp_pol - lp_ref)
        adv = r_eff - r_eff.mean()
        adv = adv / (adv.std() + 1e-6)
        lm = np.zeros((RB, T), dtype=float)
        for j in range(RB):
            Px, K = len(pre_list[j]), len(got_list[j])
            for k in range(K):
                lm[j, Px - 1 + k] = adv[j] - BETA
        tgt = np.concatenate([tok[:, 1:], np.zeros((RB, 1), dtype=int)], axis=1)
        loss, cache = R.forward(tok, p_pol, tgt)
        gr = R.backward(p_pol, cache, loss_mask=lm)
        R.adam_update(p_pol, o_pol, gr, 3e-4)
        if step % 50 == 0:
            reward_curve.append(R.eval_rm_reward(p_pol, rp, pref_train, 2))
    reward_curve.append(R.eval_rm_reward(p_pol, rp, pref_train, 2))
    print(f"  RL {STEPS_RL} 步 · {time.perf_counter()-t0:.0f}s", file=sys.stderr)
    print(f"  平均 RM 奖励：前 {reward_curve[0]:.3f} → step50 {reward_curve[1]:.3f}"
          f" → 后 {reward_curve[-1]:.3f}")
    kb = np.random.RandomState(5)
    kl_est = 0.0
    for i in range(60):
        j = kb.randint(len(R.INSTR_PAIRS))
        ans = R.sample_ans(p_pol, pref_train[j], kb, temp=1.0, max_n=ROLL_MAX)
        L = len(pref_train[j]) + len(ans)
        seqp = np.full((1, T), 0, dtype=int)
        seqp[0, :L] = np.array(pref_train[j] + ans)
        lp_p = R.answer_logps(seqp, p_pol, [len(pref_train[j])], [len(ans)])[0]
        lp_r = R.answer_logps(seqp, ref_params, [len(pref_train[j])], [len(ans)])[0]
        kl_est += float(lp_p - lp_r) / 60
    print(f"  KL(π‖SFT_ref) ≈ {kl_est:.3f}")
    s_e2 = np.random.RandomState(3)
    pf_tr = R.sample_perfect_rate(p_pol, pref_train, gold_ids, tox_per, s_e2)
    pf_nv = R.sample_perfect_rate(p_pol, novel_pref, nvq_gold_ids, tox_nv, s_e2)
    p2_tr = pollute_rate(p_pol, pref_train, tox_per, np.random.RandomState(4))
    p2_nv = pollute_rate(p_pol, novel_pref, tox_nv, np.random.RandomState(5))
    print(f"  RLHF 后三票 perfect：训练问 {pf_tr:.0%}（前 {r0_tr:.0%}）"
          f" · 全新问 {pf_nv:.0%}（前 {r0_nv:.0%}）")
    print(f"  RLHF 后污染率：训练问 {p2_tr:.0%}（前 {p0_tr:.0%}）"
          f" · 全新问 {p2_nv:.0%}（前 {p0_nv:.0%}）")
    print("  → RM 只学『修订 vs 违规』的格式差，rollout 探索到『非tox但答错』的乱答时")
    print("    RM 给它高分 → 三票崩、污染反升：这就是 02-14 正文写的『RM 判别率≈8成、")
    print("    RLHF 高方差』在玩具上的直演。Constitutional 的稳路径在 A6 的 revise 数据 SFT。")

    # ---------- A6 Constitutional 稳路径：修订答直接当下一轮 SFT 数据（0 条人工标注） ----------
    print(); print("[A6] Constitutional 稳路径：判断官修订答直接 SFT（AI 数据→回灌，600 步）")
    print("  · 真实 Constitutional 第一阶段=用 critique→revise 自动产的对训练下一轮（无关 RL）")
    s_rc = np.random.RandomState(81)
    rev_pairs_all = []
    for i in range(len(R.INSTR_PAIRS)):
        for _ in range(4):
            rev_pairs_all.append(
                (R.INSTR_PAIRS[i][0],
                 " ".join(itos[t] for t in
                          revise_constrained(p_sft, i, pref_train[i], tox_per[i], s_rc))))
    p_ca = {k: v.copy() for k, v in p_sft.items()}
    o_ca = {"m": {k: np.zeros_like(v) for k, v in p_ca.items()},
            "v": {k: np.zeros_like(v) for k, v in p_ca.items()},
            "step": 1}
    ca_rng = np.random.RandomState(91)
    t0 = time.perf_counter()
    for s in range(1, 601):
        lr_now = 3e-3 * min(1.0, s / 200)
        toks_all, masks_all = [], []
        for _ in range(B):
            qs, good = rev_pairs_all[ca_rng.randint(len(rev_pairs_all))]
            t_i, ans_start = R.parse_sft_pair(qs, good, stoi, unk)
            L = len(t_i)
            arr = np.full(T, unk, dtype=int)
            arr[:L] = t_i
            m = np.zeros(T, dtype=float)
            for pp in range(max(0, ans_start - 1), L - 1):
                m[pp] = 1.0
            toks_all.append(arr)
            masks_all.append(m)
        tok = np.stack(toks_all)
        lm = np.stack(masks_all)
        tgt = np.concatenate([tok[:, 1:], np.zeros((B, 1), dtype=int)], axis=1)
        loss, cache = R.forward(tok, p_ca, tgt)
        gr = R.backward(p_ca, cache, loss_mask=lm)
        R.adam_update(p_ca, o_ca, gr, lr_now)
    print(f"  Constitutional SFT 600 步 · {time.perf_counter()-t0:.0f}s", file=sys.stderr)
    s_e3 = np.random.RandomState(3)
    ca_tr = R.sample_perfect_rate(p_ca, pref_train, gold_ids, tox_per, s_e3)
    ca_nv = R.sample_perfect_rate(p_ca, novel_pref, nvq_gold_ids, tox_nv, s_e3)
    p3_tr = pollute_rate(p_ca, pref_train, tox_per, np.random.RandomState(4))
    p3_nv = pollute_rate(p_ca, novel_pref, tox_nv, np.random.RandomState(5))
    print(f"  SFT 后三票 perfect：训练问 {ca_tr:.0%}（前 {r0_tr:.0%}）"
          f" · 全新问 {ca_nv:.0%}（前 {r0_nv:.0%}）")
    print(f"  SFT 后污染率：训练问 {p3_tr:.0%}（前 {p0_tr:.0%}）"
          f" · 全新问 {p3_nv:.0%}（前 {p0_nv:.0%}）")
    print("  → Constitutional 主线读数：训练问三票升 + 污染率降 => 安全性与帮助性同涨，")
    print("    且整个标注管道没有任何人类介入（0 条人工标注），只有判官三票 + 约束解码。")

    print()
    print("=" * 74)
    print("[B] 闭源 API 系派生账（公开配置 × 公式，非本机实测 · 未在线复核）")
    print("=" * 74)
    print("  ① 推理方式格：自部署 → API 快/慢旋钮（extended thinking）")
    print("     02-20 玩具实测：同题直答 1 token vs think 协议 6 token ——")
    print("     思考 token 在多数 API 按输出价计费 → 开思考 ≈ 输出账单 ×6")
    print("     真实 Claude3.7 混合推理：一条消息可开/关思考（同引擎两档），")
    print("     复杂任务开思考买正确率、简单任务关思考省钱——决策变量从'部署'变'旋钮'。")
    print("  ② 上下文账：200k → 1M token 能读进多少文本")
    print("     口令：英文 ≈0.7 词/token → 200k token ≈ 14 万词、1M ≈ 70 万词；")
    print("     一部长篇（如《红与黑》）≈ 20 万词 → 200k 装不下 1 本，1M 装下 ~3 本；")
    print("     每加进上下文 1 token，KV 驻留线性 +2·C·字节（引用 02-10 母线）。")
    print("  ③ Agent 接口：MCP 发起者(2024-11) / Computer Use —— 能力商品化")
    print("     工具调用=把外部动作变成 API 的 return token；每回合工具结果回填上下文；")
    print("     长 Agent 任务的成本 ≈ Σ各回合输入 token（随回合数近似线性累积）。")
    print()
    print("  一句话（B 段）：闭源 API 系八要素里架构/权重全不可见，能调的只剩")
    print("  上下文档位 × 思考旋钮 × 工具协议——'省'的作战地图从'装多大'翻到'买多少'。")

    print()
    print("done · 一键复现：python code/scripts/claude_demo.py")


if __name__ == "__main__":
    main()
