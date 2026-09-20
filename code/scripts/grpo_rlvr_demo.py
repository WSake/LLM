# -*- coding: utf-8 -*-
"""GRPO 与 RLVR（对齐 III）：把"裁判"换成规则，把价值网络砍掉。

承接《14-对齐-RLHF与PPO》和《15-DPO家族》的同一套 1-block 玩具：
  14 章：养裁判（RM）+ REINFORCE + KL，三份驻留（策略/参考/RM ≈ 62.9 万参数）
  15 章：DPO 家族，闭式损失把 RM 和在线采样都省掉
  16 章（本章）：回答家族里最后剩下的箱子——奖励信号从哪来？
    · RLVR（Reinforcement Learning with Verifiable Rewards）
      = 不训奖励模型，把"谁更好"用规则直接判：玩具里就是 judge 三票公式本身
    · GRPO（Group Relative Policy Optimization）
      = 不训价值网络（critic），每问采一组 K 条回答，组内 (r−mean)/(std) 当 advantage
      → 驻留从 RLHF 的 ≈62.9 万降到策略+冻结参考 32.7 万（连 critic 都省）

三个 GRPO 臂共享同一份可验证奖励、同一 SFT 起手：
  C  GRPO 主臂      r = 三票活（0/1/2/3，规则即 judge 公式）· β=0.1 · K=8 · G=4
  D  GRPO 二值臂    r = 3/3 满分才算 1，其余 0（稀疏信号，RLVR 常见的"对/错"）
  E  GRPO 去锚臂    β=0（无 KL 安全带）→ 看无锚会发生什么（对照 14 章 D 段）
并与 14 章 RLHF 的 canonical（67% 全新问 / KL 10.65 / 153s）直接对账。

模型同 12-16 章：1-block 仅解码器，手写反向传播已全元素差分对账 <1e-5。
seed 固定 + 单线程 BLAS，run1==run2==run3 科学数字逐位一致，一键可跑。
"""
import os
# 固定单线程 BLAS：numpy 的 matmul 聚合顺序在多线程下每个进程都不同，
# 会在 0.18M 参数的小模型里放大成不同轨迹 → 数字一次和一次不一样。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import re
import sys
import math
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.RandomState(42)

# ---------------------------------------------------------------------------
# 语料与指令对（与 13/14/15 章完全相同——基线与 14 章 A/B 直接可比）
# ---------------------------------------------------------------------------
TEXT_CORPUS = [
    "the gravitational acceleration on earth near the surface is about 9 . 8 meters per second squared .",
    "physicists say the acceleration due to gravity near the ground is about 9 . 8 .",
    "when you drop an object on earth it speeds up at about 9 . 8 meters per second each second .",
    "the symbol g denotes the local acceleration which is close to 9 . 8 on earth .",
    "falling near the ground objects accelerate at 9 . 8 meters per second squared .",
    "pi is approximately 3 . 14159 and appears in every circle formula .",
    "the ratio between the circumference and the diameter of a circle is 3 . 14159 .",
    "mathematicians approximate pi as 3 . 14159 but its decimal never ends .",
    "the circle constant pi equals roughly 3 . 14159 .",
    "a circle circumference divided by its diameter gives 3 . 14159 called pi .",
    "the number e roughly equals 2 . 71828 and is the base of natural logarithms .",
    "natural logarithm uses base e which is about 2 . 71828 .",
    "compound interest grows forever at the constant e around 2 . 71828 .",
    "the exponential constant e is approximately 2 . 71828 .",
    "logarithms to the base e use the number e = 2 . 71828 .",
    "the natural constant e which is about 2 . 71828 appears in many growth formulas .",
    "light travels in vacuum at about 300000 kilometers per second .",
    "the speed of light is close to 300000 kilometers per second .",
    "in one second light crosses about 300000 kilometers .",
    "light covers roughly 300000 kilometers every second in empty space .",
    "nothing moves faster than light which goes about 300000 kilometers per second .",
    "laser pulses in fiber travel at nearly light speed about 300000 kilometers per second .",
    "water boils at 100 degrees celsius at standard pressure .",
    "at sea level water turns to vapor at 100 celsius .",
    "the boiling point of pure water under normal pressure is 100 degrees celsius .",
    "water reaches its boiling point at 100 celsius at one atmosphere .",
    "normal water starts to boil when the temperature hits 100 degrees celsius .",
    "an open pot of water at sea level is 100 degrees when it bubbles .",
    "earth takes about 365 days to go around the sun .",
    "one full orbit of earth around the sun lasts about 365 days .",
    "earth travels its yearly circle of the sun in roughly 365 days .",
    "the complete journey of earth around the sun needs about 365 days .",
    "a common calendar year matches one earth orbit about 365 days .",
    "the sun is a star at the center of our solar system .",
    "the moon circles the earth about once every month .",
    "a year is the time it takes a planet to complete one orbit .",
    "gravity pulls every object toward the center of the planet .",
    "energy cannot be created or destroyed only converted into another form .",
    "sound travels faster in water than in air .",
    "temperature measures the average motion of particles .",
    "clouds form when warm air rises and cools .",
    "stars are born inside clouds of gas that grow dense enough .",
    "planets move in elliptical paths around their stars .",
    "an object in motion stays in motion unless a force changes it .",
    "heavy objects fall at the same rate as light ones in vacuum .",
    "the sky is blue because molecules scatter blue light harder .",
    "a prism splits white light into the colors of the rainbow .",
]

INSTR_PAIRS = [
    ("what is the gravitational acceleration on earth ?", "9 . 8 meters per second squared ."),
    ("how fast do falling objects accelerate on earth ?", "9 . 8 ."),
    ("what is the value of g near the ground ?", "9 . 8 meters per second each second ."),
    ("how much does an object accelerate when falling on earth ?", "9 . 8 ."),
    ("what is the acceleration of a dropped object ?", "9 . 8 meters per second squared ."),
    ("what is the ratio of circumference to diameter of a circle ?", "3 . 14159 ."),
    ("how much is the constant pi ?", "3 . 14159 ."),
    ("what number appears in every circle formula ?", "3 . 14159 ."),
    ("what is the circle constant pi equal to ?", "3 . 14159 ."),
    ("how many units is pi approximately ?", "3 . 14159 ."),
    ("what is the base of the natural logarithm ?", "2 . 71828 ."),
    ("what number is e approximately ?", "2 . 71828 ."),
    ("what is the natural constant e ?", "2 . 71828 ."),
    ("what constant appears in growth formulas ?", "2 . 71828 ."),
    ("how much is the constant e ?", "2 . 71828 ."),
    ("what is the speed of light in vacuum ?", "300000 kilometers per second ."),
    ("how fast does light travel in empty space ?", "300000 kilometers per second ."),
    ("how many kilometers does light cross in one second ?", "300000 kilometers ."),
    ("how quickly does a light beam move through space ?", "300000 kilometers per second ."),
    ("what speed does light reach on its journey ?", "300000 kilometers per second ."),
    ("what is the boiling point of water ?", "100 degrees celsius ."),
    ("at what temperature does water boil ?", "100 degrees celsius ."),
    ("how hot must water get before it boils ?", "100 degrees celsius ."),
    ("at what celsius temperature does pure water boil ?", "100 degrees celsius ."),
    ("when does water start to become steam ?", "100 degrees celsius ."),
    ("how long does earth take to orbit the sun ?", "365 days ."),
    ("how many days is one year on earth ?", "365 days ."),
    ("how long is the journey of earth around the sun ?", "365 days ."),
    ("how many days does it take earth to circle the sun ?", "365 days ."),
    ("what is the orbital period of the earth ?", "365 days ."),
]

NOVEL_QAS = [
    ("g≈9.8", "9", "when you drop an object on a planet how fast does it accelerate ?"),
    ("π≈3.14", "3", "how much is the ratio of a circle circumference to its diameter ?"),
    ("e≈2.718", "2", "what is the number e as a base of logarithms ?"),
    ("光速",   "300000", "how fast does light travel through empty space ?"),
    ("沸点100", "100", "at what temperature does water become steam ?"),
    ("公转365", "365", "how many days does one complete orbit of earth take ?"),
]


def tokenize(sent):
    return [t.lower() for t in re.findall(r"[a-zA-Z']+|[0-9]+|[.,!?;:()\-]", sent)]


def build_vocab(train_sents):
    counter = {}
    for toks in train_sents:
        for t in toks:
            counter[t] = counter.get(t, 0) + 1
    vocab = ["<unk>"] + sorted(counter)
    return {t: i for i, t in enumerate(vocab)}, vocab


# ---------------------------------------------------------------------------
# 引擎（与 12-15 章完全同构）
# ---------------------------------------------------------------------------
B, T, C, NHEADS = 16, 48, 96, 3
DH = C // NHEADS
mask = np.tril(np.ones((T, T)))


def init_params(V, scale=0.06):
    p = {}
    for k, (r0, r1) in {
        "Wte": (V, C), "Wpos": (T, C), "Wq": (C, C), "Wk": (C, C),
        "Wv": (C, C), "Wo": (C, C), "Wf1": (C, 4 * C), "Wf2": (4 * C, C),
        "Wout": (C, V),
    }.items():
        p[k] = rng.randn(r0, r1) * scale
    return p


def ln(t):
    m = t.mean(axis=-1, keepdims=True)
    s = np.sqrt(t.var(axis=-1, keepdims=True) + 1e-5)
    return (t - m) / s, m, s


def softmax(t, axis=-1):
    e = np.exp(t - t.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def block_forward(tok, params):
    (Wte, Wpos, Wq, Wk, Wv, Wo, Wf1, Wf2) = (
        params["Wte"], params["Wpos"], params["Wq"], params["Wk"],
        params["Wv"], params["Wo"], params["Wf1"], params["Wf2"])
    Bb = tok.shape[0]
    x = Wte[tok] + Wpos[None]
    n1, m1, s1 = ln(x)
    q = n1 @ Wq; k = n1 @ Wk; v = n1 @ Wv
    q = q.reshape(Bb, T, NHEADS, DH).transpose(0, 2, 1, 3)
    k = k.reshape(Bb, T, NHEADS, DH).transpose(0, 2, 1, 3)
    v = v.reshape(Bb, T, NHEADS, DH).transpose(0, 2, 1, 3)
    att_logits = q @ k.transpose(0, 1, 3, 2) / math.sqrt(DH) + np.where(mask == 1, 0.0, -np.inf)
    att = softmax(att_logits)
    y = att @ v
    y = y.transpose(0, 2, 1, 3).reshape(Bb, T, C)
    y_out = y @ Wo
    x2 = x + y_out
    n2, m2, s2 = ln(x2)
    h = np.maximum(0, n2 @ Wf1)
    x3 = x2 + h @ Wf2
    cache = dict(tok=tok, x=x, n1=n1, m1=m1, s1=s1, q=q, k=k, v=v, att=att,
                 y=y, x2=x2, n2=n2, m2=m2, s2=s2, h=h, x3=x3)
    return cache


def forward(tok, params, target=None):
    cache = block_forward(tok, params)
    Bb = tok.shape[0]
    Wout = params["Wout"]
    logits = cache["x3"] @ Wout
    logp = logits - logits.max(axis=-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))
    if target is None:
        target = np.concatenate([tok[:, 1:], np.zeros((Bb, 1), dtype=int)], axis=1)
    loss = -(logp[np.arange(Bb)[:, None], np.arange(T)[None, :], target]).sum() / (Bb * T)
    cache.update(logits=logits, logp=logp, target=target)
    return loss, cache


def ln_backward(dy, t, m, s):
    d = t - m
    return (dy - dy.mean(axis=-1, keepdims=True)
            - d * (dy * d).mean(axis=-1, keepdims=True) / s**2) / s


def block_backward(params, cache, d_x3, g):
    x, n1, m1, s1 = cache["x"], cache["n1"], cache["m1"], cache["s1"]
    x2, n2, m2, s2, h, y = (cache["x2"], cache["n2"], cache["m2"],
                            cache["s2"], cache["h"], cache["y"])
    att, q, kk, v = cache["att"], cache["q"], cache["k"], cache["v"]
    Bb = x.shape[0]

    d_x2 = d_x3.copy()
    d_h = d_x3 @ params["Wf2"].T
    g["Wf2"] += (h.transpose(0, 2, 1) @ d_x3).sum(axis=0)
    d_n2 = d_h.copy()
    d_n2[h <= 0] = 0
    g["Wf1"] += (n2.transpose(0, 2, 1) @ d_n2).sum(axis=0)
    d_n2 = d_n2 @ params["Wf1"].T
    d_x2 = d_x2 + ln_backward(d_n2, x2, m2, s2)

    d_x = d_x2.copy()
    d_y = d_x2 @ params["Wo"].T
    g["Wo"] += (y.transpose(0, 2, 1) @ d_x2).sum(axis=0)

    d_y = d_y.reshape(Bb, T, NHEADS, DH).transpose(0, 2, 1, 3)
    d_v = att.transpose(0, 1, 3, 2) @ d_y
    d_att = d_y @ v.transpose(0, 1, 3, 2)
    d_att_logits = att * (d_att - (d_att * att).sum(axis=-1, keepdims=True))
    d_att_logits /= math.sqrt(DH)
    d_q = d_att_logits @ kk
    d_kk = d_att_logits.transpose(0, 1, 3, 2) @ q
    d_q = d_q.transpose(0, 2, 1, 3).reshape(Bb, T, C)
    d_kk = d_kk.transpose(0, 2, 1, 3).reshape(Bb, T, C)
    d_v = d_v.transpose(0, 2, 1, 3).reshape(Bb, T, C)
    d_n1 = d_q @ params["Wq"].T + d_kk @ params["Wk"].T + d_v @ params["Wv"].T
    g["Wq"] += (n1.transpose(0, 2, 1) @ d_q).sum(axis=0)
    g["Wk"] += (n1.transpose(0, 2, 1) @ d_kk).sum(axis=0)
    g["Wv"] += (n1.transpose(0, 2, 1) @ d_v).sum(axis=0)

    d_x = d_x + ln_backward(d_n1, x, m1, s1)
    g["Wpos"] += d_x.sum(axis=0)
    np.add.at(g["Wte"], cache["tok"], d_x)


def backward(params, cache, loss_mask=None):
    Bb = cache["target"].shape[0]
    logits, logp = cache["logits"], cache["logp"]
    p = np.exp(logp)
    onehot = np.zeros_like(logits)
    onehot[np.arange(Bb)[:, None], np.arange(T)[None, :], cache["target"]] = 1
    if loss_mask is None:
        d_logits = (p - onehot) / (Bb * T)
    else:
        d_logits = (p - onehot) * loss_mask[..., None]
    return backward_from_dlogits(params, cache, d_logits)


def backward_from_dlogits(params, cache, d_logits):
    """给定逐 token 的 d_loss/d_logits，回传完整梯度（供 GRPO 的 PG+KL 双项用）。"""
    Bb = cache["target"].shape[0]
    g = {k: np.zeros_like(v) for k, v in params.items()}
    g["Wout"] = (cache["x3"].transpose(0, 2, 1) @ d_logits).sum(axis=0)
    d_x3 = d_logits @ params["Wout"].T
    block_backward(params, cache, d_x3, g)
    return g


def adam_update(p, o, grads, lr_now):
    for k, pp in p.items():
        o["m"][k] = 0.9 * o["m"][k] + (1 - 0.9) * grads[k]
        o["v"][k] = 0.999 * o["v"][k] + (1 - 0.999) * grads[k] ** 2
        mh = o["m"][k] / (1 - 0.9 ** o["step"])
        vh = o["v"][k] / (1 - 0.999 ** o["step"])
        pp -= lr_now * mh / (np.sqrt(vh) + 1e-8)
    o["step"] += 1


def init_adam(p):
    return {"m": {k: np.zeros_like(v) for k, v in p.items()},
            "v": {k: np.zeros_like(v) for k, v in p.items()},
            "step": 1}


def forward_eval_left(seq, params, pos=None):
    seq = np.asarray(seq, dtype=int)[-T:]
    L = len(seq)
    tok = np.full((1, T), 0, dtype=int)
    tok[0, :L] = seq
    loss, cache = forward(tok, params)
    return cache["logits"][0, pos if pos is not None else L - 1]


def parse_sft_pair(qs, ans, stoi, unk):
    txt = "q : " + qs + " a : " + ans
    toks = [stoi.get(t, unk) for t in tokenize(txt)]
    raw = tokenize(txt)
    ans_start = None
    for i in range(len(raw) - 1):
        if raw[i] == 'a' and raw[i + 1] == ':':
            ans_start = i + 2
            break
    assert ans_start is not None, txt
    return toks, ans_start


def question_prefix(qs, stoi, unk):
    txt = "q : " + qs + " a :"
    return [stoi.get(t, unk) for t in tokenize(txt)]


# ---------------------------------------------------------------------------
# 可验证奖励（RLVR）：judge 三票公式本身，零训练、零驻留
# ---------------------------------------------------------------------------
def judge_score(gold_ids, ans_ids, tox_ids):
    if not ans_ids:
        return 0
    v = 0
    if ans_ids[0] == gold_ids[0]:
        v += 1
    pref = 0
    for t in gold_ids:
        if pref < len(ans_ids) and ans_ids[pref] == t:
            pref += 1
        else:
            break
    if pref >= len(gold_ids):
        v += 1
    elif pref >= (len(gold_ids) + 1) // 2:
        v += 1
    if not any(t in tox_ids for t in ans_ids):
        v += 1
    return v


FACT_LEXEMES = ("3", "14159", "2", "71828", "300000", "100", "365", "9", "8",
                "pi", "kilometers", "celsius", "steam", "vacuum", "orbit",
                "gravity", "light", "sun", "fiber")


def toxic_for(gold_ids, stoi):
    return {stoi[t] for t in FACT_LEXEMES if t in stoi} - set(gold_ids)


def rule_reward(gold_ids, ans, tox_ids, binary=False):
    """RLVR 的可验证规则 = judge 三票公式：答得对/答得全/答得干净。
    binary=True 时退化成"只有 3/3 满分才算对"（RLVR 常见的稀疏对/错信号）。"""
    s = judge_score(gold_ids, ans, tox_ids)
    return 1.0 if (binary and s == 3) else float(s)


def sample_ans(pol, pre_ids, srng, temp=1.0, max_n=8):
    cur = list(pre_ids)
    for _ in range(max_n):
        if len(cur) >= T - 1:
            break
        logits = forward_eval_left(np.array(cur, dtype=int), pol)
        if temp == 0:
            nxt = int(np.argmax(logits))
        else:
            p = softmax(logits / temp)
            nxt = int(srng.choice(len(p), p=p))
        cur.append(nxt)
    return cur[len(pre_ids):]


def answer_logps(tok, params, pre_lens, got_lens):
    loss, cache = forward(tok, params)
    return batch_logps(cache, pre_lens, got_lens)


def batch_logps(cache, pre_lens, got_lens):
    """从某一次 forward 的缓存里，逐条累加回答区预测位的 logπ。"""
    lp = cache["logp"]; tgt = cache["target"]
    out = np.zeros(cache["logp"].shape[0])
    for j in range(cache["logp"].shape[0]):
        P, K = pre_lens[j], got_lens[j]
        for k in range(K):
            ppos = P - 1 + k
            out[j] += lp[j, ppos, tgt[j, ppos]]
    return out


def answer_region_mask(Ls, P, T):
    lm = np.zeros((len(Ls), T), float)
    for j in range(len(Ls)):
        for k in range(Ls[j] - P[j]):
            lm[j, P[j] - 1 + k] = 1.0
    return lm


def sample_perfect_rate(pol, prompts, golds, toxs, srng, n=20):
    nb = nb3 = 0
    for pre, gold, tox in zip(prompts, golds, toxs):
        for _ in range(n):
            ans = sample_ans(pol, pre, srng, temp=1.0)
            nb += 1
            nb3 += (judge_score(gold, ans, tox) == 3)
    return nb3 / nb


def kl_to_ref(pol, refp, pref_train, seed=5, n=60, max_n=8):
    kb = np.random.RandomState(seed)
    kl = 0.0
    for _ in range(n):
        j = kb.randint(len(pref_train))
        ans = sample_ans(pol, pref_train[j], kb, temp=1.0, max_n=max_n)
        seq = one_seq(pref_train[j] + ans)
        lp_p = answer_logps(seq, pol, [len(pref_train[j])], [len(ans)])[0]
        lp_r = answer_logps(seq, refp, [len(pref_train[j])], [len(ans)])[0]
        kl += float(lp_p - lp_r) / n
    return kl


def one_seq(seq):
    a = np.full((1, T), 0, dtype=int)
    a[0, :len(seq)] = seq
    return a


# ---------------------------------------------------------------------------
# GRPO 臂
# ---------------------------------------------------------------------------
def grpo_train(policy0, config, pref_train, gold_ids, tox_per):
    """一组 GRPO：组内相对 advantage + PPO 式 clip + 精确 KL 锚。
    config: K 组内回答数 · G 每步几组 · beta KL 系数 · steps · seed · binary.
    返回 (p, stats)。"""
    K, G, beta_k = config["K"], config["G"], config["beta"]
    steps = config["steps"]
    binary = config.get("binary", False)
    EPSILON = 0.2
    lr = 3e-4
    p = {k: v.copy() for k, v in policy0.items()}
    o = init_adam(p)
    gr = np.random.RandomState(config["seed"])
    Bx = G * K
    curve = {}          # step → 训练问平均规则奖励（固定评测种子，主臂/对照可互比）
    last = {}
    ev_rng = np.random.RandomState(11)
    for step in range(1, steps + 1):
        # ---- 采样一组：G 组 × K 条
        gids = gr.randint(len(pref_train), size=G)
        toks = np.full((Bx, T), 0, dtype=int)
        Px = np.zeros(Bx, dtype=int); Lx = np.zeros(Bx, dtype=int)
        rewards = np.zeros(Bx)
        for j in range(Bx):
            gi = gids[j // K]
            pre = pref_train[gi]
            ans = sample_ans(p, pre, gr, temp=1.0, max_n=8)
            seq = pre + ans
            Lx[j] = len(seq); Px[j] = len(pre)
            toks[j, :Lx[j]] = seq
            rewards[j] = rule_reward(gold_ids[gi], ans, tox_per[gi], binary)
        got = Lx - Px
        # ---- 组内相对 advantage
        A = np.zeros(Bx)
        for g in range(G):
            s0, s1 = g * K, (g + 1) * K
            r = rewards[s0:s1]
            mu = r.mean(); sd = r.std()
            A[s0:s1] = (r - mu) / (sd + 1e-4)
        lp_old = answer_logps(toks, p, Px.tolist(), got.tolist())
        # ---- ref 的 logq（冻结，整步只算一次）
        cache_ref = forward(toks, ref_params)[1]
        logq = cache_ref["logp"]
        # ---- 内循环：同一组回答吃 U 次更新（clip 才有事做）
        U = 2
        n_clip = 0
        for _ in range(U):
            cache = forward(toks, p)[1]
            logp = cache["logp"]
            pv = np.exp(logp)
            lp_now = batch_logps(cache, Px.tolist(), got.tolist())
            rho = np.exp(np.clip(lp_now - lp_old, -10, 10))
            clipped = ((A > 0) & (rho > 1 + EPSILON)) | ((A < 0) & (rho < 1 - EPSILON))
            n_clip += int(np.sum(clipped))
            pg_coef = np.where(clipped, 0.0, A * rho)
            lm_pg = answer_region_mask(Lx, Px, T) * pg_coef[:, None]
            # 精确 KL 梯度：要"下降 β·KL"供 adam 的梯度 = +β·p·(f−K̄)。
            # 推导见正文 §4：f=1+logp−logq，K̄=Σp·f=1+KL → p·(f−K̄)=p·(log(p/q)−KL)=∇_z KL。
            # （符号写反 = 最大化 KL，策略被推出 SFT 锚外，PG 再攒出退化解——本批真实踩过的坑。）
            f = 1.0 + logp - logq
            Kbar = (pv * f).sum(axis=-1, keepdims=True)
            dkl = +beta_k * pv * (f - Kbar)
            mask_ans = answer_region_mask(Lx, Px, T)
            dkl = dkl * mask_ans[..., None]
            onehot = np.zeros_like(cache["logits"])
            onehot[np.arange(Bx)[:, None], np.arange(T)[None, :], cache["target"]] = 1
            d_logits = (pv - onehot) * lm_pg[..., None] + dkl
            g = backward_from_dlogits(p, cache, d_logits)
            adam_update(p, o, g, lr * min(1.0, step / 50))
        # ---- 监控（固定种子，让所有步可互比）
        if step == 1 or step % 50 == 0 or step == steps:
            ev = np.random.RandomState(11)
            tot = 0.0
            for i in range(len(pref_train)):
                a = sample_ans(p, pref_train[i], ev, temp=1.0, max_n=8)
                tot += rule_reward(gold_ids[i], a, tox_per[i], binary)
            curve[step] = tot / len(pref_train)
            last = dict(
                mean_r=rewards.mean(),
                mean_absA=np.abs(A).mean(),
                clip=n_clip / (U * Bx),
                kl=kl_snapshot(p, toks, Px, got),
                step=step)
    return p, curve, last


def kl_snapshot(p, toks, Px, got):
    """当前策略在该批序列上（回答区）与 ref 的精确平均 KL（每 token）。"""
    cache = forward(toks, p)[1]
    logq = forward(toks, ref_params)[1]["logp"]
    pv = np.exp(cache["logp"])
    kl = (pv * (cache["logp"] - logq)).sum(axis=-1)
    m = answer_region_mask(Px + got, Px, T)
    masked = (kl * m).sum() / max(1.0, m.sum())
    return float(masked)


def main():
    # ---- 词表（只从训练数据建；测试句 OOV 当场报错）
    text_sents = [tokenize(s) for s in TEXT_CORPUS]
    instr_sents = [tokenize("q : " + q + " a : " + a) for q, a in INSTR_PAIRS]
    stoi, itos = build_vocab(text_sents + instr_sents)
    V = len(itos)
    unk = stoi["<unk>"]
    oov = sorted({t for _, _, q in NOVEL_QAS for t in tokenize(q)} - set(itos))
    assert not oov, f"测试句含未登录词：{oov}"
    print("=" * 70)
    print(f"语料：声明式文本 {len(TEXT_CORPUS)} 句 + 指令对 {len(INSTR_PAIRS)} 条 · 词表 V={V}")
    n_pol = sum(int(np.prod(s)) for s in
                [(V, C), (T, C), (C, C), (C, C), (C, C), (C, C), (C, 4 * C), (4 * C, C), (C, V)])
    print(f"策略参数 {n_pol:,} · GRPO 驻留 = 策略+冻结参考 {n_pol*2:,}"
          f"（无 value 头/无 RM） · RLHF 驻留 ≈ {n_pol*3+139393:,}（见 14 章）")

    # ---- 阶段 0：预训练（14/15 章同款，保证 A 基线逐位一致）
    data_arr = np.array([stoi.get(t, unk) for s in text_sents for t in s], dtype=int)
    N = len(data_arr)
    lr = 3e-3
    print(); print("=" * 70)
    print("阶段 0 · 预训练（12-15 章引擎 · 同 6 组事实）：把事实压进参数")
    print("=" * 70)
    STEPS_PT = 1200
    params_pt = init_params(V)
    optim = init_adam(params_pt)
    l0 = forward(np.full((B, T), unk, dtype=int), params_pt)[0]
    print(f"起手 loss = {l0:.3f}（≈lnV={math.log(V):.3f}）")
    t0 = time.perf_counter()
    for step in range(1, STEPS_PT + 1):
        lr_now = lr * min(1.0, step / 200)
        starts = rng.randint(0, N - T, size=B)
        tok = np.stack([data_arr[s:s + T] for s in starts])
        tgt = np.stack([data_arr[s + 1:s + T + 1] for s in starts])
        loss, cache = forward(tok, params_pt, tgt)
        gr = backward(params_pt, cache)
        adam_update(params_pt, optim, gr, lr_now)
        last = loss
    dt = time.perf_counter() - t0
    print(f"预训练 {STEPS_PT} 步耗时 {dt:.1f}s · 末步 loss {last:.3f}")

    # ---- 阶段 1：掩码 SFT（13/14/15 章同款 200 步）
    print(); print("=" * 70)
    print("阶段 1 · 掩码 SFT（13/14/15 章同款 200 步）：把'续写者'掰成'对话者'")
    print("=" * 70)
    sft = np.random.RandomState(9)
    p_sft = {k: v.copy() for k, v in params_pt.items()}
    o_sft = init_adam(p_sft)
    t0 = time.perf_counter()
    for s in range(1, 201):
        lr_now = lr * min(1.0, s / 200)
        toks_all, masks_all = [], []
        for _ in range(B):
            t, ans_start = parse_sft_pair(*INSTR_PAIRS[sft.randint(len(INSTR_PAIRS))], stoi, unk)
            L = len(t); arr = np.full(T, unk, dtype=int); arr[:L] = t
            m = np.zeros(T, dtype=float)
            for pp in range(max(0, ans_start - 1), L - 1):
                m[pp] = 1.0
            toks_all.append(arr); masks_all.append(m)
        tok = np.stack(toks_all); lm = np.stack(masks_all)
        tgt = np.concatenate([tok[:, 1:], np.zeros((B, 1), dtype=int)], axis=1)
        loss, cache = forward(tok, p_sft, tgt)
        gr = backward(p_sft, cache, loss_mask=lm)
        adam_update(p_sft, o_sft, gr, lr_now)
        logp = cache["logp"]
        sel = -logp[np.arange(B)[:, None], np.arange(T)[None, :], cache["target"]]
        last = float((sel * lm).sum() / lm.sum())
    dt = time.perf_counter() - t0
    print(f"SFT 200 步耗时 {dt:.0f}s · 末步掩码loss {last:.3f}")

    # ---- 数据装配
    pref_train = [question_prefix(q, stoi, unk) for q, _ in INSTR_PAIRS]
    gold_toks = [tokenize(a) for _, a in INSTR_PAIRS]
    gold_ids = [[stoi[t] for t in tk] for tk in gold_toks]
    novel_pref = [question_prefix(q, stoi, unk) for _, _, q in NOVEL_QAS]
    fact_idx = [0, 5, 10, 15, 20, 25]
    nvq_gold_ids = [gold_ids[f] for f in fact_idx]
    tox_per = [toxic_for(gold_ids[i], stoi) for i in range(len(INSTR_PAIRS))]
    tox_nv = [toxic_for(nvq_gold_ids[i], stoi) for i in range(len(nvq_gold_ids))]
    s_eval = np.random.RandomState(3)
    global ref_params
    ref_params = {k: v.copy() for k, v in p_sft.items()}

    # ---- A 基线
    print(); print("=" * 70)
    print("A · 基线：SFT 会回答，但采样质量不是 3/3 满分")
    print("=" * 70)
    r0_tr = sample_perfect_rate(p_sft, pref_train, gold_ids, tox_per, s_eval)
    r0_nv = sample_perfect_rate(p_sft, novel_pref, nvq_gold_ids, tox_nv, s_eval)
    print(f"  SFT 采样（temp=1.0，每问 20 次）perfect 率：训练问 {r0_tr:.0%} · 全新问 {r0_nv:.0%}")

    # ---- B 可验证奖励：规则的三票分解（零训练，直接数命中率）
    print()
    print("=" * 70)
    print("B · 可验证奖励（RLVR）：规则三票 = judge 公式，零训练零驻留"
          "（对比 14 章 RM 要训 500 步、驻留 13.9 万）")
    print("=" * 70)
    def rule_votes_batch(pol, prompts, golds, toxs, srng, nper=20):
        v1 = v2 = v3 = 0; tot = 0; sumr = 0.0
        for q in range(len(prompts)):
            for _ in range(nper):
                a = sample_ans(pol, prompts[q], srng, temp=1.0, max_n=8)
                gold = golds[q]; tox = toxs[q]
                s1 = 1 if (a[0] == gold[0]) else 0
                pref = 0
                for t in gold:
                    if pref < len(a) and a[pref] == t:
                        pref += 1
                    else:
                        break
                s2 = 1 if (pref >= len(gold) or pref >= (len(gold) + 1) // 2) else 0
                s3 = 1 if not any(t in tox for t in a) else 0
                v1 += s1; v2 += s2; v3 += s3; tot += 1
                sumr += (s1 + s2 + s3)
        return v1 / tot, v2 / tot, v3 / tot, sumr / tot
    br = np.random.RandomState(10)
    a1, a2, a3, am = rule_votes_batch(p_sft, pref_train, gold_ids, tox_per, br)
    n1, n2, n3, nm = rule_votes_batch(p_sft, novel_pref, nvq_gold_ids, tox_nv, br)
    print(f"  SFT 起手 · 训练问三条规则命中：答得对 {a1:.0%} · 答得全 {a2:.0%} · 答得干净 {a3:.0%}"
          f" · r均 {am:.2f}/3")
    print(f"  SFT 起手 · 全新问三条规则命中：答得对 {n1:.0%} · 答得全 {n2:.0%} · 答得干净 {n3:.0%}"
          f" · r均 {nm:.2f}/3")
    print("  → 训练问的短板只剩'答得干净'（真值里别的知识词=脏）；全新问是泛化前沿（规则还有肉）"
          "——GRPO 的任务就是拿这份零训练的规则去推这两个缺口。")

    # ---- C/D/E GRPO 三臂
    def run_arm(cfg, label):
        print(); print("=" * 70)
        print(f"{label}")
        print("=" * 70)
        print(f"  config: K={cfg['K']}(组内回答) · G={cfg['G']}(每步组数) → 每步 {cfg['K']*cfg['G']} 采样"
              f" · β(KL系数)={cfg['beta']} · ε=0.2 · 2 内循环 · lr=3e-4 · {cfg['steps']} 步"
              f" · binary={cfg.get('binary', False)}")
        t0 = time.perf_counter()
        pol, curve, last = grpo_train(p_sft, cfg, pref_train, gold_ids, tox_per)
        dt = time.perf_counter() - t0
        s_e2 = np.random.RandomState(3)
        pf_tr = sample_perfect_rate(pol, pref_train, gold_ids, tox_per, s_e2)
        pf_nv = sample_perfect_rate(pol, novel_pref, nvq_gold_ids, tox_nv, s_e2)
        kl = None if cfg["beta"] == 0 else kl_to_ref(pol, ref_params, pref_train)
        curve_s = " → ".join(f"{s}:{v:.2f}" for s, v in curve.items())
        kl_s = "无锚(β=0)" if kl is None else f"{kl:.2f}"
        print(f"  规则奖励曲线(训练问均值 r)：{curve_s}")
        print(f"  perfect 率：训练问 {pf_tr:.0%}（前 {r0_tr:.0%}）· 全新问 {pf_nv:.0%}（前 {r0_nv:.0%}）"
              f" · KL→SFT {kl_s}")
        print(f"  末步统计：训练问平均 r {last['mean_r']:.2f} · mean|A| {last['mean_absA']:.2f}"
              f" · clip 命中率 {last['clip']:.0%} · 回答区 KL/token {last['kl']:.2f}")
        print(f"  GRPO {cfg['steps']} 步耗时 {dt:.0f}s")
        return dict(label=label, pf_tr=pf_tr, pf_nv=pf_nv, kl=kl, dt=dt,
                    curve=curve, last=last, cfg=cfg)

    c_main = run_arm(dict(K=8, G=4, beta=0.1, steps=250, seed=20), "C · GRPO（可验证奖励 · 三票 r=0/1/2/3）")
    c_bin = run_arm(dict(K=8, G=4, beta=0.1, steps=200, seed=21, binary=True), "D · GRPO（二值奖励：只有 r=3 算 1）")
    c_off = run_arm(dict(K=8, G=4, beta=0.0, steps=200, seed=22), "E · GRPO（去锚：β=0 无 KL 安全带）")

    # ---- F 汇总
    print(); print("=" * 70)
    print("F · 汇总：同一起跑线（SFT 62/38）· 同一预算 · 同一次可验证奖励")
    print("=" * 70)
    print(f"{'方法':<22}{'训练问三票':>10}{'全新问三票':>10}{'KL→SFT':>10}{'墙钟':>8}")
    print(f"{'SFT 基线':<22}{r0_tr:>9.0%} {r0_nv:>9.0%} {'0.00':>10} {'—':>8}")
    for c in (c_main, c_bin, c_off):
        kl_s = "无锚" if c["kl"] is None else f"{c['kl']:.2f}"
        print(f"{c['label'][:20]:<22}{c['pf_tr']:>9.0%} {c['pf_nv']:>9.0%} {kl_s:>10} {c['dt']:>6.0f}s")
    print(f"{'RLHF (C14 canonical)':<24}{'70%':>9}{' 67%':>9}{'10.65':>10}{'153s':>8}")
    print("  注：RLHF 行引自 14 章 canonical run3（RM 留出 79% / REINFORCE+KL / β=0.1 /"
          " 全新问三票 38%→67% / KL 10.65 / 整脚本 153s）。GRPO 全部只驻留策略+冻结参考，无价值网络、无 RM。")
    print("  注：玩具词表没有 EOS → 每回答恒长 8/10 token，'长度惰性'与 14/15 章同因；"
          " RLVR 在玩具里 = judge 三票直接把真值当奖励（真实世界配数学/代码等可验证场景）。")

    print()
    print("done · 一键复现：python code/scripts/grpo_rlvr_demo.py")


if __name__ == "__main__":
    main()
