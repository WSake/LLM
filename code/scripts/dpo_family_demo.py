# -*- coding: utf-8 -*-
"""DPO 家族（对齐 II）：把"喜欢"变成免 RL 的闭式损失。

承接《14-对齐-RLHF与PPO》的同一套 1-block 玩具（预训练 → 掩码 SFT → 高温偏好对 / RM），
继续走对齐主线。RLHF 那条路要"养裁判（RM）+ 在线 rollout + 三份驻留"，这章回答一个
更省钱的问题：**能不能直接在偏好对上写一个闭式损失，把 RM 和在线采样都省掉？**

  DPO   (2023)：从 RLHF 最优解出发把 RM 消掉 → 策略 + 冻结参考 + 偏好对的监督损失
  IPO   (2024)：把 DPO 的 logistic 换成长方误差（带目标 margin），治 DPO 的过优化
  KTO   (2024)：不要成对，只要"好 / 坏"单标签（对应玩具裁判的 3 分 vs 不是 3 分）
  ORPO  (2024)：连参考模型都省掉，SFT 里直接加 odds-ratio 惩罚
  SimPO (2024)：也省参考模型，换成"平均 token 概率 + 边距"的对照

五个 arm 用与 14 章 B 完全相同的 383 组偏好对（留出 58）、同一个 SFT 起手、同一预算
跑完，交叉对照"留出对判别率 / 训练与全新问三票 / KL / 墙钟"，并能直接跟 14 章
RLHF 的数字对账（RLHF 的"显式 RM + 500 步"在本脚本 B 段也完整重放：留出 79%）。

模型同 12-14 章：1-block 仅解码器（LN → 因果 MHA → 残差 → LN → FFN → 残差 → 输出头），
手写反向传播已全元素差分对账 <1e-5。纯 CPU 可跑，seed 固定，全部数字一键复现。
"""
import os
# 固定单线程 BLAS：numpy 的 matmul 聚合顺序在多线程下每个进程都不同，
# 会在 0.18M 参数的小模型里放大成不同的轨迹 → 数字一次和一次不一样。
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
# 语料与指令对（与 13/14 章完全相同——A 的基线与 14 章 A/B 直接可比）
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
# 引擎（与 12-14 章完全同构）
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


def init_rm_params(V, scale=0.06):
    """奖励模型：同一引擎读到 block 输出为止（无 Wout）+ wr/br 标量头。"""
    p = {}
    for k, (r0, r1) in {
        "Wte": (V, C), "Wpos": (T, C), "Wq": (C, C), "Wk": (C, C),
        "Wv": (C, C), "Wo": (C, C), "Wf1": (C, 4 * C), "Wf2": (4 * C, C),
    }.items():
        p[k] = rng.randn(r0, r1) * scale
    p["wr"] = rng.randn(C) * scale
    p["br"] = np.zeros(1)
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
    g = {k: np.zeros_like(v) for k, v in params.items()}
    g["Wout"] = (cache["x3"].transpose(0, 2, 1) @ d_logits).sum(axis=0)
    d_x3 = d_logits @ params["Wout"].T
    block_backward(params, cache, d_x3, g)
    return g


def rm_forward(seq, rp, Ls):
    cache = block_forward(seq, rp)
    Bb = seq.shape[0]
    h_last = cache["x3"][np.arange(Bb), Ls - 1, :]
    r = h_last @ rp["wr"] + rp["br"]
    return r, cache, h_last


def rm_backward(rp, cache, h_last, d_r, Ls):
    g = {k: np.zeros_like(v) for k, v in rp.items()}
    Bb = cache["x3"].shape[0]
    d_x3 = np.zeros((Bb, T, C))
    d_x3[np.arange(Bb), Ls - 1, :] = d_r[:, None] * rp["wr"][None, :]
    block_backward(rp, cache, d_x3, g)
    g["wr"] += h_last.T @ d_r
    g["br"] += d_r.sum()
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
# 偏好裁判与采样（与 14 章完全一致，保证同一份偏好对）
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
    """一批完整序列：逐条累加"回答区预测位"的 logπ。"""
    loss, cache = forward(tok, params)
    lp = cache["logp"]; tgt = cache["target"]
    out = np.zeros(tok.shape[0])
    for j in range(tok.shape[0]):
        P, K = pre_lens[j], got_lens[j]
        for k in range(K):
            ppos = P - 1 + k
            out[j] += lp[j, ppos, tgt[j, ppos]]
    return out


def sample_perfect_rate(pol, prompts, golds, toxs, srng, n=20):
    nb = nb3 = 0
    for pre, gold, tox in zip(prompts, golds, toxs):
        for _ in range(n):
            ans = sample_ans(pol, pre, srng, temp=1.0)
            nb += 1
            nb3 += (judge_score(gold, ans, tox) == 3)
    return nb3 / nb


# ---------------------------------------------------------------------------
# DPO 家族零件
# ---------------------------------------------------------------------------
def pair_batch(idx_list, pairs, pref_train):
    """一批 pair 下标 → (tok_c, tok_r, Lc, Lr, P)。回答区等长（玩具无 EOS）。"""
    Bx = len(idx_list)
    tok_c = np.full((Bx, T), 0, dtype=int); tok_r = np.full((Bx, T), 0, dtype=int)
    Lc = []; Lr = []; P = []
    for n in range(Bx):
        i, cw, rw = pairs[idx_list[n]]
        P.append(len(pref_train[i]))
        cs = pref_train[i] + cw; rs = pref_train[i] + rw
        Lc.append(len(cs)); Lr.append(len(rs))
        tok_c[n, :Lc[-1]] = cs; tok_r[n, :Lr[-1]] = rs
    return tok_c, tok_r, np.array(Lc), np.array(Lr), np.array(P)


def answer_region_mask(Ls, P, T):
    lm = np.zeros((len(Ls), T), float)
    for j in range(len(Ls)):
        for k in range(Ls[j] - P[j]):
            lm[j, P[j] - 1 + k] = 1.0
    return lm


def one_seq(seq):
    a = np.full((1, T), 0, dtype=int)
    a[0, :len(seq)] = seq
    return a


def log_odds_of(lp_sum, K):
    a = np.clip(lp_sum / K, -20, -1e-6)
    return a - np.log1p(-np.exp(a))


def pref_rank_acc(pol, refp, sub_pairs, pref_train, beta, mode):
    """在给定偏好对列表上，用'自己的隐式奖励'判谁更好 → 判别率。
    mode: dpo/ipo = β·(logπ−logπ_ref) 之差；orpo = log-odds 之差；simpo = β·平均logp之差。
    平局记 0.5 分——SFT 起步（无偏好信号）会得到约 50%。"""
    tot = good = 0.0
    for i, cw, rw in sub_pairs:
        P = len(pref_train[i])
        cs = pref_train[i] + cw; Lc = len(cs); Lcw = Lc - P
        rs = pref_train[i] + rw; Lr = len(rs); Lrw = Lr - P
        seq_c, seq_r = one_seq(cs), one_seq(rs)
        if mode in ("dpo", "ipo"):
            dc = (answer_logps(seq_c, pol, [P], [Lcw]) - answer_logps(seq_c, refp, [P], [Lcw])).item()
            dr = (answer_logps(seq_r, pol, [P], [Lrw]) - answer_logps(seq_r, refp, [P], [Lrw])).item()
            diff = dc - dr
        elif mode == "orpo":
            ow = log_odds_of(answer_logps(seq_c, pol, [P], [Lcw]).item(), Lcw)
            ol = log_odds_of(answer_logps(seq_r, pol, [P], [Lrw]).item(), Lrw)
            diff = ow - ol
        else:  # simpo
            diff = beta * (answer_logps(seq_c, pol, [P], [Lcw]).item() / Lcw
                           - answer_logps(seq_r, pol, [P], [Lrw]).item() / Lrw)
        if diff > 0:
            good += 1.0
        elif diff == 0:
            good += 0.5
        tot += 1.0
    return good / tot


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
    n_rm = sum(int(np.prod(s)) for s in
               [(V, C), (T, C), (C, C), (C, C), (C, C), (C, C), (C, 4 * C), (4 * C, C)]) + C + 1
    print(f"策略参数 {n_pol:,} · DPO 家族驻留 = 策略+冻结参考 {n_pol*2:,}"
          f" · RLHF 驻留 = 3×策略+RM ≈ {n_pol*3+n_rm:,}（见 14 章）")

    # ---- 阶段 0：预训练（14 章同款，保证基线逐位一致）
    data_arr = np.array([stoi.get(t, unk) for s in text_sents for t in s], dtype=int)
    N = len(data_arr)
    lr = 3e-3
    print(); print("=" * 70)
    print("阶段 0 · 预训练（12-14 章引擎 · 同 6 组事实）：把事实压进参数")
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

    # ---- 阶段 1：掩码 SFT（13/14 章同款 200 步）
    print(); print("=" * 70)
    print("阶段 1 · 掩码 SFT（13/14 章同款 200 步）：把'续写者'掰成'对话者'")
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
    ref_params = {k: v.copy() for k, v in p_sft.items()}

    # ---- A 基线
    print(); print("=" * 70)
    print("A · 基线：SFT 会回答，但采样质量不是 3/3 满分")
    print("=" * 70)
    r0_tr = sample_perfect_rate(p_sft, pref_train, gold_ids, tox_per, s_eval)
    r0_nv = sample_perfect_rate(p_sft, novel_pref, nvq_gold_ids, tox_nv, s_eval)
    print(f"  SFT 采样（temp=1.0，每问 20 次）perfect 率：训练问 {r0_tr:.0%} · 全新问 {r0_nv:.0%}")

    # ---- B 偏好对 + RM（14 章 B 段原样重放；顺带采集 KTO 的"好/坏"单标签池）
    print(); print("=" * 70)
    print("B · 偏好对 + 奖励模型（14 章同款）：RLHF 的'养裁判'基线")
    print("=" * 70)
    rm_rng = np.random.RandomState(7)
    pairs = []
    kto_pool = []
    for i in range(len(INSTR_PAIRS)):
        tox_i = tox_per[i]
        ans, sc = [], []
        for temp in (2.0, 3.0):
            for _ in range(12):
                a = sample_ans(p_sft, pref_train[i], rm_rng, temp=temp, max_n=10)
                ans.append(a)
                sc.append(judge_score(gold_ids[i], a, tox_i))
                kto_pool.append((i, a, sc[-1] == 3))
        hi_idx = int(np.argmax(sc))
        for k in range(len(ans)):
            if k != hi_idx and sc[k] < sc[hi_idx]:
                pairs.append((i, ans[hi_idx], ans[k]))
    N_pair = len(pairs)
    print(f"  偏好对：{N_pair} 组（SFT 高温采样 24 条/问 · judge 当'人类裁判'挑最好）")
    n_train = int(N_pair * 0.85)
    train_pairs, test_pairs = pairs[:n_train], pairs[n_train:]
    print(f"  留出验证：{len(test_pairs)} 组")

    rp = init_rm_params(V)
    for k in ("Wte", "Wpos", "Wq", "Wk", "Wv", "Wo", "Wf1", "Wf2"):
        rp[k] = p_sft[k].copy()
    o_rm = init_adam(rp)
    RM_STEPS, RM_B = 500, 8
    t0 = time.perf_counter()
    for step in range(1, RM_STEPS + 1):
        lr_now = 3e-3 * min(1.0, step / 100)
        idx = rm_rng.randint(len(train_pairs), size=RM_B)
        c_seq = [pref_train[train_pairs[j][0]] + train_pairs[j][1] for j in idx]
        r_seq = [pref_train[train_pairs[j][0]] + train_pairs[j][2] for j in idx]
        Lc = [len(c) for c in c_seq]; Lr = [len(r) for r in r_seq]
        tok_c = np.full((RM_B, T), 0, dtype=int); tok_r = np.full((RM_B, T), 0, dtype=int)
        for j in range(RM_B):
            tok_c[j, :Lc[j]] = c_seq[j]; tok_r[j, :Lr[j]] = r_seq[j]
        rc, cc, hc = rm_forward(tok_c, rp, np.array(Lc))
        rr, cr, hr = rm_forward(tok_r, rp, np.array(Lr))
        sig = 1.0 / (1.0 + np.exp(-(rc - rr)))
        d_rc = sig - 1.0
        d_rr = 1.0 - sig
        gc = rm_backward(rp, cc, hc, d_rc, np.array(Lc))
        gr = rm_backward(rp, cr, hr, d_rr, np.array(Lr))
        for k in gc:
            gc[k] += gr[k]
        adam_update(rp, o_rm, gc, lr_now)
    dt = time.perf_counter() - t0
    print(f"  RM 训练 {RM_STEPS} 步耗时 {dt:.0f}s")
    correct = 0; nt = min(400, len(test_pairs))
    for j in range(nt):
        tg = test_pairs[j]
        c_seq = pref_train[tg[0]] + tg[1]; r_seq = pref_train[tg[0]] + tg[2]
        c = one_seq(c_seq); rr = one_seq(r_seq)
        r_c = rm_forward(c, rp, np.array([len(c_seq)]))[0][0]
        r_r = rm_forward(rr, rp, np.array([len(r_seq)]))[0][0]
        correct += (r_c > r_r)
    rm_acc = correct / nt
    print(f"  留出偏好对判别率：{rm_acc:.0%}（显式 RM · 训 500 步才有这判别力）")

    sft_rank_a = pref_rank_acc(p_sft, ref_params, test_pairs, pref_train, 0.1, "dpo")
    print(f"  SFT 起手隐式判别率 ≈ {sft_rank_a:.0%}（DPO/IPO 家族的'0 分'：无偏好信号=平局）")

    # ---- C DPO（2023）
    print(); print("=" * 70)
    print("C · DPO（2023）：免 RM 闭式损失——把 RLHF 目标里的奖励模型消掉")
    print("=" * 70)
    STEPS, PB = 300, 8
    beta_d = 0.5   # 探针扫描 β∈{0.05,0.3,0.5,1.0}：0.05 时 KL 崩坏质量归零，0.5 最佳
    p_d = {k: v.copy() for k, v in p_sft.items()}
    o_d = init_adam(p_d)
    rd = np.random.RandomState(15)
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        idx = rd.randint(len(train_pairs), size=PB)
        tok_c, tok_r, Lc, Lr, P = pair_batch(idx, train_pairs, pref_train)
        lw = answer_logps(tok_c, p_d, P, Lc - P)
        ll = answer_logps(tok_r, p_d, P, Lr - P)
        lw_r = answer_logps(tok_c, ref_params, P, Lc - P)
        ll_r = answer_logps(tok_r, ref_params, P, Lr - P)
        z = beta_d * ((lw - lw_r) - (ll - ll_r))
        sig = 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))
        # 惯例：backward(loss_mask) 输出 Σ mask·(−logp) 的梯度。
        # L=−logσ(z) → dL/dlogp 选 = −β(1−σ) → mask 选 = +β(1−σ)；拒绝 = −β(1−σ)。符号写反=把它当最大化跑。
        lm_c = answer_region_mask(Lc, P, T) * (beta_d * (1.0 - sig))[:, None]
        lm_r = answer_region_mask(Lr, P, T) * (-beta_d * (1.0 - sig))[:, None]
        gr_c = backward(p_d, forward(tok_c, p_d)[1], loss_mask=lm_c)
        gr_r = backward(p_d, forward(tok_r, p_d)[1], loss_mask=lm_r)
        for k in gr_c:
            gr_c[k] += gr_r[k]
        adam_update(p_d, o_d, gr_c, 3e-4 * min(1.0, step / 50))
        if step in (50, 150, 300):
            acc_tr = pref_rank_acc(p_d, ref_params, train_pairs[:40], pref_train, beta_d, "dpo")
            acc_te = pref_rank_acc(p_d, ref_params, test_pairs, pref_train, beta_d, "dpo")
            print(f"    step {step}: 训练对判别率 {acc_tr:.0%} · 留出对判别率 {acc_te:.0%}")
    dt_d = time.perf_counter() - t0
    acc_te_d = pref_rank_acc(p_d, ref_params, test_pairs, pref_train, beta_d, "dpo")
    kl_d = kl_to_ref(p_d, ref_params, pref_train)
    s_eval2 = np.random.RandomState(3)
    pf_tr_d = sample_perfect_rate(p_d, pref_train, gold_ids, tox_per, s_eval2)
    pf_nv_d = sample_perfect_rate(p_d, novel_pref, nvq_gold_ids, tox_nv, s_eval2)
    print(f"  DPO {STEPS} 步耗时 {dt_d:.0f}s · 留出判别率 {acc_te_d:.0%}（显式 RM 500 步 = {rm_acc:.0%}）")
    print(f"  perfect 率：训练问 {pf_tr_d:.0%}（前 {r0_tr:.0%}）· 全新问 {pf_nv_d:.0%}（前 {r0_nv:.0%}）"
          f" · KL→SFT ≈ {kl_d:.2f}")

    # ---- D IPO（2024）
    print(); print("=" * 70)
    print("D · IPO（2024）：把 logistic 换成长方误差（target margin），治 DPO 过优化")
    print("=" * 70)
    beta_i, m_ip = 0.1, 2.0  # margin 探针：0.25 时推不动（判别率卡 52%≈SFT），2.0 才到优质区
    p_pipo = {k: v.copy() for k, v in p_sft.items()}
    o_i = init_adam(p_pipo)
    ri = np.random.RandomState(16)
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        idx = ri.randint(len(train_pairs), size=PB)
        tok_c, tok_r, Lc, Lr, P = pair_batch(idx, train_pairs, pref_train)
        lw = answer_logps(tok_c, p_pipo, P, Lc - P)
        ll = answer_logps(tok_r, p_pipo, P, Lr - P)
        lw_r = answer_logps(tok_c, ref_params, P, Lc - P)
        ll_r = answer_logps(tok_r, ref_params, P, Lr - P)
        gap = beta_i * ((lw - lw_r) - (ll - ll_r))
        d = np.clip(gap - m_ip, -3.0, 3.0)
        # L=½(gap−m)² → dL/dlogp 选 = +2βd → mask 选 = −2βd；拒绝 = −mask。
        w = -2.0 * d * beta_i
        lm_c = answer_region_mask(Lc, P, T) * w[:, None]
        lm_r = answer_region_mask(Lr, P, T) * (-w)[:, None]
        gr_c = backward(p_pipo, forward(tok_c, p_pipo)[1], loss_mask=lm_c)
        gr_r = backward(p_pipo, forward(tok_r, p_pipo)[1], loss_mask=lm_r)
        for k in gr_c:
            gr_c[k] += gr_r[k]
        adam_update(p_pipo, o_i, gr_c, 3e-4 * min(1.0, step / 50))
        if step in (50, 150, 300):
            acc_tr = pref_rank_acc(p_pipo, ref_params, train_pairs[:40], pref_train, beta_i, "dpo")
            acc_te = pref_rank_acc(p_pipo, ref_params, test_pairs, pref_train, beta_i, "dpo")
            print(f"    step {step}: 训练对判别率 {acc_tr:.0%} · 留出对判别率 {acc_te:.0%}")
    dt_i = time.perf_counter() - t0
    acc_te_i = pref_rank_acc(p_pipo, ref_params, test_pairs, pref_train, beta_i, "dpo")
    kl_i = kl_to_ref(p_pipo, ref_params, pref_train)
    pf_tr_i = sample_perfect_rate(p_pipo, pref_train, gold_ids, tox_per, s_eval2)
    pf_nv_i = sample_perfect_rate(p_pipo, novel_pref, nvq_gold_ids, tox_nv, s_eval2)
    print(f"  IPO {STEPS} 步耗时 {dt_i:.0f}s · 留出判别率 {acc_te_i:.0%} ·"
          f" perfect 训练问 {pf_tr_i:.0%} · 全新问 {pf_nv_i:.0%} · KL→SFT ≈ {kl_i:.2f}")

    # ---- E KTO（2024）
    print(); print("=" * 70)
    print("E · KTO（2024）：不要成对——'好/坏'单位标签（裁判 3 分 vs 不是 3 分）")
    print("=" * 70)
    good_pool = [e for e in kto_pool if e[2]]
    bad_pool = [e for e in kto_pool if not e[2]]
    lam_u = len(good_pool) / max(1, len(bad_pool))
    print(f"  单标签池：好 {len(good_pool)} 条（judge=3）· 坏 {len(bad_pool)} 条（judge<3）"
          f" · λ_u/λ_w = {lam_u:.2f}（按类别数量配平）")
    beta_k = 0.5
    KB, STKP = 32, 150   # 探针：b=8 时 KL 无界崩坏，b=32 且 150 步早停才对（β 也更大）
    p_k = {k: v.copy() for k, v in p_sft.items()}
    o_k = init_adam(p_k)
    rk = np.random.RandomState(17)
    lam_pool = np.array([1.0 if e[2] else lam_u for e in kto_pool])  # 好=λ_w=1 · 坏=λ_u
    t0 = time.perf_counter()
    for step in range(1, STKP + 1):
        idx = rk.randint(len(kto_pool), size=KB)
        tokx = np.full((KB, T), 0, dtype=int)
        Lx = []; Px = []; okx = np.zeros(KB, dtype=bool)
        for j in range(KB):
            i, aa, gj_ = kto_pool[idx[j]]
            okx[j] = gj_
            seqj = pref_train[i] + aa
            Px.append(len(pref_train[i])); Lx.append(len(seqj))
            tokx[j, :len(seqj)] = seqj
        Px = np.array(Px); Lx = np.array(Lx)
        rj = beta_k * (answer_logps(tokx, p_k, Px, Lx - Px)
                       - answer_logps(tokx, ref_params, Px, Lx - Px))
        z_ref = rj.mean()
        # L_good = σ(z_ref − r) → dL/dr = −σ(1−σ)；L_bad = σ(r − z_ref) → dL/dr = +σ(1−σ)
        dL = np.zeros(KB)
        u = z_ref - rj[okx]
        s_ = 1.0 / (1.0 + np.exp(-np.clip(u, -40, 40)))
        dL[okx] = -s_ * (1.0 - s_)
        u = rj[~okx] - z_ref
        s_ = 1.0 / (1.0 + np.exp(-np.clip(u, -40, 40)))
        dL[~okx] = s_ * (1.0 - s_)
        w = -dL * beta_k * lam_pool[idx]   # mask = −dL/dlogp（别忘负号）
        lm_x = answer_region_mask(Lx, Px, T) * w[:, None]
        gr = backward(p_k, forward(tokx, p_k)[1], loss_mask=lm_x)
        adam_update(p_k, o_k, gr, 3e-4 * min(1.0, step / 50))
    dt_k = time.perf_counter() - t0
    acc_te_k = pref_rank_acc(p_k, ref_params, test_pairs, pref_train, beta_k, "dpo")
    kl_k = kl_to_ref(p_k, ref_params, pref_train)
    pf_tr_k = sample_perfect_rate(p_k, pref_train, gold_ids, tox_per, s_eval2)
    pf_nv_k = sample_perfect_rate(p_k, novel_pref, nvq_gold_ids, tox_nv, s_eval2)
    print(f"  KTO {STKP} 步耗时 {dt_k:.0f}s · 留出判别率 {acc_te_k:.0%} ·"
          f" perfect 训练问 {pf_tr_k:.0%} · 全新问 {pf_nv_k:.0%} · KL→SFT ≈ {kl_k:.2f}")

    # ---- F ORPO（2024）
    print(); print("=" * 70)
    print("F · ORPO（2024）：连参考模型都省——SFT 里直接加 odds-ratio 惩罚")
    print("=" * 70)
    lam_or = 1.0
    p_o = {k: v.copy() for k, v in p_sft.items()}
    o_o = init_adam(p_o)
    ro = np.random.RandomState(18)
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        idx = ro.randint(len(train_pairs), size=PB)
        tok_c, tok_r, Lc, Lr, P = pair_batch(idx, train_pairs, pref_train)
        Kc = Lc - P; Kr = Lr - P
        aw = answer_logps(tok_c, p_o, P, Kc) / Kc
        al = answer_logps(tok_r, p_o, P, Kr) / Kr
        ow = log_odds_of(aw * Kc, Kc)
        ol = log_odds_of(al * Kr, Kr)
        sig = 1.0 / (1.0 + np.exp(-(ow - ol)))
        f = lam_or * (1.0 - sig)
        denw = np.clip(1.0 - np.exp(np.clip(aw, -20, -1e-6)), 1e-4, None)
        denl = np.clip(1.0 - np.exp(np.clip(al, -20, -1e-6)), 1e-4, None)
        # L = −log σ(ow−ol) → 选 dL/dlogp = −(1/Kc)·(1+λ(1−σ)/(1−p̄)) → mask = −dL/dlogp
        lm_c = answer_region_mask(Lc, P, T) * ((1.0 / Kc) * (1.0 + f / denw))[:, None]
        lm_r = answer_region_mask(Lr, P, T) * (-(f / denl) / Kr)[:, None]
        gr_c = backward(p_o, forward(tok_c, p_o)[1], loss_mask=lm_c)
        gr_r = backward(p_o, forward(tok_r, p_o)[1], loss_mask=lm_r)
        for k in gr_c:
            gr_c[k] += gr_r[k]
        adam_update(p_o, o_o, gr_c, 3e-4 * min(1.0, step / 50))
    dt_o = time.perf_counter() - t0
    acc_te_o = pref_rank_acc(p_o, None, test_pairs, pref_train, 1.0, "orpo")
    pf_tr_o = sample_perfect_rate(p_o, pref_train, gold_ids, tox_per, s_eval2)
    pf_nv_o = sample_perfect_rate(p_o, novel_pref, nvq_gold_ids, tox_nv, s_eval2)
    print(f"  ORPO {STEPS} 步耗时 {dt_o:.0f}s · 留出判别率 {acc_te_o:.0%} ·"
          f" perfect 训练问 {pf_tr_o:.0%} · 全新问 {pf_nv_o:.0%}（无参考模型 → 无 KL）")

    # ---- G SimPO（2024）
    print(); print("=" * 70)
    print("G · SimPO（2024）：连参考模型也省——平均 token 概率 + 边距 γ")
    print("=" * 70)
    beta_s, g_sp = 2.0, 0.0   # 探针：β=1 时质量崩（全新 5%），模型没真在分离；β=2 且 γ=0 到达优质区
    p_s = {k: v.copy() for k, v in p_sft.items()}
    o_s = init_adam(p_s)
    rs = np.random.RandomState(19)
    t0 = time.perf_counter()
    for step in range(1, STEPS + 1):
        idx = rs.randint(len(train_pairs), size=PB)
        tok_c, tok_r, Lc, Lr, P = pair_batch(idx, train_pairs, pref_train)
        Kc = Lc - P; Kr = Lr - P
        aw = answer_logps(tok_c, p_s, P, Kc) / Kc
        al = answer_logps(tok_r, p_s, P, Kr) / Kr
        dd = beta_s * (aw - al) - g_sp
        sig = 1.0 / (1.0 + np.exp(-np.clip(dd, -40, 40)))
        # L = −log σ(β(aw−al)−γ) → mask 选 = −dL/dlogp = β(1−σ)/Kc；拒绝 = −β(1−σ)/Kr
        w = beta_s * (1.0 - sig)
        lm_c = answer_region_mask(Lc, P, T) * (w / Kc)[:, None]
        lm_r = answer_region_mask(Lr, P, T) * (-w / Kr)[:, None]
        gr_c = backward(p_s, forward(tok_c, p_s)[1], loss_mask=lm_c)
        gr_r = backward(p_s, forward(tok_r, p_s)[1], loss_mask=lm_r)
        for k in gr_c:
            gr_c[k] += gr_r[k]
        adam_update(p_s, o_s, gr_c, 3e-4 * min(1.0, step / 50))
    dt_s = time.perf_counter() - t0
    acc_te_s = pref_rank_acc(p_s, None, test_pairs, pref_train, beta_s, "simpo")
    pf_tr_s = sample_perfect_rate(p_s, pref_train, gold_ids, tox_per, s_eval2)
    pf_nv_s = sample_perfect_rate(p_s, novel_pref, nvq_gold_ids, tox_nv, s_eval2)
    print(f"  SimPO {STEPS} 步耗时 {dt_s:.0f}s · 留出判别率 {acc_te_s:.0%} ·"
          f" perfect 训练问 {pf_tr_s:.0%} · 全新问 {pf_nv_s:.0%}（无参考模型 → 无 KL）")

    # ---- H 汇总 + 过优化对照
    print(); print("=" * 70)
    print("H · 汇总：同一份偏好对 · 同一预算（300 步） · 同一 SFT 起手")
    print("=" * 70)
    print(f"{'方法':<8}{'留出判别率':>10}{'训练问三票':>10}{'全新问三票':>10}{'KL→SFT':>9}{'墙钟':>7}")
    rows = [
        ("SFT 基线", sft_rank_a, r0_tr, r0_nv, 0.0, 0),
        ("显式RM(B14)", rm_acc, r0_tr, r0_nv, 0.0, 0),
        ("DPO",         acc_te_d, pf_tr_d, pf_nv_d, kl_d, dt_d),
        ("IPO",         acc_te_i, pf_tr_i, pf_nv_i, kl_i, dt_i),
        ("KTO",         acc_te_k, pf_tr_k, pf_nv_k, kl_k, dt_k),
        ("ORPO",        acc_te_o, pf_tr_o, pf_nv_o, None, dt_o),
        ("SimPO",       acc_te_s, pf_tr_s, pf_nv_s, None, dt_s),
        ("RLHF(C14)",   0.79, 0.70, 0.67, 10.65, -1),
    ]
    for name, a1, a2, a3, kl, dtv in rows:
        kl_s = "—" if kl is None else f"{kl:.2f}"
        dtv_s = "153s" if dtv < 0 else (f"{dtv:.0f}s" if dtv else "—")
        print(f"{name:<8}{a1:>8.0%} {a2:>9.0%} {a3:>9.0%} {kl_s:>9} {dtv_s:>7}")
    print("  注：RLHF(C14) 行引自 14 章 canonical run3（RM 留出 79% / RLHF 后三票 70-67% /"
          " KL 10.65 / 全脚本 153s）——在线 rollout + 三份驻留的完整 RLHF；本脚本 B 段重放其 RM 部分。")
    print("  注：'留出判别率'各 method 用自己家族的打分器；DPO/KTO 与显式 RM 的口径可公度（优劣排序）。")
    print("  注：玩具词表没有 EOS → 每回答恒长 8/10 token，SimPO 长度归一与 14 章的'长度奖励代理'"
          " 同样惰性——长度偏置在这套玩具里测不出来（见正文 §7 的诚实声明）。")

    print()
    print("done · 一键复现：python code/scripts/dpo_family_demo.py")


if __name__ == "__main__":
    main()
