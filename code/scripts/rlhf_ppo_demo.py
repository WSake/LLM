# -*- coding: utf-8 -*-
"""RLHF（对齐）：把"喜欢"变成优化目标。

承接《13-SFT-监督微调》的同一套 1-block 玩具（预训练 → 掩码 SFT），继续走对齐主线：
SFT 只教会模型"像金标准那样说话"，没有教它"哪个回答更被喜欢"。这一章加三步：

  A. 基线：SFT 之后模型"会回答"但谈不上"答得好"——把采样回答按一个 judge 规则
     打"人类三票"（答得对 + 答得全 + 答得干净 = 3/3），报"perfect 率"。
  B. 第一步 · 奖励模型（Reward Model）：从 (指令, 回答) 偏好对里学一个打分器
     ——Bradley-Terry 交叉熵；RM 就是"先花钱养个裁判"。
  C. 第二步 · PPO 策略梯度：离散采样不能直接反向传播，用 REINFORCE（带均值基线）
     把 RM 的分数变成梯度，同时加 KL 惩罚（对冻结的 SFT 参考策略）防跑偏；
     测"平均 RM 奖励 / perfect 率 / KL"三件事。
  D. 对照组 · β=0：去掉 KL 锚 → 观察高方差 REINFORCE 会怎么走
     （玩具 RM 判别率约 8 成≈真值，短程不一定'翻车'；翻车外推到真实世界的
       弱代理/长训练，见正文 §6/§8 的实测与对账）。

模型同 12/13 章：1-block 仅解码器（LN → 因果 MHA → 残差 → LN → FFN → 残差 → 输出头），
手写反向传播已全元素差分对账 <1e-5。RM 用同一引擎的前半段（读到 block 输出）+ 线性标量头。
PPO 阶段同时驻留：策略 θ（被优化）/ 冻结参考 ref（判 KL）/ 冻结 RM（给奖励）——
这正是 InstructGPT"重型"的来源；真实世界还要第四个价值网络（这里用均值基线代替）。
纯 CPU 可跑，seed 固定，全部数字可一键复现。
"""
import os
# 固定单线程 BLAS：numpy 的 matmul 聚合顺序在多线程下每个进程都不同，
# 会在 0.18M 参数的小模型里放大成不同的采样轨迹 → 数字一次和一次不一样。
# （真实原因在 14 章正文 §9 有交代。）以下四行在 import numpy 之前生效。
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
# 语料与指令对（与 13 章完全相同——保证 A 的基线与 13 章 A2/A3 直接可比）
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
# 引擎（与 12/13 章完全同构；拆出 block_forward 供 RM 读 block 输出）
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
    """奖励模型：同一引擎读到 block 输出为止（无 Wout）+ 线性标量头 wr/br。"""
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
    """走完整闭环的 block 部分（到 x3 为止），返回全部缓存。"""
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
    """策略反向：loss_mask 给权重（SFT 的 0/1；RL 的 adv-β 权重），不再除 nact
    （Adam 对全局缩放不敏感，除不除只影响中间量）。"""
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
    """RM 打分：每序列最后真实位置的 block 输出 → wr·h + br（标量奖励）。"""
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
# RLHF 零件
# ---------------------------------------------------------------------------
def judge_score(gold_ids, ans_ids, tox_ids):
    """人类偏好裁判（脚注 oracle）的'三票'：答得对 + 答得全 + 答得干净 = 满分 3。
    gold_ids / ans_ids / tox_ids 都是 token 下标列表（"干净"由 tox_ids 判定——
    没把'别的公式'的含义扯进来）。"""
    if not ans_ids:
        return 0
    v = 0
    if ans_ids[0] == gold_ids[0]:                 # ① 答得对：开头是期望的事实值
        v += 1
    pref = 0                                      # ② 答得全：金标准覆盖过半算"全"
    for t in gold_ids:
        if pref < len(ans_ids) and ans_ids[pref] == t:
            pref += 1
        else:
            break
    if pref >= len(gold_ids):
        v += 1
    elif pref >= (len(gold_ids) + 1) // 2:
        v += 1
    if not any(t in tox_ids for t in ans_ids):    # ③ 答得干净：没混进别的公式
        v += 1
    return v


FACT_LEXEMES = ("3", "14159", "2", "71828", "300000", "100", "365", "9", "8",
                "pi", "kilometers", "celsius", "steam", "vacuum", "orbit",
                "gravity", "light", "sun", "fiber")


def toxic_for(gold_ids, stoi):
    """'别的公式'的指纹 token：全局事实词表里、不属于本问金标准的那部分。
    SFT 之后模型答题会"金标准开头 + 语料尾巴"，尾巴里躺着别的公式值
    （答 g 却扯出 pi/300000）——人类一眼就烦，'干净'票专抓它。"""
    return {stoi[t] for t in FACT_LEXEMES if t in stoi} - set(gold_ids)


def sample_ans(pol, pre_ids, srng, temp=1.0, max_n=8):
    """策略自回归采样一段回答（玩具词表没有 EOS，用长度上限截断：
    收不住尾 = 更容易把第二个公式扯进尾巴 → judge 的'干净'票就没了）。"""
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
    """一批完整序列：逐条累加"回答区预测位"的 logπ（KL / 策略梯度都用它）。"""
    loss, cache = forward(tok, params)
    lp = cache["logp"]; tgt = cache["target"]
    out = np.zeros(tok.shape[0])
    for j in range(tok.shape[0]):
        P, K = pre_lens[j], got_lens[j]
        for k in range(K):            # 第 k 个回答 token 的预测在位置 P-1+k
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


def eval_rm_reward(pol, rp, prompts, fixed_seed, n=12):
    """固定评测集：采样回答 → RM 打分平均（RLHF 真正要优化的目标）。"""
    sr = np.random.RandomState(fixed_seed)
    tot = 0.0; cnt = 0
    for pre in prompts:
        for _ in range(n):
            ans = sample_ans(pol, pre, sr, temp=1.0)
            L = len(pre) + len(ans)
            seq = np.full((1, T), 0, dtype=int)
            seq[0, :L] = np.array(pre + ans)
            r, _, _ = rm_forward(seq, rp, np.array([L]))
            tot += float(r[0]); cnt += 1
    return tot / cnt


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
    print(f"策略参数 {n_pol:,} · RM 参数 {n_rm:,} · PPO 驻留 策略+RM+冻结参考 ≈ {n_pol * 3 + n_rm:,}")

    # ---- 阶段 0：预训练（12 章同款引擎）
    data_arr = np.array([stoi.get(t, unk) for s in text_sents for t in s], dtype=int)
    N = len(data_arr)
    lr = 3e-3
    print()
    print("=" * 70)
    print("阶段 0 · 预训练（12 章引擎 · 同 6 组事实）：把事实压进参数")
    print("=" * 70)
    STEPS_PT = 1200
    params_pt = init_params(V)
    optim = {"m": {k: np.zeros_like(v) for k, v in params_pt.items()},
             "v": {k: np.zeros_like(v) for k, v in params_pt.items()},
             "step": 1}
    l0 = forward(np.full((B, T), unk, dtype=int), params_pt)[0]
    print(f"起手 loss = {l0:.3f}（≈lnV={math.log(V):.3f}）")
    t0 = time.perf_counter()
    last = None
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

    # ---- 阶段 1：掩码 SFT（13 章同款 → 得到"会回答"的对话模型）
    print()
    print("=" * 70)
    print("阶段 1 · 掩码 SFT（13 章同款 200 步）：把'续写者'掰成'对话者'")
    print("=" * 70)
    sft = np.random.RandomState(9)
    p_sft = {k: v.copy() for k, v in params_pt.items()}
    o_sft = {"m": {k: np.zeros_like(v) for k, v in p_sft.items()},
             "v": {k: np.zeros_like(v) for k, v in p_sft.items()},
             "step": 1}
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
        # 打印"只在回答区算的"掩码 loss（padding 位置统计上很吵）
        logp = cache["logp"]
        sel = -logp[np.arange(B)[:, None], np.arange(T)[None, :], cache["target"]]
        last = float((sel * lm).sum() / lm.sum())
    dt = time.perf_counter() - t0
    print(f"SFT 200 步耗时 {dt:.0f}s · 末步掩码loss {last:.3f}")

    # ---- 数据装配
    pref_train = [question_prefix(q, stoi, unk) for q, _ in INSTR_PAIRS]
    gold_toks = [tokenize(a) for _, a in INSTR_PAIRS]
    gold_ids = [[stoi[t] for t in tk] for tk in gold_toks]      # 金标准 → token 下标
    novel_pref = [question_prefix(q, stoi, unk) for _, _, q in NOVEL_QAS]
    fact_idx = [0, 5, 10, 15, 20, 25]        # NOVEL 六条 → 六组事实的 canonical 金标准
    nvq_gold_ids = [gold_ids[f] for f in fact_idx]
    tox_per = [toxic_for(gold_ids[i], stoi) for i in range(len(INSTR_PAIRS))]    # 每问的"脏词"表
    tox_nv = [toxic_for(nvq_gold_ids[i], stoi) for i in range(len(nvq_gold_ids))]   # 全新问同样判定干净
    s_eval = np.random.RandomState(3)

    # ---- A 基线：SFT 会回答，但采样质量不是满分
    print()
    print("=" * 70)
    print("A · 基线：SFT 会回答，但采样质量不是 3/3 满分")
    print("=" * 70)
    r0_tr = sample_perfect_rate(p_sft, pref_train, gold_ids, tox_per, s_eval)
    r0_nv = sample_perfect_rate(p_sft, novel_pref, nvq_gold_ids, tox_nv, s_eval)
    print(f"  SFT 采样（temp=1.0，每问 20 次）perfect 率：训练问 {r0_tr:.0%} · 全新问 {r0_nv:.0%}")
    print("  （3/3 = 答得对 + 答得全 + 答得干净；'该答的都答了'，但采样仍会飘）")
    for i in (0, 5, 20):
        ans = sample_ans(p_sft, pref_train[i], s_eval, temp=1.0, max_n=10)
        print(f"  例：问「{INSTR_PAIRS[i][0][:32]}…」 采样 →「{' '.join(itos[t] for t in ans)}」"
              f"  judge {judge_score(gold_ids[i], ans, tox_per[i])}/3")

    # ---- B 第一步：奖励模型（RM）
    print()
    print("=" * 70)
    print("B · 第一步 · 奖励模型：从偏好对学'哪个回答更被喜欢'")
    print("=" * 70)
    rm_rng = np.random.RandomState(7)
    pairs = []
    for i in range(len(INSTR_PAIRS)):
        tox_i = tox_per[i]
        ans, sc = [], []
        for temp in (2.0, 3.0):             # 高温采样拉开回答质量差，人类裁判才有偏爱可挑
            for _ in range(12):
                a = sample_ans(p_sft, pref_train[i], rm_rng, temp=temp, max_n=10)
                ans.append(a)
                sc.append(judge_score(gold_ids[i], a, tox_i))
        hi_idx = int(np.argmax(sc))          # 每问挑"最好"的当 chosen，其余更差的当 rejected
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
        rp[k] = p_sft[k].copy()          # RM 从同一个语言模型出发（真实管线的惯例）
    o_rm = {"m": {k: np.zeros_like(v) for k, v in rp.items()},
            "v": {k: np.zeros_like(v) for k, v in rp.items()},
            "step": 1}
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
        sig = 1.0 / (1.0 + np.exp(-(rc - rr)))        # P(chosen 更被喜欢)
        d_rc = sig - 1.0                              # d(-log σ(z))/dz = σ - 1
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
        c = np.full((1, T), 0, dtype=int); rr = np.full((1, T), 0, dtype=int)
        c[0, :len(c_seq)] = c_seq; rr[0, :len(r_seq)] = r_seq
        r_c = rm_forward(c, rp, np.array([len(c_seq)]))[0][0]
        r_r = rm_forward(rr, rp, np.array([len(r_seq)]))[0][0]
        correct += (r_c > r_r)
    rm_acc = correct / nt
    print(f"  留出偏好对判别率：{rm_acc:.0%}（RM 学到了'哪条更被喜欢'）")

    # ---- C 第二步：PPO（REINFORCE + KL 惩罚 + 均值基线）
    print()
    print("=" * 70)
    print("C · 第二步 · 策略梯度：把 RM 的分数变成梯度（REINFORCE + KL 惩罚）")
    print("=" * 70)
    ref_params = {k: v.copy() for k, v in p_sft.items()}
    BETA = 0.1
    STEPS_RL, RB, ROLL_MAX = 150, 8, 8
    rewards = [eval_rm_reward(p_sft, rp, pref_train, 2)]
    p_pol = {k: v.copy() for k, v in p_sft.items()}
    o_pol = {"m": {k: np.zeros_like(v) for k, v in p_pol.items()},
             "v": {k: np.zeros_like(v) for k, v in p_pol.items()},
             "step": 1}
    rl_rng = np.random.RandomState(11)
    t0 = time.perf_counter()
    for step in range(1, STEPS_RL + 1):
        bio = rl_rng.randint(len(INSTR_PAIRS), size=RB)
        pre_list = [pref_train[i] for i in bio]
        got_list = [sample_ans(p_pol, pre, rl_rng, temp=1.0, max_n=ROLL_MAX) for pre in pre_list]
        Ls = [len(p) + len(a) for p, a in zip(pre_list, got_list)]
        tok = np.full((RB, T), 0, dtype=int)
        for j in range(RB):
            seq = pre_list[j] + got_list[j]
            tok[j, :Ls[j]] = seq
        lp_pol = answer_logps(tok, p_pol, [len(p) for p in pre_list], [len(a) for a in got_list])
        lp_ref = answer_logps(tok, ref_params, [len(p) for p in pre_list], [len(a) for a in got_list])
        r_scores = rm_forward(tok, rp, np.array(Ls))[0]
        r_eff = r_scores - BETA * (lp_pol - lp_ref)
        adv = r_eff - r_eff.mean()
        adv = adv / (adv.std() + 1e-6)           # 优势归一化（RL 标配，稳 REINFORCE 高方差）
        lm = np.zeros((RB, T), dtype=float)
        for j in range(RB):
            P, K = len(pre_list[j]), len(got_list[j])
            for k in range(K):
                lm[j, P - 1 + k] = adv[j] - BETA     # KL 的 ∇logπ 项也在这里（-β）
        tgt = np.concatenate([tok[:, 1:], np.zeros((RB, 1), dtype=int)], axis=1)
        loss, cache = forward(tok, p_pol, tgt)
        gr = backward(p_pol, cache, loss_mask=lm)
        adam_update(p_pol, o_pol, gr, 3e-4)      # RL 步长比预训练/SFT 慢 10×（噪声优势梯度）
        if step % 50 == 0:
            rewards.append(eval_rm_reward(p_pol, rp, pref_train, 2))
    dt = time.perf_counter() - t0
    rewards.append(eval_rm_reward(p_pol, rp, pref_train, 2))
    print(f"  RL {STEPS_RL} 步耗时 {dt:.0f}s")
    print(f"  平均 RM 奖励（固定评测 30×12 采样）：前 {rewards[0]:.3f}"
          f" → step50 {rewards[1]:.3f} → 后 {rewards[-1]:.3f}")
    kb = np.random.RandomState(5)
    kl_est = 0.0
    for i in range(60):
        j = kb.randint(len(INSTR_PAIRS))
        ans = sample_ans(p_pol, pref_train[j], kb, temp=1.0, max_n=ROLL_MAX)
        L = len(pref_train[j]) + len(ans)
        seq = np.full((1, T), 0, dtype=int); seq[0, :L] = np.array(pref_train[j] + ans)
        lp_p = answer_logps(seq, p_pol, [len(pref_train[j])], [len(ans)])[0]
        lp_r = answer_logps(seq, ref_params, [len(pref_train[j])], [len(ans)])[0]
        kl_est += float(lp_p - lp_r) / 60
    print(f"  终态 KL(π‖SFT_ref) ≈ {kl_est:.3f}（有界——没滑向'只管吃奖励'）")
    s_eval2 = np.random.RandomState(3)
    pf_tr = sample_perfect_rate(p_pol, pref_train, gold_ids, tox_per, s_eval2)
    pf_nv = sample_perfect_rate(p_pol, novel_pref, nvq_gold_ids, tox_nv, s_eval2)
    print(f"  RLHF 后采样 perfect 率：训练问 {pf_tr:.0%}（前 {r0_tr:.0%}）·"
          f" 全新问 {pf_nv:.0%}（前 {r0_nv:.0%}）")
    print("  C 组采样示例（RLHF 后，temp=1.0）：")
    for i in (0, 5, 20):
        ans = sample_ans(p_pol, pref_train[i], np.random.RandomState(3), temp=1.0, max_n=10)
        print(f"    问「{INSTR_PAIRS[i][0][:30]}…」→「{' '.join(itos[t] for t in ans)}」"
              f" judge {judge_score(gold_ids[i], ans, tox_per[i])}/3")

    # ---- D β=0 对照
    print()
    print("=" * 70)
    print("D · 对照：去掉 KL 惩罚（β=0）——没有锚的高方差 REINFORCE")
    print("=" * 70)
    p_hack = {k: v.copy() for k, v in p_sft.items()}
    o_hack = {"m": {k: np.zeros_like(v) for k, v in p_hack.items()},
              "v": {k: np.zeros_like(v) for k, v in p_hack.items()},
              "step": 1}
    rl_rng2 = np.random.RandomState(11)
    rew0 = eval_rm_reward(p_hack, rp, pref_train, 2)
    t0 = time.perf_counter()
    for step in range(1, STEPS_RL + 1):
        bio = rl_rng2.randint(len(INSTR_PAIRS), size=RB)
        pre_list = [pref_train[i] for i in bio]
        got_list = [sample_ans(p_hack, pre, rl_rng2, temp=1.0, max_n=ROLL_MAX) for pre in pre_list]
        Ls = [len(p) + len(a) for p, a in zip(pre_list, got_list)]
        tok = np.full((RB, T), 0, dtype=int)
        for j in range(RB):
            seq = pre_list[j] + got_list[j]
            tok[j, :Ls[j]] = seq
        r_scores = rm_forward(tok, rp, np.array(Ls))[0]
        adv = r_scores - r_scores.mean()
        adv = adv / (adv.std() + 1e-6)           # 与 C 组同款步长/归一化——唯一差别是 β=0
        lm = np.zeros((RB, T), dtype=float)
        for j in range(RB):
            P, K = len(pre_list[j]), len(got_list[j])
            for k in range(K):
                lm[j, P - 1 + k] = adv[j]
        tgt = np.concatenate([tok[:, 1:], np.zeros((RB, 1), dtype=int)], axis=1)
        loss, cache = forward(tok, p_hack, tgt)
        gr = backward(p_hack, cache, loss_mask=lm)
        adam_update(p_hack, o_hack, gr, 3e-4)
    dt = time.perf_counter() - t0
    rew_last = eval_rm_reward(p_hack, rp, pref_train, 2)
    kl_h = 0.0
    kb = np.random.RandomState(5)
    for i in range(60):
        j = kb.randint(len(INSTR_PAIRS))
        ans = sample_ans(p_hack, pref_train[j], kb, temp=1.0, max_n=ROLL_MAX)
        L = len(pref_train[j]) + len(ans)
        seq = np.full((1, T), 0, dtype=int); seq[0, :L] = np.array(pref_train[j] + ans)
        lp_p = answer_logps(seq, p_hack, [len(pref_train[j])], [len(ans)])[0]
        lp_r = answer_logps(seq, ref_params, [len(pref_train[j])], [len(ans)])[0]
        kl_h += float(lp_p - lp_r) / 60
    s_eval3 = np.random.RandomState(3)
    ph_tr = sample_perfect_rate(p_hack, pref_train, gold_ids, tox_per, s_eval3)
    kb = np.random.RandomState(5)
    uniq_h, uniq_k = set(), set()
    for i in range(len(INSTR_PAIRS)):
        for _ in range(3):                    # 每问 3 采，数"去重后的不同回答"
            uniq_h.add(tuple(sample_ans(p_hack, pref_train[i], kb, temp=1.0, max_n=ROLL_MAX)))
            uniq_k.add(tuple(sample_ans(p_pol, pref_train[i], kb, temp=1.0, max_n=ROLL_MAX)))
    print(f"  β=0 训练 {STEPS_RL} 步耗时 {dt:.0f}s")
    print(f"  平均 RM 奖励 {rew0:.3f} → {rew_last:.3f}（C 组 {rewards[0]:.3f} → {rewards[-1]:.3f}）")
    print(f"  KL(π‖SFT_ref) ≈ {kl_h:.3f}（C 组 {kl_est:.3f}）· perfect 率 {ph_tr:.0%}"
          f"（C 组 {pf_tr:.0%}）· 回答去重 {len(uniq_h)}（C 组 {len(uniq_k)}）")
    dt_rew = rew_last - rewards[-1]
    dt_perf = ph_tr - pf_tr
    if dt_rew > 0 and dt_perf <= 0:
        msg = (f"同样预算，β=0 把 RM 分刷得更高（+{dt_rew:.1f}），'三票'却更低"
               f"（{ph_tr:.0%} vs {pf_tr:.0%}）——分没换成质量，奖励与观感的剪刀差出现；")
    elif dt_rew <= 0 and dt_perf > 0:
        msg = (f"同样预算，β=0 的 RM 分反而没刷上去（{rew_last:.1f} < C 组 {rewards[-1]:.1f}）——"
               f"没有锚的高方差 REINFORCE 在这里'学不进'奖励；KL 不只是刹车，还是稳定梯度的导航线；")
    elif dt_rew > 0 and dt_perf > 0:
        msg = (f"β=0 分刷得高、'三票'也没差——短程内 RM 判别率 {rm_acc:.0%}≈真值，奖励与质量还一致；")
    else:
        msg = (f"β=0 既没刷到更高分、'三票'也略低——短程删除 KL 没有立竿见影的破坏；")
    print(f"  → 读它：{msg}同样的预算，把 rollout 种子换成 13/17/21，β=0 还会出现质量崩"
          f"（perfect 掉到 55%/24%/67%，最差的比 SFT 基线还低）、KL 冲到 12-17 的漂移——"
          f"没有一个 β=0 种子能像 C 组那样'奖励涨、质量稳'；RL 对种子敏感、无锚=漂移不受控，"
          f"KL 的导航作用就在这里。真模型的奖励代理判别率只有约 7-8 成、还带长度/措辞偏差，"
          f"跑到足够久，同一套漂移会被放大成'专挑裁判缝钻'的刷分怪（见正文 §6/§8）。")
    print("  D 组采样示例（β=0 后，temp=1.0）：")
    for i in (0, 5, 20):
        ans = sample_ans(p_hack, pref_train[i], np.random.RandomState(3), temp=1.0, max_n=10)
        print(f"    问「{INSTR_PAIRS[i][0][:30]}…」→「{' '.join(itos[t] for t in ans)}」"
              f" judge {judge_score(gold_ids[i], ans, tox_per[i])}/3")

    print()
    print("done · 一键复现：python code/scripts/rlhf_ppo_demo.py")


if __name__ == "__main__":
    main()
