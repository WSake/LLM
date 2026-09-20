# -*- coding: utf-8 -*-
"""SFT（监督微调）：把"续写者"掰成"对话者"。

承接《12-预训练-Next-Token-Prediction》的同一套 1-block 玩具：
先在声明式语料上预训练（把 6 组科学事实"知道"），再在「指令 → 期望回答」对上做 SFT。
四段实验：

  A1. 症状：预训练模型面对 `q : … ? a :` 只会"续写"，不会"回答"
      ——事实它全知道（12 章 A3 已验证），但指令格式从没见过 → 答案 top-1 几乎不中
  A2. 掩码 SFT 的机械：只在回答区算交叉熵（loss masking），解释"掩码在干嘛"；
      掩码 vs 朴素同一预算训练——答案都学得会，差别在问题区 CE（朴素把梯度
      花在"复述用户问题"上，掩码从不学问题词）
  A3. 泛化：全新句式的问题（任何数据里都没整体出现过），SFT 后命中 6/6、
      SFT 前同 cue 几乎全错 → "会"和"会表达"是两件事
  A4. 质量 >> 数量：30 条"5 句式多样" vs 30 条"1 句式重复"，
      同一预算同一批数——只有多样化训练过的模型答得上全新问题

模型同 12 章：1-block 仅解码器（LN → 因果 MHA → 残差 → LN → FFN → 残差 → 输出头），
手写反向传播已全元素差分对账 <1e-5。SFT 只是多了一个 loss_mask 参数。
评估一律用"左对齐 + 末尾取读"，与 SFT 训练时"指令对贴在窗口开头"的位置布局一致
（Wpos 可学习，绝对位置对不上会判错；这是比 12 章更严格的同分布评估）。
纯 CPU 可跑，seed 固定，全部数字可一键复现。
"""
import re
import sys
import math
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.RandomState(42)

# ---------------------------------------------------------------------------
# 阶段 0 语料：声明式英文句子（科学事实 × 6 + 通用文本），与 12 章同型、同 6 组事实
# （地球重力 9.8 / π / e / 光速 / 沸点 100 / 公转 365）。预训练只做一件事：
# 让模型在"陈述句"里把 6 组事实压进参数。
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

# ---------------------------------------------------------------------------
# 阶段 1 数据：指令对（q…? a…）——答案区第一个 token 恒等于事实值。
# 6 个事实 × 5 种问法 = 30 条，刻意多样化（多样性就是"质量"的一部分）。
# ---------------------------------------------------------------------------
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

# 阶段 2 测试：全新句式的问题（训练/预训练语料都没有整体出现过），只改措辞、
# 换组合，不换"事实值"；不引入任何训练词表外的词（脚本内 assert）。
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
    """词表只从训练数据（声明式文本 + 指令对）建；测试句若出现 OOV 会被标 <unk>，
    因此刻意保证 NOVEL_QAS 全部词都在词表内（脚本里 assert）。"""
    counter = {}
    for toks in train_sents:
        for t in toks:
            counter[t] = counter.get(t, 0) + 1
    vocab = ["<unk>"] + sorted(counter)                 # min_freq=1：小语料全保真
    stoi = {t: i for i, t in enumerate(vocab)}
    return stoi, vocab


# ---------------------------------------------------------------------------
# 模型：与 12 章完全同构的 1-block 仅解码器。唯一改动：forward/backward 接受
# loss_mask —— 只在"回答区"位置的预测上算梯度（SFT 的心脏）。
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


def forward(tok, params, target=None, loss_mask=None):
    """tok: (B,T) int；loss_mask: (B,T) float（1=该"预测位"参与交叉熵）。
    target 缺省=左右错位（train_step 显式传下一位置流）。"""
    (Wte, Wpos, Wq, Wk, Wv, Wo, Wf1, Wf2, Wout) = (
        params["Wte"], params["Wpos"], params["Wq"], params["Wk"],
        params["Wv"], params["Wo"], params["Wf1"], params["Wf2"], params["Wout"])
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
    logits = x3 @ Wout
    logp = logits - logits.max(axis=-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))
    if target is None:
        target = np.concatenate([tok[:, 1:], np.zeros((Bb, 1), dtype=int)], axis=1)
    if loss_mask is None:
        loss_mask = np.ones_like(target, dtype=float)
    nact = loss_mask.sum()
    loss = -(logp[np.arange(Bb)[:, None], np.arange(T)[None, :], target] * loss_mask).sum() / nact
    cache = dict(x=x, n1=n1, m1=m1, s1=s1, q=q, k=k, v=v, att=att,
                 y=y, x2=x2, n2=n2, m2=m2, s2=s2, h=h, x3=x3,
                 logits=logits, logp=logp, target=target, loss_mask=loss_mask, nact=nact)
    return loss, cache


def ln_backward(dy, t, m, s):
    d = t - m
    return (dy - dy.mean(axis=-1, keepdims=True)
            - d * (dy * d).mean(axis=-1, keepdims=True) / s**2) / s


def backward(params, cache):
    """与 12 章/01-Transformer 完全同构的手写反向（已全元素差分对账）；
    差别只有一处：d_logits 乘上 loss_mask 并按有效位数归一。"""
    g = {k: np.zeros_like(v) for k, v in params.items()}
    logits, logp, att, q, kk, v = (cache["logits"], cache["logp"], cache["att"],
                                   cache["q"], cache["k"], cache["v"])
    x, n1, m1, s1 = cache["x"], cache["n1"], cache["m1"], cache["s1"]
    x2, n2, m2, s2, h, x3, y = (cache["x2"], cache["n2"], cache["m2"],
                                cache["s2"], cache["h"], cache["x3"], cache["y"])
    target = cache["target"]
    lm = cache["loss_mask"]
    Bb = target.shape[0]

    p = np.exp(logp)
    onehot = np.zeros_like(logits)
    onehot[np.arange(Bb)[:, None], np.arange(T)[None, :], target] = 1
    d_logits = (p - onehot) * lm[..., None] / cache["nact"]

    g["Wout"] = (x3.transpose(0, 2, 1) @ d_logits).sum(axis=0)
    d_x3 = d_logits @ params["Wout"].T

    d_x2 = d_x3.copy()
    d_h = d_x3 @ params["Wf2"].T
    g["Wf2"] = (h.transpose(0, 2, 1) @ d_x3).sum(axis=0)
    d_n2 = d_h.copy()
    d_n2[h <= 0] = 0
    g["Wf1"] = (n2.transpose(0, 2, 1) @ d_n2).sum(axis=0)
    d_n2 = d_n2 @ params["Wf1"].T
    d_x2 = d_x2 + ln_backward(d_n2, x2, m2, s2)

    d_x = d_x2.copy()
    d_y = d_x2 @ params["Wo"].T
    g["Wo"] = (y.transpose(0, 2, 1) @ d_x2).sum(axis=0)

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
    g["Wq"] = (n1.transpose(0, 2, 1) @ d_q).sum(axis=0)
    g["Wk"] = (n1.transpose(0, 2, 1) @ d_kk).sum(axis=0)
    g["Wv"] = (n1.transpose(0, 2, 1) @ d_v).sum(axis=0)

    d_x = d_x + ln_backward(d_n1, x, m1, s1)
    g["Wpos"] = d_x.sum(axis=0)
    np.add.at(g["Wte"], cache["input_tok"], d_x)
    return g


def forward_eval_left(seq, params, pos=None):
    """左对齐评估：前缀从位置 0 起、读末尾 logits——与 SFT 训练时"指令对贴在窗口
    开头"的位置布局一致（Wpos 可学习，绝对位置错位会判错）。pos 可指定读哪个位置。"""
    seq = np.asarray(seq, dtype=int)[-T:]
    L = len(seq)
    tok = np.full((1, T), 0, dtype=int)
    tok[0, :L] = seq
    cache = forward(tok, params)[1]
    return cache["logits"][0, pos if pos is not None else L - 1]


def sample(seq, params, n_new, temp=0.9, seed=None):
    r = np.random.RandomState(seed) if seed is not None else rng
    cur = list(seq)
    for _ in range(n_new):
        if len(cur) >= T:
            cur = cur[-(T - 1):]
        logits = forward_eval_left(np.array(cur, dtype=int), params)
        if temp == 0:
            nxt = int(np.argmax(logits))
        else:
            p = softmax(logits / temp)
            nxt = int(r.choice(len(p), p=p))
        cur.append(nxt)
    return cur


def train_step(params, optim, data_arr, N, lr_now):
    """预训练步：随机窗口 + 错位 target，全位置参与。"""
    starts = rng.randint(0, N - T, size=B)
    tok = np.stack([data_arr[s:s + T] for s in starts])
    tgt = np.stack([data_arr[s + 1:s + T + 1] for s in starts])
    loss, cache = forward(tok, params, tgt)
    cache["input_tok"] = tok
    grads = backward(params, cache)
    # Adam（β1 0.9，β2 0.999，basic-step-1 项）
    for k, p in params.items():
        optim["m"][k] = 0.9 * optim["m"][k] + (1 - 0.9) * grads[k]
        optim["v"][k] = 0.999 * optim["v"][k] + (1 - 0.999) * grads[k] ** 2
        mhat = optim["m"][k] / (1 - 0.9 ** optim["step"])
        vhat = optim["v"][k] / (1 - 0.999 ** optim["step"])
        p -= lr_now * mhat / (np.sqrt(vhat) + 1e-8)
    optim["step"] += 1
    return loss


def parse_sft_pair(qs, ans, stoi, unk):
    """拼成 q : <问题> ? a : <答案> 并确定答案区起点。"""
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
    """问题前缀 → 到 `a :` 为止的 token 列表（预测第一个答案 token 的输入）。"""
    txt = "q : " + qs + " a :"
    return [stoi.get(t, unk) for t in tokenize(txt)]


def main():
    # ---- 词表（只从训练数据建；测试句 OOV 当场报错）
    text_sents = [tokenize(s) for s in TEXT_CORPUS]
    instr_sents = [tokenize("q : " + q + " a : " + a) for q, a in INSTR_PAIRS]
    stoi, itos = build_vocab(text_sents + instr_sents)
    V = len(itos)
    unk = stoi["<unk>"]
    print("=" * 70)
    print(f"语料：声明式文本 {len(TEXT_CORPUS)} 句 + 指令对 {len(INSTR_PAIRS)} 条 · 词表 V={V}")
    oov = sorted({t for _, _, q in NOVEL_QAS for t in tokenize(q)} - set(itos))
    assert not oov, f"测试句含未登录词：{oov}"
    print("测试句未登录词：无 ✅（测试问法全部只复用训练词表里的词）")
    # 参数个数从形状算（不真初始化——那会白白消耗一整条随机数流，把"固定 seed 可复现"带偏）
    n_params = sum(int(np.prod(s)) for s in
                   [(V, C), (T, C), (C, C), (C, C), (C, C), (C, C), (C, 4 * C), (4 * C, C), (C, V)])
    print(f"模型参数：{n_params:,}")

    # ---- 阶段 0：预训练（12 章同款——声明式语料，把事实压进参数）
    data_arr = np.array([stoi.get(t, unk) for s in text_sents for t in s], dtype=int)
    N = len(data_arr)
    lr = 3e-3
    print()
    print("=" * 70)
    print(f"阶段 0 · 预训练（12 章同款引擎 · 同 6 组事实的声明式语料）：声明式 {N} token")
    print("=" * 70)
    STEPS_PT = 1200
    params_pt = init_params(V)
    optim = {"m": {k: np.zeros_like(v) for k, v in params_pt.items()},
             "v": {k: np.zeros_like(v) for k, v in params_pt.items()},
             "step": 1}
    l0, _ = forward(np.full((B, T), unk, dtype=int), params_pt)
    print(f"起手 loss = {l0:.3f}（≈lnV={math.log(V):.3f}）")
    t0 = time.perf_counter()
    last = None
    for step in range(1, STEPS_PT + 1):
        lr_now = lr * min(1.0, step / 200)
        last = train_step(params_pt, optim, data_arr, N, lr_now)
        if step % 400 == 0:
            print(f"  step {step:5d}  loss {last:6.3f}  ppl {math.exp(last):6.2f}")
    dt = time.perf_counter() - t0
    print(f"预训练 {STEPS_PT} 步耗时 {dt:.1f}s · 末步 loss {last:.3f}")
    print("  （玩具已把 6 组事实背进参数——下面 A1 就看它会不会「说」出来）")

    # ---- SFT 数据与评估器
    sft_pairs = [parse_sft_pair(q, a, stoi, unk) for q, a in INSTR_PAIRS]
    novel_pref = [(name, exp, question_prefix(q, stoi, unk)) for name, exp, q in NOVEL_QAS]

    def answer(p, pref_ids):
        return int(np.argmax(forward_eval_left(np.array(pref_ids, dtype=int), p)))

    def novel_acc(p):
        return sum(itos[answer(p, pref)] == exp for _, exp, pref in novel_pref)

    def train_acc(p):
        got = 0
        for qq, aa in INSTR_PAIRS:
            pref = question_prefix(qq, stoi, unk)
            if itos[answer(p, pref)] == tokenize(aa)[0]:
                got += 1
        return got

    # ---- A1 症状：纯预训练面对指令
    print()
    print("=" * 70)
    print("实验 A1 · 症状：预训练模型面对指令不会回答（会但不表达）")
    print("=" * 70)
    hits_pt = 0
    for name, exp, pref in novel_pref:
        top = itos[answer(params_pt, pref)]
        ok = top == exp
        hits_pt += ok
        qtxt = " ".join(itos[t] for t in pref).replace("q : ", "").replace(" a :", "")[:36]
        print(f"  {name:<9} 问「{qtxt}」 → 续写为 {top:<7}（期望答案开头 {exp}）{'✅' if ok else '❌'}")
    print(f"  → 纯预训练 top-1 命中 {hits_pt}/{len(NOVEL_QAS)}（事实它全知道，但指令格式没学过）")
    for name, exp, pref in novel_pref[:2]:
        outs = sample(pref, params_pt, 16, temp=0.9, seed=7)
        qtxt = " ".join(itos[t] for t in pref).replace("q : ", "").replace(" a :", "")
        show = qtxt + "  →  " + " ".join(itos[t] for t in outs)
        print(f"  └ 采样续写（{name}）：{show[:100]}")

    # ---- A2 掩码的机械：只在回答区算账
    print()
    print("=" * 70)
    print("实验 A2 · 掩码 vs 朴素：同一份指令数据、同一预算、同一批数")
    print("=" * 70)
    Ls = np.array([len(t) for t, _ in sft_pairs])
    ans_frac = np.array([(len(t) - a) / len(t) for t, a in sft_pairs])
    print(f"  指令平均 {np.mean(Ls):.1f} token；回答区平均占 {np.mean(ans_frac):.0%}"
          f" → 掩码剔除 {(1 - np.mean(ans_frac)):.0%} 的预测位（问题词不进梯度）")

    SFT_SEED, STEPS_SFT = 9, 200

    def sft_train(pairs_list, masked, seed=SFT_SEED):
        r = np.random.RandomState(seed)
        p = {k: v.copy() for k, v in params_pt.items()}
        o = {"m": {k: np.zeros_like(v) for k, v in p.items()},
             "v": {k: np.zeros_like(v) for k, v in p.items()},
             "step": 1}
        t0 = time.perf_counter()
        last = None
        for s in range(1, STEPS_SFT + 1):
            lr_now = lr * min(1.0, s / 200)
            toks_all, masks_all = [], []
            for _ in range(B):
                t, ans_start = pairs_list[r.randint(len(pairs_list))]
                L = len(t); arr = np.full(T, unk, dtype=int); arr[:L] = t
                if masked:
                    m = np.zeros(T, dtype=float)
                    for pp in range(max(0, ans_start - 1), L - 1):
                        m[pp] = 1.0
                else:
                    m = np.ones(T, dtype=float) * (np.arange(T) < L - 1)
                toks_all.append(arr); masks_all.append(m)
            tok, lm = np.stack(toks_all), np.stack(masks_all)
            tgt = np.concatenate([tok[:, 1:], np.zeros((B, 1), dtype=int)], axis=1)
            loss, cache = forward(tok, p, tgt, lm)
            cache["input_tok"] = tok
            gr = backward(p, cache)
            for k, pp in p.items():
                o["m"][k] = 0.9 * o["m"][k] + 0.1 * gr[k]
                o["v"][k] = 0.999 * o["v"][k] + 0.001 * gr[k] ** 2
                mh = o["m"][k] / (1 - 0.9 ** o["step"]); vh = o["v"][k] / (1 - 0.999 ** o["step"])
                pp -= lr_now * mh / (np.sqrt(vh) + 1e-8)
            o["step"] += 1
            last = loss
            if s in (50, 100, 200):
                print(f"    [{('掩码' if masked else '朴素')}] step {s:4d}  loss {loss:6.3f}")
        dt = time.perf_counter() - t0
        return p, last, dt

    def ce_regions(p):
        """30 条指令对逐条左对齐过一遍：问题区 / 回答区的平均 -logp（期望越小越好）。"""
        n_q, n_a, ce_q, ce_a = 0, 0, 0.0, 0.0
        for t, ans_start in sft_pairs:
            L = len(t)
            tok = np.full((1, T), 0, dtype=int)
            tok[0, :L] = t
            logp = forward(tok, p)[1]["logp"][0]
            tgt = np.concatenate([tok[0, 1:], [0]])   # 与训练同款错位：tgt[p] = tok[p+1]
            sel = logp[np.arange(T), tgt]
            qmask = np.arange(T) + 1 < ans_start
            amask = (np.arange(T) + 1 >= ans_start) & (np.arange(T) + 1 < L)
            ce_q += -sel[qmask].sum(); n_q += int(qmask.sum())
            ce_a += -sel[amask].sum(); n_a += int(amask.sum())
        return ce_q / n_q, ce_a / n_a

    print(f"  （从预训练权重继续，{STEPS_SFT} 步 · 掩码/朴素共用同一批数）")
    params_m, last_m, dt_m = sft_train(sft_pairs, masked=True)
    params_n, last_n, dt_n = sft_train(sft_pairs, masked=False)
    q_m, a_m = ce_regions(params_m)
    q_n, a_n = ce_regions(params_n)
    print(f"  掩码 SFT：{dt_m:.0f}s · 训练问 {train_acc(params_m)}/30 · 全新问 {novel_acc(params_m)}/6"
          f" · 问题区CE {q_m:.3f} / 回答区CE {a_m:.3f}")
    print(f"  朴素 SFT：{dt_n:.0f}s · 训练问 {train_acc(params_n)}/30 · 全新问 {novel_acc(params_n)}/6"
          f" · 问题区CE {q_n:.3f} / 回答区CE {a_n:.3f}")
    print(f"  → 两版都学会了回答（回答区 CE 都≈0）；差别留在了问题区：朴素把预算花在"
          f"'复述问题词'上（CE {q_n:.2f}），掩码从头到尾不学问题词（CE {q_m:.2f}）")

    # ---- A3 泛化总表
    print()
    print("=" * 70)
    print("实验 A3 · 泛化总表：全新句式问题，SFT 前（纯预训练） vs SFT 后（掩码）")
    print("=" * 70)
    print(f"{'知识':<10}{'问题（语料里没整体出现过的新问法）':<44}{'期望':<8}{'SFT前':<8}{'SFT后':<8}")
    for name, exp, pref in novel_pref:
        qtxt = " ".join(itos[t] for t in pref).replace("q : ", "").replace(" a :", "")
        print(f"{name:<10}{qtxt[:44]:<44}{exp:<8}{itos[answer(params_pt, pref)]:<8}{itos[answer(params_m, pref)]:<8}")
    print(f"  命中：SFT 前 {hits_pt}/6 vs SFT 后 {novel_acc(params_m)}/6"
          f"（随机初始化对照≈1/251≈0.4%，纯猜）")

    # ---- A4 质量 >> 数量
    print()
    print("=" * 70)
    print("实验 A4 · 质量 >> 数量：同 30 条、同预算、同步数——多样性 vs 重复")
    print("=" * 70)
    dup_txt = []
    for i in range(0, len(INSTR_PAIRS), 5):
        q, a = INSTR_PAIRS[i]
        for _ in range(5):
            dup_txt.append((q, a))
    dup_pairs = [parse_sft_pair(q, a, stoi, unk) for q, a in dup_txt]
    dup_seen = [(f"v{j // 5 + 1}", tokenize(INSTR_PAIRS[j][1])[0], question_prefix(INSTR_PAIRS[j][0], stoi, unk))
                for j in range(0, len(INSTR_PAIRS), 5)]

    def seen_acc(p):
        return sum(itos[answer(p, pre)] == exp for _, exp, pre in dup_seen)

    params_d, _, dt_d = sft_train(dup_pairs, masked=True)
    print(f"  多样（5 句式 × 6 事实）：见过的问法 {train_acc(params_m)}/30 · 全新问法 {novel_acc(params_m)}/6（复用 A2 掩码模型）")
    print(f"  重复（1 句式 × 30 条，其实只有 6 种问法 ×5 复本）"
          f"：见过的问法 {seen_acc(params_d)}/6 · 全新问法 {novel_acc(params_d)}/6 · {dt_d:.0f}s")
    print(f"  → 条数一样、预算一样：重复组把见过的 6 种问法背得滚瓜烂熟，但一换新问法就塌到"
          f" {novel_acc(params_d)}/6；决定泛化的不是数量，是条与条之间的多样性")

    print()
    print("done · 一键复现：python code/scripts/sft_demo.py")


if __name__ == "__main__":
    main()
