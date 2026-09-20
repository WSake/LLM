# -*- coding: utf-8 -*-
"""预训练 = Next Token Prediction：从文本流里"长出"知识与续写能力。

配合《02-核心原理/12-预训练-Next-Token-Prediction.md》使用。四段实验：

  A1. shift 对齐与数据复用：一段 N token 的文本 = N-1 个训练样本，
      每个 token 同时充当"前一个位置的答案"与"后一个位置的依据"
  A2. 训练看懂 loss：纯 CPU 训练一个极简 1-block transformer，
      随机初始化起手 loss ≈ ln(V)，训练后降到可读范围（ppl = exp(loss)）
  A3. 事实泛化：语料里反复出现的"知识"（g≈9.8 / π≈3.14 / e≈2.718 …）
      在全新句式上也能被续对——对照随机初始化模型全错
  A4. 自回归采样：把学到的模型把玩起来，看续写质量与路径依赖

模型是 01-Transformer 那套"手写反向传播已全元素差分对账 <1e-5"的
1-block 仅解码器（LN → 因果 MHA → 残差 → LN → FFN → 残差 → 输出头），
本脚本只补上：数据装载、Adam 优化器、训练循环、采样三样东西。
纯 CPU 可跑，seed 固定，全部数字可一键复现。word-level 简化（英文，
中文需子词，见 08-Tokenizer）——语料与超参刻意极小，"玩具"是演示机制。
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
# 语料：多主题英文知识句。每个"知识"以 5 种句式重复出现（给统计加权），
# 另配一个"泛化测试句"（训练语料里不出现）——验证知识跨句式迁移。
# ---------------------------------------------------------------------------
CORPUS = [
    # —— 知识 1：g ≈ 9.8（重力加速度）
    "the gravitational acceleration on earth near the surface is about 9 . 8 meters per second squared .",
    "physicists say the acceleration due to gravity near the ground is about 9 . 8 .",
    "when you drop an object on earth it speeds up at about 9 . 8 meters per second each second .",
    "the symbol g denotes the local acceleration which is close to 9 . 8 on earth .",
    "falling near the ground objects accelerate at 9 . 8 meters per second squared .",
    # —— 知识 2：π ≈ 3.14159
    "pi is approximately 3 . 14159 and appears in every circle formula .",
    "the ratio between the circumference and the diameter of a circle is 3 . 14159 .",
    "mathematicians approximate pi as 3 . 14159 but its decimal never ends .",
    "the circle constant pi equals roughly 3 . 14159 .",
    "a circle circumference divided by its diameter gives 3 . 14159 called pi .",
    # —— 知识 3：e ≈ 2.71828
    "the number e roughly equals 2 . 71828 and is the base of natural logarithms .",
    "natural logarithm uses base e which is about 2 . 71828 .",
    "compound interest grows forever at the constant e around 2 . 71828 .",
    "the exponential constant e is approximately 2 . 71828 .",
    "logarithms to the base e use the number e = 2 . 71828 .",
    "the natural constant e which is about 2 . 71828 appears in many growth formulas .",
    # —— 知识 4：光速 ≈ 300000 km/s
    "light travels in vacuum at about 300000 kilometers per second .",
    "the speed of light is close to 300000 kilometers per second .",
    "in one second light crosses about 300000 kilometers .",
    "light covers roughly 300000 kilometers every second in empty space .",
    "nothing moves faster than light which goes about 300000 kilometers per second .",
    "laser pulses in fiber travel at nearly light speed about 300000 kilometers per second .",
    # —— 知识 5：水在 100 度开
    "water boils at 100 degrees celsius at standard pressure .",
    "at sea level water turns to vapor at 100 celsius .",
    "the boiling point of pure water under normal pressure is 100 degrees celsius .",
    "water reaches its boiling point at 100 celsius at one atmosphere .",
    "normal water starts to boil when the temperature hits 100 degrees celsius .",
    "an open pot of water at sea level is 100 degrees when it bubbles .",
    # —— 知识 6：地球公转 ≈ 365 天
    "earth takes about 365 days to go around the sun .",
    "one full orbit of earth around the sun lasts about 365 days .",
    "earth travels its yearly circle of the sun in roughly 365 days .",
    "the complete journey of earth around the sun needs about 365 days .",
    "a common calendar year matches one earth orbit about 365 days .",

    # —— 通用文本（给模型提供"句子语法/主题惯例"）
    "the sun is a star at the center of our solar system .",
    "the moon circles the earth about once every month .",
    "a year is the time it takes a planet to complete one orbit .",
    "gravity pulls every object toward the center of the planet .",
    "energy cannot be created or destroyed only converted into another form .",
    "atoms combine to form molecules and molecules form materials .",
    "the air around us is mostly nitrogen and oxygen .",
    "sound travels faster in water than in air .",
    "voltage measures the electric pressure in a circuit .",
    "a battery stores chemical energy and delivers electric energy .",
    "red light has a longer wavelength than blue light .",
    "green plants use sunlight to make sugar by photosynthesis .",
    "roots absorb water and minerals from the soil .",
    "the heart pumps blood through the body every second .",
    "oxygen enters the blood in the lungs .",
    "the brain controls movement memory and language .",
    "temperature measures the average motion of particles .",
    "mountain air is colder because the pressure is lower .",
    "rain falls when water droplets in clouds become heavy enough .",
    "the seasons change because the earth axis is tilted .",
    "a telescope collects light and magnifies distant objects .",
    "mirrors reflect light according to a simple law of angles .",
    "sound is a pressure wave that travels through a medium .",
    "stars are born inside clouds of gas that grow dense enough .",
    "planets move in elliptical paths around their stars .",
    "comets are small icy bodies with long bright tails .",
    "rivers carry sediment downhill toward the ocean .",
    "the ocean is salty because rivers carry dissolved minerals .",
    "wind is caused by air moving from high pressure to low pressure .",
    "clouds form when warm air rises and cools .",
    "copper and gold are good conductors of electricity .",
    "glass is a bad conductor and is used to insulate wires .",
    "water expands when it freezes which is why ice floats .",
    "salt lowers the freezing point of water .",
    "a magnet has a north pole and a south pole .",
    "a compass needle points toward magnetic north .",
    "a prism splits white light into the colors of the rainbow .",
    "the sky is blue because molecules scatter blue light harder .",
    "sunsets look red because the light travels through more air .",
    "an object in motion stays in motion unless a force changes it .",
    "heavy objects fall at the same rate as light ones in vacuum .",
    "a rocket works by pushing gas backward very fast .",
    "orbiting astronauts appear weightless because they keep falling .",
    "a pendulum swings back and forth with a steady period .",
]


def tokenize(sent):
    return [t.lower() for t in re.findall(r"[a-zA-Z']+|[0-9]+|[.,!?;:()\-]", sent)]


def build_vocab(sents, min_freq=1):
    """word-level 词表。语料很小（~900 token），min_freq=1 即全保真、无 <unk>；
    真实预训练语料海量，才有子词与低频裁剪问题（见 08-Tokenizer）。"""
    counter = {}
    for s in sents:
        for t in s:
            counter[t] = counter.get(t, 0) + 1
    vocab = ["<unk>"] + sorted(t for t, c in counter.items() if c >= min_freq)
    stoi = {t: i for i, t in enumerate(vocab)}
    itos = vocab
    return stoi, itos, counter


# ---------------------------------------------------------------------------
# 模型：1-block 仅解码器（与 01-Transformer 同款结构，反向已全元素对账）。
# LN 无参（简化，教学见 07-归一化）；位置嵌入可学习。
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


def forward(tok, params, target=None):
    """tok: (B,T) int；target: (B,T) 可选（缺省=左右错位，供评估用）。
    训练时 target 传真实的"下一位置流"。batch 由输入推导。"""
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
    loss = -logp[np.arange(Bb)[:, None], np.arange(T)[None, :], target].mean()
    cache = dict(x=x, n1=n1, m1=m1, s1=s1, q=q, k=k, v=v, att=att,
                 y=y, x2=x2, n2=n2, m2=m2, s2=s2, h=h, x3=x3,
                 logits=logits, logp=logp, target=target)
    return loss, cache


def ln_backward(dy, t, m, s):
    d = t - m
    return (dy - dy.mean(axis=-1, keepdims=True)
            - d * (dy * d).mean(axis=-1, keepdims=True) / s**2) / s


def backward(params, cache):
    """与 01-Transformer 完全同构的手写反向（那边已全元素差分对账 <1e-5）。"""
    g = {k: np.zeros_like(v) for k, v in params.items()}
    logits, logp, att, q, kk, v = (cache["logits"], cache["logp"], cache["att"],
                                   cache["q"], cache["k"], cache["v"])
    x, n1, m1, s1 = cache["x"], cache["n1"], cache["m1"], cache["s1"]
    x2, n2, m2, s2, h, x3, y = (cache["x2"], cache["n2"], cache["m2"],
                                cache["s2"], cache["h"], cache["x3"], cache["y"])
    target = cache["target"]
    Bb = target.shape[0]

    p = np.exp(logp)
    onehot = np.zeros_like(logits)
    onehot[np.arange(Bb)[:, None], np.arange(T)[None, :], target] = 1
    d_logits = (p - onehot) / (Bb * T)

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


def forward_eval(seq, params):
    """单序列评估：seq (L,) int；返回最末位置 logits (V,)。
    位置总用 0..T-1、取 seq 尾部窗口（简化：玩具只看最近 48 个 token 的上下文）。"""
    seq = np.asarray(seq, dtype=int)[-T:]
    L = len(seq)
    tok = np.full((1, T), 0, dtype=int)
    tok[0, T - L:] = seq
    cache = forward(tok, params)[1]
    return cache["logits"][0, T - 1]


def sample(seq, params, n_new, temp=0.8, seed=None):
    """自回归续写：给定前缀 seq (list)，续 n_new 个 token。"""
    r = np.random.RandomState(seed) if seed is not None else rng
    cur = list(seq)
    for _ in range(n_new):
        if len(cur) >= T:
            cur = cur[-(T - 1):]
        logits = forward_eval(np.array(cur, dtype=int), params)
        if temp == 0:
            nxt = int(np.argmax(logits))                 # 贪心：T=0 就是取峰
        else:
            p = softmax(logits / temp)
            nxt = int(r.choice(len(p), p=p))
        cur.append(nxt)
    return cur


def train_step(params, optim, data_arr, N, lr_now):
    starts = rng.randint(0, N - T, size=B)
    tok = np.stack([data_arr[s:s + T] for s in starts])
    tgt = np.stack([data_arr[s + 1:s + T + 1] for s in starts])   # 下一位置流
    loss, cache = forward(tok, params, tgt)
    cache["input_tok"] = tok
    grads = backward(params, cache)
    # Adam
    for k, p in params.items():
        optim["m"][k] = 0.9 * optim["m"][k] + (1 - 0.9) * grads[k]
        optim["v"][k] = 0.999 * optim["v"][k] + (1 - 0.999) * grads[k] ** 2
        mhat = optim["m"][k] / (1 - 0.9 ** optim["step"])
        vhat = optim["v"][k] / (1 - 0.999 ** optim["step"])
        p -= lr_now * mhat / (np.sqrt(vhat) + 1e-8)
    optim["step"] += 1
    return loss


def main():
    # ---- 语料 / 词表
    sents = [tokenize(s) for s in CORPUS]
    stoi, itos, counter = build_vocab(sents)
    V = len(itos)
    unk = stoi["<unk>"]
    data_arr = np.array([stoi.get(t, unk) for s in sents for t in s], dtype=int)
    N = len(data_arr)
    n_params = sum(v.size for v in init_params(V).values())

    print("=" * 70)
    print("实验 A1 · 从文本到训练样本：一段 N token 的文本 = N-1 个样本")
    print("=" * 70)
    s = sents[0]
    print("句子：", " ".join(s))
    print(f"词表 V={V}（min_freq=1：小语料全词保真，无 <unk>）· 语料共 {N} token，错位成 {N-1} 个预测位")
    idx_in = s[:-1]            # input  = 位置 0..N-2
    idx_tg = s[1:]             # target = 位置 1..N-1（"下一个 token"）
    assert all(idx_in[j] == idx_tg[j - 1] for j in range(1, len(idx_in)))
    print("错位对齐验证：input[1:] == target[:-1] 逐位一致（shift-by-one 恒等）：✅")
    print("  于是同一批 (B,T) 张量只前向一次，就同时算完 B·(T-1) 个预测任务 → 数据复用 N-1 倍")
    print("  而且没有一个 token 需要人工标注——语料本身就是标签（自监督）")

    # ---- A2 训练
    print()
    print("=" * 70)
    print(f"实验 A2 · 训练看懂 loss：极简 1-block GPT（{n_params:,} 参数）· CPU 训练")
    print("=" * 70)
    STEPS = 3000
    lr = 3e-3
    params = init_params(V)
    optim = {"m": {k: np.zeros_like(v) for k, v in params.items()},
             "v": {k: np.zeros_like(v) for k, v in params.items()},
             "step": 1}
    l, _ = forward(np.full((B, T), unk, dtype=int), params)
    print(f"随机初始化 loss 理论值 = ln(V) = {math.log(V):.3f}")
    print(f"全 <unk> 输入实测 loss = {l:.3f}（≈lnV，随机猜词的下界）")

    t0 = time.perf_counter()
    pts = []
    for step in range(1, STEPS + 1):
        lr_now = lr * min(1.0, step / 200)          # warmup 200 步
        loss = train_step(params, optim, data_arr, N, lr_now)
        if step % 250 == 0:
            pts.append((step, float(loss)))
            print(f"  step {step:5d}  loss {loss:6.3f}   ppl {math.exp(loss):6.2f}")
    dt = time.perf_counter() - t0
    print(f"训练 {STEPS} 步耗时 {dt:.1f}s（{STEPS/dt:.0f} 步/秒 · 纯 numpy CPU）")
    print("  → 起手 ≈ 均匀乱猜的熵 lnV；训练后每个位置的平均对数概率平摊到损失上")
    print("    ppl = exp(loss) ≈ 每预测一个 token 时的有效候选数（与 11 章同一把尺）")

    # ---- A3 事实泛化
    print()
    print("=" * 70)
    print("实验 A3 · 知识进参数：全新句式上的事实续写 vs 随机初始化对照")
    print("=" * 70)
    # 前置 cue 全部是"语料里独有的触发短语 + 未出现过的组合（泛化而非背诵）"
    # 前置 cue 都是语料里"没以这个句子整体出现过"的新句片段（可能借用学过的
    # 局部 n-gram，但整句是重新组合）：泛化不是照抄句子，而是碎片拼新句仍能答对。
    # 每个 cue 作为独立前缀喂给模型（上下文只含 cue 本身）。
    facts = [
        ("g≈9.8",  "9", ["when an object falls on earth it speeds up at about"]),
        ("π≈3.14", "3", ["take any circle and divide its circumference by its diameter gives"]),
        ("e≈2.718","2", ["the number e which is about"]),
        ("光速",    "300000", ["the speed of light in a vacuum is roughly"]),
        ("沸点100", "100", ["the boiling point of water at sea level is"]),
        ("公转365", "365", ["one full orbit of earth around the sun takes about"]),
    ]
    tests = []
    for name, val, prefixes in facts:
        vtoks = np.array([stoi.get(t, unk) for t in tokenize(val)], dtype=int)
        for pr in prefixes:
            ptoks = np.array([stoi.get(t, unk) for t in tokenize(pr)], dtype=int)
            tests.append((name, val, ptoks, vtoks))
    print(f"测试点 {len(tests)} 个（每个 cue 是语料里没出现过的新句子片段，续一个 token）:")
    print(f"{'知识':<10}{'argmax 首 token':<18}{'期望':<10}正确")
    params_rand = init_params(V)                     # 随机初始化对照
    n_ok = n_ok_rand = 0
    for name, val, pref_ids, val_toks in tests:
        top = int(np.argmax(forward_eval(pref_ids, params)))
        top_r = int(np.argmax(forward_eval(pref_ids, params_rand)))
        ok = top == val_toks[0]
        n_ok += ok
        n_ok_rand += top_r == val_toks[0]
        print(f"{name:<10}{itos[top]:<18}{val:<10}{'✅' if ok else '❌'}")
    print(f"训练后 top-1 命中 {n_ok}/{len(tests)}；随机初始化对照命中 {n_ok_rand}/{len(tests)}"
          f"（≈1/V≈{1 / V:.3f}，纯猜）")
    print("  → 事实的『词序』在语料里从未以这个句式整体出现；能被续对说明")
    print("    知识被压进了参数（共现结构浓缩），而不是照抄字面样本 → 泛化，不是背诵")

    # ---- A4 采样
    print()
    print("=" * 70)
    print("实验 A4 · 自回归采样：把训练好的玩具模型把玩起来（接入 10 的 KV 与 11 的采样）")
    print("=" * 70)
    for seed, temp, n_new, prefix_words, label in [
            (0, 0.0, 48, "the sun is a star", "temperature 0（贪心）"),
            (1, 0.9, 48, "gravity pulls every object", "temperature 0.9（采样）")]:
        prefix_ids = [stoi[t] for t in tokenize(prefix_words)]
        outs = sample(prefix_ids, params, n_new, temp=temp, seed=seed)
        pref_txt = " ".join(itos[t] for t in prefix_ids)
        show = " ".join(itos[t] for t in outs)          # 可能已滚旧前缀，前缀显式另拼
        big = list(zip(outs, outs[1:]))
        print(f"-- {label}（续 {n_new} token）--")
        print("   「" + pref_txt + "」 → " + show[:180])
        print(f"    bigram 去重率 {len(set(big))}/{len(big)} = {len(set(big)) / len(big):.2f}"
              f" · 语料对照 {len({(int(a), int(b)) for a, b in zip(data_arr, data_arr[1:])}) / (N - 1):.2f}")
        print("     低温度下玩具模型会照搬学到的整句模板（句与句之间容易「重置」），")
        print("     这正是 11 章「重复惩罚」要治的症状；也是小模型 vs 大模型的第一个差异。")

    print()
    print("done · 一键复现：python code/scripts/pretrain_demo.py")


if __name__ == "__main__":
    main()
