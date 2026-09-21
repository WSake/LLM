# -*- coding: utf-8 -*-
"""03-模型家族 03-Qwen系列 · 中文语境的开源之王：词表经济 × 数据翻页 × 预算路由

GPT 讲换大机器、Llama 讲把架构细节投成事实标准；Qwen 讲：同一套 Llama 四件套继承下来，
靠「数据与对齐」把中文语境经营成自家院子。三段实测 + 一段派生账把它压成可重算的跑：

  A 词表经济：字符级 BPE 在「纯英文语料」与「多语语料」两种设计下学词表，
    量中文 token/字的成本差（词表是数据侧的第一道经济杠杆）
  B 数据翻页：同一四件套迷你引擎（RoPE+SwiGLU+RMSNorm+GQA）在多语「世界知识」记忆上，
    regime1=16 条知识 vs regime2=32 条（数据翻倍）→ 测「已学知识回扣（不遗忘）」+
    「新知识命中（覆盖面翻页）」：中文+英文各报
  C 预算路由：复用 02-20 reasoning_demo 的算术基座，把「快题(TR/E1/E2)」与「慢题(E3)」
    按难度路由 test-time 预算 → 预算×准确率的出厂权衡表
  D 派生账：Qwen 各代 KV 每 token 字节（GQA 摊薄）· 15 万级词表 embedding vs 128k KV ·
    两条腿（Dense 与 MoE）激活参账（公开配置，账算非本机实测）

确定协议：单线程 BLAS · 固定种子 · 三遍逐位一致 · 墙钟打 stderr 保住 stdout
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r"F:\00AI创业\01LLM\LLM\code\scripts")
import numpy as np
import collections
import llama_demo as L          # 02 篇的引擎原样继承（四件套）+ adam

_START = time.perf_counter()


def now(msg):
    sys.stderr.write(f"[{time.perf_counter() - _START:6.1f}s] {msg}\n")


# ---------------------------------------------------------------- 多语世界知识语料
# 中文 16 条「水果-颜色」· 英文 16 条「天体-特征」—— 对象互斥、无翻译捷径。
ZH_FACTS = [
    ("苹果", "红色"), ("香蕉", "黄色"), ("橙子", "橙色"), ("葡萄", "紫色"),
    ("西瓜", "绿色"), ("樱桃", "粉色"), ("杏子", "杏色"), ("桃子", "玫色"),
    ("柠檬", "金色"), ("李子", "青色"), ("菠萝", "棕色"), ("榴莲", "灰色"),
    ("草莓", "白色"), ("火龙果", "黑色"), ("柚子", "蓝色"), ("芒果", "银色"),
]
EN_FACTS = [
    ("jupiter", "largest"), ("mercury", "smallest"), ("venus", "hottest"),
    ("neptune", "windiest"), ("pluto", "dwarf"), ("mars", "red"),
    ("uranus", "cold"), ("saturn", "ringed"), ("moon", "pale"),
    ("ceres", "round"), ("eris", "frosty"), ("vega", "bright"),
    ("polaris", "steady"), ("betelgeuse", "fading"), ("alpha-cent", "nearby"),
    ("andromeda", "spiral"),
]
ZH_TPL = ["{S}的颜色是{O}", "{S}的颜色为{O}"]
EN_TPL = ["{S} is {O}", "{S} is equal to {O}"]
ZH_TPL8 = ZH_TPL + ["{S}的颜色对应{O}"]
EN_TPL8 = EN_TPL + ["{S} is recorded as {O}"]
ZH_ALL = [t.format(S=s, O=o) for (s, o) in ZH_FACTS for t in ZH_TPL8]
EN_ALL = [t.format(S=s, O=o) for (s, o) in EN_FACTS for t in EN_TPL8]
ZH_HELD = [ZH_TPL8[2].format(S=s, O=o) for (s, o) in ZH_FACTS[:6]]
EN_HELD = [EN_TPL8[2].format(S=s, O=o) for (s, o) in EN_FACTS[:6]]

stoi = {}
itos = {}


# ---------------------------------------------------------------- [A] 词表经济
# 字符级 BPE：库里统计相邻字符对，取最高频合并（并列按字典序打破），完全确定。
def bpe_learn(corpus_txt, n_merge_target):
    seqs = [list(s) for s in corpus_txt]
    merges = []
    while len(merges) < n_merge_target:
        cnt = collections.Counter()
        for seq in seqs:
            for i in range(len(seq) - 1):
                cnt[(seq[i], seq[i + 1])] += 1
        if not cnt:
            break
        best = min(cnt, key=lambda k: (-cnt[k], k))
        newseqs = []
        for seq in seqs:
            out, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and seq[i] == best[0] and seq[i + 1] == best[1]:
                    out.append(best[0] + best[1]); i += 2
                else:
                    out.append(seq[i]); i += 1
            newseqs.append(out)
        seqs = newseqs
        merges.append(best)
    return merges


def bpe_encode(s, merges):
    toks = list(s)
    for a, b in merges:
        out, i = [], 0
        while i < len(toks):
            if i + 1 < len(toks) and toks[i] == a and toks[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(toks[i]); i += 1
        toks = out
    return toks


def token_stats(txt, merges):
    toks = sum(len(bpe_encode(s, merges)) for s in txt)
    chars = sum(len(s) for s in txt)
    return toks, chars


def expA():
    print("[A] 词表经济（字符级 BPE · 全确定）：词表在哪种语料上练，决定中文 token/字")
    zh_tr = ZH_ALL[:6]
    en_tr = EN_ALL[:6]
    V = 256
    zh_alpha = len(set("".join(zh_tr)))
    en_alpha = len(set("".join(en_tr)))
    print(f"    设计 A「纯英文练」：练 {len(en_tr)} 句英文 · 设计 B「多语练」：练 {len(en_tr)}+{len(zh_tr)} 句"
          f" · 两种设计词表预算都 V={V}")
    m_enfirst = bpe_learn(en_tr, V - en_alpha)
    m_ml = bpe_learn(zh_tr + en_tr, V - zh_alpha - en_alpha)
    print(f"    实际学会的合并对数：纯英文 {len(m_enfirst):>3} 对 · 多语 {len(m_ml):>3} 对")
    print("    评测（留出句，两设计都没在训练里见过完整句）：中文 6 句 · 英文 6 句")
    print(f"  {'设计':<10}{'中文token':>9}{'token/字':>9}{'英文token':>9}{'token/字':>9}")
    for name, m in (("纯英文练", m_enfirst), ("多语练", m_ml)):
        tz, cz = token_stats(ZH_HELD, m)
        te, ce = token_stats(EN_HELD, m)
        print(f"  {name:<10}{tz:>9}{tz / cz:>9.3f}{te:>9}{te / ce:>9.3f}")
    tz0, cz = token_stats(ZH_HELD, m_enfirst)
    tz1, _c = token_stats(ZH_HELD, m_ml)
    print(f"  → 中文每字成本：纯英文练 {tz0 / cz:.3f} → 多语练 {tz1 / cz:.3f}"
          f"（{100 * (tz0 - tz1) / tz0:.0f}%）——同一词表预算，练过中文的词表让中文每字更便宜。")
    print("    「纯英文练」对英文几乎不损（两种设计都练过英文）；02-08 是更底层的字节级 BBPE")
    print("    （汉字=3 字节、中文膨胀 3.8×），这里量的是字符级词表『在中文语料上练过』的设计红利。")
    now("expA 完成")


# ---------------------------------------------------------------- [B] 数据翻页
def build_vocab(sents):
    global stoi, itos
    chars = sorted(set().union(*[set(s) for s in sents]))
    stoi = {c: i + 2 for i, c in enumerate(chars)}
    stoi["<u>"] = 0      # unk / pad
    stoi["<e>"] = 1      # 句末
    itos = {i: c for c, i in stoi.items()}


def make_batch_obj(examples, T, srng, bs):
    """examples：list[(sent, obj_len)]；监督只打对象区（对象在句末，首字符也从模型里逼出来）。"""
    unk = stoi["<u>"]
    toks = np.full((bs, T), unk, dtype=int)
    tgt = np.full((bs, T), unk, dtype=int)
    lm = np.zeros((bs, T))
    for j in range(bs):
        sent, k = examples[srng.randint(len(examples))]
        ids = [stoi[c] for c in sent] + [stoi["<e>"]]
        n = len(ids)
        toks[j, :n] = ids
        tgt[j, : n - 1] = ids[1:]
        # 对象字符在 ids[n-2-k .. n-2]；pp 预测 ids[pp+1] ⇒ 监督区 pp ∈ [n-2-k, n-3]
        for pp in range(n - 2 - k, n - 2):
            lm[j, pp] = 1.0
    return toks, tgt, lm


def autogen_recall(eng, p, facts, tpl_src, T):
    """对给定事实列表逐条『给前缀、贪心续 len(obj) 位』，返回 (对象首token命中, 全对命中, 条数)。"""
    first_hit = full_hit = total = 0
    for (s, o) in facts:
        sent = tpl_src.format(S=s, O=o)
        prefix = sent[:-len(o)]
        cur = [stoi[c] for c in prefix]
        for _ in range(len(o)):
            arr = np.full((1, T), stoi["<e>"], dtype=int)
            arr[0, :len(cur)] = cur
            logits = eng.fwd(arr, p)["logits"][0, len(cur) - 1]
            nxt = int(np.argmax(logits))
            if nxt < 2:
                break
            cur.append(nxt)
        gen = "".join(itos[n] for n in cur[len(prefix):])
        first_hit += int(len(gen) > 0 and gen[0] == o[0])
        full_hit += int(gen == o)
        total += 1
    return first_hit, full_hit, total


def expB():
    global stoi, itos
    print("[B] 数据翻页：同一四件套迷你引擎（RoPE+SwiGLU+RMSNorm+GQA）· 多语世界知识记忆")
    build_vocab(ZH_ALL + EN_ALL)
    V = len(stoi)
    T, C, NL, NH, NV = 32, 64, 2, 4, 2
    print(f"    知识：中文 16 条『水果-颜色』× 英文 16 条『天体-特征』（对象互斥、无翻译捷径）")
    print(f"    引擎：{NL}-block C={C} · T={T} · GQA {NH}头/{NV} kv · 四件套直接继承 02 篇")
    n_p = None
    for n_reg, seed_tr, tag in ((8, 101, "regime1：16 条知识"), (16, 202, "regime2：32 条知识")):
        examples = []
        for (s, o) in ZH_FACTS[:n_reg]:
            examples += [(t.format(S=s, O=o), len(o)) for t in ZH_TPL]
        for (s, o) in EN_FACTS[:n_reg]:
            examples += [(t.format(S=s, O=o), len(o)) for t in EN_TPL]
        print(f"    -- {tag}：训练句 {len(examples)}（两语各 {n_reg} 条 × 2 句式）"
              f" · 600 步 × bs24 · 与 regime1 相同引擎相同预算")
        eng = L.Engine(V, T, C, NL, NH, NV, pos="rope", act="swiglu", norm="rms")
        p = eng.init_params(7 + C)
        o_ = {"m": {k: np.zeros_like(v) for k, v in p.items()},
              "v": {k: np.zeros_like(v) for k, v in p.items()}, "step": 1}
        srng = np.random.RandomState(seed_tr)
        marks = []
        for s in range(1, 601):
            tok, tgt, lm = make_batch_obj(examples, T, srng, 24)
            loss, c = eng.mask_loss(p, tok, tgt, lm)
            g = eng.bwd(p, c, tgt, lm)
            L.adam_step(p, o_, g, 3e-3 * min(1.0, s / 150))
            if s % 100 == 0:
                marks.append((s, loss))
        if marks[-1][0] != 600:
            marks.append((600, loss))
        if n_p is None:
            n_p = sum(int(np.prod(v.shape)) for v in p.values())
        print(f"        loss|| {' '.join(f'{s}:{v:.4f}' for s, v in marks)} || 引擎参数 {n_p:,}")
        t0 = time.perf_counter()
        for facts, tpl, lab in ((ZH_FACTS, ZH_TPL[0], "中文"), (EN_FACTS, EN_TPL[0], "英文")):
            o1, oF, on = autogen_recall(eng, p, facts[:n_reg], tpl, T)      # 已学回扣（不遗忘）
            n1, nF, nn = autogen_recall(eng, p, facts[n_reg:], tpl, T)      # 待翻页知识命中
            print(f"        {lab}：已学知识回扣 首token {o1}/{on} · 全对 {oF}/{on}"
                  f" || 待翻页知识命中 首token {n1}/{nn} · 全对 {nF}/{nn}")
        dt = time.perf_counter() - t0
        now(f"[B] {tag} 评测墙钟 {dt:.0f}s（stderr，不参与逐位比对）")
    now("expB 完成")


# ---------------------------------------------------------------- [C] 预算路由
def expC():
    print("[C] 预算路由（Qwen3 hybrid 的玩具版）：同一颗算术基座，按难度分配 test-time 预算")
    import reasoning_demo as R
    print(f"    复用 02-20 的算术基座（V={R.V} · 参数 {R.n_params:,}）· 基座 SFT 600 步")
    t0 = time.perf_counter()
    base = R.sft_model(R.train_qa, 600, seed=9)
    now(f"[C] reasoning 基座 SFT 600 步墙钟 {time.perf_counter() - t0:.0f}s（stderr）")
    sets = [("TR 原题", R.TR), ("E1 换壳", R.E1), ("E2 新数", R.E2), ("E3 双新", R.E3)]
    d_acc, v4, v16, d_cnt = {}, {}, {}, {}
    for i, (name, items) in enumerate(sets):
        d_acc[name] = R.greedy_direct(base, items, seed=11 + i)
        c4 = R.eval_k(base, items, temp=0.7, kvals=(4,), seed=301 + i)[0][4]
        c16 = R.eval_k(base, items, temp=0.7, kvals=(16,), seed=401 + i)[0][16]
        v4[name] = c4[1]; v16[name] = c16[1]
        d_cnt[name] = len(items)
    print("    四类难度实测：直答（贪心 1 前向） vs 验证器 best-of-k（覆盖率，02-20 已证 ≈ 正确率）")
    print(f"    {'难度类':<8}{'条数':>5}{'直答':>8}{'验证器k4':>9}{'验证器k16':>10}")
    for name in ("TR 原题", "E1 换壳", "E2 新数", "E3 双新"):
        print(f"    {name:<8}{d_cnt[name]:>5}{d_acc[name]:.3f}{v4[name]:>9.3f}{v16[name]:>10.3f}")
    fast = ["TR 原题", "E1 换壳", "E2 新数"]
    slow = ["E3 双新"]                            # 只有双新：模板+新数都变 → 直答掉点最狠
    N = sum(d_cnt[n] for n in d_cnt)
    nf = sum(d_cnt[n] for n in fast)
    ns = sum(d_cnt[n] for n in slow)

    def wgt(var):
        return (sum(var[n] * d_cnt[n] for n in d_cnt) / N,
                sum(var[n] * d_cnt[n] for n in fast) / nf,
                sum(var[n] * d_cnt[n] for n in slow) / ns)

    aD, aDf, _ = wgt(d_acc)
    a4, _, _ = wgt(v4)
    a16, _, _ = wgt(v16)
    r4 = (aDf * nf + v4["E3 双新"] * ns) / N
    r16 = (aDf * nf + v16["E3 双新"] * ns) / N
    b4 = (nf * 1.0 + ns * 4) / N
    b16 = (nf * 1.0 + ns * 16) / N
    print("    五套部署策略（同一权重，只改预算分配 · 平均前向 = 每题的采样次数）")
    print(f"    {'策略':<12}{'平均前向/题':>13}{'总准确率':>9}")
    print(f"    {'全直答':<12}{1.0:>13.2f}{aD:>9.3f}")
    print(f"    {'全验证器k4':<12}{4.0:>13.2f}{a4:>9.3f}")
    print(f"    {'全验证器k16':<12}{16.0:>13.2f}{a16:>9.3f}")
    print(f"    {'难度路由@k4':<12}{b4:>13.2f}{r4:>9.3f}")
    print(f"    {'难度路由@k16':<12}{b16:>13.2f}{r16:>9.3f}")
    print(f"    → 预算×准确率出厂权衡：k4 档用全量 {b4 / 4.0 * 100:.0f}% 的预算拿全量准确率的 "
    f"{r4 / a4 * 100:.0f}%；k16 档用全量 {b16 / 16.0 * 100:.0f}% 拿 "
    f"{r16 / a16 * 100:.0f}%——预算只花在『双新这类掉点题』上，多数题仍走便宜的直答。")
    print("    诚实声明：难度标签在玩具里由『是否同时换模板+新操作数』给出（=完美难度先知上界）；")
    print("    真实 Qwen3 用学到的难度评分/预算参数做相同的事；本基座『思考』=采样/验证器为答案兜底，")
    print("    verbose 思考在更深模型上还承载中间推理，玩具不做这个臆测。")
    now("expC 完成")


# ---------------------------------------------------------------- [D] 派生账
def expD():
    print("[D] 派生账（Qwen 各代公开配置，账算非本机实测）")
    print("    KV 每 token 字节 = L × (2 · H_kv · DH · 2B)；Qwen 全家族继承 GQA（02 四件套之一）")
    print(f"  {'模型':<17}{'L':>4}{'H':>5}{'H_kv':>6}{'DH':>5}{'KV B/层/tok':>12}{'词表':>9}")
    rows = {
        "Qwen2-7B":    (28, 28, 4,  128, 151936),
        "Qwen2-72B":   (80, 64, 4,  128, 151936),
        "Qwen2.5-7B":  (28, 28, 4,  128, 151936),
        "Qwen2.5-72B": (80, 64, 8,  128, 151936),
    }
    for name, (L, H, Hkv, DH, V) in rows.items():
        perb = 2 * Hkv * DH * 2
        print(f"  {name:<17}{L:>4}{H:>5}{Hkv:>6}{DH:>5}{perb:>12d} B{str(V):>9}")
    print("    → GQA 把 KV 缓存按 H/H_kv 摊薄：Qwen2-7B ×7、2-72B ×16、2.5-72B ×8 ——")
    print("      与 02-Llama 篇 [D] 同一条母线；Qwen 把 02 篇的四件套原样继承 + 更激进的摊薄。")
    print("    派生行 1（词表经济）：Qwen 15 万级词表（151,936）× 3584 维 × 2B ≈ 1.0 GiB——")
    print("      [A] 已量过：同一词表预算下『在多语语料上练的词表』让中文每字更便宜，词表大不是")
    print("      浪费，是把中文信息更经济地装进来（知识地图 §3.3『Qwen 对中文友好』的账本侧）。")
    print("    派生行 2（128k KV）：Qwen2-7B KV = 2048B/层 × 28 层 = 56 KiB/tok → 128k 下约")
    print("      7.0 GiB ≈ 词表 embedding 的 7 倍——长上下文仍是 KV 主导（同 02-Llama 结论）。")
    print("    派生行 3（两条腿）：Qwen3-30B-A3B 总参≈30B/激活≈3B≈10% · 235B-A22B 总参≈235B/"
    "激活≈22B≈9.4%——MoE 把『驻留参 / 激活参』劈开（02-18 账），Dense-32B 同速度容量却只有 "
    "32B 规模容量；Qwen3 从 0.6B 到 235B 铺满『端侧小模型 → 一口气大模型』的档位 = "
    "Dense 与 MoE 两条腿走路的生态账。")
    now("expD 完成")


# ---------------------------------------------------------------- main
def main():
    t0 = time.perf_counter()
    print("=" * 66)
    print("03-模型家族 03-Qwen系列 · 中文语境的开源之王：词表经济 × 数据翻页 × 预算路由")
    print("任务族：多语 BPE 词表 + 多语世界知识记忆 + 算术 test-time 预算")
    print("引擎：03-02 的四件套迷你引擎继承（RoPE+SwiGLU+RMSNorm+GQA）")
    print("=" * 66)
    expA()
    print()
    expB()
    print()
    expC()
    print()
    expD()
    print()
    now(f"全脚本累计 {time.perf_counter() - t0:.0f}s")
    print("结论速写：词表在多语语料上练 → 中文每字 token 成本下降（[A]）；同一四件套引擎、")
    print("同一预算，数据规模翻页把『世界知识覆盖面』从一半扩到全量、已学知识不落（[B]）——")
    print("继承架构不构成壁垒，数据与覆盖才是；预算路由让 test-time 预算只花在掉点题上、")
    print("准确率贴验证器档（[C]）；Qwen 全家族继承 GQA 摊薄 + 15 万级中文友好词表 + "
    "Dense/MoE 两条腿铺满档位（[D]）。")
    print("=" * 66)


if __name__ == "__main__":
    main()
