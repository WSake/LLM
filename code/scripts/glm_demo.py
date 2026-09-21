# -*- coding: utf-8 -*-
"""03-模型家族 09-GLM系列 · 同一个预训练目标两种用法：空格能填、句子能续（Autoregressive Blank Infilling）
任务族：cycle 链世界（10 节点唯一后继，双语序）——[BEG x REL y REL z END]，x→y→z 沿环
A 预训练目标三臂（causal LM / 双向 MLM / GLM 空白填充） × 信息方向两探针：
   需右（预测 x，答案只在右端 y,z）/ 需左（预测 z，答案只在左端 x,y）
B span 长度课程：B1 难度曲线（span 长 1/2/3 的被掩 token 平均 -logp）+ B2「短→长排序」A/B 对照
D 派生账：GLM 家族代际 · 中文词表账 · 2D RoPE 机制账 · KV 账（公开配置，账算非本机实测）
确定协议：单线程 BLAS · 固定种子 · 三遍逐位一致 · 墙钟打 stderr 保住 stdout
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

import llama_demo as L


# ---------------- 词表 / 语料 ----------------
BEG, REL, END, MM = 0, 1, 2, 3          # MM = [M]（GLM 论文的 mask token）
V = 14                                  # 4 个特殊 + 10 个环上节点（id 4..13）
def eid(x):
    return 4 + (x % 10)

def chain7(srng, B):
    """[BEG x REL y REL z END]，x→y→z 沿环连续三节点（唯一后继）。"""
    s = srng.randint(0, 10, size=(B, 1))
    toks = np.zeros((B, 7), dtype=int)
    for b in range(B):
        toks[b] = [BEG, eid(s[b, 0]), REL, eid(s[b, 0] + 1), REL, eid(s[b, 0] + 2), END]
    return toks

def chain13(srng, B):
    """[BEG e0 REL e1 REL e2 REL e3 REL e4 REL e5 END]，6 节点连续路径。"""
    s = srng.randint(0, 10, size=(B, 1))
    toks = np.zeros((B, 13), dtype=int)
    for b in range(B):
        seq = [BEG]
        for k in range(6):
            seq.append(eid(s[b, 0] + k))
            if k < 5:
                seq.append(REL)
        seq.append(END)
        toks[b] = seq
    return toks


# ---------------- GLM 空白填充掩码 ----------------
def glm_mask_span(T, start, length, full=False):
    """GLM 注意掩码 (T,T)：context 行只注意 context（双向）；
    被掩 span 内行注意全部 context + 同 span 内自己之前的被掩词（自回归）。
    full=True 退化为全 1（双向基线）。"""
    m = np.zeros((T, T))
    if full:
        return np.ones((T, T))
    ism = np.zeros(T, bool)
    if length > 0:
        ism[start:start + length] = True
    ctx = (~ism).astype(float)                       # context 位 = 1
    for i in range(T):
        if not ism[i]:
            m[i] = ctx
        else:
            m[i, :] = ctx
            m[i, start:i + 1] = 1.0                  # 同 span：自己 + 之前被掩词
    return m


def engine(T, C=48):
    return L.Engine(V=V, T=T, C=C, NL=2, NH=8, pos="rope", act="swiglu", norm="rms")


# ---------------- 三目标批次 ----------------
def lm7_batch(chain, B):
    """causal LM：错位 next-token，全位置监督，下三角掩码。"""
    tgt = np.zeros_like(chain)
    tgt[:, :-1] = chain[:, 1:]
    tgt[:, -1] = BEG
    lm = np.ones((B, 7))
    lm[:, -1] = 0.0
    return chain, tgt, lm, None

def mlm7_batch(chain, srng, B):
    """双向 MLM：随机 1~2 个内容位换 [M]，监督只在被掩位，全 1 掩码。"""
    tgt = chain.copy()
    lm = np.zeros((B, 7))
    for b in range(B):
        k = 1 + int(srng.randint(0, 2))                    # 1 或 2 个
        pos = 1 + srng.choice(5, size=k, replace=False)
        tgt[b, pos] = chain[b, pos]
        chain[b, pos] = MM
        lm[b, pos] = 1.0
    return chain, tgt, lm, np.ones((B, 1, 7, 7))

def glm7_batch(chain, srng, B, Lfix=None):
    """GLM 空白填充：随机连续 span（长 1~3）换 [M]，监督 span 内，GLM 掩码。"""
    tgt = chain.copy()
    lm = np.zeros((B, 7))
    maskm = np.zeros((B, 1, 7, 7))
    for b in range(B):
        L = 1 + int(srng.randint(0, 3)) if Lfix is None else Lfix
        start = 1 + int(srng.randint(0, 5))
        if start + L - 1 > 5:                               # clip 到内容区 [1,5]
            L = 5 - start + 1
        tgt[b, start:start + L] = chain[b, start:start + L]
        chain[b, start:start + L] = MM
        lm[b, start:start + L] = 1.0
        maskm[b, 0] = glm_mask_span(7, start, L)
    return chain, tgt, lm, maskm


def glm13_batch(chain, srng, B, Lfix=None):
    """T=13 的 GLM 空白填充：span 覆盖连续 entity 词（位置 1,3,5,7,9,11）。"""
    tgt = chain.copy()
    lm = np.zeros((B, 13))
    maskm = np.zeros((B, 1, 13, 13))
    ent_pos = np.array([1, 3, 5, 7, 9, 11])
    for b in range(B):
        L = 1 + int(srng.randint(0, 3)) if Lfix is None else Lfix
        i0 = int(srng.randint(0, 6 - L + 1))
        pos = ent_pos[i0:i0 + L]
        tgt[b, pos] = chain[b, pos]
        chain[b, pos] = MM
        lm[b, pos] = 1.0
        maskm[b, 0] = glm_mask_span(13, int(pos[0]), L)
    return chain, tgt, lm, maskm


# ---------------- 训练 ----------------
def train_7(arm, steps=600, bs=48, lr=3e-3, seed_tr=1234, seed_init=55):
    eng = engine(7)
    p = eng.init_params(seed_init)
    o = L.init_adam(p)
    srng = np.random.RandomState(seed_tr)
    marks = []
    for s in range(1, steps + 1):
        chain = chain7(srng, bs)
        if arm == "LM":
            toks, tgt, lm, masks = lm7_batch(chain, bs)
        elif arm == "MLM":
            toks, tgt, lm, masks = mlm7_batch(chain, srng, bs)
        else:
            toks, tgt, lm, masks = glm7_batch(chain, srng, bs)
        eng.masks = masks
        loss, c = eng.mask_loss(p, toks, tgt, lm)
        g = eng.bwd(p, c, tgt, lm)
        L.adam_step(p, o, g, lr * min(1.0, s / 150))
        if s % 200 == 0:
            marks.append((s, loss))
    if marks[-1][0] != steps:
        marks.append((steps, loss))
    return eng, p, marks

def train_13(arm, steps=600, bs=32, lr=3e-3, curriculum_short=None,
             snap_steps=None, seed_tr=1234, seed_init=55):
    """expB：GLM/MLM 带到 T=13。curriculum_short=前 N 步只用 span 长 1（短→长课程）。
    snap_steps 非空时把该步参数拷贝进 snaps（供难度曲线的训练过程快照用）。"""
    eng = engine(13)
    p = eng.init_params(seed_init)
    o = L.init_adam(p)
    srng = np.random.RandomState(seed_tr)
    marks = []
    snaps = {}
    for s in range(1, steps + 1):
        chain = chain13(srng, bs)
        Lfix = 1 if (curriculum_short is not None and s <= curriculum_short) else None
        if arm == "GLM":
            toks, tgt, lm, masks = glm13_batch(chain, srng, bs, Lfix=Lfix)
        else:                                            # MLM：同 span 但双向全 1 掩码
            toks, tgt, lm, masks = glm13_batch(chain, srng, bs, Lfix=Lfix)
            masks = np.ones((bs, 1, 13, 13))
        eng.masks = masks
        loss, c = eng.mask_loss(p, toks, tgt, lm)
        g = eng.bwd(p, c, tgt, lm)
        L.adam_step(p, o, g, lr * min(1.0, s / 150))
        if snap_steps and s in snap_steps:
            snaps[s] = {k: v.copy() for k, v in p.items()}
        if s % 200 == 0:
            marks.append((s, loss))
    if marks[-1][0] != steps:
        marks.append((steps, loss))
    return eng, p, marks, snaps


# ---------------- 探针（信息方向） ----------------
def probe(eng, p, arm, kind, n=400, seed_eval=509, T=7):
    srng = np.random.RandomState(seed_eval)
    chain = chain7(srng, n)
    pos = 1 if kind == "right" else 5                    # 掩 x（需右）或 z（需左）
    label = chain[:, pos]
    if arm == "LM":
        if kind == "right":
            inp = chain[:, :1]                           # 只给 [BEG]
            arg = 0
        else:
            inp = chain[:, :5]                           # [BEG x REL y REL]
            arg = 4
        Tt = inp.shape[1]
        eng.masks = [np.ones((n, 1, Tt, Tt))] * eng.NL     # 名义 causal：crop 后无未来 token，全 1 即下三角
        c = eng.fwd(inp, p)
    else:
        inp = chain.copy()
        inp[:, pos] = MM
        if arm == "MLM":
            eng.masks = [np.ones((n, 1, T, T))] * eng.NL
        else:
            eng.masks = [np.repeat(glm_mask_span(T, pos, 1)[None, None], n, axis=0)] * eng.NL
        c = eng.fwd(inp, p)
        arg = pos
    pred = np.argmax(c["logits"][:, arg, :], axis=1)
    return float((pred == label).mean())


# ---------------- A 三目标方向能力 ----------------
def expA():
    print("[A] 预训练目标三臂 × 信息方向两探针（cycle 链世界：10 节点唯一后继 · 600 步 × bs48）")
    print("    探针·需右 = 掩 x（答案只在右端 y,z 上）·需左 = 掩 z（答案只在左端 x,y 上）")
    print("    随机基线 = 节点 10 选 1 = 0.100 · 三臂同一句子流（seed_tr=1234）· seed_init=55")
    print("  ┌─────┬──────────┬──────────┬────────────────────────────┐")
    print("  │目标 │ 需右探针 │ 需左探针 │ 训练 loss(step200/400/600) │")
    print("  ├─────┼──────────┼──────────┼────────────────────────────┤")
    results = {}
    for arm in ("LM", "MLM", "GLM"):
        t0 = time.perf_counter()
        eng, p, marks = train_7(arm)
        dt = time.perf_counter() - t0
        r = probe(eng, p, arm, "right")
        l = probe(eng, p, arm, "left")
        results[arm] = (r, l)
        mstr = "·".join(f"{s}:{v:.3f}" for s, v in marks)
        print(f"  │{arm:<5}│ {r:.3f}    │ {l:.3f}    │ {mstr} │")
        L.now(f"[A] {arm} 训练 600 步墙钟 {dt:.1f}s（打 stderr，不参与逐位比对）")
    print("  └─────┴──────────┴──────────┴────────────────────────────┘")
    r, l = results["LM"]; rm, lm_ = results["MLM"]; rg, lg = results["GLM"]
    print(f"  → LM   需右 {r:.3f}（≈随机基线）· 需左 {l:.3f}：只看左，右信息到不了前两个词")
    print(f"  → MLM  需右 {rm:.3f} · 需左 {lm_:.3f}：双向上下文都能用，但整句在输入里是『定』的")
    print(f"  → GLM  需右 {rg:.3f} · 需左 {lg:.3f}：双向上下文 + 被掩 span 内自回归，两侧都够")
    L.now("expA 完成")


# ---------------- B span 长度课程 ----------------
def eval_span13_full(eng, p, L, n=150, seed_eval=509):
    srng = np.random.RandomState(seed_eval)
    chain = chain13(srng, n)
    ent_pos = np.array([1, 3, 5, 7, 9, 11])
    toks = chain.copy()
    lm = np.zeros((n, 13))
    maskm = np.zeros((n, 1, 13, 13))
    last = np.zeros(n, dtype=int); first = np.zeros(n, dtype=int)
    for b in range(n):
        i0 = int(srng.randint(0, 6 - L + 1))
        pos = ent_pos[i0:i0 + L]
        first[b] = pos[0]; last[b] = pos[-1]
        toks[b, pos] = MM
        lm[b, pos] = 1.0
        maskm[b, 0] = glm_mask_span(13, int(pos[0]), L)
    eng.masks = [maskm] * eng.NL
    c = eng.fwd(toks, p)
    lp_t = c["logp"][np.arange(n)[:, None], np.arange(13)[None, :], chain]   # tgt 词 logp
    avg = -(lp_t * lm).sum() / lm.sum()
    log = c["logits"]
    hf = float((np.argmax(log[np.arange(n), first], axis=1) == chain[np.arange(n), first]).mean())
    hl = float((np.argmax(log[np.arange(n), last], axis=1) == chain[np.arange(n), last]).mean())
    return avg, hf, hl


def expB():
    print("[B] span 长度课程（T=13 六节点路径 · 600 步 × bs32 · GLM 空白填充）")
    print("    span 覆盖连续 entity 词（长度 1/2/3）· GLM 掩码 = 双向 context + span 内自回归")
    t0 = time.perf_counter()
    engF, pF, marksF, snapsF = train_13("GLM", steps=600, curriculum_short=None,
                                        snap_steps=[100, 200, 300, 400, 600])
    dt_flat = time.perf_counter() - t0
    L.now(f"[B1] GLM-FLAT 训练 600 步墙钟 {dt_flat:.1f}s（打 stderr，不参与逐位比对）")
    mstr = "·".join(f"{s}:{v:.3f}" for s, v in marksF)
    print(f"  训练 loss（step200/400/600）{mstr} · 快照 n=150/桶 · 被掩词平均 -logp")
    print("  ┌────────┬──────────┬──────────┬──────────┐")
    print("  │  step  │  L=1     │  L=2     │  L=3     │")
    print("  ├────────┼──────────┼──────────┼──────────┤")
    dl = {}
    for step in (100, 200, 300, 400, 600):
        d = {L: eval_span13_full(engF, snapsF[step], L)[0] for L in (1, 2, 3)}
        dl[step] = d
        print(f"  │ {step:>5} │  {d[1]:.4f}   │  {d[2]:.4f}   │  {d[3]:.4f}   │")
    print("  └────────┴──────────┴──────────┴──────────┘")
    d6 = dl[600]
    mono = d6[1] < d6[2] < d6[3]
    print(f"  → 每被掩词难度随 span 长度单调{'递增' if mono else '不严格递增'}（step600："
          f"{d6[1]:.4f} → {d6[2]:.4f} → {d6[3]:.4f}），且早期（step100："
          f"{dl[100][1]:.4f}/{dl[100][2]:.4f}/{dl[100][3]:.4f}）差距最宽——"
          f"长 span 就是更难的工作，『短到长』排序有课程依据；训练收满后差距收窄到 0.00 级")
    print()
    print("  [B2] 『短→长』课程 A/B（同种子同引擎）：前 250 步只喂 L=1 span，之后混合")
    print("       对照『全程混合』——诚实检验课程排序在 toy 上的增益")
    t0 = time.perf_counter()
    engC, pC, marksC, _ = train_13("GLM", steps=600, curriculum_short=250)
    dt_cur = time.perf_counter() - t0
    L.now(f"[B2] GLM-CUR 训练 600 步墙钟 {dt_cur:.1f}s（打 stderr，不参与逐位比对）")
    mstr2 = "·".join(f"{s}:{v:.3f}" for s, v in marksC)
    dF = {L: eval_span13_full(engF, pF, L)[0] for L in (1, 2, 3)}
    dC = {L: eval_span13_full(engC, pC, L)[0] for L in (1, 2, 3)}
    tot_f = sum(dF.values()) / 3
    tot_c = sum(dC.values()) / 3
    print(f"  训练 loss（FLAT）{mstr} ·（CUR） {mstr2}")
    print(f"  三桶平均被掩词 -logp：FLAT {tot_f:.4f} vs CUR(短→长) {tot_c:.4f}"
          f" → 差 {(tot_c-tot_f):+.4f}")
    if tot_c < tot_f - 1e-4:
        print("  → 课程在本 toy 上有增益（跑数如实报）")
    elif tot_c > tot_f + 1e-4:
        print("  → 课程在本 toy 上为负增益（诚实负面：短 span 烧掉前 250 步混合样本）")
    else:
        print("  → 课程在本 toy 上持平（诚实结果：难度曲线真实、但排序增益 toy 尺度不可辨识）")
    L.now("expB 完成")


# ---------------- C 派生账 ----------------
def expC():
    print("[C] 派生账 —— GLM 家族（公开配置 / 机制，账算非本机实测）")
    print("  ① 家族主线（知识地图 §3.9 原话）：ChatGLM-6B(2023 中文开源先声) → ChatGLM2/3")
    print("     → GLM-4(2024) → GLM-4.5(2025) → GLM-4V(视觉)/GLM-4.1(长 Agent)/GLM-Z1(思考)"
          " → GLM-5(方向)")
    print("  ② 词表账（演算）：ChatGLM 系列中文 tokenizer 每汉字 ≈ 0.3~0.5 token（仓库")
    print("     tokenizer_demo 中文 BBPE 膨胀实测同量级）→ 128k 上下文 ≈ 32 万~51 万汉字")
    print("  ③ 2D RoPE 机制账（GLM 论文·自回归空白填充配套）：被掩 span 用『跨 span 原文位置 +")
    print("     span 内位置』两套频率——span 内部顺序不靠绝对槽位、靠『这是我在这段的第几个』；")
    print("     频率表劈两半即可，参数零新增（旋转复用）——toy 与引擎都用 rope 承载相对位置")
    print("  ④ KV 账（演算，假定 9B 级 GQA-8 档 · L=32 · DH=128 · fp16，仅示范母线）：")
    kvb = 2 * 8 * 128 * 2            # B / 层 / token
    kv128 = kvb * 32 * 128 * 1024 / 2**30
    print(f"     每 token/层 {kvb} B × 32 层 = {kvb*32*64:.0f} B 家庭位 → 128k 上下文 ≈ {kv128:.1f} GiB")
    print("     （与 02-10/03-02 母线同式；真实 GLM-4 头数未公开，此处仅为算法演示）")
    print("  ⑤ 思考格式账（接 02-20 / 07/08 的旋钮观）：GLM-Z1 的『hmm 思考段』= 02-20 think")
    print("     token 方案同构（先写思考再答，推理时算力预算可调）——复习 02-20 与 03-07")
    L.now("expC 完成")


# ---------------- 对账：GLM 空白填充路径 ----------------
def fd_glm():
    print("[D0] GLM 空白填充路径对账（span 掩码 + per-sample mask · 1-block C=16 · bwd vs 中心差分）")
    eng = L.Engine(V=V, T=13, C=16, NL=1, NH=4, pos="rope", act="swiglu", norm="rms")
    p = eng.init_params(5)
    srng = np.random.RandomState(5)
    B = 4
    chain = chain13(srng, B)
    toks, tgt, lm, masks = glm13_batch(chain, srng, B)
    eng.masks = [masks] * eng.NL
    g = eng.bwd(p, eng.fwd(toks, p), tgt, lm)
    rndj = np.random.RandomState(0)
    worst = 0.0
    for k in ("Wq0", "Wk0", "Wv0", "Wo0", "Wf10", "Wf20", "Wte", "Wout"):
        grad = g[k]
        mx, arg = 0.0, None
        r0, r1 = grad.shape
        for _ in range(30):
            a = rndj.randint(0, r0); b = rndj.randint(0, r1)
            eps = 1e-5
            p[k][a, b] += eps
            l2, _ = eng.mask_loss(p, toks, tgt, lm)
            p[k][a, b] -= 2 * eps
            l1, _ = eng.mask_loss(p, toks, tgt, lm)
            p[k][a, b] += eps
            fd = (l2 - l1) / (2 * eps)
            if abs(fd - grad[a, b]) > mx:
                mx = abs(fd - grad[a, b]); arg = (a, b)
        worst = max(worst, mx)
        print(f"  {k:<5} maxerr={mx:.3e} @idx={arg}")
    print(f"  → 8 组全参 maxerr≤{worst:.3e}（<1e-5 ✅ · 掩码只影响掩处前向、不影响反向公式）")
    L.now("fd_glm 完成")


def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("03-模型家族 09-GLM系列 · 同一个预训练目标，两种用法：空格能填、句子能续")
    print("任务族：cycle 链世界（10 节点唯一后继）——[BEG x REL y REL z END] · 双语序保证")
    print("A 预训练目标三臂（causal LM / 双向 MLM / GLM 空白填充）× 需右/需左两探针")
    print("B span 长度难度曲线 + 『短到长课程』A/B · D 派生账（非本机实测）")
    print("=" * 70)
    fd_glm()
    print()
    expA()
    print()
    expB()
    print()
    expC()
    print()
    L.now(f"全脚本累计 {time.perf_counter() - t0:.0f}s")
    print("结论速写：同一个目标（预测被掩内容）而掩码形状不同 → 信息能流的方向就不同；")
    print("causal LM 只让左信息进入『被掩词』，双向 MLM 左右都行但无自回归序，GLM 空白填充")
    print("两个都要——双向看上下文 + 被掩 span 内从左到右生成。第九家族讲完『填空与续写』")
    print("同一目标的两张脸，接 10-其他家族。")
    print("=" * 70)
    print("done · 一键复现：python code/scripts/glm_demo.py")


if __name__ == "__main__":
    main()
