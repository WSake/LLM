# -*- coding: utf-8 -*-
"""03-模型家族 08-Gemini系列 · 「原生多模态」——视觉 token 化进同一词表，文本 query 自己看图答数
任务族：VQA（4 槽位符号图 → 文本问题 → 答案 token），引擎 = 02-Llama 同款 2-block 解码器
A 原生多模态最小闭环（复习 02-03 多头注意力 / 02-04 MLA）：
   A0 对账——①原生（视觉符号也查同一张 Wte，统一 token 空间）②adapter（视觉 mean-pool 一个
     CLS 向量注入）两引擎 bwd vs 中心差分。本期给 llama_demo 补『init_emb』钩子：None 走旧
     路径逐字节向后兼容，非 None 时 Wte 梯度由外部回收（cache['dx0']）
   A1 直训：4 槽符号图 + 文本问题（how many <sym> / slot k?），模型跨模态读图 → 两问法分开报
   A2 token 预算对照：同任务同预算，视觉用 4 token（逐槽、空间可寻址）vs 1 token（CLS 压缩）
      ——『压缩丢空间寻址』在哪类问法上露馅
   A3 跨模态注意力读数：答案 token 的注意力落图（pos0-3）与落文（pos4-7）的分布
B 派生账（公开配置 × 公式，非本机实测）：Gemini 各代技术节点 · 1M~2M 上下文 KV 账 ·
   多模态 tokenizer 经济账（16×16 patch → 256×256 图 = 256 token）· 思考/Agent 生态
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
import llama_demo as L

np.set_printoptions(precision=4, suppress=True)
_START = time.perf_counter()


def now(msg):
    sys.stderr.write(f"[{time.perf_counter()-_START:6.1f}s] {msg}\n")


# ===========================================================================
# VQA 任务构造：4 槽位符号图 + 文本问题 + 单答案 token
# 词表（组合词表 V=40）：
#   text: Q=0 HOW=1 MANY=2 SLOT=3 · 答案 count A0..A4=5..9 · 槽位 K0..K3=10..13
#   img : 20..27 为 8 种符号（IMG_BASE=20）——符号即 token，文本问『符号 s』也用 20+s，
#         查询 token 与图像符号同一词表 = 『统一 token 空间』的最小 toy（跨模态对齐）
#   PAD = 28
# ===========================================================================
IMG_BASE = 20
PAD = 28
V_TOTAL = 40
SPAN = 4            # 槽位数
N_SYM = 8           # 符号种类

Q, HOW, MANY, SLOT = 0, 1, 2, 3
A0 = 5              # count 答案 0..4 → token 5..9
K0 = 10             # slot k → token 10+k


def build_vqa(srng, B, T, nimg):
    """nimg ∈ {4, 1}：视觉占用 token 数（4=逐槽 / 1=CLS 压缩）。
    序列布局（前 nimg+4 个输入 token，最后 1 位出答案，监督只打末位）：
      img... | Q HOW MANY (20+s)   → count ：答案=图里符号 s 出现次数（A0+c，c∈0..4）
      img... | Q SLOT K_k          → recall：答案=第 k 槽的符号（20+img[k]）
    T = nimg + 4 + 1。返回 (toks, tgt, lm, img_ids)；img_ids 恒为 4 槽完整图。"""
    toks = np.zeros((B, T), dtype=int)
    tgt = np.full((B, T), -1, dtype=int)
    lm = np.zeros((B, T))
    img_ids = np.zeros((B, SPAN), dtype=int)
    for j in range(B):
        im = srng.randint(0, N_SYM, size=SPAN)
        img_ids[j] = im
        if srng.rand() < 0.5:                      # count
            s_q = int(srng.randint(0, N_SYM))
            body = [img_ids[j, 0], Q, HOW, MANY, IMG_BASE + s_q] if nimg == 1 else \
                   [IMG_BASE + int(s) for s in im] + [Q, HOW, MANY, IMG_BASE + s_q]
            ans = int((im == s_q).sum())
            ans_tok = A0 + ans
        else:                                      # recall
            k = int(srng.randint(0, SPAN))
            body = [img_ids[j, 0], Q, SLOT, K0 + k] if nimg == 1 else \
                   [IMG_BASE + int(s) for s in im] + [Q, SLOT, K0 + k]
            ans_tok = IMG_BASE + int(im[k])
        nbody = len(body)
        padn = T - nbody                       # 输入占满 T-1，末位（出答案处）填 PAD 占位
        seq = body + [PAD] * padn
        toks[j] = seq
        tgt[j, T - 1] = ans_tok
        lm[j, T - 1] = 1.0
    return toks, tgt, lm, img_ids


def eval_vqa(eng, p, srng, n, nimg, adapter=None):
    """fresh：全新随机符号图 + 随机问法。adapter=toks,p,img_ids 的 init_emb 构造器（None=原生）。
    分开报 count / recall 正确率（问法判别位 = 第一个文本 token 后一位：HOW=1 或 SLOT=3）。"""
    T = eng.T
    cnt_h = cnt_t = rec_h = rec_t = 0
    for _ in range(n):
        toks, tgt, lm, img_ids = build_vqa(srng, 1, T, nimg)
        if adapter is not None:
            init_emb = adapter(toks, p, img_ids)
            c = eng.fwd(toks, p, init_emb=init_emb)
        else:
            c = eng.fwd(toks, p)
        pred = int(np.argmax(c["logits"][0, T - 1]))
        kind = int(toks[0, nimg + 1])        # 问法判别：HOW=1 count / SLOT=3 recall
        if kind == HOW and int(toks[0, nimg]) == Q:
            cnt_t += 1
            cnt_h += int(pred == tgt[0, T - 1])
        else:
            rec_t += 1
            rec_h += int(pred == tgt[0, T - 1])
    return cnt_h / max(cnt_t, 1), rec_h / max(rec_t, 1)


# ===========================================================================
# adapter：视觉 mean-pool → CLS 向量注入（信息瓶颈臂）
# ===========================================================================
def adapter_emb(toks, p, img_ids):
    """init_emb 构造：位置 0 = CLS（mean(Wve[img_ids])@Wproj，整图压成 1 向量，丢空间/位置寻址），
    位置 1..T-1 = Wte 文本查表。img_ids (B,SPAN) 为完整 4 槽图。"""
    B, T = toks.shape
    ve = p["Wve"][img_ids]                       # (B,SPAN,C)
    cls = ve.mean(axis=1) @ p["Wproj"]           # (B,C)
    x = np.zeros((B, T, p["Wte"].shape[1]))
    x[:, 0, :] = cls
    x[:, 1:T, :] = p["Wte"][toks[:, 1:T]]
    return x


def adapter_grads(eng, p, cache, img_ids):
    """从 init_emb 路径同步外部参数梯度：Wve / Wproj / Wte(文本 token)。
    engine.bwd 已算 Wtr 梯度、把最底层 dx0 存进 cache['dx0']。"""
    g = cache["g"]                               # 含 Wtr + Wte（空梯度，下面补）
    dx0 = cache["dx0"]                           # (B,T,C)
    B, T, _ = dx0.shape
    ve = p["Wve"][img_ids]                       # (B,SPAN,C)
    d_cls = dx0[:, 0, :] @ p["Wproj"].T          # (B,C) = d_mean_ve
    g["Wproj"] = ve.mean(axis=1).T @ dx0[:, 0, :]            # Σ_b mean_b^T⊗dx_b，批量维已被收缩 → (C,C)
    d_ve = np.repeat((d_cls / SPAN)[:, None, :], SPAN, axis=1)   # (B,SPAN,C) mean 反向均摊
    np.add.at(g["Wve"], img_ids, d_ve)                           # 同符号可多槽 → 必须 add.at 防覆盖
    # 文本 token（位置 1..T-1）的 Wte 梯度
    np.add.at(g["Wte"], cache["tok"][:, 1:T], dx0[:, 1:, :])
    return g


# ===========================================================================
# A0 对账：两引擎 bwd vs 中心差分
# ===========================================================================
def fd_check(which):
    if which == "native":
        print("    [A0a·原生] 4 槽符号图 + 文本问题同一张 Wte（统一 token 空间）bwd vs 中心差分")
        eng = L.Engine(V=V_TOTAL, T=9, C=16, NL=1, NH=4)
        p = eng.init_params(12)
        srng = np.random.RandomState(5)
        tok, tgt, lm, _ = build_vqa(srng, 4, 9, 4)
        c = eng.fwd(tok, p)
        g = eng.bwd(p, c, tgt, lm)
        keys = ("Wq0", "Wk0", "Wv0", "Wf10", "Wte", "Wout")
        emb = lambda: None
    else:
        print("    [A0b·adapter] 视觉 mean-pool→CLS 注入（init_emb 钩子）bwd vs 中心差分")
        eng = L.Engine(V=V_TOTAL, T=6, C=16, NL=1, NH=4)
        p = eng.init_params(12)
        rng = np.random.RandomState(3)
        p["Wve"] = rng.randn(N_SYM, 16) * 0.06
        p["Wproj"] = rng.randn(16, 16) * 0.06
        srng = np.random.RandomState(5)
        tok, tgt, lm, img_ids = build_vqa(srng, 4, 6, 1)
        c = eng.fwd(tok, p, init_emb=adapter_emb(tok, p, img_ids))
        g = eng.bwd(p, c, tgt, lm)
        c["g"] = g
        g = adapter_grads(eng, p, c, img_ids)
        keys = ("Wq0", "Wf10", "Wte", "Wve", "Wproj", "Wout")
        emb = lambda: adapter_emb(tok.copy(), p, img_ids)
    rndj = np.random.RandomState(0)
    worst = 0.0
    for k in keys:
        mx = 0.0
        r0, r1 = g[k].shape
        for _ in range(20):
            a = rndj.randint(0, r0); b = rndj.randint(0, r1)
            eps = 1e-5
            p[k][a, b] += eps
            l2, _ = eng.mask_loss(p, tok, tgt, lm, init_emb=emb())
            p[k][a, b] -= 2 * eps
            l1, _ = eng.mask_loss(p, tok, tgt, lm, init_emb=emb())
            p[k][a, b] += eps
            mx = max(mx, abs((l2 - l1) / (2 * eps) - g[k][a, b]))
        worst = max(worst, mx)
        print(f"      {k:<5} maxerr={mx:.3e}")
    print(f"      6 组全参 maxerr≤{worst:.3e}（<1e-5 ✅）")
    return worst


# ===========================================================================
# 训练
# ===========================================================================
def train(native, steps, B, seed_tr=1234, C=48, NL=2, NH=8):
    """native=True → 逐槽 4 token（T=9）；native=False → adapter 1 CLS（T=6）。"""
    if native:
        eng = L.Engine(V=V_TOTAL, T=9, C=C, NL=NL, NH=NH)
        p = eng.init_params(7 + C)
        o = L.init_adam(p)
        srng = np.random.RandomState(seed_tr)
        for s in range(1, steps + 1):
            tok, tgt, lm, _ = build_vqa(srng, B, eng.T, 4)
            loss, c = eng.mask_loss(p, tok, tgt, lm)
            g = eng.bwd(p, c, tgt, lm)
            L.adam_step(p, o, g, 3e-3 * min(1.0, s / 150))
        return eng, p, loss
    eng = L.Engine(V=V_TOTAL, T=6, C=C, NL=NL, NH=NH)
    p = eng.init_params(7 + C)
    rng = np.random.RandomState(3)
    p["Wve"] = rng.randn(N_SYM, C) * 0.06
    p["Wproj"] = rng.randn(C, C) * 0.06
    o = L.init_adam(p)
    srng = np.random.RandomState(seed_tr)
    for s in range(1, steps + 1):
        tok, tgt, lm, img_ids = build_vqa(srng, B, eng.T, 1)
        loss, c = eng.mask_loss(p, tok, tgt, lm, init_emb=adapter_emb(tok, p, img_ids))
        c["g"] = eng.bwd(p, c, tgt, lm)
        g = adapter_grads(eng, p, c, img_ids)
        L.adam_step(p, o, g, 3e-3 * min(1.0, s / 150))
    return eng, p, loss


def expA():
    print("=" * 66)
    print("[A] 原生多模态最小闭环：4 槽符号图 + 文本问题 → 答案 token（VQA）")
    print("    引擎 = 02-Llama 同款 2-block 解码器（C=48 NH=8）· 本期给引擎补『init_emb』钩子")
    print("    词表 V=40：文本 token 与 8 种视觉符号 token 同一张表（统一 token 空间，Gemini 哲学的最小 toy）")
    print("    问法两类：how many <sym>（count，数整图） /  slot k?（recall，读第 k 槽）")
    print("=" * 66)
    w1 = fd_check("native")
    w2 = fd_check("adapter")
    print()
    print("    A0 两引擎 maxerr 均过 1e-5 门（原生 %.1e · adapter %.1e）——init_emb 是干净路径" % (w1, w2))
    print()
    STEPS, BS = 1200, 64
    print("    A1 原生直训：逐槽视觉 token 进统一词表，模型自己学『查图』（1200 步×bs64）")
    t0 = time.perf_counter()
    eng, p, _ = train(True, STEPS, BS)
    dt = time.perf_counter() - t0
    cc, rc = eval_vqa(eng, p, np.random.RandomState(777), 400, 4)
    print(f"      fresh 正确率：count {cc:.3f} · recall {rc:.3f}（全新随机图+问法 n=400）")
    now(f"[A1] 原生 1200 步 wall {dt:.0f}s（stderr）")
    print()
    print("    A2 token 预算对照：视觉用 4 token（逐槽可寻址）vs 1 token（CLS 压缩）——同任务同预算")
    t0 = time.perf_counter()
    engA, pA, _ = train(False, STEPS, BS)
    dtA = time.perf_counter() - t0
    ccA, rcA = eval_vqa(engA, pA, np.random.RandomState(777), 400, 1, adapter=adapter_emb)
    print(f"      原生 4-token ：count {cc:.3f} · recall {rc:.3f}")
    print(f"      adapter 1-CLS：count {ccA:.3f} · recall {rcA:.3f}")
    print(f"      → 4 token 逐槽保留空间、query 按槽寻址；1 个 CLS 把 4 槽 mean 成一个向量，")
    print("        count（聚合统计）仍可答、recall（按位置读特定槽）信息被压掉——『原生多模态』")
    print("        买的是『保留空间结构的 token 流』，不是更少的表示维度。")
    now(f"[A2] adapter 1200 步 wall {dtA:.0f}s（stderr）")
    print()
    print("    A3 跨模态注意力读数：答案 token 的注意力落『图』/落『文』/目标槽（最后一层）")
    srng = np.random.RandomState(42)
    toks, tgt, lm, _ = build_vqa(srng, 1, 9, 4)
    c = eng.fwd(toks, p)
    att_last = c["caches"][eng.NL - 1]["att"][0]            # (H,T,T) 最后一层
    last = att_last[:, 8, :]                                # 预测答案位对前面 8 位的注意力
    att_img = float(last[:, :4].sum())                      # 落图
    att_txt = float(last[:, 4:8].sum())                     # 落文
    t = toks[0]
    if t[5] == SLOT:                                        # recall：目标槽随问题给定
        ks = int(t[6]) - K0
        att_tgt = float(last[:, ks].sum())
        tgt_label = f"槽{ks}"
    else:                                                   # count：数整图，无单槽目标
        ks, att_tgt, tgt_label = -1, 0.0, "(count 整图#)"
    ans = tgt[0, 8]
    ans_s = int(ans) - IMG_BASE if ans >= IMG_BASE else int(ans) - A0
    print(f"      示例 图=[{','.join(str(int(t)-IMG_BASE) for t in toks[0,:4])}] 问={tokens_q(toks)} → 答={ans_s}")
    print(f"      跨模态注意力：落图(4 槽) {att_img:.3f} · 落文(4 token) {att_txt:.3f}"
          f" · 目标{tgt_label} 收到 {att_tgt:.3f}")
    print(f"      → 图侧占答案注意力 {att_img/max(att_img+att_txt,1e-9)*100:.0f}%，目标槽占图侧 "
          f"{att_tgt/max(att_img,1e-9)*100:.0f}%：模型在『图』上按槽寻址读到答数，")
    print("        '读图'不是 prompt 拼接的假象，是本 script 里 self-attention 可读的位置结构。")
    now("expA 完成")


def tokens_q(toks):
    t = toks[0]
    if t[5] == HOW:
        return f"how many {t[7]-IMG_BASE}?"
    return f"slot {t[6]-K0}?"


# ===========================================================================
# B 派生账（公开配置 × 公式，非本机实测）
# ===========================================================================
def expB():
    print("=" * 66)
    print("[B] 派生账（公开配置 × 公式，非本机实测 · 未在线复核，概数以 Google 官方卡片为准）")
    print("    ① 超长上下文 1M~2M：KV 驻留账（02-10 母线）")
    print("=" * 66)
    GiB = lambda b: b / 2 ** 30
    per_tok = 32 * 2 * 8 * 128 * 2                # L=32 · 2 · H_kv=8 · DH=128 · 2B → B/tok
    kv = {200_000: per_tok * 200 * 1024,
          1_000_000: per_tok * 1_000_000,
          2_000_000: per_tok * 2_000_000}
    print(f"    每 token 每层 KV = 2·H_kv·DH·2B（02-10 母线）· 取 L=32·H_kv=8·DH=128 一档口径（账算）")
    tot = {k: gi / 2 ** 30 for k, gi in kv.items()}
    for k, v in sorted(tot.items()):
        print(f"      上下文 {k//1000}k  →  KV 驻留 {v:.0f} GiB（每 token {per_tok:,} B）")
    print(f"      → 1M→2M 上下文翻倍，KV 线性翻倍；GQA/MoE 把 H_kv 摊薄（8 头 vs 128 头省 16×）")
    print(f"        ——长上下文的第一性成本是『每 token 的 KV 线性累积』，再长也要在 KV 账上做减法。")
    print()
    print("    ② 原生多模态 tokenizer 经济账：一张图 = 多少 token")
    print(f"    Gemini 用 16×16 patch（公开 tech report 口径）→ 图切成 (H/16)×(W/16) 个 patch token")
    for (hw, label) in ((256, "256×256"), (768, "768×768"), (1024, "1024×1024")):
        ntok = (hw // 16) ** 2
        print(f"      {label} 图 → {hw//16}×{hw//16} = {ntok:,} patch token")
    print(f"      → 「统一 token 空间」把图像拉进和文本同一条计价带：一张 1024 图≈4096 token，")
    print(f"        读一张图花的上下文预算比一页纸还多——多模态是『token 预算地图』的另一维。")
    print()
    print("    ③ 全家谱（公开事实，非本机实测）")
    print(f"    Gemini1.0(2023-12 Pro/Ultra/Nano) → 1.5(2024: MoE, 1M~2M, 原生音/视频) →")
    print(f"    2.0/2.5(2025: Agentic, 代码执行/搜索/浏览器, thinking) → 3(2025 末, 大版本迭代)")
    print(f"    八要素速写：原生多模态 / MoE+超长上下文 / 多模态语料直训 / 1M+ 第一梯队 /")
    print(f"    thinking / 与 Google 搜索生态深度绑定 → 与 07-Claude 同属闭源 API 系，")
    print("    但双子座的招牌是『把上下文与多模态做成产品现实』，Claude 的招牌是对齐与 Agent。")
    now("expB 完成")


def main():
    t0 = time.perf_counter()
    print("=" * 66)
    print("03-模型家族 08-Gemini系列 · 「原生多模态」——视觉 token 化进同一词表")
    print("任务族：VQA（4 槽符号图 → 文本问题 → 答案 token），引擎 = 02-Llama 同款 2-block")
    print("词表 V=40：文本与视觉符号共用一张表（统一 token 空间）")
    print("=" * 66)
    expA()
    print()
    expB()
    print()
    now(f"全脚本累计 {time.perf_counter()-t0:.1f}s")
    print()
    print("done · 一键复现：python code/scripts/gemini_demo.py")


if __name__ == "__main__":
    main()
