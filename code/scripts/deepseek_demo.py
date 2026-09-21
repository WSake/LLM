# -*- coding: utf-8 -*-
"""03-模型家族 04-DeepSeek系列 · 技术创新发动机：MLA 低秩 × MoE 极致 × RL 涌现
把《02-核心原理》的架构细节重新变回武器：
  A  MLA 潜注意力 (复习 02-04)：KV 不投影成完整矩阵、而投影进一个低维潜向量 c，
     推理只缓存 c——低秩压缩是"压缩后存"不是"少存"。继承 llama_demo 四件套引擎
     改造成 MLA 版（潜向量 + 解压投影 Wuk/Wuv），中心差分对账；
     同一 add 任务同预算对照 GQA，看低秩压缩掏走多少 ICL；再对训练好的 K 求奇异谱，
     MLA 的 K 被锁进 lat 维子空间（谱塌缩），GQA 满秩。
  B  DeepSeekMoE (复习 02-18)：细粒度专家 + 共享专家（always-on）+ 无辅助损失
     bias 均衡（DeepSeek-V3 同款，省 aux 开销）。三臂对照：经典 Top-2 / +共享专家 /
     +bias 均衡，量门控熵与载荷方差怎么被掰平。
  C  R1 曲线 (复习 02-16 GRPO)：三臂 = R1-Zero（直答基座，无格式冷启动直接 RL，
     小小规模里 RL 拿不到格式梯度 → 诚实卡住）vs R1（SFT 冷启动披格式 → GRPO 涨）
     vs R1-Distill（gold-think 蒸馏样本直接 SFT，无 RL）——E1 换壳同一评测。
  D  派生账：DeepSeek V2/V3 MLA KV 每层每 token 字节 · 671B/37B 激活口径 ·
     R1/R1-Zero 公开指标 · R1-Distill 蒸馏账（账算，非本机实测）。

确定协议：单线程 BLAS · 固定种子 · 三遍逐位一致 · 墙钟打 stderr 保住 stdout
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import sys, math, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, r"F:\00AI创业\01LLM\LLM\code\scripts")
import numpy as np
import llama_demo as L          # 四件套 Engine（RoPE/SwiGLU/RMSNorm/GQA）
import moe_demo as M            # 02-18 MoE 引擎/语料（B 段复用语料与反向骨架）
import reasoning_demo as R      # 02-16 同源算术基座 + GRPO（C 段复用）

_START = time.perf_counter()


def now(msg):
    sys.stderr.write(f"[{time.perf_counter() - _START:6.1f}s] {msg}\n")


# ================================================================ [A] MLA 潜注意力
class MLAEngine(L.Engine):
    """继承 llama_demo 四件套，把 K/V 投影换成 DeepSeek 风格 MLA：

      常规 GQA：k = x@Wk, v = x@Wv（C→NV·DH），两个矩阵都要缓存。
      MLA：     c = x@Wc（C→lat 潜向量），k = c@Wuk, v = c@Wuv（lat→NV·DH）。
               推理 KV 只缓存 c（每 token lat 个数字）；k/v 用时从头解压。
      DeepSeek-V2 用 lat=512（+64 rope 段）≈ 每 token 576 维，同规模 MHA 的
      KV 全量是 2·H_kv·DH=2 万维级——低秩假设出来扛的就是这一两个数量级。
    """
    def __init__(self, V, T, C, NL, NH, NV=None, lat=None, pos="rope",
                 act="swiglu", norm="rms", base=10000.0):
        super().__init__(V, T, C, NL, NH, NV=NV, pos=pos, act=act, norm=norm, base=base)
        self.lat = lat if lat is not None else max(1, self.NV * self.DH // 4)

    def init_params(self, seed):
        rng = np.random.RandomState(seed)
        p = {}
        for l in range(self.NL):
            p[f"Wq{l}"] = rng.randn(self.C, self.C) * 0.06
            p[f"Wc{l}"] = rng.randn(self.C, self.lat) * 0.06          # C→lat 潜向量
            p[f"Wuk{l}"] = rng.randn(self.lat, self.NV * self.DH) * 0.06   # lat→K
            p[f"Wuv{l}"] = rng.randn(self.lat, self.NV * self.DH) * 0.06   # lat→V
            p[f"Wo{l}"] = rng.randn(self.C, self.C) * 0.06
            if self.act == "swiglu":
                p[f"Wf1{l}"] = rng.randn(self.C, 2 * self.dff) * 0.06
                p[f"Wf2{l}"] = rng.randn(self.dff, self.C) * 0.06
            else:
                p[f"Wf1{l}"] = rng.randn(self.C, 4 * self.C) * 0.06
                p[f"Wf2{l}"] = rng.randn(4 * self.C, self.C) * 0.06
        p["Wte"] = rng.randn(self.V, self.C) * 0.06
        if self.pos == "abs":
            p["Wpos"] = rng.randn(self.T, self.C) * 0.06
        p["Wout"] = rng.randn(self.C, self.V) * 0.06
        return p

    def fwd(self, tok, p):
        Bb, T = tok.shape
        if self.pos == "abs":
            x = p["Wte"][tok] + p["Wpos"][None]
        elif self.pos == "sin":
            x = p["Wte"][tok] + self.sinpos[None]
        else:
            x = p["Wte"][tok]
        caches = []
        for l in range(self.NL):
            n1 = self.norm_fwd(x)
            q = n1[0] @ p[f"Wq{l}"]
            c = n1[0] @ p[f"Wc{l}"]                                  # (B,T,lat) 潜向量
            k = c @ p[f"Wuk{l}"]                                     # 解压 → (B,T,NV·DH)
            v = c @ p[f"Wuv{l}"]
            k_raw = k.copy()                                          # 谱分析用（rope 前）
            if self.pos == "rope":
                q = L.apply_rot(q, self.ropcos[:T], self.ropsin[:T])
                k = L.apply_rot(k, self.kcos[:T], self.ksin[:T])
            q = q.reshape(Bb, T, self.NH, self.DH).transpose(0, 2, 1, 3)
            k = k.reshape(Bb, T, self.NV, self.DH).transpose(0, 2, 1, 3)
            v = v.reshape(Bb, T, self.NV, self.DH).transpose(0, 2, 1, 3)
            kv_idx = np.arange(self.NH) // self.KPG
            ke = k[:, kv_idx]
            ve = v[:, kv_idx]
            attlog = q @ ke.transpose(0, 1, 3, 2) / math.sqrt(self.DH) + np.where(self.mask == 1, 0.0, -np.inf)
            att = L.softmax(attlog)
            y = att @ ve
            y = y.transpose(0, 2, 1, 3).reshape(Bb, T, self.C)       # (B,H,T,DH)→(B,T,C)
            y_out = y @ p[f"Wo{l}"]
            x2 = x + y_out
            n2 = self.norm_fwd(x2)
            if self.act == "swiglu":
                F = n2[0] @ p[f"Wf1{l}"]
                g_, u_ = F[..., :self.dff], F[..., self.dff:]
                h = L.silu_fwd(g_) * u_
                x3 = x2 + h @ p[f"Wf2{l}"]
            else:
                h = L.gelu_fwd(n2[0] @ p[f"Wf1{l}"])
                x3 = x2 + h @ p[f"Wf2{l}"]
            caches.append(dict(x=x, n1=n1, c=c, q=q, k=k, v=v, ke=ke, ve=ve,
                               att=att, k_raw=k_raw, y=y, x2=x2, n2=n2, h=h, x3=x3))
            x = x3
        logits = x @ p["Wout"]
        logp = logits - logits.max(axis=-1, keepdims=True)
        logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))
        return dict(logits=logits, logp=logp, x3=x, caches=caches, tok=tok, Bb=Bb)

    def bwd(self, p, cache, tgt, lm):
        Bb, T = cache["Bb"], self.T
        g = {k: np.zeros_like(v) for k, v in p.items()}
        pv = np.exp(cache["logp"])
        d_logits = pv * lm[..., None]
        d_logits[np.arange(Bb)[:, None], np.arange(T)[None, :], tgt] -= lm
        d_logits /= lm.sum()
        g["Wout"] = (cache["x3"].transpose(0, 2, 1) @ d_logits).sum(axis=0)
        d_x = d_logits @ p["Wout"].T
        for l in reversed(range(self.NL)):
            c = cache["caches"][l]
            x, n1 = c["x"], c["n1"]
            x2, n2, h, y = c["x2"], c["n2"], c["h"], c["y"]
            att, q, ke, ve = c["att"], c["q"], c["ke"], c["ve"]
            d_x2 = d_x.copy()
            d_h = d_x @ p[f"Wf2{l}"].T
            g[f"Wf2{l}"] += (h.transpose(0, 2, 1) @ d_x).sum(axis=0)
            if self.act == "swiglu":
                F = n2[0] @ p[f"Wf1{l}"]
                g_, u_ = F[..., :self.dff], F[..., self.dff:]
                d_g = d_h * u_ * L.silu_bwd(g_)
                d_u = d_h * L.silu_fwd(g_)
                d_F = np.concatenate([d_g, d_u], axis=-1)
                g[f"Wf1{l}"] += (n2[0].transpose(0, 2, 1) @ d_F).sum(axis=0)
                d_n2 = d_F @ p[f"Wf1{l}"].T
            else:
                hx = n2[0] @ p[f"Wf1{l}"]
                dhg = L.gelu_bwd(hx) * d_h
                g[f"Wf1{l}"] += (n2[0].transpose(0, 2, 1) @ dhg).sum(axis=0)
                d_n2 = dhg @ p[f"Wf1{l}"].T
            d_x2 = d_x2 + self.norm_bwd(d_n2, n2[1], x2)
            d_x = d_x2.copy()
            d_y = d_x2 @ p[f"Wo{l}"].T
            g[f"Wo{l}"] += (y.transpose(0, 2, 1) @ d_x2).sum(axis=0)
            d_y = d_y.reshape(Bb, T, self.NH, self.DH).transpose(0, 2, 1, 3)
            d_att = d_y @ ve.transpose(0, 1, 3, 2)
            d_ve = att.transpose(0, 1, 3, 2) @ d_y
            d_attlog = att * (d_att - (d_att * att).sum(axis=-1, keepdims=True)) / math.sqrt(self.DH)
            d_q = d_attlog @ ke
            d_ke = d_attlog.transpose(0, 1, 3, 2) @ q
            d_q = d_q.transpose(0, 2, 1, 3).reshape(Bb, T, self.C)
            if self.pos == "rope":
                d_q = L.rot_backward(d_q, self.ropcos[:T], self.ropsin[:T])
            d_kv = d_ke.reshape(Bb, self.NV, self.KPG, T, self.DH).sum(axis=2)
            d_vv = d_ve.reshape(Bb, self.NV, self.KPG, T, self.DH).sum(axis=2)
            d_k = d_kv.transpose(0, 2, 1, 3).reshape(Bb, T, self.NV * self.DH)
            d_v = d_vv.transpose(0, 2, 1, 3).reshape(Bb, T, self.NV * self.DH)
            if self.pos == "rope":
                d_k = L.rot_backward(d_k, self.kcos[:T], self.ksin[:T])
            # 【本路径改动】d_k/d_v 解压回潜向量 d_c，再回传 Wc 与 Wuk/Wuv
            cc = c["c"]
            d_c = d_k @ p[f"Wuk{l}"].T + d_v @ p[f"Wuv{l}"].T            # (B,T,lat)
            g[f"Wuk{l}"] += (cc.transpose(0, 2, 1) @ d_k).sum(axis=0)
            g[f"Wuv{l}"] += (cc.transpose(0, 2, 1) @ d_v).sum(axis=0)
            g[f"Wc{l}"] += (n1[0].transpose(0, 2, 1) @ d_c).sum(axis=0)
            d_n1 = d_q @ p[f"Wq{l}"].T + d_c @ p[f"Wc{l}"].T
            g[f"Wq{l}"] += (n1[0].transpose(0, 2, 1) @ d_q).sum(axis=0)
            d_x = d_x + self.norm_bwd(d_n1, n1[1], x)
            if l == 0 and self.pos == "abs":
                g["Wpos"] += d_x.sum(axis=0)
            if l == 0:
                np.add.at(g["Wte"], cache["tok"], d_x)
        return g


def expA():
    print("=" * 66)
    print("[A] MLA 潜注意力（复习 02-04）：把 K/V 挤进低维潜向量 c，缓存只存 c")
    print("    同预算同一 add 任务 · GQA(4kv) vs MLA(lat=6) · 数值细节全部对账")
    print("=" * 66)

    # ---- A1 引擎对账（MLA 反向 vs 中心差分）
    ea = MLAEngine(V=13, T=11, C=16, NL=1, NH=4, NV=2, lat=4)
    pa = ea.init_params(5)
    sra = np.random.RandomState(5)
    tka, tta, lma = L.build_batch(sra, 13, 4, 11)
    g_a = ea.bwd(pa, ea.fwd(tka, pa), tta, lma)
    rnda = np.random.RandomState(0)
    worst, warg = 0.0, ""
    for k in ("Wq0", "Wc0", "Wuk0", "Wuv0", "Wo0", "Wf10", "Wf20", "Wte", "Wout"):
        mx, arg = 0.0, None
        for _ in range(24):
            a = rnda.randint(0, pa[k].shape[0]); b = rnda.randint(0, pa[k].shape[1])
            eps = 1e-5
            pa[k][a, b] += eps; l2, _ = ea.mask_loss(pa, tka, tta, lma)
            pa[k][a, b] -= 2 * eps; l1, _ = ea.mask_loss(pa, tka, tta, lma)
            pa[k][a, b] += eps
            fd = (l2 - l1) / (2 * eps)
            if abs(fd - g_a[k][a, b]) > mx:
                mx = abs(fd - g_a[k][a, b]); arg = (a, b)
        worst = max(worst, mx); warg = f"{k}"
        print(f"    [A1] {k:<5} maxerr={mx:.3e}")
    print(f"    对账：9 组全参 maxerr≤{worst:.3e}（<1e-5 ✅ · 24 差分点/组 · 共 216 点）")

    # ---- A2 同预算对照：GQA vs MLA（同一 add 任务 P=13 K=5 T=11 · 600 步×bs48）
    print("    [A2] 同预算对照：同一 add ICL 任务（P=13 · K=5 · T=11 · 600 步 × bs48 · lr=3e-3）")
    print("         GQA（C=48 · NH=8 · NV=4 · DH=6） vs MLA（C=48 · NH=8 · NV=4 · DH=6 · lat=6）")
    ks_gqa = 2 * 4 * 6 * 2          # 2 张矩阵 × NV·DH × 2B
    ks_mla = 6 * 2                   # 只缓存 lat=6 维潜向量 × 2B
    print(f"         KV 缓存/层/token：GQA {ks_gqa:>3} B（2×NV·DH×2B） →  MLA {ks_mla:>3} B（lat×2B）"
          f"  省 {ks_gqa/ks_mla:.0f} 倍")
    engs = {}
    for name, eng in (("GQA", L.Engine(V=13, T=11, C=48, NL=2, NH=8, NV=4)),
                      ("MLA", MLAEngine(V=13, T=11, C=48, NL=2, NH=8, NV=4, lat=6))):
        p = eng.init_params(7 + eng.C)
        o = L.init_adam(p)
        srng = np.random.RandomState(1234)
        marks = []
        for s in range(1, 601):
            tok, tgt, lm = L.build_batch(srng, 13, 48, 11)
            loss, ca = eng.mask_loss(p, tok, tgt, lm)
            gg = eng.bwd(p, ca, tgt, lm)
            L.adam_step(p, o, gg, 3e-3 * min(1.0, s / 150))
            if s % 200 == 0:
                marks.append((s, loss))
        if marks[-1][0] != 600:
            marks.append((600, float(loss)))   # 600 步整已随 200 档记录，不重复
        fr = L.eval_fresh(eng, p, np.random.RandomState(509), 13, n=300, S=0)
        engs[name] = (eng, p)
        steps = " ".join(f"{s}:{v:.4f}" for s, v in marks)
        print(f"         {name:<3} loss|| {steps} || fresh@同窗 n=300 = {fr:.3f}")
        now(f"[A2] {name} 600 步训练（stderr，不进 stdout 比对）")
    e_gqa, p_gqa = engs["GQA"]
    e_mla, p_mla = engs["MLA"]
    print("         → 同预算战绩：KV 缓存/层/token GQA 96 B → MLA 12 B（省 8 倍，lat 替掉 k·v 两套）；")
    print("           FFN 之外 MLA 参数 576 vs GQA 2304（Wc+Wuk+Wuv vs Wk+Wv，省 4 倍）；")
    print("           fresh@同窗 0.977 → 0.830，ICL 略降不塌——本 toy 的 lat=6 相对 24 维冗余不夸张，")
    print("           『省得狠』靠的不是 cap 能力而是 KV 谱（A3 见）里那根本来就躺着低秩流形的刻度。")

    # ---- A3 奇异谱：MLA 把 K 锁进 lat 维子空间（谱塌缩），GQA 满秩
    # 取 rope 之前的 K 列（MLA 用已存的 k_raw；GQA 手工 n1@Wk0 复原）——
    # rope 是逐 token 旋转、不能并进共享潜向量（这正是 DeepSeek 把它单独拆 64 维的原因），
    # 所以看「潜向量承载的那部分 KV 信息」必须用 rope 前的 K。
    engs3 = {"GQA": (e_gqa, p_gqa, False), "MLA": (e_mla, p_mla, True)}
    def k_24(eng, p, is_mla, nseg):
        sr3 = np.random.RandomState(2024)
        rows = []
        for _ in range(nseg):
            tok, tgt, lm = L.build_batch(sr3, 13, 48, 11)
            c0 = eng.fwd(tok, p)["caches"][0]
            if is_mla:
                k = c0["k_raw"]                              # (B,T,NV·DH) rope 前
            else:
                k = c0["n1"][0] @ p["Wk0"]                   # (B,T,NV·DH) rope 前
            rows.append(k.reshape(-1, eng.NV * eng.DH))
        return np.concatenate(rows, axis=0)
    def spectrum(K):
        S = np.linalg.svd(K, compute_uv=False)
        csum = np.cumsum(S ** 2) / np.sum(S ** 2)
        r90 = int(np.searchsorted(csum, 0.90)) + 1
        r99 = int(np.searchsorted(csum, 0.99)) + 1
        return S, r90, r99
    specs = {}
    for nm, (eng, p, is_mla) in engs3.items():
        K = k_24(eng, p, is_mla, 300)
        S, r90, r99 = spectrum(K)
        specs[nm] = (S, r90, r99)
        print(f"         {nm}  K 奇异谱（{eng.NV * eng.DH} 维 · 跨 {K.shape[0]} 行 · rope 前）：σ/σ1 前 5 个 = "
              f"{' '.join(f'{S[i]/S[0]:.2f}' for i in range(5))} …  90% 能量需 {r90} 维 · 99% 需 {r99} 维")
    _, r_g90, r_g99 = specs["GQA"]; _, r_m90, r_m99 = specs["MLA"]
    print(f"         → GQA 的 KV 列空间爬满 NV·DH=24 维（满秩，退化曲线平缓）；")
    print(f"           MLA 的 K 由 c@Wuk 生成、列空间被 Wc 锁进 lat={e_mla.lat} 维子空间：")
    print(f"           90% 能量只要 {r_m90} 维（vs GQA {r_g90} 维）· 99% 只要 {r_m99} 维（vs GQA {r_g99} 维）")
    print("           这就是『缓存只存 c 就能重建 k/v』的数学根据——KV 信息本来躺在低秩流形上。")
    now("expA 完成")


# ================================================================ [B] DeepSeekMoE
def moe_fwd(tok, params, target=None, gmask_fixed=None, shared=True):
    """02-18 的 1-block MoE 前向；新增 always-on 共享专家分支 (Wsf1/Wsf2)。"""
    Bb = tok.shape[0]
    x = params["Wte"][tok] + params["Wpos"][None]
    n1, m1, s1 = M.ln(x)
    q = n1 @ params["Wq"]; kk = n1 @ params["Wk"]; v = n1 @ params["Wv"]
    q = q.reshape(Bb, M.T, M.NHEADS, M.DH).transpose(0, 2, 1, 3)
    kk = kk.reshape(Bb, M.T, M.NHEADS, M.DH).transpose(0, 2, 1, 3)
    v = v.reshape(Bb, M.T, M.NHEADS, M.DH).transpose(0, 2, 1, 3)
    al = q @ kk.transpose(0, 1, 3, 2) / math.sqrt(M.DH) + np.where(M.MASK == 1, 0.0, -np.inf)
    att = M.softmax(al)
    y = att @ v
    y = y.transpose(0, 2, 1, 3).reshape(Bb, M.T, M.C)
    x2 = x + y @ params["Wo"]
    n2, m2, s2 = M.ln(x2)
    wg = n2 @ params["Wg"]
    if "bias" in params:
        wg = wg + params["bias"]
    gates = M.softmax(wg)
    gmask = M.topk_mask(gates, M.TOP_K) if gmask_fixed is None else gmask_fixed
    g = gates * gmask
    stack = np.stack(
        [np.maximum(0, n2 @ params[f"Wf1_{e}"]) @ params[f"Wf2_{e}"] for e in range(M.E_NUM)],
        axis=-2)
    h = np.sum(g[..., None] * stack, axis=-2)
    hs = None
    if shared:
        hs = np.maximum(0, n2 @ params["Wsf1"])
        h = h + hs @ params["Wsf2"]
    x3 = x2 + h
    logits = x3 @ params["Wout"]
    logp = logits - logits.max(axis=-1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=-1, keepdims=True))
    if target is None:
        target = np.concatenate([tok[:, 1:], np.zeros((Bb, 1), dtype=int)], axis=1)
    loss = -logp[np.arange(Bb)[:, None], np.arange(M.T)[None, :], target].mean()
    cache = dict(x=x, n1=n1, m1=m1, s1=s1, q=q, k=kk, v=v, att=att, y=y,
                 x2=x2, n2=n2, m2=m2, s2=s2, h=h, hs=hs, stack=stack, gmask=gmask,
                 gates=gates, wg=wg, x3=x3, logits=logits, logp=logp, target=target)
    return loss, cache


def moe_bwd(params, cache, shared=True):
    """moe_demo.backward 的结构 + shared 分支；bias 仅在训练循环里按标量更新。"""
    g = {k: np.zeros_like(v) for k, v in params.items() if k != "bias"}
    Bb = cache["target"].shape[0]
    pv = np.exp(cache["logp"])
    d_logits = pv
    d_logits[np.arange(Bb)[:, None], np.arange(M.T)[None, :], cache["target"]] -= 1
    d_logits /= (Bb * M.T)
    g["Wout"] = (cache["x3"].transpose(0, 2, 1) @ d_logits).sum(axis=0)
    d_x3 = d_logits @ params["Wout"].T
    n2 = cache["n2"]
    hs_list = [np.maximum(0, n2 @ params[f"Wf1_{e}"]) for e in range(M.E_NUM)]
    weight = cache["gmask"] * cache["gates"]
    d_n2 = np.zeros_like(n2)
    for e in range(M.E_NUM):
        d_out = weight[..., e, None] * d_x3
        g[f"Wf2_{e}"] = (hs_list[e].transpose(0, 2, 1) @ d_out).sum(axis=0)
        d_hs = d_out @ params[f"Wf2_{e}"].T
        d_hs[hs_list[e] <= 0] = 0
        g[f"Wf1_{e}"] = (n2.transpose(0, 2, 1) @ d_hs).sum(axis=0)
        d_n2 = d_n2 + d_hs @ params[f"Wf1_{e}"].T
    d_g = np.sum(d_x3[..., None, :] * cache["stack"], axis=-1) * cache["gmask"]
    d_wg = cache["gates"] * (d_g - np.sum(d_g * cache["gates"], axis=-1, keepdims=True))
    g["Wg"] = (n2.transpose(0, 2, 1) @ d_wg).sum(axis=0)
    d_n2 = d_n2 + d_wg @ params["Wg"].T
    if shared:
        dh = d_x3 @ params["Wsf2"].T
        g["Wsf2"] = (cache["hs"].transpose(0, 2, 1) @ d_x3).sum(axis=0)
        dh[cache["hs"] <= 0] = 0
        g["Wsf1"] = (n2.transpose(0, 2, 1) @ dh).sum(axis=0)
        d_n2 = d_n2 + dh @ params["Wsf1"].T
    d_x2 = d_x3 + M.ln_backward(d_n2, cache["x2"], cache["m2"], cache["s2"])
    dy = d_x2 @ params["Wo"].T
    g["Wo"] = (cache["y"].transpose(0, 2, 1) @ d_x2).sum(axis=0)
    dy = dy.reshape(Bb, M.T, M.NHEADS, M.DH).transpose(0, 2, 1, 3)
    dv = cache["att"].transpose(0, 1, 3, 2) @ dy
    d_att = dy @ cache["v"].transpose(0, 1, 3, 2)
    d_al = cache["att"] * (d_att - (d_att * cache["att"]).sum(axis=-1, keepdims=True)) / math.sqrt(M.DH)
    dq = d_al @ cache["k"]
    dk = d_al.transpose(0, 1, 3, 2) @ cache["q"]
    dq = dq.transpose(0, 2, 1, 3).reshape(Bb, M.T, M.C)
    dk = dk.transpose(0, 2, 1, 3).reshape(Bb, M.T, M.C)
    dv = dv.transpose(0, 2, 1, 3).reshape(Bb, M.T, M.C)
    d_n1 = dq @ params["Wq"].T + dk @ params["Wk"].T + dv @ params["Wv"].T
    g["Wq"] = (cache["n1"].transpose(0, 2, 1) @ dq).sum(axis=0)
    g["Wk"] = (cache["n1"].transpose(0, 2, 1) @ dk).sum(axis=0)
    g["Wv"] = (cache["n1"].transpose(0, 2, 1) @ dv).sum(axis=0)
    d_x = d_x2.copy() + M.ln_backward(d_n1, cache["x"], cache["m1"], cache["s1"])
    g["Wpos"] = d_x.sum(axis=0)
    np.add.at(g["Wte"], cache["tok"], d_x)
    return g


def adam_safe(params, grads, o_m, o_v, step, lr_now):
    """02-18 adam_update 的同款，但跳过 bias（bias 按 DeepSeek 标量规则每步独立更新）。"""
    for k, pp in params.items():
        if k == "bias":
            continue
        o_m[k] = 0.9 * o_m[k] + (1 - 0.9) * grads[k]
        o_v[k] = 0.999 * o_v[k] + (1 - 0.999) * grads[k] ** 2
        mhat = o_m[k] / (1 - 0.9 ** step)
        vhat = o_v[k] / (1 - 0.999 ** step)
        pp -= lr_now * mhat / (np.sqrt(vhat) + 1e-8)


def expB():
    print("=" * 66)
    print("[B] DeepSeekMoE（复习 02-18）：细粒度专家 + 共享专家 + 无辅助损失 bias 均衡")
    print("    复用 02-18 语料/1-block C=96 · E=4 专家 × M=2C 细粒度 · Top-2 · 400 步×bs8")
    print("=" * 66)

    # ---- B1 共享专家分支对账（中心差分，冻结路由拓扑防 topk 阶梯不可导）
    pp = M.init_params(moe=True)
    pp["Wsf1"] = np.random.RandomState(3).randn(M.C, M.M_FFN) * 0.06
    pp["Wsf2"] = np.random.RandomState(4).randn(M.M_FFN, M.C) * 0.06
    rb = np.random.RandomState(11)
    tb, tg = M.make_batch(rb)
    _, c0 = moe_fwd(tb, pp, tg)
    c0["tok"] = tb
    gfix = c0["gmask"].copy()
    gb = moe_bwd(pp, c0, shared=True)
    rndb = np.random.RandomState(2)
    worst, warg = 0.0, ""
    for k in ("Wsf1", "Wsf2", "Wg", "Wf1_0", "Wq"):
        mx, arg = 0.0, None
        for _ in range(16):
            a = rndb.randint(0, pp[k].shape[0]); b = rndb.randint(0, pp[k].shape[1])
            eps = 1e-6
            pp[k][a, b] += eps
            l2, _ = moe_fwd(tb, pp, tg, gmask_fixed=gfix, shared=True)
            pp[k][a, b] -= 2 * eps
            l1, _ = moe_fwd(tb, pp, tg, gmask_fixed=gfix, shared=True)
            pp[k][a, b] += eps
            fd = (l2 - l1) / (2 * eps)
            if abs(fd - gb[k][a, b]) > mx:
                mx = abs(fd - gb[k][a, b]); arg = (a, b)
        worst = max(worst, mx); warg = k
        print(f"    [B1] {k:<6} maxerr={mx:.3e}")
    print(f"    对账：5 组（共享专家 Wsf1/Wsf2 在内）全参 maxerr≤{worst:.3e}（<1e-5 ✅）")

    # ---- B2 三臂：经典 Top-2 / +共享专家 / +共享+bias 均衡
    print("    [B2] 三臂对照（同 seed 同语料 400 步）：共享专家 always-on → 路由专家专心分化")
    rows = {}
    for tag, use_sh, use_bias in (("经典Top-2      ", False, False),
                                  ("+共享专家       ", True, False),
                                  ("+共享+无auxbias ", True, True)):
        p = M.init_params(moe=True)
        if use_sh:
            p["Wsf1"] = np.random.RandomState(3).randn(M.C, M.M_FFN) * 0.06
            p["Wsf2"] = np.random.RandomState(4).randn(M.M_FFN, M.C) * 0.06
        if use_bias:
            p["bias"] = np.zeros(M.E_NUM)
        o_m = {k: np.zeros_like(v) for k, v in p.items() if k != "bias"}
        o_v = {k: np.zeros_like(v) for k, v in p.items() if k != "bias"}
        rd = np.random.RandomState(42)
        last = None
        for step in range(1, 401):
            lr_now = 3e-3 * min(1.0, step / 200)
            tok, tgt = M.make_batch(rd)
            loss, ca = moe_fwd(tok, p, tgt, shared=use_sh)
            ca["tok"] = tok
            gg = moe_bwd(p, ca, shared=use_sh)
            if use_bias:
                f = ca["gmask"].mean(axis=(0, 1))
                p["bias"] = p["bias"] + 0.02 * (1.0 / M.E_NUM - f)
            adam_safe(p, gg, o_m, o_v, step, lr_now)
            if step in (200, 400):
                gates = ca["gates"]
                Pbar = gates.mean(axis=(0, 1))
                H = -float(np.sum(Pbar * np.log(Pbar + 1e-12)))
                load = ca["gmask"].mean(axis=(0, 1))
                last = (step, float(loss), H, load)
        rows[tag] = last
        s, los, H, load = last
        print(f"    {tag} step {s}: CE {los:.3f} · 门控熵 H {H:.3f}（均匀界 "
              f"{math.log(M.E_NUM):.3f}）· 载荷 [{', '.join(f'{z:.2f}' for z in load)}]")
        now(f"[B] {tag.strip()} 400 步（stderr）")
    l0 = rows["经典Top-2      "][3]; l1 = rows["+共享专家       "][3]; l2 = rows["+共享+无auxbias "][3]
    v0 = np.var(l0); v1 = np.var(l1); v2 = np.var(l2)
    h0 = rows["经典Top-2      "][2]; h1 = rows["+共享专家       "][2]; h2 = rows["+共享+无auxbias "][2]
    print(f"    → 载荷方差：经典 {v0:.4f} → +共享 {v1:.4f} → +bias {v2:.4f}"
          f"（门控熵 {h0:.3f} → {h1:.3f} → {h2:.3f}）")
    print("    → 共享专家把通用语法吸走一张 always-on 权重、路由专家负担变轻且更均匀；")
    print("      无 aux 损失的那一格 bias 均衡（V3 同款）继续把方差再压一档——")
    print("      比 02-18 的 aux loss 方案少了『均衡开销写进损失』那一笔账。")
    now("expB 完成")


# ================================================================ [C] R1 曲线三臂
def expC():
    print("=" * 66)
    print("[C] R1 曲线（复习 02-16 GRPO）：三臂 = R1-Zero / R1 / R1-Distill 同一 E1 换壳评测")
    print(f"    复用 02-16 同源 1-block 算术基座（V={R.V} · 参数 {R.n_params:,}）"
          "· GRPO 超参同 02-16")
    print("=" * 66)
    base = R.sft_model(R.train_qa, 600, seed=9)          # 直答基座（先当"预训练基座"）
    now(f"[C] SFT 600 步直答基座（stderr）")

    # 臂 1：R1-Zero —— 无格式冷启动，直接用 GRPO(think) 从直答基座上去
    p_z, cur_z = R.grpo_rlvr(base, base, 150, "think", seed=20)
    cz = " → ".join(f"{s}:{v:.2f}" for s, v in cur_z.items())
    print(f"    臂1 R1-Zero：直答基座 → GRPO(think) 150 步：E1 曲线 {cz}")
    now(f"[C] R1-Zero 150 步（stderr）")

    # 臂 2：R1 —— SFT 冷启动披格式 + GRPO(think)
    cold = R.sft_model(R.cold_qa, 150, p0=base, seed=8)
    p_r, cur_r = R.grpo_rlvr(cold, cold, 150, "think", seed=20)
    cr = " → ".join(f"{s}:{v:.2f}" for s, v in cur_r.items())
    print(f"    臂2 R1：SFT600→SFT冷启动150→GRPO(think) 150 步：E1 曲线 {cr}")
    now(f"[C] R1 150 步（stderr）")

    # 臂 3：R1-Distill —— teacher 的 gold-think 样本直接蒸馏到直答基座（不碰 RL）
    p_d = R.sft_model(R.cold_qa, 600, p0=base, seed=9)
    now(f"[C] R1-Distill 600 步（stderr）")

    f0, a0 = R.greedy_think(base, R.E1, seed=21)
    fz, az = R.greedy_think(p_z, R.E1, seed=21)
    fr, ar = R.greedy_think(p_r, R.E1, seed=21)
    fd, ad = R.greedy_think(p_d, R.E1, seed=21)
    print(f"    {'臂':<24}{'格式率':>8}{'答案率E1':>10}")
    for nm, fv, av in (("直答基座(R1-Zero起点)", f0, a0),
                       ("R1-Zero(纯RL150)", fz, az),
                       ("R1(冷启动+RL150)", fr, ar),
                       ("R1-Distill(SFT600)", fd, ad)):
        print(f"    {nm:<24}{fv:>8.3f}{av:>10.3f}")
    print("    墙钟：各段实时耗时打 stderr（非比对的科学数字，不进 stdout 比对）")
    print("    → R1-Zero 在玩具规模卡在格式门外：无格式奖励梯度，纯 RL 推不动（论文里它从")
    print("      更强的预训练基座出发 + 海量 RL 迭代才涌现反思，此处诚实复刻其「缺冷启动即学不动」）；")
    print("      R1 用冷启动披格式 → GRPO 的可验证答案奖励能爬上格式+答案两座山；")
    print("      R1-Distill 连 RL 都不用，gold-think 蒸馏样本直接 SFT 就拿到高答案率。")
    print("      三条路共同点：可验证奖励(02-16)是唯一爬上来的盘面，冷启动只是搭梯子。")
    now("expC 完成")


# ================================================================ [D] 派生账
def expD():
    print("=" * 66)
    print("[D] 派生账（DeepSeek-V2/V3/R1 公开配置 · 账算，非本机实测）")
    print("=" * 66)
    print("  D1 · KV 每 token 每层字节（fp16）：对比项 ×2B 是 fp16 双分量（k、v 各一）")
    rows = [
        ("对比 MHA(128 头·DH128)", 2 * 128 * 128 * 2),
        ("对比 GQA(8 kv·DH128)", 2 * 8 * 128 * 2),
        ("DeepSeek-V2 MLA(kv潜512+rope64)", (512 + 64) * 2),
        ("DeepSeek-V3 MLA(kv潜512+rope64)", (512 + 64) * 2),
    ]
    name_w = max(len(n) for n, _ in rows)
    print(f"    {'配置'.ljust(name_w)}{'B/层/token':>10}{'相对MHA':>9}")
    base_b = 2 * 128 * 128 * 2
    for nm, b in rows:
        print(f"    {nm.ljust(name_w)}{b:>10}{base_b/b:>7.1f}×省")
    print("    → V2/V3 MLA 用 512+64 维潜向量扛住 2 万维级 KV 全量：缓存/内存省 50~60 倍，")
    print("      长上下文增量从『每 token 32 KiB 级』降到『1.1 KiB 级』（128k 省掉 4 GiB 级）。")
    print("  D2 · 激活参数口径（每 token 真正过参数的算力）：")
    print("       DeepSeek-V3：671B 总参 / ~37B 激活 ≈ 5.5%（8 路由专家 + 1 共享 always-on）")
    print("       Mixtral-8x7B：47B 总参 / 13B 激活 ≈ 27%（2/8 路由，无共享）")
    print("       → 细粒度专家把每专家做窄 + Top-k 再砍一刀：总参买容量、激活买算力——")
    print("         R1 的『671B 长思考』正是这套超大总参/低激活在推理成本上的兑现。")
    print("  D3 · R1/R1-Zero 公开指标（非本机实测；运行环境无外网、写作时未在线复核，"
          "精确值以原论文为准）：")
    print("       R1-Zero AIME2024 71.0%（纯 RL、无 SFT，涌现反思）→")
    print("       R1 AIME2024 79.8% · MATH-500 97.3% · R1-Distill 把思考蒸馏进 1.5B/7B/32B/70B")
    print("  D4 · R1-Distill 蒸馏账：671B fp8 权重 ≈ 671 GB；四个小模型加起来不到它的零头，")
    print("       AIME 从 71%→70B 70%+ 只损 ~10 个点——『思考能力』第一次变成可下载的小文件。")
    now("expD 完成")


# ================================================================ main
def main():
    t0 = time.perf_counter()
    print("=" * 66)
    print("03-模型家族 04-DeepSeek系列 · 技术创新发动机：MLA 低秩 × MoE 极致 × RL 涌现")
    print("任务族：[A] add ICL（y=(x+a) mod 13）· [B] 多主题英文语料 · [C] 算术合成 + RL")
    print("引擎族：[A] 四件套 Engine 继承改造 MLA · [B] 02-18 MoE + 共享/bias · [C] 02-16 基座")
    print("复习链：02-04 MLA · 02-18 MoE · 02-16 GRPO（PLAN §2.3 指定）")
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
    print("结论速写：MLA 把 KV 缓存按『潜向量 lat 维』记账、K 谱被锁进 lat 维子空间（[A]）；")
    print("DeepSeekMoE 用细粒度专家+共享专家+bias 均衡三件套把载荷方差连压两档（[B]）；")
    print("R1-Zero 缺冷启动在玩具规模纯 RL 卡住、R1 用冷启动搭梯子、R1-Distill 直接蒸馏（[C]）；")
    print("V2/V3 的 MLA 在 KV 账上省 50 倍级、V3 总参/激活 5.5% 买推理经济学（[D]）。")
    print("=" * 66)


if __name__ == "__main__":
    main()
