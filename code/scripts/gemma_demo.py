# -*- coding: utf-8 -*-
"""03-模型家族 06-Gemma系列 · 「端侧是另一本账」+ 交替注意力的全局锚点放对位置
任务族：y=(a+x) mod P（add，ICL），引擎 = 02-Llama 同款 2-block 解码器（C=48 NH=8 NV=4）
A 交替注意力布局四臂（复习 02-17 长上下文 / 05-Mistral SWA）：
   同 1200 步量级预算，掩码逐层可换（本期给 llama_demo 补每层 masks）——
   GG=全全局 / LL=全滑窗 W3（末位 xq 回望只容 (x5,y5) 一对完整演示 + xq 自身 / 窗内对 1）/
   LG=底层滑窗 W3+顶层全局 / GL=底层全局+顶层滑窗 W3（哪一侧更需要全局锚，交给测试裁决）
   三遍逐位一致；每布局再测「推理时改窗」全窗口径（SWA 是训练期决策的交叉复核）。
B 派生账（公开配置换算，非本机实测）：交替注意力打分对数账（真实 Gemma2 尺度）·
   端侧装载账（手机 4 GiB / 桌面 8 / 40 GiB × fp16 与 int4）· 256k 大词表 embedding
   占比账（小模型 1B 逼近 1/3，端侧第二本账）。
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
import llama_demo as L

np.set_printoptions(precision=4, suppress=True)
_START = time.perf_counter()


def now(msg):
    sys.stderr.write(f"[{time.perf_counter()-_START:6.1f}s] {msg}\n")


def band_mask(T, W):
    """滑窗因果掩码：m[t,j]=1 iff 0 ≤ t−j < W；W≥T 即退化 full 因果。"""
    idx = np.arange(T)
    m = idx[:, None] - idx[None, :]
    return ((m >= 0) & (m < W)).astype(float)


def full_mask(T):
    return np.tril(np.ones((T, T)))


def pair_visible(T, W):
    """t=T−1（预测位置）的滑窗里能看见的『完整 (x_i,y_i) 演示对』个数（同 05 篇口径）。"""
    lo = max(0, T - W)
    n = 0
    for i in range(T):
        if (2 * i) >= lo and (2 * i + 1) < T - 1:
            n += 1
    return n


def sp_cost(T, W):
    """滑窗打分对数 = Σ_t min(t+1, W)（self 含在对角线；W≥T 退 full）。"""
    W = min(W, T)
    return W * (W + 1) // 2 + W * (T - W)


# ===========================================================================
# [A] 交替注意力布局四臂（复习 02-17 / 05-SWA）
# ===========================================================================
def train_layout(layout, steps, bs, seed_tr=1234):
    """layout = ["full", "W3"] 等；返回 (eng, p, 终CE)。"""
    eng = L.Engine(V=13, T=11, C=48, NL=2, NH=8, NV=4)
    eng.masks = [db("full" if m == "full" else f"W{int(m[1:])}") for m in layout]
    p = eng.init_params(7 + eng.C)
    o = L.init_adam(p)
    srng = np.random.RandomState(seed_tr)
    for s in range(1, steps + 1):
        tok, tgt, lm = L.build_batch(srng, 13, bs, eng.T)
        loss, c = eng.mask_loss(p, tok, tgt, lm)
        g = eng.bwd(p, c, tgt, lm)
        L.adam_step(p, o, g, 3e-3 * min(1.0, s / 150))
    return eng, p, float(loss)


def db(spec):
    """布局 spec 字符 → 掩码矩阵。full → 因果全连通；W* → 滑窗 W。"""
    T = 11
    return full_mask(T) if spec == "full" else band_mask(T, int(spec[1:]))


def switch_masks(eng, specs, T=11):
    eng.masks = [db(s) for s in specs]


def fd_check_masks():
    """per-layer mask 路径引擎对账：选 LG（逐层掩码不同）bwd vs 中心差分。"""
    print("    per-layer mask 路径对账：LG 布局（底层滑窗 W3 · 顶层全局）bwd vs 中心差分")
    eng = L.Engine(V=13, T=11, C=48, NL=2, NH=8, NV=4)
    switch_masks(eng, ["W3", "full"])
    p = eng.init_params(55)
    srng = np.random.RandomState(5)
    tok, tgt, lm = L.build_batch(srng, 13, 4, 11)
    g = eng.bwd(p, eng.fwd(tok, p), tgt, lm)
    rndj = np.random.RandomState(0)
    worst = 0.0
    for k in ("Wq0", "Wq1", "Wf10", "Wf21", "Wte", "Wout"):
        mx = 0.0
        r0, r1 = g[k].shape
        for _ in range(20):
            a = rndj.randint(0, r0); b = rndj.randint(0, r1)
            eps = 1e-5
            p[k][a, b] += eps
            l2, _ = eng.mask_loss(p, tok, tgt, lm)
            p[k][a, b] -= 2 * eps
            l1, _ = eng.mask_loss(p, tok, tgt, lm)
            p[k][a, b] += eps
            mx = max(mx, abs((l2 - l1) / (2 * eps) - g[k][a, b]))
        worst = max(worst, mx)
        print(f"      {k:<5} maxerr={mx:.3e}")
    print(f"      6 组全参 maxerr≤{worst:.3e}（<1e-5 ✅）")
    return worst


def expA():
    print("=" * 66)
    print("[A] 交替注意力布局四臂（复习 02-17 长上下文 / 05-SWA 回望半径）")
    print("    引擎 = 02-Llama 同款 2-block add-ICL（C=48 NH=8 NV=4）· 本期给引擎补『每层掩码』")
    print("    W=3 滑窗：末位 xq 的回望只容 (x5,y5) 一对完整演示 + xq 自身（窗内完整演示对=1）；")
    print("    mod-13 加法下 1 对即定出 a（可辨识）——但『读全场』的校准取决于哪层是全局：")
    print("=" * 66)
    _ = fd_check_masks()
    STEPS, BS = 1000, 96
    print()
    print(f"    四布局同预算 1000 步×bs96（seed_tr=1234 · init=7+C）：")
    print(f"    {'布局':<6}{'底层掩码':<8}{'顶层掩码':<8}{'终CE':>8}  fresh@各布局  fresh@全窗    fresh@全滑窗")
    layouts = {
        "GG": ["full", "full"],
        "LL": ["W3", "W3"],
        "LG": ["W3", "full"],
        "GL": ["full", "W3"],
    }
    for name, layout in layouts.items():
        t0 = time.perf_counter()
        eng, p, ce = train_layout(layout, STEPS, BS)
        dt = time.perf_counter() - t0
        # fresh@训练布局原样
        fr_self = L.eval_fresh(eng, p, np.random.RandomState(509), 13, n=400, S=0)
        # 推理时改窗：全放开 vs 全滑窗 W3
        switch_masks(eng, ["full", "full"])
        fr_full = L.eval_fresh(eng, p, np.random.RandomState(509), 13, n=400, S=0)
        switch_masks(eng, ["W3", "W3"])
        fr_swa = L.eval_fresh(eng, p, np.random.RandomState(509), 13, n=400, S=0)
        switch_masks(eng, layout)   # 还原
        print(f"    {name:<6}{layout[0]:<8}{layout[1]:<8}{ce:>8.3f}  {fr_self:7.3f}  {fr_full:7.3f}    {fr_swa:7.3f}")
        now(f"[A] {name} 训练 {STEPS} 步 wall {dt:.0f}s（stderr）")
    print("    → 信息上 (x5,y5) 一对即定出 a → 四布局终局全解锁（fresh@各布局 0.975-1.000）：")
    print("      布局决定的是『学习速度 × 窗口耐受』，不是能否学会。")
    print("      05『训练期锁窗』逐层化：推理放宽全窗必 OOD，且随全局层数单调——")
    print("      LL 0.130 < LG 0.203 < GL 0.485 < GG 1.000；反向收窄（GG→全滑窗）同样 OOD 0.290。")
    print("      全局锚放底层(GL)比放顶层(LG)耐受更强：底层看全局→放宽不动源表征、")
    print("      只动顶层汇总；LG 的源表征（底层滑窗）一放宽先被推歪。Gemma2 把全局层")
    print("      散布在层间并非常规『顶层汇总』，而是给局部层垫『敢做精炼』的地基。")
    now("expA 完成")


# ===========================================================================
# [B] 派生账（公开配置换算，非本机实测）
# ===========================================================================
def expB():
    print("=" * 66)
    print("[B] 派生账（公开配置 × 公式，非本机实测 · 未在线复核，概数以官方 release note 为准）")
    print("    ① 交替注意力打分对数：全部层与『2 成本全局层换长距离』")
    print("=" * 66)
    T, W, L, NG = 8192, 4096, 42, 8          # Gemma2-9B 量级：42 层、滑窗 4k、约每 5 层 1 全局
    full = T * (T + 1) // 2
    swa = sp_cost(T, W)
    n_l = L - NG
    mix = n_l * swa + NG * full
    ans_all_full = L * full
    ans_all_swa = L * swa
    print(f"  Gemma2-9B 量级：T={T}(8k) · 滑窗 W={W} · L={L} 层 · 全局 {NG} 层 / 局部 {n_l} 层")
    print(f"    全全局打分对数 = {ans_all_full:,}（=100%）")
    print(f"    {NG} 全局+{n_l} 滑窗交替 = {mix:,} = 交替/全全局 {mix/ans_all_full:.1%}（省 {1-mix/ans_all_full:.0%}）")
    print(f"    全滑窗打分对数 = {ans_all_swa:,} = 交替/全滑窗 {mix/ans_all_swa:.1%}（贵 {mix/ans_all_swa-1:.0%}）")
    print("    → 用 ~2 成的全局层把『长距离通路』买回来，账面上只比全滑窗贵 6%——")
    print("      交替注意力是『滑窗的省 × 全局的能力』的折中形态（本玩具 LG 布局的尺度版本）。")
    print()
    print("  ② 端侧装载账：三档设备 × fp16 / int4（fp16≈2B/参数，int4≈0.5B/参数，不含 KV/激活）")
    print(f"    {'模型':<12}{'参量':>6}{'fp16':>8}{'int4':>8}{'手4GiB':>7}{'桌8GiB':>7}{'桌40GiB':>8}")
    models = [
        ("Gemma3-1B",  1.1),
        ("Gemma2-2B",  2.6),
        ("Gemma3-4B",  4.0),
        ("Gemma2-9B",  9.2),
        ("Gemma3-12B", 12.0),
        ("Gemma3-27B", 27.0),
    ]
    GiB = lambda b: b / 2**30          # B → GiB（2^30）
    for name, prm in models:
        f16 = GiB(prm * 1e9 * 2)       # fp16：每参 2 字节
        i4 = GiB(prm * 1e9 * 0.5)      # int4：每参 4bit = 0.5 字节
        phone = "f16✓" if f16 <= 4 else ("int4✓" if i4 <= 4 else "✗")
        desk8 = "f16✓" if f16 <= 8 else ("int4✓" if i4 <= 8 else "✗")
        desk40 = "f16✓" if f16 <= 40 else ("int4✓" if i4 <= 40 else "✗")
        print(f"    {name:<12}{prm:>5}B{f16:>9.1f}{i4:>9.1f}{phone:>6}{desk8:>6}{desk40:>8}")
    print("    → 端侧第一性约束=内存/带宽：fp16 连 9B 都塞不进 8 GiB 桌，int4 是端侧的『默认入场券』；")
    print("      评判标准从『装多大的模型』翻到『单位算力下的质量』——这正是 Gemma 做质量参考系的原因。")
    print()
    print("  ③ 大词表 embedding 占比账：256k 词表在小模型里的驻留成本被放大")
    vs = [
        ("Gemma3-1B",   256000, 1536, 1.1e9),
        ("Gemma2-9B",   256000, 3584, 9.2e9),
        ("Gemma2-27B",  256000, 5376, 27e9),
        ("Llama3.2-3B", 128256, 3072, 3.2e9),    # 128k 词表对照
        ("Qwen2.5-1.5B",151936, 1536, 1.5e9),    # 152k 词表对照
    ]
    print(f"    {'模型':<13}{'词表':>9}{'hidden':>8}{'embedding 参数':>16}{'占总参':>9}{'fp16 GiB':>10}")
    for name, V, d, tot in vs:
        ep = V * d
        print(f"    {name:<13}{V:>9,}{d:>8}{ep:>16,}{ep/tot:>8.1%}{GiB(ep*2):>10.2f}")
    print("    → 同‘压缩 token 数’的词表红利，对小模型是『驻留负债』：Gemma3-1B 的 256k 词表")
    print("      embedding ≈0.73 GiB（≈整套 fp16 权重 2.0 GiB 的 1/3），9B/27B 摊到 5-10%——")
    print("      词表这账是固定一把，参数涨了才被稀释；端侧拷权重要连 embedding 一起进内存。")
    now("expB 完成")


def main():
    t0 = time.perf_counter()
    print("=" * 66)
    print("03-模型家族 06-Gemma系列 · 「端侧是另一本账」——交替注意力的全局锚点放对位置")
    print("任务族：y=(a+x) mod P（add，ICL）——每片段新随机 a，模型必须现场从示例读映射")
    print("引擎：02-Llama 同款 2-block 解码器（C=48 · RoPE/SwiGLU/RMSNorm/GQA 8头/4kv）")
    print("=" * 66)
    expA()
    print()
    expB()
    print()
    now(f"全脚本累计 {time.perf_counter()-t0:.1f}s")
    print()
    print("done · 一键复现：python code/scripts/gemma_demo.py")


if __name__ == "__main__":
    main()
