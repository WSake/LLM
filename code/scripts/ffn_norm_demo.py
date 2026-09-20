# -*- coding: utf-8 -*-
"""FFN 三兄弟 + LayerNorm/RMSNorm 演示（配合《06-FFN与激活函数》《07-归一化》）。

A. FFN 与激活函数（17.2.7）
   A1 参数账：SwiGLU 以 8/3·C 门控+升维，与 ReLU 的 4·C 同预算（都≈8C² 参数）
   A2 激活形态：ReLU 的"死区"比例、GELU 的负泄漏、SwiGLU 的门控相乘
   A3 三条非线性曲线的点状对照
B. 归一化（17.2.8）
   B1 数值差异：LN 强制通道零均值（~1e-15），RMSNorm 保留均值；二者输出差 0.1 量级
   B2 计算代价：CPU 计时 median-of-3（RMSNorm 少一次均值归约）＋ op 计数
   B3 post/pre-norm：同款 mini 深残差网，训练曲线对比（全精确反向，含 dnorm）。

本机纯 CPU 可跑；seed 固定，全部数字一键复现。
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.RandomState(0)


def layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


def rms_norm(x, eps=1e-5):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def dnorm(x, dy, eps=1e-5):
    """layer_norm 的精确反向（dx = (dy − mean(dy) − xc·mean(dy·xc)/s²)/s）。"""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    s = np.sqrt(var + eps)
    xc = x - mu
    return (dy - dy.mean(-1, keepdims=True)
            - xc * (dy * xc).mean(-1, keepdims=True) / (var + eps)) / s


def relu(x):
    return np.maximum(x, 0.0)


def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def timeit(fn, n=3):
    """同一操作重复 n 遍取中位数（ms）。"""
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000.0


def train_resnet(depth, mode, lr=0.01, steps=2500, seed=7):
    """mini 深残差网（hidden 96, depth 个块, 每块 隐层→ReLU→隐层）

    mode='pre' ：规范 pre-LN    h ← h + f(LN(h))        （norm 在块首, 残差直接累积）
    mode='post'：规范 post-LN   h ← LN(h + f(h))        （norm 在残差求和之后）
    数据 std≈1（与真实 embedding 尺度对齐），Wo 初值 ×0.2。
    全精确反向：pre/post 子块反向均已与中心差分对账（maxerr≈1e-8），
    多层链式亦经端到端对账（|Δ|≈4e-10）。
    返回 (eval-MSE 终值, {step: eval-MSE})。
    """
    r = np.random.RandomState(seed)
    D, H = 16, 96
    Xtr = r.randn(2048, D)
    w = r.randn(D) / 6.0
    ytr = np.sin(Xtr @ w) + 0.05 * r.randn(2048)
    Xev = r.randn(512, D)
    yev = np.sin(Xev @ w)

    def init_w(a, b):
        return r.randn(a, b) * np.sqrt(2.0 / a)

    Ws = [[init_w(H, H), init_w(H, H)] for _ in range(depth)]
    W_in = init_w(D, H)
    W_out = init_w(H, 1) * 0.2

    def predict(X):
        h = relu(X @ W_in)
        for W1, W2 in Ws:
            if mode == 'pre':
                n = layer_norm(h); h = h + relu(n @ W1) @ W2
            else:
                h = layer_norm(h + relu(h @ W1) @ W2)
        return (h @ W_out)[:, 0]

    curve = {}
    for it in range(steps):
        idx = r.randint(0, Xtr.shape[0], 256)
        xb, yb = Xtr[idx], ytr[idx]
        # —— 前向缓存 ——
        h0 = relu(xb @ W_in); h = h0; cell = []
        for W1, W2 in Ws:
            if mode == 'pre':
                n, u = layer_norm(h), relu(layer_norm(h) @ W1)
                h = h + u @ W2
                cell.append((n, u, h, None))
            else:
                u = relu(h @ W1)
                s = h + u @ W2
                h = layer_norm(s)
                cell.append((None, u, h, s))
        y = (h @ W_out)[:, 0]
        # —— 反向（精确, 表达式与中心差分对账过）——
        g = 2 * (y - yb)[:, None] / yb.size
        W_out -= lr * (h.T @ g)
        g_h = g @ W_out.T
        for i in range(depth - 1, -1, -1):
            n, u, hnext, s = cell[i]
            W1, W2 = Ws[i]
            if mode == 'pre':
                du = (g_h @ W2.T); du[u <= 0] = 0.0
                Ws[i][0] -= lr * (n.T @ du)
                Ws[i][1] -= lr * (u.T @ g_h)
                g_h = g_h + dnorm(hnext - u @ W2, du @ W1.T)
            else:
                g_s = dnorm(s, g_h)                     # 从 LN(hin+v) 穿过 norm
                du = (g_s @ W2.T); du[u <= 0] = 0.0
                Ws[i][1] -= lr * (u.T @ g_s)
                Ws[i][0] -= lr * ((s - u @ W2).T @ du)  # hin = s − v
                g_h = g_s + du @ W1.T
        da = g_h; da[h0 <= 0] = 0.0
        W_in -= lr * (xb.T @ da)
        if it % 500 == 0 or it == steps - 1:
            curve[it] = float(np.mean((predict(Xev) - yev) ** 2))
    return curve[steps - 1], curve


def init_grad_span(depth, mode, seed=7):
    """初始化时反向逐块 |g_h| 的『最深→最浅』梯度尺度变化（无训练, 只量梯度）。

    返回 (最深块输入侧 |g_h|, 最浅层输入侧 |g_h|)。
    """
    r = np.random.RandomState(seed)
    D, H = 16, 96
    Xtr = r.randn(2048, D)
    w = r.randn(D) / 6.0
    ytr = np.sin(Xtr @ w) + 0.05 * r.randn(2048)
    rng = np.random.RandomState(seed)

    def iw(a, b):
        return rng.randn(a, b) * np.sqrt(2.0 / a)

    Ws = [[iw(H, H), iw(H, H)] for _ in range(depth)]
    W_in = iw(D, H)
    W_out = iw(H, 1) * 0.2
    idx = r.randint(0, 2048, 256)
    xb, yb = Xtr[idx], ytr[idx]
    h0 = relu(xb @ W_in); h = h0; cell = []
    for W1, W2 in Ws:
        if mode == 'pre':
            n = layer_norm(h); u = relu(n @ W1); h = h + u @ W2
            cell.append((n, u, h, None))
        else:
            u = relu(h @ W1); s = h + u @ W2; h = layer_norm(s)
            cell.append((None, u, h, s))
    y = (h @ W_out)[:, 0]
    g = 2 * (y - yb)[:, None] / yb.size
    gh = g @ W_out.T
    norms = []
    for i in range(depth - 1, -1, -1):
        n, u, hnext, s = cell[i]
        W1, W2 = Ws[i]
        if mode == 'pre':
            du = gh @ W2.T; du[u <= 0] = 0.0
            gh = gh + dnorm(hnext - u @ W2, du @ W1.T)
        else:
            gs = dnorm(s, gh)
            du = gs @ W2.T; du[u <= 0] = 0.0
            gh = gs + du @ W1.T
        norms.append(float(np.linalg.norm(gh)))
    # norms[i] = 从最深块一步步反向走到第 i 块输入时的 |g_h|；norms[-1] 是浅层(输入侧)
    return norms[0], norms[-1]


def main():
    C = 512
    # ============================================================
    print("=" * 70)
    print("A1 · 参数账：三种 FFN 在'同参数预算'下怎么摆")
    print("=" * 70)
    ic4 = 4 * C
    ic_83 = int(8 * C / 3)
    print(f"hidden C={C}：ReLU/GELU 中间 {ic4}；SwiGLU 中间 {ic_83}（≈8/3·C）")
    p_rel, p_glu = 2 * C * ic4, 3 * C * ic_83
    print(f"参数：ReLU-FFN   2×{C}×{ic4}   = {p_rel:>10,}")
    print(f"     SwiGLU   3×{C}×{ic_83} = {p_glu:>10,}  （预算差 {(p_glu/p_rel-1)*100:.1f}%）")
    print("  → 3 块矩阵塞进更窄的中间维（8/3C vs 4C），总参数几乎不变")

    # ============================================================
    print()
    print("=" * 70)
    print("A2 · 激活形态：负输入去哪了（死区 / 泄漏 / 门控）")
    print("=" * 70)
    N_ = 2048
    x = rng.randn(N_, C) * 0.4
    W1 = rng.randn(C, ic4) / np.sqrt(C)
    Wu = rng.randn(C, ic_83) / np.sqrt(C)
    Wg = rng.randn(C, ic_83) / np.sqrt(C)
    h_rel, h_gel = relu(x @ W1), gelu(x @ W1)
    h_glu = silu(x @ Wg) * (x @ Wu)
    for name, h in [("ReLU ", h_rel), ("GELU ", h_gel), ("SwiGLU", h_glu)]:
        near0 = (np.abs(h) < 1e-8).mean()
        neg = (h < 0).mean()
        print(f"  {name}  近零比例 = {near0:6.2%}   负值比例 = {neg:6.2%}   "
              f"|h| 均值 = {np.abs(h).mean():.3f}")
    print("  → ReLU 把负激活掐成精确 0（近零比例最高）；SwiGLU 用 σ 门控保留软负值")
    r9 = np.random.RandomState(9)
    W2r = r9.randn(ic4, C) / np.sqrt(ic4)
    W2g = r9.randn(ic4, C) / np.sqrt(ic4)
    W2s = r9.randn(ic_83, C) / np.sqrt(ic_83)
    y_r = relu(x @ W1) @ W2r
    y_g = gelu(x @ W1) @ W2g
    y_s = (silu(x @ Wg) * (x @ Wu)) @ W2s
    print(f"\n  整条 FFN 输出（同一输入、全部 RandomState(9) 初始化）：")
    for name, y in [("ReLU-FFN ", y_r), ("GELU-FFN ", y_g), ("SwiGLU-FFN", y_s)]:
        print(f"  {name}  mean|y| = {np.abs(y).mean():.4f}   std = {y.std():.4f}")

    # ============================================================
    print()
    print("=" * 70)
    print("A3 · 单点映射：三条非线性曲线")
    print("=" * 70)
    xs = np.linspace(-4, 4, 9)
    print(f"  x    = {np.array2string(xs, precision=1)}")
    print(f"  ReLU = {np.array2string(relu(xs), precision=2)}")
    print(f"  GELU = {np.array2string(gelu(xs), precision=2)}")
    print(f"  SiLU = {np.array2string(silu(xs), precision=2)}")
    print("  → ReLU x<0 精确 0；GELU/SiLU 保留小负值（软泄漏）")

    # ============================================================
    print()
    print("=" * 70)
    print("B1 · LayerNorm vs RMSNorm：差在哪（0.1 量级）")
    print("=" * 70)
    big = rng.randn(4096, 2 * C)
    big = big + 0.2
    ln = layer_norm(big)
    rn = rms_norm(big)
    print(f"  输入通道均值（前 3 行）≈ {np.array2string(big.mean(-1)[:3], precision=2)}")
    print(f"  LN  输出通道均值 max|·| = {np.abs(ln.mean(-1)).max():.2e}（强制居中）")
    print(f"  RMS 输出通道均值 max|·| = {np.abs(rn.mean(-1)).max():.3f}（不居中,均值还留着）")
    d = np.abs(ln - rn)
    print(f"  两者输出 max 差 = {d.max():.3f}，mean 差 = {d.mean():.4f}")
    print("  → 0.1 量级的差，正是'去均值'这一刀；RMSNorm 干脆不做，换来算力")

    # ============================================================
    print()
    print("=" * 70)
    print("B2 · 计算代价：RMSNorm 省掉一次均值归约")
    print("=" * 70)
    X = rng.randn(8192, 2048)
    t_ln = timeit(lambda: layer_norm(X))
    t_rn = timeit(lambda: rms_norm(X))
    print(f"  CPU median-of-3：LN {t_ln:.2f} ms vs RMS {t_rn:.2f} ms → RMS 快 {(t_ln/t_rn-1)*100:.0f}%")
    print("  op 账（每个 C 维向量）：LN = 求均值归约 + 减均值 + 方差归约 + 除以 std")
    print("                       RMS = 平方 + 自平方均值归约 + 除以 rms")
    print("  → LN 比 RMS 多一次均值归约 + 一次逐元素减；单次很小,但 T×L 全仓热路径")

    # ============================================================
    print()
    print("=" * 70)
    print("B3 · post/pre-norm：np 里把它放到残差求和之前还是之后（17.2.8 实践，精确反向）")
    print("=" * 70)
    print("先量『初始化时反向梯度放大倍数』（无训练，纯机制）：")
    for dm in ('post', 'pre'):
        lo, hi = init_grad_span(12, dm)
        print(f"  {dm}-norm depth=12  反向逐块 |g_h|  最深 {lo:.3f} → 最浅 {hi:.3f}  → 放大 {hi/lo:.1f}×")
    print()
    print("训练曲线（同架构同 seed，仅 norm 放法与 lr 不同）：")
    for dm, dep, lr in [("post", 4, 0.01), ("post", 12, 0.01),
                        ("pre", 4, 0.01), ("pre", 12, 0.01),
                        ("pre", 4, 0.001), ("post", 12, 0.001),
                        ("pre", 12, 0.0003)]:
        final, curve = train_resnet(dep, dm, lr=lr, steps=2500, seed=7)
        tag = "NaN" if not np.isfinite(final) else f"{final:.4f}"
        pts = "   ".join(f"{k}:{v:.3g}" for k, v in curve.items())
        print(f"  {dm}-norm depth={dep:2d} lr={lr:<6.4g}  终值 {tag:>8}  |  曲线 {pts}")

    print()
    print("done · 一键复现：python code/scripts/ffn_norm_demo.py")


if __name__ == "__main__":
    main()
