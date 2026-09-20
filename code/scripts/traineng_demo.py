# -*- coding: utf-8 -*-
"""
traineng_demo.py —— 04-训练体系 09-训练工程（§17.3.9）的确定性账本 + 断点续训实证
================================================================================
承接 06/07/08：06 给了 78.2 GiB 四件套账（权重13.0+梯度13.0+Adam 52.2）、07 给了七种
切法、08 给了框架选择。本篇补"训练工程的纪律"——显存预算单 / 精度 / checkpoint / 日志：

  [A] 显存五件套预算单：权重+梯度+优化器+激活+通信缓冲
      全参 7B vs LoRA r=16 7B 各一张单（账算，非本机实测）
  [B] 精度三角账：BF16/FP16/FP32/FP8 规格 + 小梯度落入次正规数的损失缩放演示
      （fp16/fp32 用 np.finfo 机器校验、bf16 用 torch.finfo 机器校验；fp8 为公开规格常数）
  [C] checkpoint 断点续训实证（本机真跑）：torch CPU 上真训练一个小字符串语言模型，
      模拟"step K 崩溃 → 从存档续训"，四路对照：
        ① 全量存档（模型+优化器+RNG）续训 vs 无中断参考 → 逐位一致（无损）
        ② 只丢优化器状态（RNG 在）续训 → 第一步更新就分叉（Δ_opt）
        ③ 只丢 RNG（优化器在）续训 → 第一步 batch/丢层就错位（Δ_rng）
      再讲"没有存档的崩溃成本"：等于把崩溃前的功课重付一遍
  [D] 日志诊断器：给一段 loss 曲线判"健康/死平/发散"（规则，账算非实测）
      + 监测仪表盘 + 关键参数经验值表

除 [C] 外均为确定性账算/规格常数/规则，可复现；[C] 为本机真实训练——
torch CPU 单线程、固定 seed、固定数据顺序（含每 epoch randperm 与 dropout，
都用全局 RNG，凑齐"存档必须含 RNG"的教学条件），科学数字逐位复现，
只有墙钟在跑次间浮动。整脚本纯 CPU，墙钟 <15 s（I/O 与训练主导）。
"""
import sys, time, os, tempfile, math

if sys.stdout.encoding and sys.stdout.encoding.lower() in ("gbk", "gb2312"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

GIB = 2 ** 30
t0 = time.perf_counter()

print("=" * 66)
print("traineng_demo：训练工程的纪律（04 训练体系 §17.3.9）——预算单·精度·断点续训·日志")
print("=" * 66)
print("[0] 口径：承接 06 章 4 件套账（78.2 GiB）与 07 章切法；本章补运行时件（激活/通信）"
      "与工程纪律。[A][B][D] = 账算/规格常数/规则，标注'非本机实测'；"
      "[C] 段在本机 CPU 上真训练一个小 LM 做断点续训对照 = 本机真跑。")

# ---------------------------------------------------------------------------
# [A] 显存五件套预算单（账算，非本机实测）
# ---------------------------------------------------------------------------
print("\n[A] 显存五件套（权重 / 梯度 / 优化器状态 / 激活 / 通信缓冲，§4.9）预算单（账算，非本机实测）")
N7 = 7e9
FP16B, FP32B = 2, 4
w7 = N7 * FP16B / GIB                 # 权重 fp16
g7 = N7 * FP16B / GIB                 # 梯度 fp16
o7 = N7 * FP32B * 2 / GIB             # Adam 一阶+二阶（fp32 各一份）
static7 = w7 + g7 + o7                # 静态三件套
B_, S_, d7, L7, H7 = 4, 2048, 4096, 32, 32
act7 = B_ * S_ * d7 * 2 * 2 * L7 / GIB           # 激活(激活检查点开)：每层驻留 ~2 份 block 输入
scores_off = B_ * H7 * S_ * S_ * L7 * 2 / GIB    # 注意分数（检查点关）主导项
comm7 = g7                                        # 通信缓冲：DDP 一次全参梯度
total7 = static7 + act7 + comm7
# LoRA r=16 全挂 q/k/v/o+gate/up/down（7 个线性），7B 可训练约 30M 参数
LORA_PARAMS = 30e6
lw = LORA_PARAMS * FP16B / GIB
lg = LORA_PARAMS * FP16B / GIB
lo = LORA_PARAMS * FP32B * 2 / GIB
lc = LORA_PARAMS * FP16B / GIB
lora_train = lw + lg + lo + lc
total_lora = w7 + lora_train + act7
save_ratio = (1 - total_lora / total7) * 100

assert abs(static7 - 78.2) < 0.1, f"static7={static7:.2f} 应≈78.2"
assert abs(act7 - 4.0) < 1e-9
assert abs(scores_off - 32.0) < 1e-6
print(f"    7B 全参（40 GiB 桌 · B=4 · S=2048）")
print(f"      权重  fp16            {w7:8.1f} GiB")
print(f"      梯度  fp16            {g7:8.1f} GiB")
print(f"      优化器 Adam fp32×2    {o7:8.1f} GiB  ← 静态账 78.2 里的 2/3")
print(f"      激活(检查点开)·B·S 账  {act7:8.1f} GiB  （注意分数若不复算≈{scores_off:.0f} GiB，8×）")
print(f"      通信缓冲 DDP 梯度一份  {comm7:8.1f} GiB")
print(f"      五件套合计            {total7:8.1f} GiB  → 40 GiB 桌单卡放不下；78.2 只是'静态'口径")
print(f"    7B LoRA r=16（冻结主权重，可训练约 {LORA_PARAMS/1e6:.0f}M 参数）")
print(f"      冻结主权重 fp16       {w7:8.1f} GiB")
print(f"      LoRA 可训练四件小计    {lora_train:8.2f} GiB （w {lw:.2f}+g {lg:.2f}+Adam {lo:.2f}+通信 {lc:.2f}）")
print(f"      激活(不省，随前向形状) {act7:8.1f} GiB")
print(f"      合计                  {total_lora:8.1f} GiB  → 40 GiB 桌✓ 余量 ~{40-total_lora:.0f} GiB")
print(f"    读法：LoRA 把五件套从 {total7:.1f} 压到 {total_lora:.1f} GiB，省 {save_ratio:.1f}% ——"
      f"省的件全在'随可训练参数量'的梯度/优化器/通信，激活 {act7:.1f} GiB 一分不少；")
print(f"          序列 S 再翻倍，激活往上涨，长上下文 + LoRA 也要盯激活。")

# ---------------------------------------------------------------------------
# [B] 精度三角账：IEEE 规格（fp16/fp32 用 np.finfo、bf16 用 torch.finfo 机器校验）
# ---------------------------------------------------------------------------
print("\n[B] 精度三角账：BF16/FP16/FP32/FP8 规格（IEEE 常数；fp16/fp32→np.finfo、bf16→torch.finfo 均为本机校验）")
f16, f32 = np.finfo(np.float16), np.finfo(np.float32)
bf = torch.finfo(torch.bfloat16)
assert float(f16.max) == 65504.0 and abs(float(f16.eps) - 2**-10) < 1e-12 and abs(float(f16.tiny) - 2**-14) < 1e-12
assert abs(float(f32.tiny) - 2**-126) < 1e-45 and abs(float(f32.eps) - 2**-23) < 1e-20
assert abs(float(bf.eps) - 2**-7) < 1e-12 and abs(float(bf.tiny) - 2**-126) < 1e-45 and float(bf.max) > 3e38
spec_lines = [
    ("FP32", 8, 23, f32.max, f32.tiny, f32.eps, 4, "[参考] 训练早期/数值敏感处"),
    ("FP16", 5, 10, f16.max, f16.tiny, f16.eps, 2, "范围窄·尾数宽→需要 loss 缩放"),
    ("BF16", 8, 7, bf.max, bf.tiny, bf.eps, 2, "[主力] 范围同 FP32·免缩放"),
    ("FP8-E4M3", 4, 3, 448.0, 2**-6, 2**-3, 1, "前向/激活（DeepSeek-V3 示范）"),
    ("FP8-E5M2", 5, 2, 57344.0, 2**-14, 2**-1, 1, "反向/梯度·动态 per-tensor 缩放"),
]
hdr = "    " + " | ".join(["格式", "指数", "尾数", "最大", "最小正规", "eps", "每参B"])
print(hdr); print("    " + "-" * (len(hdr) - 4))
for name, e, m, mx, mn, eps, pb, note in spec_lines:
    print(f"    {name:<8} | {e:>2} | {m:>2} | {mx:>11.6g} | {mn:>11.6g} | {eps:>10.6g} | {pb:>3}  {note}")
# 7B 主权重各档
print(f"    7B 主权重单档显存：FP32 {N7*FP32B/GIB:>6.1f} GiB · FP16/BF16 {N7*FP16B/GIB:>6.1f} · FP8 {N7*1/GIB:>6.1f} GiB（账算）")
# 次正规数演示：为什么 FP16 要 loss 缩放
print("\n    loss 缩放演示（本机逐值换算，IEEE 正确舍入）：梯度 g=1e-7")
x = 1e-7
st16 = float(np.float16(np.float32(x)))                       # 直接进 FP16
scaled = float(np.float16(np.float32(x) * 256.0)) / 256.0      # ×2^8 缩放→换算→缩回
bf16v = float(torch.tensor(x, dtype=torch.bfloat16).float())   # 进 BF16
err16, errsc, errbf = abs(st16-x)/x, abs(scaled-x)/x, abs(bf16v-x)/x
print(f"      FP16 直接存 {st16:.3e}（次正规数，相对误差 {err16*100:.1f}%）")
print(f"      FP16 ×2^8 缩放存后还原 {scaled:.3e}（变正规数，相对误差 {errsc*100:.2f}%）")
print(f"      BF16 直接存 {bf16v:.3e}（仍在正规范围，相对误差 {errbf*100:.2f}%）")
assert err16 > 0.05 and errsc < 0.01 and errbf < 0.01
print("    → FP16 小梯度的位宽掉进次正规数，loss 缩放把它抬回正规范围；BF16 范围=FP32，免缩放。")

# ---------------------------------------------------------------------------
# [C] checkpoint 断点续训实证（本机真跑）
# ---------------------------------------------------------------------------
print("\n[C] checkpoint 断点续训实证（本机真跑：torch CPU 小语言模型，GRU-96，固定语料）")
torch.manual_seed(0)
torch.set_num_threads(1)

TEXT = ("大模型训练不只是把模型放进显卡跑起来，真正花时间的往往是那些看不见的功夫：能算清显存、"
        "会断点续训、看得懂训练日志。显存不是猜的，是一笔一笔算出来的。权重一份、梯度一份、"
        "优化器状态是大头，Adam会把每个参数再存两份高精度副本。激活也要占地方，序列越长越凶，"
        "所以长上下文训练要开激活检查点。通信缓冲跟着并行策略走，卡越多，每卡要同步的越少。"
        "精度要分清楚。BF16是训练主力，范围和FP32一样大，尾数少一点没关系。FP16范围窄，小梯度会"
        "掉进次正规数，所以要给损失乘一个缩放因子。FP8更省，但只有超大模型才值得，前向用E4M3，"
        "反向用E5M2。检查点是救命稻草，训练动辄卡好几天，一断电全没了。所以每隔一段距离就存一次档，"
        "把模型、优化器、步数、随机数种子全部存下来，崩溃了从最近的一档继续。日志要看得懂，loss一路"
        "往下走是健康，loss不动先怀疑学习率太低，loss发散那是学习率太高或者梯度爆炸。训练本身是一场"
        "漫长的修行，会算账、会存档、会看曲线，就是修行里最值钱的三件基本功。同一个学习率参数命不同"
        "的成群：预训练喜欢较大的学习率但数据海量，微调怕飞速遗忘要用小学习率，LoRA只管很小的子空间"
        "所以学习率又能给到1e-4数量级。随机种子要固定，分布式训练各路要种子一致，否则两份报告读不出"
        "同一个数字。梯度裁剪是最后的保险，等于给每一步的方向封一个上限。显存水位每步都看，batch再大"
        "也打不过硬件的墙。评估时丢掉dropout，让运气不要参与翻牌。训练日志最值得盯的不是某一步，而是"
        "这一段斜率，下降说明方向对，平台期说明要么换学习率要么换数据。半途崩溃不可怕，可怕的是存档"
        "偷懒，少存了一个随机状态，续训出来的曲线就和你以为的对不上。这些都做对了，剩下的就是耐心加"
        "算力，让死账本告诉你该等多久。")

chars = sorted(set(TEXT))
ch2i = {c: i for i, c in enumerate(chars)}
V = len(chars)
ids = [ch2i[c] for c in TEXT]
n = len(ids)
TT, STRIDE, BS = 48, 24, 12
windows, targets = [], []
for i in range(0, n - TT, STRIDE):
    windows.append(ids[i:i + TT])
    targets.append(ids[i + 1:i + TT + 1])
X = torch.tensor(windows, dtype=torch.long)
Y = torch.tensor(targets, dtype=torch.long)
Wn = len(windows)
STEPS_PER_EPOCH = (Wn + BS - 1) // BS
T_STEPS, K = 84, 36   # 总步数 84，崩溃点 36（=每 epoch 步数 3 × 12，恰在 epoch 边界）

class CharLM(nn.Module):
    def __init__(s, voc, d=96, p=0.4):
        super().__init__()
        s.emb = nn.Embedding(voc, d)
        s.gru = nn.GRU(d, d, batch_first=True)
        s.proj = nn.Linear(d, voc)
        s.drop = nn.Dropout(p)
    def forward(s, x):
        h, _ = s.gru(s.drop(s.emb(x)))
        return s.proj(s.drop(h))

def make_net():
    torch.manual_seed(0)
    return CharLM(V)

def make_opt(net):
    return torch.optim.Adam(net.parameters(), lr=5e-3)

def train_run(net, opt, a, b, seed=None, rng_state=None):
    """从全局步 a 训到 b。每 epoch 边界 randperm 全局 RNG；dropout 也取全局 RNG。
    断点续训语义：seed 是'新进程'的先验，rng_state 若给出则覆盖之。"""
    if seed is not None:
        torch.manual_seed(seed)
    if rng_state is not None:
        torch.set_rng_state(rng_state)
    order = None
    losses = []
    for s in range(a, b):
        if s % STEPS_PER_EPOCH == 0:
            order = torch.randperm(Wn)
        bx, by = X[order[:BS]], Y[order[:BS]]
        opt.zero_grad()
        logits = net(bx)
        loss = F.cross_entropy(logits.reshape(-1, V), by.reshape(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses

S0 = 12345
ct0 = time.perf_counter()
# ① 无中断参考（一条完整曲线）
netB = make_net(); optB = make_opt(netB)
base = train_run(netB, optB, 0, T_STEPS, seed=S0)
# ② 崩溃在 K：有全量存档则续训
netP = make_net(); optP = make_opt(netP)
pre = train_run(netP, optP, 0, K, seed=S0)          # 与 base[0:K] 应逐位一致
ckp = {"model": netP.state_dict(), "optim": optP.state_dict(),
       "step": K, "rng": torch.get_rng_state()}
ckpt_path = os.path.join(tempfile.gettempdir(), "ckpt_demo_kk.pt")
torch.save(ckp, ckpt_path)
loaded = torch.load(ckpt_path, weights_only=True)
netR = make_net(); optR = make_opt(netR)
netR.load_state_dict(loaded["model"]); optR.load_state_dict(loaded["optim"])
res = train_run(netR, optR, K, T_STEPS, seed=999, rng_state=loaded["rng"])
# ② 只丢"优化器状态"（恢复 RNG、重置优化器）：step36 更新前与无中断一致，更新后开始分叉
netO = make_net(); optO = make_opt(netO)
netO.load_state_dict(loaded["model"])
noo = train_run(netO, optO, K, K + 2, seed=999, rng_state=loaded["rng"])
# ③ 只丢"RNG"（模型+优化器都在，但新进程 seed、不恢复 rng）：step36 更新前就已分叉
netN = make_net(); optN = make_opt(netN)
netN.load_state_dict(loaded["model"]); optN.load_state_dict(loaded["optim"])
nor = train_run(netN, optN, K, K + 2, seed=999)
ct1 = time.perf_counter()

pre_ok = all(abs(a1-b1) <= 0.0 for a1, b1 in zip(pre, base[:K]))
res_ok = all(abs(a2-b2) <= 0.0 for a2, b2 in zip(res, base[K:]))
gap_res = max(abs(a2-b2) for a2, b2 in zip(res, base[K:]))
dq = abs(noo[0] - base[K])                 # ② 更新前偏差（应 =0：RNG 在，batch/mask 一样）
dq_opt = abs(noo[1] - base[K + 1])         # ② 更新后偏差（只丢优化器 的净效应）
dr = abs(nor[0] - base[K])                 # ③ 更新前偏差（只丢 RNG 的第一步效应）
dr_rng = abs(nor[1] - base[K + 1])         # ③ 更新后偏差（RNG 效应的延续）
assert pre_ok and res_ok and gap_res == 0.0
assert dq == 0.0 and dr > 0.0 and dq_opt > 0.0 and dr_rng > 0.0

print(f"    语料 {n} 字符 · 词表 V={V} · 窗口 {Wn} · 步 0..{T_STEPS-1}（崩溃点 {K}）")
print("    无中断 loss 曲线（每 6 步一行）")
for s in range(0, T_STEPS, 6):
    mark = "   ← 崩溃断点" if s == K else ""
    print(f"      step {s:>3}  loss {base[s]:.4f}{mark}")
print(f"    ① 全量存档(模型+优化器+RNG) 续训 vs 无中断参考：崩溃前 {K} 步{('逐位一致' if pre_ok else '不一致')}；"
      f"崩溃点续训首步 loss {res[0]:.4f} == 无中断 {base[K]:.4f}，续训 {T_STEPS-K} 步最大差 {gap_res:.6f}")
print(f"       （assert 逐位一致 ✓ 无损断点续训：存档越全，续训曲线与没崩溃一样）")
print(f"    ② 只丢'优化器状态'（RNG 在）：step36 更新前 loss {noo[0]:.4f}（与无中断 {base[K]:.4f} 逐位一致、偏差 0.0）；"
      f"step36 改用全新 Adam 更新后，step37 loss {noo[1]:.4f} vs 无中断 {base[K+1]:.4f} → 偏移 Δ{dq_opt:+.4f}"
      f"（动量/自适应尺度归零，步子一下变样）")
print(f"    ③ 只丢'RNG'（优化器在）：step36 loss {nor[0]:.4f}（≠ 无中断 {base[K]:.4f}，batch 序+dropout 一上来就错位）；"
      f"step37 偏移 Δ{dr_rng:+.4f}")
print(f"    → 存档必含：模型 + 优化器 + 步数 + RNG。漏优化器 = 续训第一步更新就变样；漏 RNG = 续训第一步数据/丢层就错位；"
      f"三者齐才与无中断完全重合（① 的 48 步最大差 0.000000）。")
print(f"      无存档直接崩溃 = 从零重付到崩溃点的全部步数（本任务 = 多付 {K} 步算力 + 期间产出低质量文本），"
      f"重启后的新随机种子还会让曲线不再可复现。")
print(f"    存储成本（接 [A] 账算）：7B 全参每档≈{total7:.0f} GiB 级；LoRA 每档≈{total_lora:.0f} GiB 级；"
      f"存档间隔就是'恢复快慢 vs 存钱/占盘'的旋钮。")
print(f"    C 段墙钟 {1000*(ct1-ct0):.1f} ms（本机真跑 · torch CPU 单线程 · 固定 seed 逐位复现）")

# ---------------------------------------------------------------------------
# [D] 日志诊断器：看得懂 loss（规则，账算非实测）
# ---------------------------------------------------------------------------
print("\n[D] 日志诊断器：给一段 loss 曲线判健康（规则；输入取 C 的真实曲线 + 合成对照）")

def diagnose(losses):
    n = len(losses)
    drop = losses[0] - losses[-1]
    span = max(losses) - min(losses)
    tail = losses[max(0, 2 * n // 4):]
    tail_var = max(tail) - min(tail)
    if losses[-1] > losses[0] + 1.0 and losses[-1] - losses[-min(len(losses), 6)] > 0.1:
        return "发散（LR 太高 / 梯度爆炸 / 数据错位）"
    if span < 0.05:
        return "死平（LR 太低 / 数据无信号 / 目标没接上）"
    if drop > 1.0 and tail_var < max(0.5, 0.35 * drop):
        return "健康（正常收敛，尾段平台可减 LR 或加步数）"
    return "观察（下降但尾段仍在动：平台未到，常见）"

print(f"    C 段真实曲线 → {diagnose(base)}")
synth = {
    "死平样本": [4.0]*20 + [3.95]*20,
    "发散样本": [4.0 + 0.05*i + 0.4*math.sin(i) for i in range(40)],
    "健康样本": [4.2*math.exp(-0.05*i) + 0.4 for i in range(40)],
}
for name, seq in synth.items():
    print(f"    合成 → {name}：{diagnose(seq)}")
print("    监测仪表盘（账算非实测）：loss / 梯度范数 / 学习率 / 累计 token / 显存水位 / 通信占比 → W&B·MLflow")
print("    关键参数经验值（§4.9，非本机实测）：预训练 LR≈3e-4 · SFT≈1e-5~2e-5 · LoRA≈1e-4~2e-4 · "
      "warmup≈前 2% 步 · grad clip≈1.0 · batch 按'每步 token'计（预训练 4M+ token/步 级）")

print(f"\n墙钟 {1000*(time.perf_counter()-t0):.1f} ms（科学数字逐位一致，只有墙钟浮动；C 段为真训练）")
os.remove(ckpt_path)
