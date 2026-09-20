# -*- coding: utf-8 -*-
"""01-基础配套实验②：GPU 到底快在哪？——先分清『并行算力』与『显存带宽』。
纯 CPU 脚本：第一段是真实计时（本机实测），第二段是按纸面参数做的算术账(非实测)。
零依赖(仅 numpy+torch，torch 缺则自动跳过)；任何机器结果可复现。
运行：python code/scripts/gpu_math_demo.py
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

print("=" * 62)
print("第 0 步 · 这台机器有没有 GPU？")
print("=" * 62)
try:
    import torch
    cuda = torch.cuda.is_available()
    n_gpu = torch.cuda.device_count() if cuda else 0
    print(f"torch {torch.__version__} | CUDA 可用: {cuda} | GPU 数: {n_gpu}")
    if cuda:
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 2**30
        print(f"GPU: {name} | 显存约 {vram_gb:.0f} GB")
    else:
        print("→ 本机无 NVIDIA 卡：本脚本全程纯 CPU 仿真，等价于『算账』；")
        print("  想验证真 GPU 时，在 Colab 先跑 `nvidia-smi` 再跑本脚本即可。")
except ImportError:
    print("未安装 torch：跳过设备检查（不影响后面的算术账）")
print()

print("=" * 62)
print("第 1 步 · 真实计时：一次矩阵乘法在本机 CPU 上有多快")
print("=" * 62)
N = 2048
a = np.random.RandomState(0).randn(N, N).astype(np.float32)
b = np.random.RandomState(1).randn(N, N).astype(np.float32)
# 预热 + 计时三次取中位，避免计时噪声
for _ in range(2):
    a @ b
t0 = time.perf_counter(); a @ b; t1 = time.perf_counter()
t0 = time.perf_counter(); a @ b; t1 = time.perf_counter()
t0 = time.perf_counter(); a @ b; t1 = time.perf_counter()
t_ms = (t1 - t0) * 1e3
flops = 2.0 * N * N * N          # 2048³ 矩阵乘 ≈ 2·N³ 次浮点运算
gflops = flops / ((t1 - t0) * 1e9)
print(f"{N}×{N} fp32 矩阵乘(CPU 实测): {t_ms:.2f} ms ≈ {gflops:.0f} GFLOP/s")
ratio = 100_000 / gflops            # 100 TFLOP/s 的消费级 GPU vs 本机 CPU
print(f"对比视角：一块消费级 GPU 的 fp16 算力约 100 TFLOP/s → 约是本机 CPU 的 {ratio:.0f} 倍量级")
print("但『算得快』不等于『读得快』——见下一步。")
print()

print("=" * 62)
print("第 2 步 · 算术账：为什么推理往往是『带宽瓶颈』(IO-bound)")
print("=" * 62)
params_7b = 7e9
bytes_fp16 = 2
bytes_fp8 = 1
w16_gb = params_7b * bytes_fp16 / 2**30
w8_gb = params_7b * bytes_fp8 / 2**30
print(f"7B 参数模型：fp16 权重 ≈ {w16_gb:.1f} GB，fp8 权重 ≈ {w8_gb:.1f} GB")
for name, bw_gbs in (("消费级显存(HBM2e 常见 ~ 800GB/s)", 800),
                     ("服务器显存(H100 ~ 3.35TB/s)", 3350)):
    t_read_ms = (params_7b * bytes_fp16) / (bw_gbs * 1e9) * 1e3
    print(f"  {name}：把 7B 参数读一遍 ≈ {t_read_ms:.2f} ms  <- 每个 token 至少读这么多")
print("推理时每个生成的 token 都要把所有权重读一遍（权重不随 token 变，只重复读取）。")
print("所以『量化 fp16→fp8』把读权重的字节砍半 → 推理速度几乎线性提升——")
print("这解释了为什么量化主要不是为了省显存，而是为了喂饱带宽。")
print()

print("=" * 62)
print("第 3 步 · 一句话定点：GPU/CPU 到底差多少")
print("=" * 62)
# 一块典型消费级 GPU 与 CPU 的数量级对比（纸面参数，非实测）
print("""GPU vs CPU 的数量级差异（通用常识量级，非本机实测）：
  · 核心数     GPU 几千个精简核  vs  CPU 十余个重核
  · 并行规模   同一时刻做几千次运算  vs  几十次
  · 显存带宽   数百 GB/s ~ TB/s     vs  数十~上百 GB/s
  · 一个断言：大模型训练/推理的墙是『带宽壁』，不是『算力壁』""")
print()
print("done · 数字可复现：python code/scripts/gpu_math_demo.py")
