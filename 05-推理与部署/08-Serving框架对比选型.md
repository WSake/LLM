# 📄 05 章 · 08-Serving框架对比选型（§5.6 · 🟢 掌握概念与选型）

> **章节**：05-推理与部署 · **站点**：7 / 10 · **难度**：[🟢 掌握概念与选型] · **前置**：05-01（两阶段 Prefill/Decode）、05-03（精度与量化）、05-05（推理并行 TP/PP/DP/EP）、05-06（Continuous Batching / Chunked Prefill）、05-07（投机解码与 PD 分离）· **衔接**：05-09（部署形态-K8s 与云端）、13-03（推理服务基础设施）、13-04（Ray：编排层）
>
> 一句话带走：**05-05~07 讲了"模型怎么摆、请求怎么调度、生成怎么加速"，本篇回答最后一个问题——"用哪个引擎上路"：先分三层（引擎内核 / 引擎壳 / 编排层），再在三强（vLLM / SGLang / TensorRT-LLM）里选主，端侧与国产分开点单，最后用 8 个场景把"NVIDIA→vLLM、极致吞吐与长前缀→SGLang、编译追求延迟下限→TensorRT-LLM、端侧→llama.cpp/Ollama/MLX、昇腾→MindIE、CPU 企业→OpenVINO"钉进决策盘。**

---

## 📑 本节大纲（plan3 全部知识点）

| # | 知识点 | 落地位置 |
|---|---|---|
| 1 | 引擎谱系一图：通用生产（vLLM）/ 高吞吐研究前沿（SGLang）/ NVIDIA 编译优化（TensorRT-LLM）/ 端侧本地（llama.cpp、Ollama）/ 老牌（TGI）/ 国产（LMDeploy、MindIE-vLLM-Ascend）/ Apple（MLX）/ CPU 跨平台（ONNX-OpenVINO）/ 编排层（Ray Serve 不是引擎是上层） | §1 / §6 |
| 2 | 主选三强对比：vLLM vs SGLang vs TensorRT-LLM（定位/优势/注意点） | §2 |
| 3 | 端侧与本地：llama.cpp（GGUF/跨平台/零依赖）、Ollama（桌面分发，不适合生产并发）、MLX（Apple 统一内存） | §3 |
| 4 | 国产算力两条线：LMDeploy（TurboMind 快/AWQ 好/生态绑 InternLM-Qwen 系）、MindIE/vLLM-Ascend（昇腾 NPU 唯一主线、与 CUDA 生态有差距需适配） | §4 |
| 5 | 选型逻辑：NVIDIA 云上大规模→vLLM 或 TensorRT-LLM；极致吞吐与长前缀→SGLang；端侧/个人→llama.cpp/Ollama/MLX；国产 NPU→MindIE；CPU 企业集成→OpenVINO | §5 |
| 6 | 边界提醒：Ollama≠生产；TGI 性能已被 vLLM 反超；Ray Serve 是编排 | §6 |

> **大纲覆盖自检**：6 点全覆盖 —— 点 1 占 §1（11 引擎归位）；点 2 占 §2；点 3 占 §3；点 4 占 §4；点 5 占 §5（含确定性"场景→点单"判定 8 场景）；点 6 占 §6（含边界三连判定）。代码落账在 §5+§6，无专属探针（05-03/05-05 同款风格，内嵌确定性代码）。

---

## §1 引擎谱系一图：先分三层，再看品牌

选引擎前的第一件事不是背品牌，而是**把 11 个名字分进三层**：

| 层 | 名字 | 一句话 |
|---|---|---|
| **引擎内核**（真正跑 token 的东西） | vLLM / SGLang / TensorRT-LLM / llama.cpp / TGI / LMDeploy / MindIE-vLLM-Ascend / MLX / ONNX-OpenVINO | 有独立调度器与 kernel，是"会算的那一层" |
| **引擎壳**（包一层的分发层） | Ollama | 壳里是 llama.cpp，自己不写 kernel，主打"拉模型即用" |
| **编排层**（把引擎编排成服务的那一层） | Ray Serve | **不是引擎是上层**，把 vLLM/SGLang 挂在里面做多模型路由（13-04 呼应） |

这个分层是全文的地基——后面所有对比、选型、边界，都建立在这里。**"Ollama 是不是引擎"、"Ray Serve 能不能替代 vLLM"这类问题的答案，在三层一摆就自己出来了。**

落到一行表（11 项归位）：

| 引擎 | 所属层 | 主要硬件 | 一句话定位 |
|---|---|---|---|
| **vLLM** | 引擎内核 | NVIDIA·通用 | PagedAttention 发明者·OpenAI 兼容 API·生产默认 |
| **SGLang** | 引擎内核 | NVIDIA·通用 | RadixAttention 前缀树·吞吐/长前缀研究前沿 |
| **TensorRT-LLM** | 引擎内核 | NVIDIA 专属 | 编译优化·FP8 原生·inflight batching |
| **llama.cpp** | 引擎内核 | CPU/Apple/低端GPU | GGUF 缔造·C/C++ 零依赖·k-quants 混合位宽 |
| **TGI** | 引擎内核 | NVIDIA·通用 | HuggingFace 老牌（2023）·性能已被 vLLM 反超 |
| **LMDeploy** | 引擎内核 | NVIDIA·国产系 | TurboMind 内核·AWQ 支持好·InternLM/Qwen 生态 |
| **MindIE/vLLM-Ascend** | 引擎内核 | 昇腾 NPU | 昇腾唯一主线·华为 MindIE + vLLM 官方昇腾后端 |
| **MLX** | 引擎内核 | Apple Silicon | Apple 统一内存框架（MLX-LM） |
| **ONNX-Runtime/OpenVINO** | 引擎内核 | CPU 跨平台 | 企业集成·X86/ARM/Intel 优化 |
| **Ollama** | 引擎壳 | 端侧分发 | llama.cpp 桌面壳·拉模型即用·不适合生产并发 |
| **Ray Serve** | 编排层 | K8s/多机 | 不是引擎是上层·把引擎打包成服务（13-04 呼应） |

> 本章节 §5.6 与 13-03（推理服务基础设施）、13-04（Ray）的关系：本篇讲"引擎之间怎么选"，13 章讲"选完之后怎么跑起来（K8s/HPA/网关）"，Ray Serve 到 13-04 才算完整账。这里只摆位置、不重复账。

---

## §2 主选三强：vLLM / SGLang / TensorRT-LLM

在 NVIDIA 云上这一个场景里，讨论收敛到三强。三个都是"真正会在生产里跑的引擎内核"，但**赢在三个不同的点上**。

### 2.1 vLLM —— 生产默认（2023 PagedAttention → 生态最大）

- **定位**：通用生产防火墙内的第一选择。把 05-04 的 PagedAttention（碎片 70%→1%）、05-06 的 Continuous Batching（GPU 64.4%→85.0%）与 Prefix Caching 全做成了默认能力。
- **优势**：生态最广——OpenAI 兼容 API 一把梭，几乎所有 LLM 框架都把 vLLM 当第一后端；模型格式接入面宽（HF safetensors 直读、GGUF/AWQ/GPTQ 量化都能喂）；部署文档、周边（`vllm serve`）、云厂商默认镜像都是它。
- **注意点**：长前缀场景不是它的最强项（PagedAttention 的 prefix cache 是哈希命中式，不是结构化的前缀树）；对"极致延迟下限"它选通用路径，不编译到极致。

### 2.2 SGLang —— 高吞吐 / 长前缀研究前沿（RadixAttention）

- **定位**：以 **RadixAttention（前缀树）** 区分 vLLM：把共享前缀做成一棵 LRU 树结构，多用户同用一个大 system prompt / RAG 同一批文档时，**相同前缀的 KV 只在树上存一份、重复使用**——这正是 05-04 复用线说的"前缀缓存 3.0×（02-17 实测）"的工程化极致。
- **优势**：前缀复用场景吞吐优势明显，是 DeepSeek/长上下文推理时代的引擎新贵；重计算调度 + 结构化前缀，天然适合"Agent 每轮重放长 system prompt"这类的反复调用。
- **注意点**：生态比 vLLM 窄（模型/量化格式覆盖略少）；"极致吞吐与长前缀"是它的专项，不是所有场景的最优。

### 2.3 TensorRT-LLM —— NVIDIA 编译优化（延迟下限）

- **定位**：把模型编译进 TensorRT engine（05-02 的 `Engine·Plan` 格式就是它），用编译期优化换运行时提速；FP8 原生、inflight batching 等是 NVIDIA 栈里最激进的工程实现。
- **优势**：单机吞吐与首 token 延迟的**下限**在 NVIDIA 官方栈里最强（要素事实：官方 benchmark，非本机实测）；配套 NVIDIA 的工具链与 Triton 推理服务器深度绑定。
- **注意点**：**编译迁移成本**——换一个模型/改一个结构要重新编译（分钟级~小时级），动态场景（模型常换）很疼；绑定 NVIDIA，不能跨芯片。

**结论**（§5 决策树的三个根）：**什么都不改的默认 = vLLM；长前缀复用/极致吞吐 = SGLang；锁定 NVIDIA + 接受编译迁移 = TensorRT-LLM。**

---

## §3 端侧与本地：llama.cpp / Ollama / MLX

### 3.1 llama.cpp —— GGUF 缔造者，零依赖跨平台

- **定位**：C/C++ 写的端侧引擎内核，无 Python、无 CUDA 强依赖，CPU/Apple/低端 GPU 都能跑。05-02 的 GGUF（单文件 + k-quants 混合位宽量化）就是它的格式。
- **一句话**：端侧/个人/边缘设备的真相引擎——Windows 桌面、Linux 服务器、树莓派、甚至手机（通过绑定）都能跑，量化后 7B FP16 13.04 GiB → INT4 ≈ 3.26 GiB（05-03 表），放得进低端显存。

### 3.2 Ollama —— llama.cpp 的桌面壳（重点：壳不是引擎）

- **定位**：套在 llama.cpp 外面的分发层，`ollama run` 拉模型即用，内置模型商店。**注意分层**：Ollama 不自己写 kernel，核心就是 llama.cpp（§1 的"引擎壳"）。
- **红线**：**不适合生产并发**。它面向个人桌面/极简部署，没有 vLLM 那套连续批调度与并发控制；把它当生产 Serving 用，吞吐和单点故障都不是这个层该背的。（§6 边界三连第 1 条。）

### 3.3 MLX —— Apple 统一内存（Mac 专属）

- **定位**：Apple 自家框架（MLX-LM 专门跑 LLM），吃 Mac 的**统一内存**优势——CPU/GPU 共享一块内存，不做 PCIe 搬运，8B 模型在 16G 统一内存的 MacBook 上可秒级加载（对比 13-02 那套 PCIe 搬运账，Mac 本地没有这个瓶颈）。
- **注意点**：只 Apple Silicon；生态和量化工具比 llama.cpp 窄，但 Apple 主推、迭代快（M 系列核心推理用得越来越多）。

---

## §4 国产算力两条线：LMDeploy 与 MindIE/vLLM-Ascend

国产线路，按"算的是 NVIDIA 生态还是昇腾 NPU"分成两条，避免混为一谈（热点选型区最容易把"国产"一键打包）。

### 4.1 LMDeploy（InternLM 团队 · NVIDIA + 自家生态）

- **定位**：开箱支持 InternLM / Qwen 全家桶，TurboMind 内核（推理快速），AWQ 支持度好（05-03 量化谱系里 AWQ 是 NVIDIA 收敛点之一）。
- **注意点**：生态绑 InternLM/Qwen 更顺手，换来的是对"更冷门模型/自建模型"的支持不如 vLLM 广——**它是主力在 NVIDIA 上跑的国产引擎**，跟昇腾是两回事（老资料把"国产 = 昇腾"是常见误区）。

### 4.2 MindIE / vLLM-Ascend（昇腾 NPU · 昇腾唯一主线）

- **定位**：昇腾 NPU（910B 等）上的唯一主线 = **华为 MindIE + vLLM 官方昇腾后端双轨**：MindIE 是华为商业化 runtime；vLLM 官方有 Ascend 后端，把 vLLM 的调度账搬上昇腾。
- **红线**：**昇腾与 CUDA 生态有差距，需算子适配**——NVIDIA 栈里天然跑的 kernel 到昇腾要换算子集，部署前要有"算子兼容性验证"这个预算项（13-03 异构部署一句带过，本篇展开成意识）。
- **一句话**：信创/昇腾机群项目 → MindIE/vLLM-Ascend；普通 NVIDIA 项目 ≠ 国产自动等于昇腾。

---

## §5 选型逻辑：8 个场景→点单（确定性判定）

把前面所有"定位/优势/注意点"收敛成一个**决策函数**：给定"硬件 + 规模 + 优化目标"就返回该点哪家。下面代码是纯 stdlib 确定性实现，与 05-03/05-05 同款小函数风格。

```python
# -*- coding: utf-8 -*-
"""05-08 Serving框架对比选型 · 简化账 + 场景→点单判定（纯 stdlib · 零网络 · 零随机）"""

import sys
import time

_wall0 = time.time()

# ---------------- 账 A：11 引擎谱系归位 ----------------
# (引擎, 所属层, 主要硬件, 一句话定位)
ENGINES = [
    ("vLLM",              "引擎内核", "NVIDIA·通用",     "PagedAttention 发明者·OpenAI 兼容 API·生产默认"),
    ("SGLang",            "引擎内核", "NVIDIA·通用",     "RadixAttention 前缀树·吞吐/长前缀研究前沿"),
    ("TensorRT-LLM",      "引擎内核", "NVIDIA 专属",     "编译优化·FP8 原生·inflight batching"),
    ("llama.cpp",         "引擎内核", "CPU/Apple/低端GPU", "GGUF 缔造·C/C++ 零依赖·k-quants 混合位宽"),
    ("TGI",               "引擎内核", "NVIDIA·通用",     "HuggingFace 老牌（2023）·性能已被 vLLM 反超"),
    ("LMDeploy",          "引擎内核", "NVIDIA·国产系",   "TurboMind 内核·AWQ 支持好·InternLM/Qwen 生态"),
    ("MindIE/vLLM-Ascend", "引擎内核", "昇腾 NPU",        "昇腾唯一主线·华为 MindIE + vLLM 官方昇腾后端"),
    ("MLX",               "引擎内核", "Apple Silicon",   "Apple 统一内存框架（MLX-LM）"),
    ("ONNX-Runtime/OpenVINO", "引擎内核", "CPU 跨平台",   "企业集成·X86/ARM/Intel 优化"),
    ("Ollama",            "引擎壳",   "端侧分发",        "llama.cpp 桌面壳·拉模型即用·不适合生产并发"),
    ("Ray Serve",         "编排层",   "K8s/多机",        "不是引擎是上层·把引擎打包成服务（13-04 呼应）"),
]

A = ["== A · Serving 引擎谱系 11 项归位（三层：引擎内核 / 引擎壳 / 编排层） ==",
     "  引擎 | 所属层 | 主要硬件 | 一句话定位"]
for name, layer, hw, note in ENGINES:
    A.append(f"  {name} | {layer} | {hw} | {note}")
cnt = {}
for _, layer, _, _ in ENGINES:
    cnt[layer] = cnt.get(layer, 0) + 1
A.append(f"  合计 11 项 = 引擎内核 {cnt['引擎内核']} + 引擎壳 {cnt['引擎壳']} + 编排层 {cnt['编排层']}")

# ---------------- 账 B：场景→点单判定（8 场景） ----------------
def pick_engine(hw, scale, opt="general", compile_ok=False, desktop=False):
    """hw: nvidia/ascend/apple/cpu；scale: prod(生产)/edge(端侧)；
       opt: general(通用)/throughput(极致吞吐)/prefix(长前缀复用)/compile(编译优化)；
       compile_ok: 是否接受编译迁移；desktop: 是否桌面分发原型。"""
    if hw == "ascend":
        return "MindIE / vLLM-Ascend（昇腾 NPU 唯一主线，需算子适配）"
    if hw == "apple":
        return "MLX（Apple 统一内存，Mac 本地推理）"
    if hw == "cpu":
        return "ONNX-Runtime / OpenVINO（CPU 跨平台企业集成）"
    if scale == "edge":
        return "Ollama（桌面分发壳，拉模型即用）" if desktop else "llama.cpp（GGUF·零依赖·跨平台）"
    # scale == prod（NVIDIA）
    if opt in ("throughput", "prefix"):
        return "SGLang（RadixAttention 前缀树·吞吐/长前缀前沿）"
    if opt == "compile" and compile_ok:
        return "TensorRT-LLM（编译优化·FP8 原生·延迟下限）"
    return "vLLM（生产默认·OpenAI 兼容·生态最广）"

SCENES = [
    ("NVIDIA 云上大规模通用生产（多模型多租户）",  "nvidia", "prod", "general", False, False),
    ("长前缀复用（多用户共享同一大 system prompt / RAG 同文档）", "nvidia", "prod", "prefix", False, False),
    ("NVIDIA 栈追求延迟下限·接受编译迁移",        "nvidia", "prod", "compile", True, False),
    ("个人笔记本零依赖本地跑（Windows/Linux/树莓派）", "nvidia", "edge", "general", False, False),
    ("桌面聊天产品原型（拉模型即用）",            "nvidia", "edge", "general", False, True),
    ("MacBook M 系列本地跑 8B（16G 统一内存）",   "apple", "edge", "general", False, False),
    ("企业内网无 GPU·CPU 集群集成",               "cpu", "prod", "general", False, False),
    ("昇腾 910B 机群信创项目调用",                "ascend", "prod", "general", False, False),
]
EXP = [
    "vLLM（生产默认·OpenAI 兼容·生态最广）",
    "SGLang（RadixAttention 前缀树·吞吐/长前缀前沿）",
    "TensorRT-LLM（编译优化·FP8 原生·延迟下限）",
    "llama.cpp（GGUF·零依赖·跨平台）",
    "Ollama（桌面分发壳，拉模型即用）",
    "MLX（Apple 统一内存，Mac 本地推理）",
    "ONNX-Runtime / OpenVINO（CPU 跨平台企业集成）",
    "MindIE / vLLM-Ascend（昇腾 NPU 唯一主线，需算子适配）",
]
FIELDS = ["hw", "scale", "opt", "compile_ok", "desktop"]
ok = 0
B = ["== B · 场景 → 点单判定（8 场景） =="]
for i, (name, *vals) in enumerate(SCENES, 1):
    got = pick_engine(**dict(zip(FIELDS, vals)))
    hit = got == EXP[i - 1]
    ok += hit
    B.append(f"  [{i}] {name} -> {got} = " + ("PASS" if hit else f"FAIL(期望 {EXP[i - 1]})"))
B.append(f"  场景判定：{ok}/8 PASS")

# ---------------- 账 C：三条边界提醒 ----------------
def warn(case):
    if case == "ollama_prod":
        return "Ollama 是桌面壳：单机上生产并发，换 vLLM/SGLang"
    if case == "tgi_legacy":
        return "TGI 老牌：吞吐已被 vLLM 反超，优先迁移 vLLM"
    if case == "ray_serve_engine":
        return "Ray Serve 是编排层：它不代替引擎，要包 vLLM/SGLang 才成立"
    raise ValueError(case)

WARN_EXP = [
    ("ollama_prod", "Ollama 是桌面壳：单机上生产并发，换 vLLM/SGLang"),
    ("tgi_legacy", "TGI 老牌：吞吐已被 vLLM 反超，优先迁移 vLLM"),
    ("ray_serve_engine", "Ray Serve 是编排层：它不代替引擎，要包 vLLM/SGLang 才成立"),
]
okc = 0
C = ["== C · 边界提醒三连（误选自查） =="]
for key, exp in WARN_EXP:
    got = warn(key)
    hit = got == exp
    okc += hit
    C.append(f"  [{key}] -> {got} = " + ("PASS" if hit else "FAIL"))
C.append(f"  边界判定：{okc}/3 PASS")

# ---------------- 一致性断言（教学自检） ----------------
assert sum(cnt.values()) == 11, "谱系 11 项恒一"
assert ok == 8, "场景判定 8/8"
assert okc == 3, "边界判定 3/3"

print("\n".join(A))
print()
print("\n".join(B))
print()
print("\n".join(C))
print(f"[wall] {time.time() - _wall0:.3f} s", file=sys.stderr)
```

```text
== A · Serving 引擎谱系 11 项归位（三层：引擎内核 / 引擎壳 / 编排层） ==
  引擎 | 所属层 | 主要硬件 | 一句话定位
  vLLM | 引擎内核 | NVIDIA·通用 | PagedAttention 发明者·OpenAI 兼容 API·生产默认
  SGLang | 引擎内核 | NVIDIA·通用 | RadixAttention 前缀树·吞吐/长前缀研究前沿
  TensorRT-LLM | 引擎内核 | NVIDIA 专属 | 编译优化·FP8 原生·inflight batching
  llama.cpp | 引擎内核 | CPU/Apple/低端GPU | GGUF 缔造·C/C++ 零依赖·k-quants 混合位宽
  TGI | 引擎内核 | NVIDIA·通用 | HuggingFace 老牌（2023）·性能已被 vLLM 反超
  LMDeploy | 引擎内核 | NVIDIA·国产系 | TurboMind 内核·AWQ 支持好·InternLM/Qwen 生态
  MindIE/vLLM-Ascend | 引擎内核 | 昇腾 NPU | 昇腾唯一主线·华为 MindIE + vLLM 官方昇腾后端
  MLX | 引擎内核 | Apple Silicon | Apple 统一内存框架（MLX-LM）
  ONNX-Runtime/OpenVINO | 引擎内核 | CPU 跨平台 | 企业集成·X86/ARM/Intel 优化
  Ollama | 引擎壳 | 端侧分发 | llama.cpp 桌面壳·拉模型即用·不适合生产并发
  Ray Serve | 编排层 | K8s/多机 | 不是引擎是上层·把引擎打包成服务（13-04 呼应）
  合计 11 项 = 引擎内核 9 + 引擎壳 1 + 编排层 1

== B · 场景 → 点单判定（8 场景） ==
  [1] NVIDIA 云上大规模通用生产（多模型多租户） -> vLLM（生产默认·OpenAI 兼容·生态最广） = PASS
  [2] 长前缀复用（多用户共享同一大 system prompt / RAG 同文档） -> SGLang（RadixAttention 前缀树·吞吐/长前缀前沿） = PASS
  [3] NVIDIA 栈追求延迟下限·接受编译迁移 -> TensorRT-LLM（编译优化·FP8 原生·延迟下限） = PASS
  [4] 个人笔记本零依赖本地跑（Windows/Linux/树莓派） -> llama.cpp（GGUF·零依赖·跨平台） = PASS
  [5] 桌面聊天产品原型（拉模型即用） -> Ollama（桌面分发壳，拉模型即用） = PASS
  [6] MacBook M 系列本地跑 8B（16G 统一内存） -> MLX（Apple 统一内存，Mac 本地推理） = PASS
  [7] 企业内网无 GPU·CPU 集群集成 -> ONNX-Runtime / OpenVINO（CPU 跨平台企业集成） = PASS
  [8] 昇腾 910B 机群信创项目调用 -> MindIE / vLLM-Ascend（昇腾 NPU 唯一主线，需算子适配） = PASS
  场景判定：8/8 PASS

== C · 边界提醒三连（误选自查） ==
  [ollama_prod] -> Ollama 是桌面壳：单机上生产并发，换 vLLM/SGLang = PASS
  [tgi_legacy] -> TGI 老牌：吞吐已被 vLLM 反超，优先迁移 vLLM = PASS
  [ray_serve_engine] -> Ray Serve 是编排层：它不代替引擎，要包 vLLM/SGLang 才成立 = PASS
  边界判定：3/3 PASS
```

（运行 `python` 复制上文 python 块 → 原始 stdout md5 `1ae065e7c9bc35e35232189f2de426c5`，LF 归一化后与 text 块一致；三进程字节全同。）

---

## §6 边界提醒：三条容易踩到底线的用法

### 6.1 Ollama ≠ 生产（引擎壳别当引擎用）

**外层最常见坑**：把 `ollama serve` 当生产 Serving 顶上去。回到 §1 的三层——Ollama 是壳，核心是 llama.cpp，**没有 vLLM 那套连续批调度、并发控制、前缀复用**。单机个人用没问题；一旦多用户并发上线，吞吐/排队/单点全不是这个层该背的，换 vLLM/SGLang。（`warn("ollama_prod")` → `"Ollama 是桌面壳：单机上生产并发，换 vLLM/SGLang"`）

### 6.2 TGI 已被 vLLM 反超（2023 老资历不自动等于强）

TGI 是 HuggingFace 2023 推的老牌引擎内核（微软也用过它顶级）。**今日的横评，TGI 的吞吐已被 vLLM 反超**（要素事实，NVIDIA/社区公开对照；非本机实测）——老项目还在 TGI 上的，优先迁 vLLM；新项目默认不看 TGI。（`warn("tgi_legacy")` → `"TGI 老牌：吞吐已被 vLLM 反超，优先迁移 vLLM"`）

### 6.3 Ray Serve 是编排层，不是引擎（它离不开引擎）

Ray Serve 上一层的编排（多模型路由、自动扩缩、混合模型流量），**但它不是引擎**——它必须包一个真正的引擎（vLLM/SGLang）才成立。把"关键路径放 Ray"但没有引擎内核支撑，等于有了调度台却没有炮。（承 13-04：`warn("ray_serve_engine")` → `"Ray Serve 是编排层：它不代替引擎，要包 vLLM/SGLang 才成立"`）

> 一个"误选自查"的 3 场景判定（`warn()` 三连）就是项目里最常被问的三个"它能不能"。

---

## §验证与锚点核对表

| 声明 | 数值 | 出处/计算方法 |
|---|---|---|
| 引擎谱系 11 项归位 | 9 内核 + 1 壳 + 1 编排 | `ENGINES` 表 + `cnt` 计数 |
| 谱系恒一 | 11 项 | `assert sum(cnt.values()) == 11` |
| 图谱三层定位 | vLLM/SGLang/TRT-LLM/llama.cpp/TGI/LMDeploy/MindIE/MLX/ONNX-OV = 引擎内核；Ollama = 壳；Ray Serve = 编排 | plan3 §5.6 点 1 / 点 6 |
| 三强对比 | vLLM=生产默认·生态最广；SGLang=长前缀/吞吐前沿；TRT-LLM=编译延迟下限 | §2（选型逻辑点 5 条目化） |
| 端侧/本地三条 | llama.cpp（GGUF 零依赖）、Ollama（壳，非生产）、MLX（Apple 统一内存） | §3（plan3 点 3） |
| 国产两条线 | LMDeploy（NVIDIA·TurboMind·AWQ）；MindIE/vLLM-Ascend（昇腾唯一主线·需算子适配） | §4（plan3 点 4） |
| 决策路径 | 昇腾→MindIE；Apple→MLX；CPU→OpenVINO；端侧→llama.cpp/Ollama；吞吐/长前缀→SGLang；编译→TRT-LLM；其它→vLLM | §5 `pick_engine()` |
| 场景判定 | 8/8 PASS | `pick_engine()` 8 场景 |
| 边界三连 | 3/3 PASS（Ollama≠生产 / TGI 反超 / Ray 是编排） | §6 `warn()` |
| 三进程 stdout | md5 `1ae065e7…`（字节全同） | 内嵌代码 3 进程实测 |

**决定性的测量锚点（本机实测）**：本篇为知识型选型，数值锚全部是站内确定性函数（谱系计数 11、场景 8/8、边界 3/3），三进程 stdout 字节全同（`1ae065e7c9bc35e35232189f2de426c5`）。**引擎间的相对 benchmark（vLLM vs SGLang 长前缀、TRT-LLM 2× 等）为要素事实（上游公开对照），非本机实测**——本机只验证了选型逻辑函数本身。

**§17.4 05 行卡位**：05-01/05-02/05-03/05-04/05-05/05-06/05-07（已绿）、**05-08（本篇，新增 ✅；覆盖:GGUF 与 llama.cpp 系槽位）**、05-09～05-10（⬜）。

---

## §常见坑（选型视角）

1. **把 Ollama/llama.cpp 混为一谈**：Ollama 是壳、llama.cpp 是引擎内核，两者要不要并发、要不要前缀复用，回答完全不同——先分三层再讨论。
2. **"国产一笔带过"**：LMDeploy 跑在 NVIDIA 上，MindIE 跑在昇腾上，两条线硬件栈完全不同，别把"国产 = 昇腾"当作默认。
3. **选 SGLang 只因"吞吐最大"**：SGLang 的吞吐优势在长前缀/结构化前置场景；通用生产不共享大前缀时，vLLM 的生态与成熟度更划算。
4. **选了 TRT-LLM 才想起编译迁移**：模型常换的动态团队上了 TRT-LLM，每次换模型/改结构都要重新编译（分钟~小时级）；接受这个成本才选它。
5. **把"引擎"和"编排"混成一件事**：Ray Serve 管多模型路由，但真正跑 token 的是它包住的 vLLM——评估并发时别只盯着编排层，要拆到引擎内核去算。

---

## 衔接下一章

下一站 05-09 `09-部署形态-K8s与云端`（§5.7）把选好的引擎"放到哪台机器上、怎么上线"讲清楚——本地 CLI → 单机多卡（vLLM TP）→ 多机 K8s 集群（自动扩缩）→ 云端托管，四级形态 + 上线要素清单 + 异构部署（CUDA/ROCm/NPU）。本篇选了"用什么引擎"，下一篇回答"引擎怎么跑"——都定下来后，13-03（推理服务基础设施）把它做成一整条可运维的服务链路。

（路线提示：05-10 `10-性能指标与压测-TTFT-TPOT` 会给本批次收束，把"选好引擎以后怎么量它"的指标口径钉死，并与 11-08 工程指标章打通；本节已经先给出一条可复跑的选型锚。）

---

## 参考资料与诚实边界

- **实测锚点（本机）**：本篇内嵌确定性代码（谱系归位 11 项 · 场景→点单 8/8 · 边界三连 3/3），三进程 stdout 字节全同 md5 `1ae065e7c9bc35e35232189f2de426c5`。
- **要素事实（上游公开，非本机实测）**：vLLM / SGLang / TensorRT-LLM 的**相对吞吐与延迟对比**；TRT-LLM "2×于开源替代"类 benchmark；SGLang 长前缀优势的论文/公开对照（LMSYS 报告）；TGI "被 vLLM 反超"的社区横评；LMDeploy TurboMind 提速与 AWQ 支持；DeepSeek/Qwen 等模型对 SGLang/vLLM 的官方支持度。全部引用为"要素事实（上游公开对照）"而非本机测量。
- **锚点引用（已交付站）**：05-02 GGUF/Engine/Plan 四格式一行定位；05-03 INT4 3.26 vs FP16 13.04 GiB；05-04 PagedAttention 碎片 70%→1%、前缀缓存 3.0×（02-17）；05-06 Continuous Batching GPU 64.4%→85.0%；13-02 PCIe vs NVLink 28.1×（Mac 本地无 PCIe 搬运）；13-04 Ray 编排层。
- 报告口径：本批所有 benchmark 均为"上游要素事实"标注，未经本机复现。
