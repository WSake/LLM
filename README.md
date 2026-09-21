<div align="center">

# 🧑‍💻 中文大模型全栈学习体系（2026）

**从 0 到能部署、能调参、能做 Agent 的完整路线 · 一份地图 + 89 个知识点 + 可运行的实验**

[![License](https://img.shields.io/github/license/WSake/LLM)](LICENSE)
[![Stars](https://img.shields.io/github/stars/WSake/LLM)](https://github.com/WSake/LLM/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)
[![中文](https://img.shields.io/badge/语言-中文-blue)]()
[![更新](https://img.shields.io/badge/更新-动态维护-brightgreen)]()

> 基于 [mlabonne/llm-course](https://github.com/mlabonne/llm-course)（Apache-2.0）整理与**大幅扩展**的中文版。
> 📢 **我们不是翻译仓库，而是一套按《大模型知识地图（2026）》重组的独立学习体系**——比原版多出 Agent / RAG 体系 / 评测 / 推理优化 / 多模态 / 安全 等 2026 年核心板块。

</div>

---

## ✨ 这是什么

你想系统地进入大模型世界，但市面上的教程要么是"收藏夹吃灰"的散装链接，要么是只能看不能跑的旧课程。这个仓库把这两件事一起解决：

| | 这仓库提供 | 解决什么问题 |
|---|---|---|
| 🗺️ **一份知识地图** | 《大模型知识地图_完整版.md》：1977 行、19 章、90+ 知识点，每个点都有掌握度标记（🔵了解/🟢掌握/🔴深入） | "我该学什么、学到什么程度、查漏补缺" |
| 📇 **一张对照索引** | [对照索引.md](00-知识地图/对照索引.md) 把 89+ 知识点与仓库目录一一映射 | "知识点在哪、写到哪了、你能帮什么忙" |
| 🧪 **能跑的实验** | 每个核心知识点配可一键运行的 notebook | "看完就手不痒，边学边跑" |
| 🛤️ **双路线规划** | 算法/研究员路线 + 应用/工程路线，从入门到专家 | "下一站往哪走" |

**不是又一份论文清单**，而是告诉你"为什么要出现 → 解决什么问题 → 怎么跑起来 → 怎么看结果"。

---

## 🚀 Quick Start（三步开始）

1. **先看地图** → [`00-知识地图/大模型知识地图_完整版.md`](00-知识地图/大模型知识地图_完整版.md)，花 1 周通读建立坐标
2. **挑路线** → 打开 [对照索引](00-知识地图/对照索引.md)，按"总览表"逐目录推进（路线正文在 [16-学习路线与里程碑](16-学习路线与里程碑) 建设中）
3. **动手指** → 打开 [`code/notebooks/`](code/notebooks/) 里已有的 6 个 notebook 一键运行（Colab 环境，免费可用；`009-中文RAG全管线.ipynb` 是 RAG 检索+评测可复现基线，`010` 是 RAGAS 式三层评测，`011` 是父子分块与元数据过滤）；零到一打基础，先跑 `code/scripts/python_basics_demo.py`（Python 工程实测，纯 CPU 可跑），再看评估口径 `ml_eval_demo.py`

> 💡 想先查漏补缺？打开 [对照索引](00-知识地图/对照索引.md)，逐条打勾，标出你的空缺——**那 90+ 个 ⬜ 也是本仓库的贡献清单**，欢迎 PR。
> 🗺️ 想看懂我们怎么打到 100K star？读 [PLAN.md](PLAN.md)（阶段化执行手册）。

---

## 📚 目录导航（与知识地图一一对应）

| 目录 | 内容 | 地图章节 |
|---|---|---|
| [`00-知识地图`](00-知识地图) | 知识地图全文 · 对照索引 · 技术栈分层 | §0 / §16 / §17 |
| [`01-基础`](01-基础) | Python 工程 / 数学 / ML(范式+sklearn) / DL / NLP / GPU，8 篇成品按依赖链排序 | §1 |
| [`02-核心原理`](02-核心原理) | Transformer → 推理模型，22 个原理点（已开篇 22 篇：时间线 + Transformer + Attention 家族 + 多头注意力 + MLA + 位置编码 + FFN与激活 + LayerNorm/RMSNorm + Tokenizer BPE/BBPE + Embedding 与词表 + KV-Cache 显存账本 + Logits 与采样 + 预训练 + SFT 监督微调 + RLHF 与 PPO + DPO 家族 + GRPO 与 RLVR + 长上下文 + MoE + 推理模型 + Scaling Law 与数据墙 + 蒸馏与 CPT） | §2 |
| [`03-模型家族`](03-模型家族) | GPT / Llama / Qwen / DeepSeek / Mistral / Gemma / Claude / Gemini 等（已开篇 00 八要素解构法 + 01 GPT系列 + 02 Llama系列 + 03 Qwen系列 + 04 DeepSeek系列 + 05 Mistral与Mixtral系列 + 06 Gemma系列） | §3 |
| [`04-训练体系`](04-训练体系) | 数据 → 预训练 → 对齐 → 微调 → 评测（已开篇 00 全链路总览 + 01 预训练数据工程 + 02 合成数据与数据飞轮 + 03 SFT 数据·指令集构建 + 04 Preference 数据与 Reward-Model + 05 RL 数据·可验证奖励 + 06 全参微调与 PEFT-LoRA-QLoRA + 07 训练并行-DP-ZeRO-FSDP-TP-PP-SP-EP + 08 训练框架横向对比 + 09 训练工程-精度-显存-检查点 + 10 训练闭环中的评测门禁） | §4 |
| [`05-推理与部署`](05-推理与部署) | 量化 / Serving / 并行 / 压测 | §5 |
| [`06-应用开发`](06-应用开发) | API / Prompt / 工具调用 / Workflow | §6 |
| [`07-应用框架`](07-应用框架) | LangGraph / LlamaIndex / MCP / DSPy … | §7 |
| [`08-RAG体系`](08-RAG体系) ⭐ | 检索 → 重排 → GraphRAG → Agentic RAG → 评测 | §8 |
| [`09-Agent体系`](09-Agent体系) ⭐ | Loop → Coding Agent → 评测 → 安全 | §9 |
| [`10-多模态`](10-多模态) | ViT / VLM / OCR / 语音 / 视频 | §10 |
| [`11-评测与可观测`](11-评测与可观测) ⭐ | Benchmark / LLM-judge / Langfuse | §11 |
| [`12-安全`](12-安全) | 注入 / 越狱 / 护栏 / 合规 | §12 |
| [`13-工程化`](13-工程化与基础设施) | GPU / K8s / Ray / LLMOps | §13 |
| [`14-垂直领域`](14-垂直领域) | 医疗 / 金融 / 法律 / 企业知识库 | §14 |
| [`15-前沿技术`](15-前沿技术) | 2026 热点滚动更新 | §15 |
| [`16-学习路线`](16-学习路线与里程碑) | 双路线 + 里程碑 + 时间线 | §18 |

> ⭐ = 本仓库差异化主战场：2026 年最核心、而原版 llm-course 完全没有的板块。

---

## 🗺️ 学习路线总览

```
【路线 A · 算法/研究员】
01-基础 → 02-核心原理 → 03-模型家族 → 04-训练体系 → 11-评测
      → 专深方向（对齐/RL / 训练效率 / 多模态数据）

【路线 B · 应用/工程】
01-基础(轻量) → 06-应用开发 → 07-应用框架 → 08-RAG → 09-Agent
      → 11-评测 → 13-工程化 → 14-垂直领域

【共同交汇带】05-推理与部署 + 11-评测 + 12-安全
```

详细分级路线、里程碑项目、推荐开源仓库见 [`16-学习路线与里程碑`](16-学习路线与里程碑)（路线正文建设中）；100K 行动手册见 [PLAN.md](PLAN.md)。

---

## 📈 当前进度

> 阶段 0（合规 + 结构 + 门面）完成后更新此表。

| 板块 | 状态 |
|---|---|
| 合规（LICENSE / NOTICE） | ✅ |
| 知识地图入库 + 对照索引 | ✅ |
| 目录结构（17 顶层目录） | ✅ |
| 正文内容（89 知识点） | 🟡 持续推进中 · 01-基础 补齐至 8 篇成品（入口 + 7 内容课按 §17.1 依赖链排序）；08-RAG 系列 6 篇成品落地；02-核心原理 开篇 22 篇（时间线 + Transformer + Attention 家族 + 多头注意力 + MLA + 位置编码 + FFN与激活 + LayerNorm/RMSNorm + Tokenizer BPE/BBPE + Embedding 与词表 + KV-Cache 显存账本 + Logits 与采样 + 预训练 + SFT 监督微调 + RLHF 与 PPO + DPO 家族 + GRPO 与 RLVR + 长上下文 + MoE + 推理模型 + Scaling Law 与数据墙 + 蒸馏与 CPT，配 `transformer_demo.py`/`attention_demo.py`/`mla_demo.py`/`rope_demo.py`/`ffn_norm_demo.py`/`tokenizer_demo.py`/`embedding_demo.py`/`kv_cache_demo.py`/`sampling_demo.py`/`pretrain_demo.py`/`sft_demo.py`/`rlhf_ppo_demo.py`/`dpo_family_demo.py`/`grpo_rlvr_demo.py`/`long_context_demo.py`/`moe_demo.py`/`reasoning_demo.py`/`scaling_demo.py`/`distil_cpt_demo.py`）；04-训练体系 开篇 10 篇（`00-全链路总览与飞轮`：数据→预训练→对齐→后训练→评测 五站流水线 + 数据飞轮，每站挂 02 系列既有实测，十站地图挂载本目录各章；`01-预训练数据工程-采集-清洗-去重`：质量过滤/去重/配比三段实测，端到端 104 条 1414→608 token 净剩 43.0%，配 `pretrain_data_demo.py`；`02-合成数据与数据飞轮`：种子扩写/质检三连/飞轮两轮闭环，60 进 25 出、坏样本 18/18 召回、两轮 25→49 零累积，配 `synth_data_demo.py`；`03-SFT数据-指令集构建`：三形态/质检三件套/条数≠token 配比实测，90 条 60/18/12、去重 4/90、算术 66.7% 条数→29.6% token、重采样回 38.8/31.8/29.4，配 `sft_data_demo.py`；`04-Preference数据与Reward-Model`：两路构造/质检三件套/长度捷径审计实测，78 对 60/18、规则对齐 60/60、启发式抓污染 2 真·1 误报、长度投票 50%/83.3% vs 真值 100%，配 `pref_data_demo.py`；`05-RL数据与可验证奖励`：奖励构造/质检/难度配比三连实测，240 输出粒度档数 2 vs 76、偷懒判据正例误杀 76.5%、难度山口信息样本 30/92.5/15%、取材搬向中档 +45.8pp，配 `rl_data_demo.py`；`06-全参微调与PEFT-LoRA-QLoRA`：全参账本/低秩谱/秩刻度三连实测，7B 全参 78.2 GiB=每参 12 字节（Adam 占 2/3）、Full ΔW 前 3 奇异值谱能量 100.0%、LoRA r4 参数 -75% 贴机器精度 vs r=2 欠参 1.68e-01，配 `peft_ft_demo.py`；`07-训练并行-DP-ZeRO-FSDP-TP-PP-SP-EP`：显存三档账/梯度货运账/气泡仿真三连实测，DP 每卡仍 78.2 GiB vs ZeRO-1/2/3@8 卡 32.5/21.1/9.8 GiB、通信·计算比 3.6%→128.2% 翻车曲线、气泡公式=上界 1F1B 省的是驻留、TP 2.1 GB/步·SP 8.0→1.0 GiB·EP 30.5 GiB，配 `parallel_demo.py`；`08-训练框架横向对比`：三类选择（模型库/预训练引擎/RL 引擎）+ 组装账 + 最大可装表，同一 78.2 GiB 组装 DDP 78.2 vs ZeRO-3/FSDP 39.1/19.6/9.8/4.9 vs Megatron ÷16 4.9、最大可装 DDP 3.6B→ZeRO-3@8 卡 28.6B→Megatron ÷16 57.3B、决策引擎 70B·8×80 全参 97.8 GiB✗→@16 卡 48.9 GiB✓、LoRA/全参双路径，配 `framework_demo.py`；`09-训练工程-精度-显存-检查点`：五件套预算单/精度三角账/断点续训/日志诊断四连，显存预算 95.3 vs LoRA 17.4 GiB 省 81.7%·激活 4.0 不省、loss 缩放 19.2%→0.12%、崩溃续训 48 步最大差 0.000000·漏优化器 Δ+0.0515 vs 漏 RNG Δ+0.0340，配 `traineng_demo.py`；`10-训练闭环中的评测门禁`：§4.10 三道门禁三连实测，同源语料 dev-PPL 260→51.6 过拟合拐点 step64 → 连续 3 次未创新低 step88 早停回滚、旧能回归分 8→4/8 @step104 触发回滚 step72、泄漏臂评测假健康 -3.55 vs hold-back 全新句仅差 0.031，配 `eval_gate_demo.py`）；03-模型家族 已开篇 6 篇（`00-八要素解构法`：全章方法论，八要素统一剖面 + 派生账本机真算（参数账 ±15% 走廊 Mixtral -0.3% / 激活两本账 27.4% vs 5.5% / KV 账 128k 16.0→8.6 GiB / 40GiB 桌判定 30.0·20.2·90.7·1258.4）+ 复习链断言 8/8 + 选型决策引擎，配 `model_family_demo.py`，墙钟 0.2 ms）+ `01-GPT系列`（GPT 家族主线：decoder + next-token 目标十年不变、规模/数据/对齐/推理时算力四件事翻页逼出 few-shot，配 `gpt_demo.py` 实测：ICL 训练中相变解锁 loss 2.5574→0.3996·fresh 0.320→1.000 同窗 / 三态 1.000·0.235·0.083 现场读映射 / 规模×预算 C=16 永不解锁·C=32 400 步边缘 0.935·C=64 400 步满格 1.000 / shot-scaling 不可辨识才爬坡 0.673→0.995 vs 素数开关 / 位置锁定负面，墙钟 522s≈8.7 分钟）+ `02-Llama系列`（Llama 家族主线：开源把"论文→权重→部署"串成一条链，RoPE/GQA/SwiGLU/RMSNorm 四件套被"谁先够、谁更省"投成事实标准，配 `llama_demo.py` 实测：引擎对账 8 组 maxerr≤4.923e-11 / 等预算配方对照同一 add 任务上都解锁 ICL 但 Llama 配方提前约 100 步穿 loss 悬崖（step 300：0.8491 vs 1.3053）·每步墙钟也省约 1/3（102s vs 146s）·参数也少约 9%（51,936 vs 57,072）/ 位置三臂 [C1] 少对+移位全塌负面·[C2] 整块平移 abs Wpos 越界 OOB·sin 回骰子 0.077·只有 rope 保部分 0.223→0.172（相对必要但不够，接 02-17 四层墙）/ KV 账 GQA 摊薄 H/H_kv 倍（3-8B ×4·405B ×16）128k 里 KV 比词表 embedding 大 16 倍，墙钟 ≈619s≈10.3 分钟）+ `03-Qwen系列`（Qwen 家族主线：中文语境的开源之王把"数据与对齐"做成家族壁垒——架构被开源摊平后，词表（练过中文让中文每字 −52%）/数据规模（同引擎同预算翻倍把覆盖面 0/8→16/16）/预算路由（难度路由 32% 预算拿回 92% 准确率）三本账拉开差距，配 `qwen_demo.py` 实测：A 词表经济 中文 1.000→0.481·英文不动 0.296·合并对 54→75 / B 数据翻页 regime1 已学 8/8·待翻页 0/8 → regime2 16/16 全回扣不遗忘（中英双报，同引擎 99,328 参数 600步×bs24）/ C 预算路由 全直答 0.737 vs 全验证器k16 0.891·难度路由@k4 1.29 前向 32% 预算拿全量 92% 准确率·@k16 2.44 拿 86%·双新 0.6 能力墙 / D 派生账 128k KV 7.0 GiB ≈ 词表 embedding 7 倍·GQA ×7/×16/×8·MoE 两条腿，墙钟 ≈100s≈1.7 分钟）+ `04-DeepSeek系列`（DeepSeek 家族主线：技术创新发动机——架构被开源摊平后把"省"做到机制级、重新发明架构差价（MLA 注意力低秩 · DeepSeekMoE 细粒度+共享+bias 均衡 · GRPO 涌现思考），配 `deepseek_demo.py` 实测（三路继承 llama/moe/reasoning 引擎）：A-MLA 同预算同一 add ICL 任务 KV 缓存 96 B→12 B 省 8 倍·参数 2304→576 省 4 倍·K 奇异谱 90% 能量 2 维 vs GQA 7 维·99% 3 维 vs 11 维 / B-DeepSeekMoE 载荷方差 0.1093→0.0097·门控熵 0.130→1.313 贴均匀界·均衡写进参数不写进损失 / C-R1 三臂 R1-Zero 缺冷启动卡格式门外 0.000·R1 冷启动+RL 格式/答案 0.937/0.603·R1-Distill 蒸馏 0.810/0.730 / D 派生账 MLA 1152 B vs MHA 65536 B 省 56.9×·V3 总参/激活 671B/37B≈5.5% vs Mixtral 27%，墙钟 ≈232s≈3.9 分钟）+ `05-Mistral与Mixtral系列`（Mistral 家族主线：MoE 与滑窗的欧洲示范——把"省"做成可卖的门槛 + 回望半径分诊，配 `mistral_demo.py` 实测（继承 llama/moe/deepseek 引擎）：A-SWA 成本账 32k/4k→4.3×·128k/4k→16.3×·回望半径分诊 full 训练推理收窄 W=10→9 悬崖 fresh 1.000→0.328·窗内剩 3..1 对救不回·W=2 统计捷径 0.463·从零滑窗 W=5 放全窗 0.562 反而退化，SWA=训练期决策 / B-MoE 五臂前三臂与 04 章 [B2] 逐位一致·同参孪生对 +aux 载荷方差 0.2165→0.0285 门控熵 0.741→1.331，共享专家与均衡机制正交 / D 派生账 Mixtral 8x7B 总参/激活 46.7B/12.9B≈27% vs V3 5.5%·SWA 不省 KV 驻留，墙钟 ≈324.9s≈5.4 分钟）+ `06-Gemma系列`（Gemma 家族主线：端侧是另一本账——Google 把小模型做成质量标杆 + 交替局部/全局注意力的全局锚点放对位置，配 `gemma_demo.py` 实测（继承 llama 引擎 + 本期补『每层掩码』扩展，None 路径向后兼容）：A 交替注意力四臂 1000 步×bs96——per-layer mask 对账 6 组 maxerr≤4.348e-11·四布局终局全解锁 fresh@各布局 0.975-1.000（信息上 (x5,y5) 一对即定 a）·布局买的是『窗口耐受』推理放宽全窗 LL 0.130<LG 0.203<GL 0.485<GG 1.000（全局层越多越稳、同为 1 个全局放底层 GL 比放顶层 LG 稳）·反向收窄 GG→全滑窗同样 OOD 0.290（05 训练期锁窗逐层化）/ B 派生账 Gemma2-9B 交替打分=全全局 79.8%·=全滑窗 106.4%·端侧装载 fp16/int4（int4=端侧默认入场券）·256k 词表 embedding 占比 35.7%（1B 逼近 1/3 驻留负债），墙钟 ≈333s≈5.6 分钟）（合计 **54 篇成品 + 13 篇提纲**） |
| 可运行实验 | 🟡 notebook 6 个 + 可复现脚本 39 个（八要素解构 派生账·复习链·选型引擎 `model_family_demo.py` / Qwen 系列 词表经济·数据翻页·预算路由·派生账 `qwen_demo.py` / DeepSeek 系列 MLA 潜注意力·MoE 三件套·R1 三臂·派生账 `deepseek_demo.py` / 评测门禁 过拟合·遗忘·泄漏三闸 `eval_gate_demo.py` / 训练工程 五件套预算单·精度三角账·断点续训·日志诊断 `traineng_demo.py` / 选型账·组装账·最大可装表·决策引擎 `framework_demo.py` / 并行 显存三档账·梯度货运账·气泡仿真 `parallel_demo.py` / 全参账本·低秩谱·秩刻度 `peft_ft_demo.py` / RL 数据 奖励粒度·质检·难度配比 `rl_data_demo.py` / Preference 数据 两路构造·质检三件套·长度捷径审计 `pref_data_demo.py` / SFT 数据 三形态·质检三件套·条数≠token 配比 `sft_data_demo.py` / 合成数据 种子扩写·质检三连·飞轮闭环 `synth_data_demo.py` / 预训练数据工程 质量过滤·MinHash 去重·token 配比 `pretrain_data_demo.py` / Python 工程 `python_basics_demo.py` / 评估口径 `ml_eval_demo.py` / GPU 账本 `gpu_math_demo.py` / Transformer 从零验证与 KV 复用 `transformer_demo.py` / Attention 家族与 MHA→GQA 显存账 `attention_demo.py` / MLA 低秩与缓存账 `mla_demo.py` / RoPE 旋转与外推 `rope_demo.py` / FFN 三兄弟与 LayerNorm/RMSNorm 实测 `ffn_norm_demo.py` / BPE 从零实现与中英 token 统计 `tokenizer_demo.py` / Embedding 查表·tied·语义近邻 `embedding_demo.py` / KV Cache 账本与 Decode 实测 `kv_cache_demo.py` / Logits 温度·Top-k·Top-p·重复惩罚 `sampling_demo.py` / 预训练错位=N-1·loss≈lnV·跨句式事实 `pretrain_demo.py` / SFT 掩码只算回答区·全新问法 1/6→6/6·多样 vs 重复 6/6 vs 3/6 `sft_demo.py` / RLHF 三步对齐·奖励模型留出 79%·β=0 去锚对照 `rlhf_ppo_demo.py` / DPO 家族五 arm 共吃 383 对·留出判别 90-98% 盖 RM 79%·全新问三票 38%→65-82% `dpo_family_demo.py` / GRPO 组内优势免 critic·规则三票·全新问三票 38%→83%·KL 1.33·二值臂 10%/0% `grpo_rlvr_demo.py` / 长上下文四层墙位置编码重定标·Wpos 零梯度·滑窗·KV 淘汰·prefix-cache 3.0× `long_context_demo.py` / MoE 总参/激活账·冻结拓扑对账·稀疏两硬事实·aux 剂量·专业化·Dense 对照 `moe_demo.py` / 推理模型基座四集·投票 vs 覆盖率 vs 验证器·能力墙·think 协议·k↔token↔KV 预算表 `reasoning_demo.py` / Scaling Law 双幂律·Hoffmann 联合拟合·等算力配比·数据墙池地板解析墙位 `scaling_demo.py` / 蒸馏白盒同位·参数偷渡 35%·温度 τ=1 最优·黑盒丢 soft·CPT 灾难性遗忘与回放 `distil_cpt_demo.py` / GPT 系列 ICL 五项实测（解锁相变·三态·规模·shot·位置锁定）`gpt_demo.py` / Llama 系列 配方对照四段实测（引擎对账·配方解锁·位置三臂·KV 摊薄）`llama_demo.py` / Mistral/Mixtral 系列 SWA 换窗分诊·MoE 五臂同参孪生对 `mistral_demo.py` / Gemma 系列 交替注意力四臂改窗耐受·派生账三本（打分账·装载账·词表占比账）`gemma_demo.py`） |

---

## 🤝 参与贡献

我们的目标是成为 **"中文大模型学习第一梯队"**，这需要社区而不是一个人。

- 📖 想学：看 [对照索引](00-知识地图/对照索引.md)，找到 ⬜ 的知识点，我们优先写更热门、更稀缺的
- ✍️ 想写：看 [CONTRIBUTING.md](CONTRIBUTING.md)，选题 + 模板 + 规范都写在里面了
- 🐛 发现问题：[提 issue](https://github.com/WSake/LLM/issues/new/choose)（纠错 / 内容请求 / Bug）
- ⭐ 支持我们：点个 Star，让更多中文学习者看到

---

## 📖 参考资料与致谢

- 上游课程：[mlabonne/llm-course](https://github.com/mlabonne/llm-course)（Apache-2.0）——感谢于本仓库整理与扩展的基础
- 结构基准：《大模型知识地图_完整版》（本仓库独有资产，2026 版）
- 课程向工具与 Colab 模块引用自上游 README（见 git 历史）

> *本仓库始终遵循上游 Apache-2.0 许可与 NOTICE 声明。*

## Licence

[Apache-2.0](LICENSE) · 详见 [NOTICE](NOTICE)（衍生声明与版权归属）。
