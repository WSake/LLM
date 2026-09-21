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
| [`03-模型家族`](03-模型家族) | GPT / Llama / Qwen / DeepSeek / Mistral / Gemma / Claude / Gemini / GLM 等（已开篇 00 八要素解构法 + 01 GPT系列 + 02 Llama系列 + 03 Qwen系列 + 04 DeepSeek系列 + 05 Mistral与Mixtral系列 + 06 Gemma系列 + 07 Claude系列·闭源API系第一家 + 08 Gemini系列·闭源API系第二家 + 09 GLM系列·填空续写一张脸 + 10 其他家族·五家五路的算力税 + 11 八要素横向对比表 + 12 家族继承与创新关系） | §3 |
| [`04-训练体系`](04-训练体系) | 数据 → 预训练 → 对齐 → 微调 → 评测（已开篇 00 全链路总览 + 01 预训练数据工程 + 02 合成数据与数据飞轮 + 03 SFT 数据·指令集构建 + 04 Preference 数据与 Reward-Model + 05 RL 数据·可验证奖励 + 06 全参微调与 PEFT-LoRA-QLoRA + 07 训练并行-DP-ZeRO-FSDP-TP-PP-SP-EP + 08 训练框架横向对比 + 09 训练工程-精度-显存-检查点 + 10 训练闭环中的评测门禁） | §4 |
| [`05-推理与部署`](05-推理与部署) | 量化 / Serving / 并行 / 压测 | §5 |
| [`06-应用开发`](06-应用开发) | API / Prompt / 工具调用 / Workflow（已开篇 01 Model API 与流式·接口基本功三件套 + 02 Prompt·从咒语到科学四账） | §6 |
| [`07-应用框架`](07-应用框架) | LangGraph / LlamaIndex / MCP / DSPy … | §7 |
| [`08-RAG体系`](08-RAG体系) ⭐ | 检索 → 重排 → GraphRAG → Agentic RAG → 评测（已开篇 02 核心流水线·总览 + 01 检索基础 + 03 混合重排 + 04 分块元数据 + 05 查询侧 + 06 上下文压缩 + 07 RAGAS 评测 + 08 向量数据库选型·先定场景再选库 + 09 Advanced 模式 + 10 Agentic RAG 决策层 + 11 GraphRAG 实体图谱 + 13 失败模式与修复·坏在哪就去哪修） | §8 |
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
| 正文内容（89 知识点） | 🟡 持续推进中 · 01-基础 补齐至 8 篇成品（入口 + 7 内容课按 §17.1 依赖链排序）；08-RAG 系列 12 篇成品落地（02 核心流水线·全图总览 + 01 检索基础 + 03 混合重排 + 04 分块元数据 + 05 查询侧 + 06 上下文压缩·检索多用 k=20 答案行 100/100 捞回 vs 生成用精 相关句压缩率 88.2%·忠实权衡曲线严格档砍 3 问答案行 + 07 RAGAS 评测 + 08 向量数据库选型·先定场景再选库 同层四索引实测（100 篇小库三索引召回全 98/100 拉平=召回分水岭在工程面→20k 合成集 IVFFlat 0.0696→1.0000·HNSW 0.9973·IVFPQ 量化封顶 0.2806·HNSW 体积 ~2×/IVFPQ 1/12·工程外壳决策引擎 9 断言）+ 09 Advanced 模式 + 10 Agentic RAG·三个决策点实测 judge 省 25.4%·多跳 2/4→4/4·盲目加深只加 token + 11 GraphRAG·实体-关系图三场对照实测 全局问题向量单次 2/11 vs 社区表一次全给·规则建图 ≈2 ms vs 向量建索引 ≈1.15 s，配 script 探针，配 3 个 notebook + 13 失败模式与修复·收束章 §8.14 七行「症状→根因→修复」诊断手册，配 `rag_diagnosis_demo.py` 六实验+台账实测：首屏 miss 2/100 混合救回 2/2 · 陪跑 16~19/20 行·证据纯度 0.06~0.21 档 · k=5/10/20 成本线 154.4/307.9/614.1 字符 · 词面断裂 3/3 miss 别名 3/3 救回 · 版本戳陈旧 3 条 vs 内容余弦 0.95+ 看不出 · 引用 naive 错位 2/3 vs 编号跟随 doc_id 0/3；方法论=信号→层→站点一表流）；02-核心原理 开篇 22 篇（时间线 + Transformer + Attention 家族 + 多头注意力 + MLA + 位置编码 + FFN与激活 + LayerNorm/RMSNorm + Tokenizer BPE/BBPE + Embedding 与词表 + KV-Cache 显存账本 + Logits 与采样 + 预训练 + SFT 监督微调 + RLHF 与 PPO + DPO 家族 + GRPO 与 RLVR + 长上下文 + MoE + 推理模型 + Scaling Law 与数据墙 + 蒸馏与 CPT，配 `transformer_demo.py`/`attention_demo.py`/`mla_demo.py`/`rope_demo.py`/`ffn_norm_demo.py`/`tokenizer_demo.py`/`embedding_demo.py`/`kv_cache_demo.py`/`sampling_demo.py`/`pretrain_demo.py`/`sft_demo.py`/`rlhf_ppo_demo.py`/`dpo_family_demo.py`/`grpo_rlvr_demo.py`/`long_context_demo.py`/`moe_demo.py`/`reasoning_demo.py`/`scaling_demo.py`/`distil_cpt_demo.py`）；04-训练体系 开篇 11 篇（`00-全链路总览与飞轮`：数据→预训练→对齐→后训练→评测 五站流水线 + 数据飞轮，每站挂 02 系列既有实测，十站地图挂载本目录各章；`01-预训练数据工程-采集-清洗-去重`：质量过滤/去重/配比三段实测，端到端 104 条 1414→608 token 净剩 43.0%，配 `pretrain_data_demo.py`；`02-合成数据与数据飞轮`：种子扩写/质检三连/飞轮两轮闭环，60 进 25 出、坏样本 18/18 召回、两轮 25→49 零累积，配 `synth_data_demo.py`；`03-SFT数据-指令集构建`：三形态/质检三件套/条数≠token 配比实测，90 条 60/18/12、去重 4/90、算术 66.7% 条数→29.6% token、重采样回 38.8/31.8/29.4，配 `sft_data_demo.py`；`04-Preference数据与Reward-Model`：两路构造/质检三件套/长度捷径审计实测，78 对 60/18、规则对齐 60/60、启发式抓污染 2 真·1 误报、长度投票 50%/83.3% vs 真值 100%，配 `pref_data_demo.py`；`05-RL数据与可验证奖励`：奖励构造/质检/难度配比三连实测，240 输出粒度档数 2 vs 76、偷懒判据正例误杀 76.5%、难度山口信息样本 30/92.5/15%、取材搬向中档 +45.8pp，配 `rl_data_demo.py`；`06-全参微调与PEFT-LoRA-QLoRA`：全参账本/低秩谱/秩刻度三连实测，7B 全参 78.2 GiB=每参 12 字节（Adam 占 2/3）、Full ΔW 前 3 奇异值谱能量 100.0%、LoRA r4 参数 -75% 贴机器精度 vs r=2 欠参 1.68e-01，配 `peft_ft_demo.py`；`07-训练并行-DP-ZeRO-FSDP-TP-PP-SP-EP`：显存三档账/梯度货运账/气泡仿真三连实测，DP 每卡仍 78.2 GiB vs ZeRO-1/2/3@8 卡 32.5/21.1/9.8 GiB、通信·计算比 3.6%→128.2% 翻车曲线、气泡公式=上界 1F1B 省的是驻留、TP 2.1 GB/步·SP 8.0→1.0 GiB·EP 30.5 GiB，配 `parallel_demo.py`；`08-训练框架横向对比`：三类选择（模型库/预训练引擎/RL 引擎）+ 组装账 + 最大可装表，同一 78.2 GiB 组装 DDP 78.2 vs ZeRO-3/FSDP 39.1/19.6/9.8/4.9 vs Megatron ÷16 4.9、最大可装 DDP 3.6B→ZeRO-3@8 卡 28.6B→Megatron ÷16 57.3B、决策引擎 70B·8×80 全参 97.8 GiB✗→@16 卡 48.9 GiB✓、LoRA/全参双路径，配 `framework_demo.py`；`09-训练工程-精度-显存-检查点`：五件套预算单/精度三角账/断点续训/日志诊断四连，显存预算 95.3 vs LoRA 17.4 GiB 省 81.7%·激活 4.0 不省、loss 缩放 19.2%→0.12%、崩溃续训 48 步最大差 0.000000·漏优化器 Δ+0.0515 vs 漏 RNG Δ+0.0340，配 `traineng_demo.py`；`10-训练闭环中的评测门禁`：§4.10 三道门禁三连实测，同源语料 dev-PPL 260→51.6 过拟合拐点 step64 → 连续 3 次未创新低 step88 早停回滚、旧能回归分 8→4/8 @step104 触发回滚 step72、泄漏臂评测假健康 -3.55 vs hold-back 全新句仅差 0.031，配 `eval_gate_demo.py`）；03-模型家族 已开篇 13 篇（`00-八要素解构法`：全章方法论，八要素统一剖面 + 派生账本机真算（参数账 ±15% 走廊 Mixtral -0.3% / 激活两本账 27.4% vs 5.5% / KV 账 128k 16.0→8.6 GiB / 40GiB 桌判定 30.0·20.2·90.7·1258.4）+ 复习链断言 8/8 + 选型决策引擎，配 `model_family_demo.py`，墙钟 0.2 ms）+ `01-GPT系列`（GPT 家族主线：decoder + next-token 目标十年不变、规模/数据/对齐/推理时算力四件事翻页逼出 few-shot，配 `gpt_demo.py` 实测：ICL 训练中相变解锁 loss 2.5574→0.3996·fresh 0.320→1.000 同窗 / 三态 1.000·0.235·0.083 现场读映射 / 规模×预算 C=16 永不解锁·C=32 400 步边缘 0.935·C=64 400 步满格 1.000 / shot-scaling 不可辨识才爬坡 0.673→0.995 vs 素数开关 / 位置锁定负面，墙钟 522s≈8.7 分钟）+ `02-Llama系列`（Llama 家族主线：开源把"论文→权重→部署"串成一条链，RoPE/GQA/SwiGLU/RMSNorm 四件套被"谁先够、谁更省"投成事实标准，配 `llama_demo.py` 实测：引擎对账 8 组 maxerr≤4.923e-11 / 等预算配方对照同一 add 任务上都解锁 ICL 但 Llama 配方提前约 100 步穿 loss 悬崖（step 300：0.8491 vs 1.3053）·每步墙钟也省约 1/3（102s vs 146s）·参数也少约 9%（51,936 vs 57,072）/ 位置三臂 [C1] 少对+移位全塌负面·[C2] 整块平移 abs Wpos 越界 OOB·sin 回骰子 0.077·只有 rope 保部分 0.223→0.172（相对必要但不够，接 02-17 四层墙）/ KV 账 GQA 摊薄 H/H_kv 倍（3-8B ×4·405B ×16）128k 里 KV 比词表 embedding 大 16 倍，墙钟 ≈619s≈10.3 分钟）+ `03-Qwen系列`（Qwen 家族主线：中文语境的开源之王把"数据与对齐"做成家族壁垒——架构被开源摊平后，词表（练过中文让中文每字 −52%）/数据规模（同引擎同预算翻倍把覆盖面 0/8→16/16）/预算路由（难度路由 32% 预算拿回 92% 准确率）三本账拉开差距，配 `qwen_demo.py` 实测：A 词表经济 中文 1.000→0.481·英文不动 0.296·合并对 54→75 / B 数据翻页 regime1 已学 8/8·待翻页 0/8 → regime2 16/16 全回扣不遗忘（中英双报，同引擎 99,328 参数 600步×bs24）/ C 预算路由 全直答 0.737 vs 全验证器k16 0.891·难度路由@k4 1.29 前向 32% 预算拿全量 92% 准确率·@k16 2.44 拿 86%·双新 0.6 能力墙 / D 派生账 128k KV 7.0 GiB ≈ 词表 embedding 7 倍·GQA ×7/×16/×8·MoE 两条腿，墙钟 ≈100s≈1.7 分钟）+ `04-DeepSeek系列`（DeepSeek 家族主线：技术创新发动机——架构被开源摊平后把"省"做到机制级、重新发明架构差价（MLA 注意力低秩 · DeepSeekMoE 细粒度+共享+bias 均衡 · GRPO 涌现思考），配 `deepseek_demo.py` 实测（三路继承 llama/moe/reasoning 引擎）：A-MLA 同预算同一 add ICL 任务 KV 缓存 96 B→12 B 省 8 倍·参数 2304→576 省 4 倍·K 奇异谱 90% 能量 2 维 vs GQA 7 维·99% 3 维 vs 11 维 / B-DeepSeekMoE 载荷方差 0.1093→0.0097·门控熵 0.130→1.313 贴均匀界·均衡写进参数不写进损失 / C-R1 三臂 R1-Zero 缺冷启动卡格式门外 0.000·R1 冷启动+RL 格式/答案 0.937/0.603·R1-Distill 蒸馏 0.810/0.730 / D 派生账 MLA 1152 B vs MHA 65536 B 省 56.9×·V3 总参/激活 671B/37B≈5.5% vs Mixtral 27%，墙钟 ≈232s≈3.9 分钟）+ `05-Mistral与Mixtral系列`（Mistral 家族主线：MoE 与滑窗的欧洲示范——把"省"做成可卖的门槛 + 回望半径分诊，配 `mistral_demo.py` 实测（继承 llama/moe/deepseek 引擎）：A-SWA 成本账 32k/4k→4.3×·128k/4k→16.3×·回望半径分诊 full 训练推理收窄 W=10→9 悬崖 fresh 1.000→0.328·窗内剩 3..1 对救不回·W=2 统计捷径 0.463·从零滑窗 W=5 放全窗 0.562 反而退化，SWA=训练期决策 / B-MoE 五臂前三臂与 04 章 [B2] 逐位一致·同参孪生对 +aux 载荷方差 0.2165→0.0285 门控熵 0.741→1.331，共享专家与均衡机制正交 / D 派生账 Mixtral 8x7B 总参/激活 46.7B/12.9B≈27% vs V3 5.5%·SWA 不省 KV 驻留，墙钟 ≈324.9s≈5.4 分钟）+ `06-Gemma系列`（Gemma 家族主线：端侧是另一本账——Google 把小模型做成质量标杆 + 交替局部/全局注意力的全局锚点放对位置，配 `gemma_demo.py` 实测（继承 llama 引擎 + 本期补『每层掩码』扩展，None 路径向后兼容）：A 交替注意力四臂 1000 步×bs96——per-layer mask 对账 6 组 maxerr≤4.348e-11·四布局终局全解锁 fresh@各布局 0.975-1.000（信息上 (x5,y5) 一对即定 a）·布局买的是『窗口耐受』推理放宽全窗 LL 0.130<LG 0.203<GL 0.485<GG 1.000（全局层越多越稳、同为 1 个全局放底层 GL 比放顶层 LG 稳）·反向收窄 GG→全滑窗同样 OOD 0.290（05 训练期锁窗逐层化）/ B 派生账 Gemma2-9B 交替打分=全全局 79.8%·=全滑窗 106.4%·端侧装载 fp16/int4（int4=端侧默认入场券）·256k 词表 embedding 占比 35.7%（1B 逼近 1/3 驻留负债），墙钟 ≈333s≈5.6 分钟）+ `07-Claude系列`（Claude 家族主线：**闭源 API 系第一大家族**——只给能力不给权重，八要素七格全闭、"推理方式"格第一次从自部署换 API；复刻对齐招牌 Constitutional AI / RLAIF 最小闭环，配 `claude_demo.py` 实测：A0 基线三票 76%/59%·污染 27%/29% → A1 红队违规 14/30 → A2 修订满分 14/14·仍污染 0/14 → A3 **RLAIF 自动偏好对 108 组（0 条人工标注，对应 02-14 人类 383 组）**→ A4 RM 800 步留出判别 **88%** → A5 **RLHF 对照负结果诚实报**（RM 奖励 -1.919→6.305、KL=25.885、无约束三票 76→30%·污染 27→70%——RM 格式捷径被 RL 劫持）→ A6 **Constitutional 稳路径 = 修订答回灌 SFT 600 步**（三票训练问 **100%**·全新问 78%、污染率 0%/19%，安全性+帮助性同涨），墙钟 ≈155s≈2.6 分钟）+ `08-Gemini系列`（Gemini 家族主线：**闭源 API 系第二家**——原生多模态 + 超长上下文（与 06-Gemma 开源版孪生镜像），招牌『统一 token 空间』把视觉直接 token 化进同一词表、不是 adapter 拼，配 `gemini_demo.py` 实测：引擎对账 A0 原生/adapter 两全套 6 组 maxerr≤4.953e-11 / A1 原生直训 1200 步 fresh count 0.900·recall 1.000 / A2 **token 预算对照** 同任务同预算 4 token 逐槽可寻址 vs 1 CLS 压缩——count（聚合量）0.994 仍可答、recall（按槽寻址）1.000→**0.343** 信息被压掉 / A3 跨模态注意力落图 33%·目标槽占图侧 **72%** / B 派生账 1M~2M KV 122→244 GiB·16×16 patch → 256²图=256 token，墙钟 ≈94s）+ `09-GLM系列`（GLM 家族主线：**智谱 GLM——同一个预训练目标，两种用法：填空与续写**（Autoregressive Blank Infilling ICLR 2023 招牌），把『填空』（双向看上下文，被掩 span 填回来）与『续写』（span 内从左到右自回归）钉进同一枚硬币，配 `glm_demo.py` 实测（复用 02-Llama 引擎 + 本期补『per-sample 掩码』扩展，None 走旧路径逐字节向后兼容）：D0 对账 8 组全参 maxerr≤4.523e-11 / A 三目标×两探针（cycle 链世界 10 节点唯一后继）600 步×bs48——causal LM 需右 **0.098**≈随机基线 0.100（右信息到不了被掩词）需左 1.000、双向 MLM 双侧 1.000、GLM 双侧 1.000+span 内自回归（掩码形状=信息能流方向）/ B1 难度曲线 span 长 1/2/3 快照 step100 0.2467/0.2953/0.5024→step600 0.0001/0.0013/0.0021 全程单调递增·B2 短→长课程 A/B（FLAT 0.0012 vs CUR 0.0003）-0.0009 方向正增益小 / C 派生账 2D RoPE 参数零新增·中文词表 0.3~0.5 token/字·KV 128k 16.0 GiB 演算·hmm 思考段接 02-20，墙钟 ≈82s）+ `10-其他家族·五家五路的算力税`（逐个家族收尾篇：Kimi K2/MiniMax M1/Hunyuan 押 MoE 稀疏税——同一激活算力 E=2→16 总参 183k→701k 放大 3.8×、激活比压到 0.261、CE400 与全新句式命中同量级抖动不单调退化（省的是激活算力、买的是驻留与通信）/ Hunyuan·Kimi 长上下文押 KV 税——128k L=32/48/64 → 16.0/24.0/32.0 GiB（纯账非本机实测）/ Doubao Seed-Thinking 押 RL 采样税——每正确回答前向 greedy 1.9 vs 验证器 best-of-16 20.6 vs RLVR 1.7，便宜 12.4×/ InternLM 押评测税——δ=10pp n=200→0.995、δ=2pp n=500→0.740、平票 0.405 压到猜硬币以下，配 `other_family_demo.py` 实测，墙钟 ≈175s）+ `11-八要素横向对比表`（§3.11 收束篇·第 1 弹：**十一个家族一张表**——14 个代表模型并排横向比差异（Attention 一格最乱六种写法 / FFN 收敛 SwiGLU 系）、跨家族派生账六家真算全落 ±15% 走廊、收敛度 8 特征×14 家（5 家 8/8 · 最低 Mistral 3/8）、全家族决策引擎八任务，配 `cross_family_demo.py` 实测：推导 7.505/7.070/46.572/9.550/7.995B · KV 母线 7.0~73.5 GiB，墙钟 0.0 s 三遍逐位一致）+ `12-家族继承与创新关系`（§3.12 收束篇·第 2 弹：**纵向看继承**——谁发明范式、谁稳定骨架、谁定义下一轮，把"谁先开源谁定义下一轮"落成台账，配 `lineage_demo.py` 实测：A 家族谱系 DAG 图算法本机真算（14 节点 / 最长代数 3 / hub 第一 Llama 5 家=开源骨架中心、DeepSeek 4 家=国产 MoE 路线 hub / LCA 分叉点）/ B 范式溯源 + 14 家普及率（RoPE+SwiGLU 14/14 共同标准 · Agent 9/14 最年轻仍在扩散）/ C 血缘量化（继承度最低 Mistral 3/8=独门最多 · 共同基底恰 2 位 RoPE+SwiGLU=谱系家规）/ D 四代引擎同一 add 任务同预算同种子真训练（gen0 古典→gen3 现代 fresh 1.000/0.990/0.983/1.000 全解锁=继承保底 · +GQA KV 192→96B 减半 · 参数 57072→51936=创新更省）/ E 范式定义者台账 + 2026 收敛态断言（7 特征 ≥10/14 事实标准 · Agent 9/14 扩散中），墙钟 294.5s≈4.9 分钟三遍逐位一致）；06-应用开发 开篇 2 篇（`01-Model-API与流式`：路线 B 概念起点·接口基本功三件套——同步 vs 流式 / token 计费 / 重试与超时，配 `api_stream_demo.py` 模拟 SSE 帧零网络三实验实测：流式≠更贵≠多次调用·重组 137 字逐字节一致·chunk 数=token 数 85·TTFT 0.010 s vs 同步全量 0.850 s/账单按 token 不收帧·误按 chunk 错 85 倍·缓存半价 /指数退避 wait=[0.5,1,2,4]·first-token 超时·429 Retry-After 覆盖·幂等纪律）+ `02-Prompt-Engineering`（从"咒语"到科学：模板账本·few-shot 示例预算·CoT 成本·偏置评测闸四账，配 `prompt_engineer_demo.py` 零网络四实验实测：六版模板累积 token 34→130·3.8×·few-shot 增量最大 / k=4 覆盖优先 4/4 vs 相似优先 3/4·冗余 0.00 vs 0.05 / CoT 输出预算 5.8×·挂靠 02-20 think≈6× / 同组 5 版 prompt 双口径榜首不同·四闸；诚实声明=词面代理非真实 ICL 增益·jieba 非真实 tokenizer）（合计 **68 篇成品 + 12 篇提纲**） |
| 可运行实验 | 🟡 notebook 6 个 + 可复现脚本 45 个（GLM 系列 三预训练目标·空白填充·span 课程 A/B·2D RoPE 账 `glm_demo.py` / Gemini 系列 统一 token 空间·原生 vs adapter 引擎对账·token 预算对照·跨模态注意力读数 `gemini_demo.py` / Claude 系列 RLAIF 最小闭环·判官三票当宪法·约束解码修订 `claude_demo.py` / Gemma 系列 交替注意力四臂改窗耐受·派生账三本（打分账·装载账·词表占比账）`gemma_demo.py` /八要素解构 派生账·复习链·选型引擎 `model_family_demo.py` / Qwen 系列 词表经济·数据翻页·预算路由·派生账 `qwen_demo.py` / DeepSeek 系列 MLA 潜注意力·MoE 三件套·R1 三臂·派生账 `deepseek_demo.py` / 评测门禁 过拟合·遗忘·泄漏三闸 `eval_gate_demo.py` / 训练工程 五件套预算单·精度三角账·断点续训·日志诊断 `traineng_demo.py` / 选型账·组装账·最大可装表·决策引擎 `framework_demo.py` / 并行 显存三档账·梯度货运账·气泡仿真 `parallel_demo.py` / 全参账本·低秩谱·秩刻度 `peft_ft_demo.py` / RL 数据 奖励粒度·质检·难度配比 `rl_data_demo.py` / Preference 数据 两路构造·质检三件套·长度捷径审计 `pref_data_demo.py` / SFT 数据 三形态·质检三件套·条数≠token 配比 `sft_data_demo.py` / 合成数据 种子扩写·质检三连·飞轮闭环 `synth_data_demo.py` / 预训练数据工程 质量过滤·MinHash 去重·token 配比 `pretrain_data_demo.py` / Python 工程 `python_basics_demo.py` / 评估口径 `ml_eval_demo.py` / GPU 账本 `gpu_math_demo.py` / Transformer 从零验证与 KV 复用 `transformer_demo.py` / Attention 家族与 MHA→GQA 显存账 `attention_demo.py` / MLA 低秩与缓存账 `mla_demo.py` / RoPE 旋转与外推 `rope_demo.py` / FFN 三兄弟与 LayerNorm/RMSNorm 实测 `ffn_norm_demo.py` / BPE 从零实现与中英 token 统计 `tokenizer_demo.py` / Embedding 查表·tied·语义近邻 `embedding_demo.py` / KV Cache 账本与 Decode 实测 `kv_cache_demo.py` / Logits 温度·Top-k·Top-p·重复惩罚 `sampling_demo.py` / 预训练错位=N-1·loss≈lnV·跨句式事实 `pretrain_demo.py` / SFT 掩码只算回答区·全新问法 1/6→6/6·多样 vs 重复 6/6 vs 3/6 `sft_demo.py` / RLHF 三步对齐·奖励模型留出 79%·β=0 去锚对照 `rlhf_ppo_demo.py` / DPO 家族五 arm 共吃 383 对·留出判别 90-98% 盖 RM 79%·全新问三票 38%→65-82% `dpo_family_demo.py` / GRPO 组内优势免 critic·规则三票·全新问三票 38%→83%·KL 1.33·二值臂 10%/0% `grpo_rlvr_demo.py` / 长上下文四层墙位置编码重定标·Wpos 零梯度·滑窗·KV 淘汰·prefix-cache 3.0× `long_context_demo.py` / MoE 总参/激活账·冻结拓扑对账·稀疏两硬事实·aux 剂量·专业化·Dense 对照 `moe_demo.py` / 推理模型基座四集·投票 vs 覆盖率 vs 验证器·能力墙·think 协议·k↔token↔KV 预算表 `reasoning_demo.py` / Scaling Law 双幂律·Hoffmann 联合拟合·等算力配比·数据墙池地板解析墙位 `scaling_demo.py` / 蒸馏白盒同位·参数偷渡 35%·温度 τ=1 最优·黑盒丢 soft·CPT 灾难性遗忘与回放 `distil_cpt_demo.py` / GPT 系列 ICL 五项实测（解锁相变·三态·规模·shot·位置锁定）`gpt_demo.py` / Llama 系列 配方对照四段实测（引擎对账·配方解锁·位置三臂·KV 摊薄）`llama_demo.py` / Mistral/Mixtral 系列 SWA 换窗分诊·MoE 五臂同参孪生对 `mistral_demo.py` / Gemma 系列 交替注意力四臂改窗耐受·派生账三本（打分账·装载账·词表占比账）`gemma_demo.py` / 其他家族 稀疏税·KV·采样·评测四账 `other_family_demo.py` / `glm_demo.py` / 八要素横向对比 全家族一张表·跨家族派生账·收敛度·决策引擎 `cross_family_demo.py` / 家族继承与创新 谱系 DAG·范式溯源与普及率·血缘量化·四代引擎继承实验·定义者台账 `lineage_demo.py`） |

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
