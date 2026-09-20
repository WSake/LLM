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
| [`02-核心原理`](02-核心原理) | Transformer → 推理模型，22 个原理点（已开篇 18 篇：时间线 + Transformer + Attention 家族 + 多头注意力 + MLA + 位置编码 + FFN与激活 + LayerNorm/RMSNorm + Tokenizer BPE/BBPE + Embedding 与词表 + KV-Cache 显存账本 + Logits 与采样 + 预训练 + SFT 监督微调 + RLHF 与 PPO + DPO 家族 + GRPO 与 RLVR + 长上下文） | §2 |
| [`03-模型家族`](03-模型家族) | GPT / Llama / Qwen / DeepSeek / Claude / Gemini 等 | §3 |
| [`04-训练体系`](04-训练体系) | 数据 → 预训练 → 对齐 → 微调 → 评测 | §4 |
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
| 正文内容（89 知识点） | 🟡 持续推进中 · 01-基础 补齐至 8 篇成品（入口 + 7 内容课按 §17.1 依赖链排序）；08-RAG 系列 6 篇成品落地；02-核心原理 开篇 18 篇（时间线 + Transformer + Attention 家族 + 多头注意力 + MLA + 位置编码 + FFN与激活 + LayerNorm/RMSNorm + Tokenizer BPE/BBPE + Embedding 与词表 + KV-Cache 显存账本 + Logits 与采样 + 预训练 + SFT 监督微调 + RLHF 与 PPO + DPO 家族 + GRPO 与 RLVR + 长上下文，配 `transformer_demo.py`/`attention_demo.py`/`mla_demo.py`/`rope_demo.py`/`ffn_norm_demo.py`/`tokenizer_demo.py`/`embedding_demo.py`/`kv_cache_demo.py`/`sampling_demo.py`/`pretrain_demo.py`/`sft_demo.py`/`rlhf_ppo_demo.py`/`dpo_family_demo.py`/`grpo_rlvr_demo.py`/`long_context_demo.py`）（合计 **32 篇成品 + 13 篇提纲**） |
| 可运行实验 | 🟡 notebook 6 个 + 可复现脚本 18 个（Python 工程 `python_basics_demo.py` / 评估口径 `ml_eval_demo.py` / GPU 账本 `gpu_math_demo.py` / Transformer 从零验证与 KV 复用 `transformer_demo.py` / Attention 家族与 MHA→GQA 显存账 `attention_demo.py` / MLA 低秩与缓存账 `mla_demo.py` / RoPE 旋转与外推 `rope_demo.py` / FFN 三兄弟与 LayerNorm/RMSNorm 实测 `ffn_norm_demo.py` / BPE 从零实现与中英 token 统计 `tokenizer_demo.py` / Embedding 查表·tied·语义近邻 `embedding_demo.py` / KV Cache 账本与 Decode 实测 `kv_cache_demo.py` / Logits 温度·Top-k·Top-p·重复惩罚 `sampling_demo.py` / 预训练错位=N-1·loss≈lnV·跨句式事实 `pretrain_demo.py` / SFT 掩码只算回答区·全新问法 1/6→6/6·多样 vs 重复 6/6 vs 3/6 `sft_demo.py` / RLHF 三步对齐·奖励模型留出 79%·β=0 去锚对照 `rlhf_ppo_demo.py` / DPO 家族五 arm 共吃 383 对·留出判别 90-98% 盖 RM 79%·全新问三票 38%→65-82% `dpo_family_demo.py` / GRPO 组内优势免 critic·规则三票·全新问三票 38%→83%·KL 1.33·二值臂 10%/0% `grpo_rlvr_demo.py` / 长上下文四层墙位置编码重定标·Wpos 零梯度·滑窗·KV 淘汰·prefix-cache 3.0× `long_context_demo.py`） |

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
