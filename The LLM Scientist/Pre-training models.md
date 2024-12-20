## 预训练是一个非常漫长且昂贵的过程，这就是为什么这不是本课程的重点。
对训练前发生的情况有一定程度的了解是件好事，但不需要实践经验

* **Data pipeline**: 预训练需要庞大的数据集（例如，[Llama 2](https://arxiv.org/abs/2307.09288) 在 2 万亿个token上进行训练），这些数据集需要使用预定义的词汇表进行过滤、标记化和整理。
* **Causal language 建模**: 了解因果语言建模和掩码语言建模之间的区别。为了进行高效的预训练，请详细了解 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 或 [gpt-neox](https://github.com/EleutherAI/gpt-neox)
* **Scaling laws**: [缩放定律](https://arxiv.org/pdf/2001.08361.pdf)根据模型大小、数据集大小和用于训练的计算量来描述预期的模型性能
* **High-Performance Computing**: 此处不在讨论范围之内，但如果您打算从头开始创建自己的 LLM（硬件、分布式工作负载等），那么了解有关 HPC 的更多知识是必不可少的

* 📚 **推荐课程**:
* [LLMDataHub](https://github.com/Zjh-819/LLMDataHub) by Junhao Zhao: 用于预训练、微调和 RLHF 的精选数据集列表
* [从头开始训练因果语言模型](https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt) by Hugging Face: 使用 transformers 库从头开始预训练 GPT-2 模型
* [TinyLlama](https://github.com/jzhang38/TinyLlama) by Zhang et al.: 更好地了解如何从头开始训练 Llama 模型
* [Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) by Hugging Face: 解释因果语言建模和掩码语言建模之间的区别，以及如何快速微调 DistilGPT-2 模型
* [Chinchilla's wild implications](https://www.lesswrong.com/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications) by nostalgebraist: 讨论缩放定律并解释它们对 LLM 的一般意义
* [BLOOM](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4) by BigScience: 描述 BLOOM 模型构建方式的 Notion 页面，其中包含有关工程部分和遇到的问题的大量有用信息
* [OPT-175 Logbook](https://github.com/facebookresearch/metaseq/blob/main/projects/OPT/chronicles/OPT175B_Logbook.pdf) by Meta: 研究日志显示哪里出了问题，什么做对了。如果您计划预先训练一个非常大的语言模型（在本例中为 175B 参数），则非常有用
* [LLM 360](https://www.llm360.ai/): 一个开源 LLM 框架，包含训练和数据准备代码、数据、指标和模型
