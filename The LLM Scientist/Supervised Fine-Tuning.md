## 预先训练的模型仅在 next-token 预测任务上进行训练
SFT 允许您调整它们以响应说明。此外，它允许您在任何数据上微调您的模型并使用它，而无需为像 OpenAI 这样的 API 付费

* **全参微调**: 全参微调是指训练模型中的所有参数。这不是一种有效的技术，但它会产生稍微好一点的结果
* [**LoRA微调**](https://arxiv.org/abs/2106.09685): 一种基于低秩适配器的参数高效技术 （PEFT）。我们只训练这些适配器，而不是训练所有参数
* [**QLoRA微调**](https://arxiv.org/abs/2305.14314): 另一个基于 LoRA 的 PEFT，以 4 位量化模型的权重，并引入分页优化器来管理内存峰值。将其与 [Unsloth](https://github.com/unslothai/unsloth) 结合使用，即可在免费的 Colab 笔记本上高效运行
* **[Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)**: 一个对用户友好且功能强大的微调工具，用于许多最先进的开源模型
* [**DeepSpeed**](https://www.deepspeed.ai/): 针对多 GPU 和多节点设置（在 Axolotl 中实现）的 LLM 进行高效的预训练和微调


📚 **推荐课程**:
* [The Novice's LLM Training Guide](https://rentry.org/llm-training) by Alpin: 微调 LLM 时要考虑的主要概念和参数概述
* [LoRA insights](https://lightning.ai/pages/community/lora-insights/) by Sebastian Raschka: 有关 LoRA 以及如何选择最佳参数的实用见解
* [微调您自己的 Llama 2 模型](https://mlabonne.github.io/blog/posts/Fine_Tune_Your_Own_Llama_2_Model_in_a_Colab_Notebook.html): 有关如何使用 Hugging Face 库微调 Llama 2 模型的动手实践教程
* [Padding Large Language Models](https://towardsdatascience.com/padding-large-language-models-examples-with-llama-2-199fb10df8ff) by Benjamin Marie: 填充因果 LLM 训练示例的最佳实践
* [LLM 微调初学者指南](https://mlabonne.github.io/blog/posts/A_Beginners_Guide_to_LLM_Finetuning.html): 有关如何使用 Axolotl 微调 CodeLlama 模型的教程



