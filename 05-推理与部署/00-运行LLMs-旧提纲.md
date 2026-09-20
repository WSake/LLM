> 🟡 **此篇为旧版提纲骨架，尚待按《大模型知识地图》展开为完整正文。**
>
> 完整知识点体系与展开优先级见 [00-知识地图/对照索引.md](../00-知识地图/对照索引.md)（认领请走 [CONTRIBUTING.md](../CONTRIBUTING.md)）。
>
---

## 这一部分侧重于学习如何构建可在生产中使用的 LLM 驱动的应用程序，重点是强化模型然后进行部署
## 由于硬件要求较高，运行 LLM 可能很困难 
如果只想通过 API（如 GPT-4）使用模型或在本地运行它。在任何情况下，修改prompt可以改进和限制应用程序的输出

* **LLM APIs**:API 是部署大型语言模型（LLM）的一种便捷方式。（如 [OpenAI](https://platform.openai.com/)、[Google](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/overview)、[Anthropic](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)、[Cohere](https://docs.cohere.com/docs) 等）和开源 LLM（如 [OpenRouter](https://openrouter.ai/)、[Hugging Face](https://huggingface.co/inference-api)、[Together AI](https://www.together.ai/) 等）
* **开源 LLMs**: [Hugging Face Hub](https://huggingface.co/models) 是寻找大型语言模型的绝佳去处。你可以在 [Hugging Face Spaces](https://huggingface.co/spaces) 中直接运行其中一些模型，或者下载并在像 [LM Studio](https://lmstudio.ai/) 这样的应用程序中本地运行它们，也可以通过命令行界面使用 [llama.cpp](https://github.com/ggerganov/llama.cpp) 或 [Ollama](https://ollama.ai/) 运行它们。
* **Prompt 工程**: 常用技术包括零样本提示、少样本提示、思维链和ReAct。这些技术在更大模型上效果更好，但也可以适应较小的模型
* **结构化输出**: 许多任务需要结构化的输出，例如严格的模板或JSON格式。像[LMQL](https://lmql.ai/)、[Outlines](https://github.com/outlines-dev/outlines)、[Guidance](https://github.com/guidance-ai/guidance)等库可以用于指导生成并遵守给定的结构


📚 **推荐课程**:
* [Run an LLM locally with LM Studio](https://www.kdnuggets.com/run-an-llm-locally-with-lm-studio) by Nisha Arya: 简短的LM Studio使用指南
* [Prompt engineering guide](https://www.promptingguide.ai/) by DAIR.AI: 带有示例的prompt技术详细指南
* [Outlines - Quickstart](https://outlines-dev.github.io/outlines/quickstart/): 大纲启用的引导生成技术列表
* [LMQL - Overview](https://lmql.ai/docs/language/overview.html): 介绍LMQL语言
