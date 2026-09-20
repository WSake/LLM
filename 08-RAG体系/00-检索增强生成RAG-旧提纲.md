> 🟡 **此篇为旧版提纲骨架，尚待按《大模型知识地图》展开为完整正文。**
>
> 完整知识点体系与展开优先级见 [00-知识地图/对照索引.md](../00-知识地图/对照索引.md)（认领请走 [CONTRIBUTING.md](../CONTRIBUTING.md)）。
>
---

## 使用RAG，大型语言模型从数据库中检索上下文文档以提高其答案的准确性
RAG是一种在不进行任何微调的情况下增强模型知识的流行方法

* **Orchestrators**: 编排器（如[LangChain](https://python.langchain.com/docs/get_started/introduction)、[LlamaIndex](https://docs.llamaindex.ai/en/stable/)、[FastRAG](https://github.com/IntelLabs/fastRAG)等）是流行的框架，用于将您的LLM与工具、数据库、记忆等连接起来，并增强其能力
* **Retrievers**: 用户指令不适合检索。可以应用不同的技术（例如，多查询检索器、[HyDE](https://arxiv.org/abs/2212.10496)等）来改写/扩展它们，从而提高性能
* **Memory**: 为了记住之前的指令和答案，像ChatGPT这样的大型语言模型和聊天机器人会将这一历史添加到它们的上下文窗口中。这个缓冲区可以通过总结（例如使用较小的语言模型）、向量存储+RAG等方法得到改进
* **Evaluation**: 我们需要评估文档检索（上下文精确度和召回率）和生成阶段（忠实度和答案相关性）。可以使用工具[Ragas](https://github.com/explodinggradients/ragas/tree/main)和[DeepEval](https://github.com/confident-ai/deepeval)来简化这一过程
    
📚 **推荐课程**:
* [Llamaindex - High-level concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts.html): 构建RAG需要了解的主要概念
* [Pinecone - Retrieval Augmentation](https://www.pinecone.io/learn/series/langchain/langchain-retrieval-augmentation/): 检索增强过程概述
* [LangChain - Q&A with RAG](https://python.langchain.com/docs/use_cases/question_answering/quickstart): 逐步教程构建典型的RAG管道
* [LangChain - Memory types](https://python.langchain.com/docs/modules/memory/types/): 不同类型的记忆及其相关用途列表
* [RAG pipeline - Metrics](https://docs.ragas.io/en/stable/concepts/metrics/index.html): 概述用于评估RAG管道的主要指标


