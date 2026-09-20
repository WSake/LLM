> 🟡 **此篇为旧版提纲骨架，尚待按《大模型知识地图》展开为完整正文。**
>
> 完整知识点体系与展开优先级见 [00-知识地图/对照索引.md](../00-知识地图/对照索引.md)（认领请走 [CONTRIBUTING.md](../CONTRIBUTING.md)）。
>
---

## 实际应用可能需要复杂的流程，包括SQL或图数据库，以及自动选择相关工具和API
这些高级技术可以改进基础解决方案并提供额外的功能

* **Query construction**: 存储在传统数据库中的结构化数据需要特定的查询语言，如SQL、Cypher、元数据等。我们可以直接将用户指令翻译成查询语句来访问数据，进行查询构建
* **Agents and tools**: 智能体通过自动选择最相关的工具来为LLM提供答案。这些工具可以像使用Google或Wikipedia一样简单，也可以像Python解释器或Jira一样复杂
* **Post-processing**: 最终步骤处理输入到LLM的内容。它通过重新排序、RAG融合和分类，增强检索到的文档的相关性和多样性
* **Program LLMs**: 像[DSPy](https://github.com/stanfordnlp/dspy)这样的框架允许你根据自动评估以编程方式优化提示和权重

📚 **推荐课程**:
* [LangChain - Query Construction](https://blog.langchain.dev/query-construction/): 关于不同类型的构造查询
* [LangChain - SQL](https://python.langchain.com/docs/use_cases/qa_structured/sql): 关于如何使用LLM与SQL数据库交互的教程，涉及文本到SQL的转换和可选的SQL代理
* [Pinecone - LLM agents](https://www.pinecone.io/learn/series/langchain/langchain-agents/): 介绍具有不同类型Agent
* [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) by Lilian Weng: 关于LLM代理的理论文章
* [LangChain - OpenAI's RAG](https://blog.langchain.dev/applying-openai-rag/): OpenAI采用的RAG策略概述，包括后处理
* [DSPy in 8 Steps](https://dspy-docs.vercel.app/docs/building-blocks/solving_your_task): 通用DSPy指南，介绍模块、签名和优化器





