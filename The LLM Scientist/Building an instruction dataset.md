## 虽然很容易从 Wikipedia 和其他网站找到原始数据，但自己制作高质量的QA对还是有些困难
与传统机器学习一样，数据集的质量将直接影响模型的质量，这就是为什么它可能是微调过程中最重要的组成部分

* **[Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)-like dataset**: 使用 OpenAI API （GPT） 从头开始生成合成数据。您可以指定种子和系统提示来创建多样化的数据集。
* **Advanced techniques**: 学习如何使用 [Evol-Instruct](https://arxiv.org/abs/2304.12244) 改进现有数据集，如何生成高质量的合成数据，就像 [Orca](https://arxiv.org/abs/2306.02707) 和 [phi-1](https://arxiv.org/abs/2306.11644) 论文一样。
* **筛选数据**: 传统技术涉及正则表达式、删除重复的数据、专注于具有大量标记的答案等
* **提示词模板**: 没有真正的标准方法来格式化说明和答案，这就是为什么了解不同的prompt模板很重要的原因，例如 [ChatML](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/chatgpt?tabs=python&pivots=programming-language-chat-ml), [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), 等.

📚 **推荐课程**:
* [准备用于指令调优的数据集](https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2) by Thomas Capelle: 探索 Alpaca 和 Alpaca-GPT4 数据集以及如何格式化它们。
* [生成 Clinical Instruction 数据集](https://medium.com/mlearning-ai/generating-a-clinical-instruction-dataset-in-portuguese-with-langchain-and-gpt-4-6ee9abfa41ae) by Solano Todeschini: 有关如何使用 GPT-4 创建合成指令数据集的教程
* [GPT 3.5 用于新闻分类](https://medium.com/@kshitiz.sahay26/how-i-created-an-instruction-dataset-using-gpt-3-5-to-fine-tune-llama-2-for-news-classification-ed02fe41c81f) by Kshitiz Sahay: 使用 GPT 3.5 创建指令数据集，以微调 Llama 2 以进行新闻分类
* [创建用于微调 LLM 的数据集](https://colab.research.google.com/drive/1GH8PW9-zAe4cXEZyOIE-T9uHXblIldAg?usp=sharing): 包含一些用于筛选数据集并上传结果的方法的笔记
* [聊天模板](https://huggingface.co/blog/chat-templates) by Matthew Carrigan: Hugging Face 关于提示模板的页面
