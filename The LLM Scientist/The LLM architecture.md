## 虽然不需要对 Transformer 架构有深入的了解，但对其输入 （token） 和输出 （logits） 有很好的理解是很重要的
原版注意力机制是另一个需要掌握的关键组件，因为会介绍它的改进版本

* **高级视图**: 重新审视编码器-解码器 Transformer 架构，更具体地说是每个现代 LLM 中使用的仅解码器 GPT 架构
* **分词化**: 了解如何将原始文本数据转换为模型可以理解的格式，这涉及将文本拆分为标记（通常是单词或子词）
* **注意力机制**: 掌握注意力机制背后的理论，包括自注意力和缩放点积注意力，这使模型在产生输出时能够专注于输入的不同部分
* **文本生成**:了解模型生成输出序列的不同方式。常见的策略包括贪心解码、束搜索、top-k 采样和原子核采样。

📚 **推荐课程**:
* [图解 Transformer](https://jalammar.github.io/illustrated-transformer/) by Jay Alammar: 对 Transformer 模型的直观解释
* [图解 GPT-2](https://jalammar.github.io/illustrated-gpt2/) by Jay Alammar: 侧重于 GPT 架构，与 Llama 的架构非常相似
* [Transformers 的视觉介绍](https://www.youtube.com/watch?v=wjZofJX0v4M&t=187s) by 3Blue1Brown: 简单易懂的 Transformers 视觉介绍
* [LLM 可视化](https://bbycroft.net/llm) by Brendan Bycroft: 令人难以置信的 LLM 内部情况的 3D 可视化
* [nanoGPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) by Andrej Karpathy: 一个 2 小时的 YouTube 视频，从头开始重新实现 GPT（面向程序员）
* [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) by Lilian Weng: 以更正式的方式介绍注意力机制
* [LLM 中的解码策略](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html): 提供代码和用于生成文本的不同解码策略的直观介绍

