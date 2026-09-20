# 📖 NLP 与 Transformer 前置：让机器读懂语言

> 本文是 **`01-基础` 第 6 篇内容课（§17.1.5 Transformer 前置：NLP 与序列建模）**。前面我们有了数学（第 2 篇）、sklearn 实操（第 4 篇）、神经网络（第 5 篇），这一课把它们全部用到**人类语言**上：让机器完成分类、理解、翻译乃至生成文本。它是 `02-核心原理` 里 Transformer 的**直接前置站**——Attention 先于 Transformer，Transformer 先于 LLM。
>
> 看完这一课，你会理解大语言模型（LLM）最核心的两件事：**"词"是怎么变成向量的**，以及**"注意力"是怎么让模型读懂上下文的**。

---

## 📑 本章目录

1. [什么是 NLP：从图灵测试到 ChatGPT](#0-什么是-nlp从图灵测试到-chatgpt)
2. [文本数字化：从句子到数字矩阵](#1-文本数字化从句子到数字矩阵)
3. [词嵌入：把词变成"有意义的向量"](#2-词嵌入把词变成有意义的向量)
4. [文本分类实战（scikit-learn）](#3-文本分类实战scikit-learn)
5. [语言模型：让机器"续写"](#4-语言模型让机器续写)
6. [注意力机制与 Transformer](#5-注意力机制与-transformer)
7. [NLP 评估与常见任务](#6-nlp-评估与常见任务)
8. [NLP × LLM：大语言模型是怎么炼成的](#7-nlp--llm大语言模型是怎么炼成的)
9. [学习路线与推荐资源](#8-学习路线与推荐资源)

---

## 0. 什么是 NLP：从图灵测试到 ChatGPT

自然语言处理（Natural Language Processing，NLP）是让计算机**理解、处理、生成人类语言**的技术。它不是单一任务，而是一整个任务族：翻译、情感分析、摘要、问答、对话、语音转文字……

### 0.1 起源时间线：一部"从规则到规模"的历史

| 年代 | 事件 | 里程碑 |
| --- | --- | --- |
| **1950** | 图灵提出"**模仿游戏**"（图灵测试） | 第一次把"机器会说话"定义为 AI 目标 |
| **1954** | 乔治城实验：机器翻译俄语→英语 | 人们一度幻想"翻译马上要解决"，结果高估了几十年 |
| **1966** | ELIZA 聊天机器人 | 用**规则模板**伪装成心理咨询师，暴露了早期的短板 |
| **1980s** | 统计方法兴起（n-gram、HMM） | 从"人工写规则"转向"从数据里学规律" |
| **2003** | Bengio 提出**神经网络语言模型** | 词向量 + 神经网络预测下一个词，奠基 LLM |
| **2013** | Mikolov 等发布 **Word2Vec** | "国王 − 男人 + 女人 ≈ 女王"惊艳全场 |
| **2014** | Bahdanau 提出**注意力机制** | 机器翻译从"死记整句"进化到"边看边译" |
| **2017** | 谷歌发布 **Transformer** | NLP 的 "iPhone 时刻" |
| **2018** | **BERT**（双向）与 **GPT**（单向生成） | 预训练大模型时代开启 |
| **2022** | **ChatGPT** 发布 | 大语言模型走向大众 |

> 🧠 一句话记住：**NLP 的历史 = "人工规则 → 统计 → 神经网络 → 大规模预训练"的不断升级。**

### 0.2 NLP 与你已学知识的连接

| NLP 环节 | 用到前面哪一课 |
| --- | --- |
| 文本 → 向量 | 第 2 篇线性代数；本课词嵌入 |
| 分词 → 词典 → 词频 | 第 4 篇 Pandas / sklearn |
| 文本分类 | 第 5 篇神经网络 / 第 4 篇逻辑回归 |
| 预测下一个词 | 第 2 篇概率（MLE）+ 交叉熵（信息论） |
| 注意力打分 | 第 2 篇点积 + softmax |

---

## 1. 文本数字化：从句子到数字矩阵

### 1.1 机器只认数字

神经网络不会读字。所有 NLP 的第一步，都是把文本变成**数字**。转化的坏办法和好办法，正是本课的主线：

```text
坏办法：只记"这个字出现过没有" → 词袋（BoW）
          │
          ▼
好办法：记住"这个词在什么语境出现" → 词嵌入（Embedding）
```

### 1.2 分词（Tokenization）：先切成"零件"

所有任务的第一步都是把句子切成词（token）。

- **英文**：按空格（"I love NLP" → `["I", "love", "NLP"]`），再处理标点、大小写。
- **中文**：没有天然空格，需要分词器（如 `jieba`）："我爱自然语言处理" → `["我", "爱", "自然语言处理"]`

```python
import jieba

text = "我爱深度学习，也喜欢自然语言处理。"
print("jieba 分词:", list(jieba.cut(text)))

en_text = "I love Deep Learning and Natural Language Processing!"
print("英文简单切分:", en_text.lower().split()[:5])
```

> 🌰 生动例子：分词像拆乐高——不先拆成小块，后面"组合计数"就没法做。中文分词难在歧义："自然语言处理"是一个词还是三个？分词器就是靠词典 + 统计猜出来的。

### 1.3 词袋模型（Bag of Words）：只数出现过没有

思路：建一个**词典**（所有出现过的词），每篇文档用一个"长度为词典大小"的向量表示，位置上的数字 = 这个词出现几次。

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = [
    "猫喜欢鱼 狗喜欢骨头",
    "狗喜欢散步 猫喜欢睡觉",
    "鸟喜欢唱歌",
]

vec = CountVectorizer(token_pattern=r"[\u4e00-\u9fa5]+")   # 中文按连续汉字切
X = vec.fit_transform(docs).toarray()

print("词典:", vec.get_feature_names_out())
print("词袋矩阵:\n", X)
# 每行 = 一个文档；每列 = 一个词的计数
```

> ⚠️ 词袋的三个明显缺点：**①丢失顺序**（"猫咬狗"和"狗咬猫"向量一模一样）；②**维度爆炸**（词典 5 万词就是 5 万维）；③把"的、了、是"这类高频词和"鲸鱼"这种关键词同等看待。

### 1.4 TF-IDF：给词"称体重"

**TF-IDF** 改进计数法的两个直觉：

- **TF（词频）**：一个词在本文档出现越多越重要。
- **IDF（逆文档频率）**：一个词在**越少**文档里出现越稀有、越有区分力（"鲸鱼"比"的"值钱）。

$$
\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \log\frac{N}{\text{DF}(w)}
$$

其中 $N$ 是文档总数，$\text{DF}(w)$ 是包含词 $w$ 的文档数。

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "猫喜欢鱼 狗喜欢骨头",
    "狗喜欢散步 猫喜欢睡觉",
    "鸟喜欢唱歌",
]

tvec = TfidfVectorizer(token_pattern=r"[\u4e00-\u9fa5]+")
X = tvec.fit_transform(docs).toarray()
weights = X.sum(axis=0)

print("词典:", tvec.get_feature_names_out())
print("各词总权重（越高越有区分力）:\n",
      sorted(zip(tvec.get_feature_names_out(), np.round(weights, 3)),
             key=lambda t: t[1], reverse=True))
```

> 🌰 生动例子：图书检索。搜"机器学习"——出现在 100 本书里，一般重要；搜"分层贝叶斯"——只出现在 2 本专业书里，**几乎能直接锁定目标**。TF-IDF 就是给"稀有又相关"的词加权。

### 1.5 N-gram：给词袋找回一点"顺序"

把"词袋"升级为"连续 n 个词的袋子"：

- 1-gram：`["我", "爱", "学习"]`
- 2-gram：`["我爱", "爱学习"]`
- 3-gram：`["我爱学习"]`

```python
def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

tokens = ["我", "爱", "学习", "机器学习"]
print("2-gram:", ngrams(tokens, 2))
```

> 📌 N-gram 能抓住"不爱"和"爱"的搭配（"我不爱学习" ≠ "我爱学习"）。它是语言模型的古老祖先，下一节会用到。

---

## 2. 词嵌入：把词变成"有意义的向量"

### 2.1 词袋的终极问题：向量里没有"语义"

"猫"和"狗"的词袋向量完全正交（不同维度），尽管它们语义相近；"苹果"（水果）和"苹果"（公司）无法区分。**人类语言的本质是"语境"，词袋把语境全扔了。**

### 2.2 核心思想：分布假设

> **"You shall know a word by the company it keeps."（看一个词的朋友，就知道它是什么词。）—— J.R. Firth, 1957**

"猫"和"狗"为什么像？因为它们都常和"宠物、喂、毛茸茸"同现。**共现（Co-occurrence）信息 = 语义信息。**

### 2.3 从零实现：共现矩阵 + SVD（经典方法）

用一个小语料造"词 × 词"共现矩阵，再用第一课的 **SVD** 压缩成低维词向量：

```python
import numpy as np

corpus = [
    "猫 喜欢 鱼",
    "狗 喜欢 骨头",
    "猫 狗 都是 宠物",
    "鱼 生活 在 水里",
    "鸟 喜欢 唱歌 树",
]
tokens = sorted({w for s in corpus for w in s.split()})
idx = {w: i for i, w in enumerate(tokens)}

# 共现矩阵：窗口=1，数"相邻出现"的次数
co = np.zeros((len(tokens), len(tokens)))
for s in corpus:
    ws = s.split()
    for i in range(len(ws) - 1):
        a, b = idx[ws[i]], idx[ws[i+1]]
        co[a, b] += 1
        co[b, a] += 1

# SVD 降到 2 维（第一课 1.10 节）
U, S, Vt = np.linalg.svd(co)
embeddings = U[:, :2] * S[:2]

for w, i in sorted(idx.items(), key=lambda t: t[1]):
    print(f"{w:4s}: ({embeddings[i,0]:6.2f}, {embeddings[i,1]:6.2f})")
```

> 🧭 观察：维度从"词典大小（个位数）"压缩到 2，但保留了结构——"猫"和"狗"共现多（都爱接"喜欢"），它们的向量应该比较近；"水里"和"树"则各占一角。**语义相近 = 向量相近**，这就是"嵌入"一词的意义：把词"嵌入"到一个能表达关系的低维空间。

### 2.4 Word2Vec：2013 年的"爆款"（2013, Mikolov 等）

SVD 方法要构建整个共现矩阵（词典 5 万 → 矩阵 25 亿格），不划算。Word2Vec 换了个思路：**直接用"预测"来学向量**。

两种架构：

- **CBOW**：用**上下文词**预测**中心词**。看"我 ___ 机器学习"，猜空格是"爱"。
- **Skip-gram**：用**中心词**预测**上下文词**。看"爱"，猜它周围可能出现"我、机器学习"。

数学目标（Skip-gram 简版）：对每个中心词 $c$ 和它的上下文词 $o$，最大化

$$
P(o \mid c) = \text{softmax}(\mathbf{u}_o \cdot \mathbf{v}_c)
$$

训练结束后，$\mathbf{v}_w$ 就是词向量。著名结果：**向量之间的加减法对应语义关系**：

$$
\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}
$$

（"国王"去掉"男性"的味道、加上"女性"的味道 ≈ "女王"）

```python
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 用上面 2.3 的 SVD 词向量做个"相似度体检"
print("猫 vs 狗 相似度:", round(cosine_sim(embeddings[idx["猫"]], embeddings[idx["狗"]]), 3))
print("猫 vs 鱼 相似度:", round(cosine_sim(embeddings[idx["猫"]], embeddings[idx["鱼"]]), 3))
```

> 🔑 一句话总结：**词嵌入 = "语境决定的向量"。** 现代 LLM 的输入层就是这样一张"查表"（Embedding 矩阵），只是词表更大、维度更高（常见 768~4096 维）。

---

## 3. 文本分类实战（scikit-learn）

做一件最经典、也最贴近生活的 NLP 任务：**短信垃圾分类**。为了无需下载数据集，我们造一个简短的模拟语料（真实项目换成真实 CSV 即可，流程完全一样）。

### 3.1 造数据 + 训练 + 评估（完整 Pipeline）

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

texts = [
    # 垃圾短信
    "恭喜您中奖了 点击链接领取大奖",
    "全场促销 低至一折 快来抢购",
    "您的账户异常 请立即回复验证",
    "免费领取红包 加微信领取",
    "限时优惠 点击链接马上参与",
    # 正常短信
    "妈妈 晚上回家吃饭吗",
    "明天下午三点开会 记得带上笔记本",
    "快递已到菜鸟驿站 请凭码取件",
    "老师通知 作业周五前提交",
    "今天天气很好 我们去公园散步吧",
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]   # 1=垃圾，0=正常

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(token_pattern=r"[\u4e00-\u9fa5]+")),
    ("clf", LogisticRegression(max_iter=1000)),
])

# 数据量很小，直接用交叉验证评估（比单次划分更诚实）
scores = cross_val_score(pipe, texts, labels, cv=5)
print("5 折交叉验证准确率:", np.round(scores, 3), "平均", round(scores.mean(), 3))

# 再演示一次标准流程：切分 → 训练 → 测试
pipe.fit(texts, labels)
news = ["恭喜您中奖了！", "请明天来参加家长会", "特价 最后一天 错过再等一年"]
pred = pipe.predict(news)
print("新短信预测（1=垃圾）:", pred)
```

> 📌 整个流程只用了几行：**分词统计（TfidfVectorizer）+ 分类器（LogisticRegression）+ Pipeline 打包**。真实项目（垃圾邮件过滤、评论情感分析、新闻分类）就是这条流水线，只是数据量大了、特征工程更讲究。

### 3.2 情感分析小例子：正面/负面评论

```python
reviews = [
    "这家店太好吃了 服务也棒",
    "味道一般 位置难找",
    "强烈推荐 性价比超高",
    "又贵又难吃 太失望了",
    "环境不错 但是上菜太慢",
]
senti = [1, 0, 1, 0, 0]

pipe2 = Pipeline([
    ("tfidf", TfidfVectorizer(token_pattern=r"[\u4e00-\u9fa5]+")),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe2.fit(reviews, senti)

for r in ["分量足 味道好", "服务态度差 再也不来"]:
    print(f"{r!r:20s} -> {'正面' if pipe2.predict([r])[0]==1 else '负面'}")
```

### 3.3 三大实战忠告

1. **数据量>算法**：先多收集数据，再考虑换花哨模型；深度学习没有大数据就是空转。
2. **类别不平衡**：垃圾短信只占 5% 时，别只用准确率（第二课学过：要看精确率/召回率/F1）。
3. **测试集必须"干净"**：文本里的预处理（去停用词、分词）只能在训练集上学，防止数据泄漏。

> 💡 进阶：学到 Transformer 后，同一任务升级为「预训练模型 + 微调」，准确率通常显著提升——但**数据准备和评估的逻辑一模一样**。

---

## 4. 语言模型：让机器"续写"

### 4.1 什么是语言模型（Language Model）

**语言模型 = 学会"下一个词/字符的概率"的系统**：

$$
P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \ldots, w_{t-1})
$$

有了这个概率，机器就能**续写**、可以给句子"打分"（哪句更像人话），甚至翻译、摘要、对话——**GPT 就是把这个目标做到极致的语言模型**。

### 4.2 起源：从 n-gram 计数到神经网络

| 阶段 | 方法 | 特点 |
| --- | --- | --- |
| **计数时代** | n-gram：统计"历史上 3 个词后面最常跟什么" | 简单但稀疏：没见过的搭配概率=0 |
| **平滑时代** | 给 n-gram 加平滑（backoff、Kneser-Ney） | 缓解稀疏，但仍是死记统计 |
| **2003 起点** | Bengio 神经网络语言模型 | 用词向量 + 神经网络**泛化**到没见过的搭配 |
| **2013+** | Word2Vec、LSTM LM | 向量化 + 长距离记忆 |
| **2017+** | Transformer（GPT） | 大规模并行 + 注意力，横扫一切 |

### 4.3 动手：字符级 n-gram 语言模型（能跑能生成）

用一个极小的"样本诗库"训练"下一个字"统计模型，然后让它**续写**：

```python
from collections import defaultdict
import numpy as np

poems = ["春眠不觉晓 处处闻啼鸟", "床前明月光 疑是地上霜", "白日依山尽 黄河入海流"]
text = "".join(poems)

# 统计转移：每种字符后面跟着什么字符，各出现几次
counts = defaultdict(lambda: defaultdict(int))
for i in range(len(text) - 1):
    counts[text[i]][text[i + 1]] += 1

# 加极小平滑：没见过的转移也给一点点概率
chars = sorted(set(text))
def next_char(c, temperature=1.0):
    freq = np.array([counts[c].get(n, 0.05) for n in chars], dtype=float)
    probs = np.exp((freq - freq.max()) / temperature)
    probs /= probs.sum()
    return np.random.default_rng(0).choice(chars, p=probs)

rng = np.random.default_rng(42)
start = "春"
gen = [start]
for _ in range(20):
    gen.append(next_char(gen[-1]))
print("生成的文字：", "".join(gen))
```

> 🧭 观察：这个"婴儿版语言模型"用的是最朴素的计数法——它已经能模仿"五言诗"的节奏（因为学过的字符组合就那么些）。**把"字符"换成"子词"，把"计数表"换成"几十亿参数的神经网络"，就是 GPT。** 原理是同一件事：学 $P(\text{下一个} \mid \text{上文})$。

### 4.4 为什么大模型不能用 n-gram

- 词典 5 万词，看前 4 个词的组合 = $5万^4$ ≈ 6×10¹⁸ 种搭配——**地球上所有文本填不满**（数据稀疏）。
- 神经网络用"向量 + 权重"把相似语境**共享**了（"猫在沙发上睡觉"和"狗在沙发上打盹"共享大量模式），所以能泛化。
- 结论：**参数多 + 数据多 = 学会"规律"而不是"背句子"。**

---

## 5. 注意力机制与 Transformer

### 5.1 起源故事：从"死记硬背"到"边翻边看"

- **2014 年（加拿大）**：Bahdanau 等人在**机器翻译**中提出**注意力机制**。此前的序列模型要把整句压成一个"脑容量有限"的向量再翻译，长句子就忘前面；注意力让模型**翻译每个词时，动态地回看源句的相关部分**——像同传译员盯着原文的对应位置。
- **2017 年（谷歌）**：论文 **《Attention Is All You Need》** 提出 **Transformer**：干脆不用循环，**全靠注意力**做序列建模。它训练可并行、能捕捉长距离关系，成为 GPT/BERT 的基石。

### 5.2 注意力的三个角色：Q、K、V

把"注意力"理解成一个"**搜索 - 匹配 - 取回**"的过程：

| 角色 | 是什么 | 生活类比 |
| --- | --- | --- |
| **Query（查询）** | "我在找什么" | 你在搜索框输入的关键词 |
| **Key（键）** | "我能提供什么" | 每个网页的标题/标签 |
| **Value（值）** | "匹配上了给什么内容" | 网页正文 |

计算三步：

**① 打分**：用**点积**衡量"这个词与那个词的匹配度"

$$
\text{score}(q, k) = q \cdot k
$$

**② 归一化 + 概率化**：用 softmax 把分数变成"该看多少"的权重（第一课学过！）

$$
\alpha_{ij} = \text{softmax}_j\!\left(\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}\right)
$$

（除以 $\sqrt{d_k}$ 是防止分数太大导致 softmax 饱和、梯度消失）

**③ 加权求和**：按权重混合所有 Value（线性组合！）

$$
\text{Output}_i = \sum_j \alpha_{ij} \mathbf{v}_j
$$

> 🧭 生动例子：读"猫坐在垫子上，它很舒服"时，"它"这个词要决定自己指谁——它会去"查询"句子里所有词："猫"匹配度高（90%），“垫子”匹配度低（10%），于是"它"的表示里**猫的语义占大头**。这就是"代词消解"在注意力里的样子。

### 5.3 动手：NumPy 实现单头自注意力

```python
import numpy as np

def softmax_rows(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

# 一句话 4 个词的"假想词向量"（真实中是嵌入层输出的向量）
# 词：["猫", "坐", "垫子", "它"]
X = np.array([
    [1.0, 0.0, 0.2],   # 猫
    [0.0, 1.0, 0.0],   # 坐
    [0.2, 0.0, 1.0],   # 垫子
    [1.0, 0.1, 0.1],   # 它（和"猫"更像）
])

# 简版：直接用 X 当 Q、K、V（真实模型有可学习的 W_Q, W_K, W_V）
Q, K, V = X, X, X
d_k = Q.shape[-1]

scores = Q @ K.T / np.sqrt(d_k)     # ① 点积打分
weights = softmax_rows(scores)      # ② softmax 权重
output = weights @ V                # ③ 加权求和

print("注意力权重（每行和为 1）:\n", np.round(weights, 3))
print("\n输出 = 按权重混合后的词向量:\n", np.round(output, 3))
```

> 🧭 看输出矩阵：`output[3]`（"它"的新表示）应该明显偏向"猫"的向量——**"它"通过注意力"看"向了猫**。整句话的每个词都互相重新表达了一遍，这就是"自注意力（Self-Attention）"。

### 5.4 多头注意力：一组不够，来八组

用 **8 组不同的 Q/K/V**（8 个头），各自"关注不同的关系"：

- 一个头关注"代词指谁"（语法）
- 一个头关注"谁和谁是主语宾语"（句法）
- 一个头关注"哪些词在同一个话题"（语义）
- 最后把 8 个头的输出拼起来再投影一次。

> 生动类比：**8 位侦探同时从不同角度重新调查案情，最后汇总成一份更完整的报告。**

### 5.5 Transformer 的整体结构

```text
输入：词向量序列（嵌入 + 位置编码）
  │
  ▼
┌─────────────────────────────────┐
│ 多头自注意力 (Multi-Head Attention)│ ──► 残差连接 + LayerNorm
│ 前馈网络 FFN（就是两层 MLP！）      │ ──► 残差连接 + LayerNorm
└─────────────────────────────────┘
  │   （这一大块叫一个 Transformer 层，GPT 有 12~96 层）
  ▼
输出：每个位置更新后的向量 → 接分类头/预测下一个词
```

### 5.6 位置编码：补上"顺序感"

注意力本身**不在乎位置**（"猫咬狗"和"狗咬猫"算出的注意力一样），所以必须给每个位置加"位置信息"。经典做法是用**正弦函数**：

$$
PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad
PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)
$$

> 🧭 直觉：为每个位置生成一个**像"指纹"一样的唯一向量**，加到词向量上——模型从此知道"第 3 个词"和"第 7 个词"的位置不同。现代模型也常改用"可学习的位置向量"。

### 5.7 掩码（Masking）：GPT 为什么"只看左边"

- **双向掩码（BERT）**：每个词都能看整句话（适合理解任务）。
- **因果掩码（GPT）**：每个词**只能看它自己和左边的词**（因为要预测下一个词，不能偷看答案）。

实现上就是把"不允许看的位置"的注意力分数设成 $-\infty$，softmax 后权重变成 0。

```python
def causal_mask(scores):
    n = scores.shape[0]
    mask = np.triu(np.full((n, n), -np.inf), k=1)   # 右上角置 -inf
    return scores + mask

print("有掩码的注意力分数:\n", np.round(causal_mask(scores), 2))
print("softmax 后（右下角全是 0 = 不许看未来）:\n",
      np.round(softmax_rows(causal_mask(scores)), 2))
```

> 🎯 到这里，Transformer 的全部核心部件已经齐了：**嵌入 → 多头自注意力 → 前馈 MLP → 位置编码 → 掩码**。GPT 就是把这些层堆很多很多遍。本课的信息量到这里已经非常接近"能读懂 GPT 论文"了。

---

## 6. NLP 评估与常见任务

### 6.1 不同任务，不同尺子

| 任务 | 常用指标 | 直觉 |
| --- | --- | --- |
| 文本分类 | 准确率、精确率、召回率、F1 | 第 4 篇学过：垃圾过滤看重精确率 |
| 语言模型 | **困惑度（Perplexity）** | 模型平均在几个词之间犹豫（第 2 篇信息论） |
| 机器翻译 | **BLEU** | 译文和参考译文的 n-gram 重合度 |
| 摘要 | ROUGE | 摘要和参考摘要的词重合度 |
| 生成对话 | 人工评估 + GPT 打分 | 自动指标往往不够，人也得看 |

### 6.2 困惑度快速回顾

$$
\mathrm{PPL} = e^{\text{平均交叉熵}} = \exp\!\left(-\frac{1}{T}\sum_t \log P(w_t^* \mid \text{上文})\right)
$$

- PPL = 10：模型每次选词，相当于在 10 个候选里犹豫 → 还不错。
- PPL = 5000：选词几乎靠猜 → 很差。
- 现代大模型在通用文本上通常能压到很低（个位数到几十）。

```python
def perplexity(log_probs):
    return float(np.exp(-np.mean(log_probs)))

print("好模型困惑度:", round(perplexity(np.array([-0.05, -0.1, -0.08])), 2))   # ≈ 1.13
print("差模型困惑度:", round(perplexity(np.array([-3.2, -2.8, -3.0])), 2))     # ≈ 19
```

---

## 7. NLP × LLM：大语言模型是怎么炼成的

### 7.1 三步走：预训练 → 微调 → 对齐

```text
① 预训练（Pretraining）
   在互联网级海量文本上，反复做"预测下一个词"（就是第 4 节的语言模型目标）
   → 学到语言规律、知识、推理能力（GPT 系列就是这条路）

② 监督微调（SFT）
   用"问题-答案"示例教它"像助手一样回答"
   → 从"会接话"变成"会答题"

③ 对齐（RLHF 等）
   用人偏好反馈进一步调整（奖励模型 + 强化学习）
   → 让它"有用、诚实、无害"
```

### 7.2 Tokenizer：为什么"词"也可以是"子词"

英文词太多（"run、runs、running"），中文更没法按"字"穷举。现代模型几乎都用 **BPE（Byte Pair Encoding）** 等**子词**分词法：

- 把"低频率"再拆小："unbelievable" → "un" + "believ" + "able"
- 常见词保持完整，罕见词拆成片段 → 词表固定（如 5 万）、OOV（生词）问题基本消失

```python
from collections import Counter

# 极小语料：{"词": 出现次数}（带 </w> 词尾标记）
corpus = Counter({"unbelievable": 3, "believable": 5, "unable": 2, "able": 4})
vocab = {w: list(w) + ["</w>"] for w in corpus}
learned_merges = []

for _ in range(6):
    pair_counts = Counter()
    for w, freq in corpus.items():
        chars = vocab[w]
        for i in range(len(chars) - 1):
            pair_counts[(chars[i], chars[i + 1])] += freq
    best = pair_counts.most_common(1)[0][0]   # 出现最多的一对 = 本轮合并
    learned_merges.append(best)

    for w in vocab:                            # 在所有词里执行本次合并
        chars, merged, i = vocab[w], [], 0
        while i < len(chars):
            if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best:
                merged.append(best[0] + best[1]); i += 2
            else:
                merged.append(chars[i]); i += 1
        vocab[w] = merged

print("学到的合并规则:", learned_merges)
print("tokenize 'unbelievable':", vocab["unbelievable"])
print("tokenize 'able':", vocab["able"])
```

> 📌 关键点：**tokenizer 是查表器，不负责理解**——理解发生在它输出的 token 进入 Transformer 之后。这也是"大模型的词表"总是固定大小的原因。

### 7.3 上下文窗口与"记忆"

- 模型的"记忆" = **上下文窗口**（如 8k~128k 个 token）。
- 超出窗口的早期内容会"被忘掉"——所以长文档用 RAG（检索增强）把相关内容"喂"进窗口。
- 注意力与"远近"无关（都能直接看到），窗口大小纯粹是**算力与显存**的限制。

### 7.4 全图景：四个数字串起整个 LLM

```text
文本 → [Tokenizer 分词] → token 数字 → [Embedding 查向量]
     → [位置编码] → 多头自注意力 × N 层 → 前馈 MLP × N 层
     → [输出层 softmax] → 下一个词的概率 → [采样] → 文本

训练时：预测真实词 → 交叉熵损失 → 反向传播 → 梯度更新（AdamW）
```

> 🎯 有没有发现？这张图的每一块，都是本课（以及第 2/4/5 篇的前置课）**亲手实现过或用数学推导过的东西**。LLM 不神秘——它就是"把本章所有积木，堆到巨大的规模"。

---

## 8. 学习路线与推荐资源

### 建议学习路径

1. **第一遍（3-5 天）**：跑通本章代码，重点看 2.3 节 SVD 词向量和 5.3 节自注意力的输出。
2. **第二遍（1 周）**：用真实数据集（如中文酒店评论）复现第 3 节分类流程；把第 5 节的自注意力代码扩展成多头。
3. **进阶**：进入《4. 大模型基础》和 **2. The LLM Scientist**——那里的"从零训练小 GPT"实践，你已经有全部前置知识。

### 📚 推荐课程（保留原推荐，按需扩充）

**动手指南**
- [Real Python - NLP with spaCy](https://realpython.com/natural-language-processing-spacy-python/)：spaCy 处理真实文本的详细指南
- [Kaggle - NLP 指南](https://www.kaggle.com/learn-guide/natural-language-processing)：Notebook 化的动手解释
- [Hugging Face 官方课程（中文）](https://huggingface.co/learn/nlp-course/zh-CN/chapter1/1)：从 tokenizer 到微调 Transformer 的标准入门

**经典必读**
- [Jay Alammar - The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)：图解 Transformer，强烈推荐
- [Jay Alammar - The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)：理解 Word2Vec 架构
- [colah"s blog - 理解 LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)：序列模型的经典好文
- [Jake Tae - 从零构建 PyTorch RNN](https://jaketae.github.io/study/pytorch-rnn/)：RNN/LSTM/GRU 的简洁实现
- [斯坦福 CS224n：NLP 与深度学习](https://web.stanford.edu/class/cs224n/)：NLP 领域最著名的课程
- [Jurafsky & Martin - Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)：NLP 权威教科书（免费在线）

> 💡 学习工具：中文分词用 [jieba](https://github.com/fxsjy/jieba)（`pip install jieba`）；想要更强的文本处理用 [spaCy](https://spacy.io/)；做 Transformer 实践用 [Hugging Face transformers](https://github.com/huggingface/transformers)。

---

## 🎓 本节小结

| 章节 | 核心收获 |
| --- | --- |
| §1 文本数字化 | 分词 → 词袋 → TF-IDF → N-gram，从"数次数"到"看权重" |
| §2 词嵌入 | 分布假设 + 共现/SVD/Word2Vec = **语义相近 → 向量相近** |
| §3 文本分类 | Tfidf + 逻辑回归 + Pipeline = 经典 NLP 全流程 |
| §4 语言模型 | 学 $P(\text{下一个} \mid \text{上文})$：从 n-gram 计数到神经网络 |
| §5 注意力 | Q/K/V + 点积 + softmax = "动态看上下文"；Transformer 由此而生 |
| §6 评估 | 分类看 F1，语言模型看困惑度 |
| §7 LLM | 预训练 → 微调 → 对齐；GPT = 大号 Transformer 语言模型 |

**到这里，`01-基础` 的六篇内容课全部完成**：Python 🐍 → 数学 🧮 → 机器学习范式与评估 📏 → sklearn 实战 🛠️ → 神经网络 🧠 → NLP 📖，再加入口篇的路线自测和收尾篇的 GPU 账本。公共地基已经铺完。

---

## ✅ 衔接下一章

本课是 **`01-基础` 的倒数第二站**，也是通往 `02-核心原理` 最直接的跳板：

- 你把 **词向量（Word2Vec/SVD）** 和 **Q/K/V 注意力** 亲手实现过——这正是 Transformer 的两根支柱。
- **路线 A（算法/研究）** → 下一篇就是 `02-核心原理/01-Self-Attention`：你会看到本课 5.3 节的"点积注意力"被换上皮位置编码、多头堆叠，长成真正的大模型骨架。
- **路线 B（应用/工程）** → 可先读 `07-GPU-CUDA与并行计算`（`01-基础` 收尾），再进 `06-应用开发`。

> 一句话带走：**Attention 先于 Transformer，Transformer 先于 LLM——你刚踩的每一步，都是后面所有"新词"的旧相识。**
