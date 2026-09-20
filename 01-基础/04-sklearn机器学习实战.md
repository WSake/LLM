# 🐍 sklearn 机器学习实战：从数据处理到模型部署一步到位

> 本文是 **`01-基础` 第 4 篇内容课（§17.1.3 机器学习 · 实战篇）**。前一篇 `02-机器学习中的数学` 给了你公式与直觉，`01-Python入门与工程能力` 给了你环境、numpy/pandas 起步与数据管线骨架；这一课把它们焊在一起：**用 Python 亲手完成"拿到数据 → 清洗 → 可视化 → 建模型 → 评估 → 用起来"的完整流程**。如果你已读过 01 篇走到"能跑、能读、会分组"的程度，这篇就是无缝接力。
>
> 目标：学完这一章，你能独立用 `NumPy / Pandas / Matplotlib / scikit-learn` 跑通一个真实的机器学习项目，并理解每个库"为什么长这样"。

---

## 📑 本章目录

1. [为什么是 Python：机器学习的第一语言](#0-为什么是-python机器学习的第一语言)
2. [环境准备与工具链](#1-环境准备与工具链)
3. [NumPy：按"矩阵"思考的数组库](#2-numpy按矩阵思考的数组库)
4. [Pandas：让表格数据"听话"](#3-pandas让表格数据听话)
5. [Matplotlib：把数据"画"出来](#4-matplotlib把数据画出来)
6. [scikit-learn：机器学习工具箱](#5-scikit-learn机器学习工具箱)
7. [端到端实战：从数据到模型](#6-端到端实战从数据到模型)
8. [与 LLM 的关联](#7-与-llm-的关联)
9. [学习路线与推荐资源](#8-学习路线与推荐资源)

---

## 0. 为什么是 Python：机器学习的第一语言

### 0.1 起源故事：从"脚本语言"到"AI 通用语"

- **1991 年（荷兰）**：**吉多·范罗苏姆（Guido van Rossum）** 发布 Python 的第一个版本。设计目标是"**可读、优雅、让程序员开心**"——所以 Python 用缩进表示代码块，读起来像大白话。
- **1995-2006 年（美国）**：数值计算库 **NumPy**（2005）把"用 Python 做科学计算"从不可能变成现实；2006 年 **SciPy**、2007 年 **scikit-learn**、2008 年 **Pandas** 相继诞生，机器学习所需的"生态"逐步齐活。
- **2010 年后**：**深度学习觉醒**——2015 年 **PyTorch** 和 **TensorFlow** 都选择 Python 作为官方语言；2017 年 **Transformer 论文** 的参考代码也是 Python。于是 Python 从"胶水语言"坐稳了 **AI 第一语言**的位子。
- 对比：R 擅长统计、MATLAB 价格昂贵且偏教学、Java/C++ 开发效率低——Python 在"**开发效率 × 生态完整 × 学习门槛低**"三者上取得了最佳平衡。

> 🧠 一句话记住：**Python 赢不是因为语言本身最强，而是因为"每个人想做的 AI 事情，都能找到现成的轮子"。**

---

### 0.2 机器学习标准流程：一张地图

任何机器学习项目，无论多大多小，都走这几步。本课的每一节恰好对应一步：

```text
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│ ① 收集数据  │ → │ ② 清洗整理  │ → │ ③ 可视侦察  │ → │ ④ 特征工程  │
│ (Pandas)   │   │ (Pandas)   │   │(Matplotlib)│   │ (NumPy/   │
└────────────┘   └────────────┘   └────────────┘   └─  sklearn)─┘
                       │                                │
                       ▼                                ▼
┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
│ ⑧ 上线使用  │ ← │ ⑦ 评估改进  │ ← │ ⑥ 训练模型  │ ← │ ⑤ 准备数据  │
│ (部署/API)  │   │(评估指标)   │   │ (sklearn)  │   │(train/test)│
└────────────┘   └────────────┘   └────────────┘   └────────────┘
```

### 0.3 与第一章数学的衔接

| 这一步做什么 | 用的 Python 工具 | 用到第一课的数学 |
| --- | --- | --- |
| 数据组织成表 | Pandas、NumPy | 矩阵、张量 |
| 特征缩放 | sklearn `StandardScaler` | 期望、方差、正态分布 |
| 线性回归 | sklearn `LinearRegression` | 矩阵方程、最小二乘、MLE |
| 分类概率 | sklearn `LogisticRegression` | softmax、交叉熵、贝叶斯 |
| 降维可视化 | sklearn `PCA` / `t-SNE` | 特征值、奇异值分解 |
| 评估模型 | sklearn 指标 | 概率、置信区间、假设检验 |
| 调参优化 | sklearn `GridSearchCV` | 数值优化的思想（搜索最优） |

> 💡 学习建议：本课每个代码块都**亲自动手运行**，然后**故意改坏一个参数**，看看会发生什么。出错不可怕，出错才是学得最牢的时候。

---

## 1. 环境准备与工具链

### 1.1 Python 基础：一分钟回顾

如果你完全没写过 Python，先建立这三个概念（其余用到再查）：

```python
# 1) 变量与基本类型
price = 29.9            # 数字
name = "苹果"            # 字符串
tags = ["甜", "脆", "红"] # 列表
stock = {"苹果": 20, "梨": 15}   # 字典：键值对

# 2) 流程控制
for tag in tags:
    if tag == "甜":
        print("这个很甜:", tag)

# 3) 函数：把逻辑包起来复用
def total_revenue(prices, quantities):
    return sum(p * q for p, q in zip(prices, quantities))

print("总营收:", total_revenue([29.9, 10], [20, 15]))
```

### 1.2 环境：Anaconda / venv 与 Jupyter

- **Anaconda**：预装好数据科学全家桶的发行版，新手最省心；缺点是把环境装得很"大"。
- **venv / conda env**：给不同项目隔离依赖的"独立小房间"。**同一个库装两个版本互相打架**是新手第一惨案，环境隔离能根治它。
- **Jupyter Notebook**：把代码、输出、图表、文字混在同一个文档里，是数据科学家的工作台。`pip install jupyter` 后运行 `jupyter notebook` 即可。
- **Google Colab**：浏览器里免费开箱即用，自带 GPU（后面训练小模型很方便）。

### 1.3 数据科学"四大件"

| 库 | 一句话定位 | 本课章节 |
| --- | --- | --- |
| **NumPy** | 把 Python 列表变成高速数组，支持矩阵运算 | §2 |
| **Pandas** | 把表格数据变成 DataFrame，方便筛选/分组/合并 | §3 |
| **Matplotlib** | 把数据画成图，一眼看出规律 | §4 |
| **scikit-learn** | 集成了几乎所有经典机器学习算法 + 预处理/评估 | §5 |

> 📌 大模型时代的补充：深度学习用的 **PyTorch** 可以理解为"带 GPU + 自动微分的 NumPy"，其 `torch.Tensor` 的用法和 NumPy 高度相似——所以把本课的 NumPy 学扎实，后面学 PyTorch 会非常快。

---

## 2. NumPy：按"矩阵"思考的数组库

### 2.1 起源故事：让 Python 跑得比 C 快

- **1995 年（美国）**：Jim Hugunin 等人开发 **Numeric**，第一次让 Python 能做正经科学计算。
- **2005 年**：**Travis Oliphant** 把 Numeric 和 Numarray 合并，发布 **NumPy**，核心是 `ndarray`（n 维数组）。
- 今天的 NumPy 相当于所有 Python 科学计算的**地基**：Pandas、scikit-learn、SciPy、Matplotlib 全部站在它上面；连 PyTorch 的编程范式都在模仿它。

为什么需要专门的数据结构？因为 Python 原生列表**存的是"对象引用"**（松散、慢），而 NumPy 数组把数字**连续紧密地排在一起**，并调用底层 C/Fortran 优化库批量计算——速度可以快 **几十到上百倍**（后面有实测）。

> 🧠 一句话记住：**NumPy 数组 = "自带高性能发动机的表格式数据"，让你用数学的"矩阵思维"而非"循环思维"写代码。**

### 2.2 创建数组：从列表到 ndarray

```python
import numpy as np

a = np.array([1, 2, 3])            # 一维数组（向量）
m = np.array([[1, 2], [3, 4]])     # 二维数组（矩阵）

zeros = np.zeros((2, 3))           # 全 0
ones  = np.ones((2, 3))            # 全 1
seq   = np.arange(0, 10, 2)        # [0 2 4 6 8]：等间隔整数
grid  = np.linspace(0, 1, 5)       # [0 0.25 0.5 0.75 1]：5 个等分点
eye   = np.eye(3)                  # 单位矩阵（第一章的 I！）

print("形状:", m.shape, "| 维度:", m.ndim, "| 元素数:", m.size)
```

#### 形状（shape）与变形（reshape）

```python
v = np.arange(12)              # 0..11 的一维数组
matrix3x4 = v.reshape(3, 4)    # 变成 3 行 4 列
print(matrix3x4)

# 变形不改变数据，只改变"怎么读"。把一张 12 个像素的图 reshape 成 3×4 或 2×2×3 都行
```

> 🌰 生动例子：把数组想成"一柜子的货"。`reshape` 只是**重新码放**（数据不变），而"形状"告诉你现在按几排几列码。

### 2.3 索引、切片与布尔掩码

```python
m = np.arange(12).reshape(3, 4)

print(m[0])            # 第 0 行：[0 1 2 3]
print(m[1, 2])         # 第 1 行第 2 列：6
print(m[:, 1])         # 所有行的第 1 列（冒号 = "全部"）
print(m[1:, :2])       # 第 1 行起、前 2 列

# 布尔掩码：按条件选数据（机器学习里超常用）
scores = np.array([78, 92, 55, 88, 60])
passed = scores >= 60
print("及格名单:", scores[passed])       # [78 92 88 60]
print("及格人数:", passed.sum())          # 布尔和 = True 的个数
```

> 🌰 生动例子：掩码就像考试后老师**打勾划线**——不改变原数据，只"挑"出符合条件的。数据集清洗 80% 的操作都是这种"条件筛选"。

### 2.4 广播（Broadcasting）：小数组自动"复制"

广播规则：**维度从右往左对齐，要么相等、要么有一个是 1，就能自动扩展**。

```python
prices = np.array([10, 20, 30])       # 3 种商品单价
discount = 0.8                         # 全场八折
final = prices * discount              # 标量 0.8 自动广播到每个元素
print("折后价:", final)                 # [ 8. 16. 24.]

# 更有用的例子：4 天×3 商品的销量表，每天都有不同的"基准销量"
sales = np.array([[10, 20, 30],
                  [12, 22, 28],
                  [11, 19, 33],
                  [15, 21, 27]])
base = np.array([10, 20, 30])          # 基准行
diff = sales - base                    # 每行自动减去基准行（广播）
print("偏离基准的销量:\n", diff)
```

> 🌰 生动例子：广播 = 店员在货架广播"全场打八折"——**一个命令，所有货架自动执行**。你不用写嵌套循环去"逐个乘"。

### 2.5 向量化：为什么"别写 for 循环"

对比"逐个元素算"和"一次性批量算"：

```python
import time

big = np.random.default_rng(0).uniform(0, 1, 1_000_000)

# 方式一：Python 循环（慢）
t0 = time.perf_counter()
result_loop = [x * 2 + 1 for x in big]
t_loop = time.perf_counter() - t0

# 方式二：NumPy 向量化（快几十倍）
t0 = time.perf_counter()
result_vec = big * 2 + 1
t_vec = time.perf_counter() - t0

print(f"循环: {t_loop*1000:.1f} ms | 向量化: {t_vec*1000:.2f} ms | 加速 {t_loop/t_vec:.0f} 倍")
```

> 📌 原理：循环在 Python 解释器里**逐元素解释执行**；向量化把操作**整个打包发给底层 C 代码**。LLM 训练同样靠"批量矩阵乘法"（GPU）而非逐个算——**向量化思维从 NumPy 一路用到 PyTorch**。

### 2.6 矩阵运算：把第一章的公式敲进电脑

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("矩阵乘法 A@B:\n", A @ B)          # 对应数学章的 AB
print("逐元素乘法 A*B:\n", A * B)         # 注意：这不是矩阵乘法！
print("转置 A.T:\n", A.T)

# 解线性方程组 Ax = b（第一章 1.8 节）
b = np.array([5, 1])
x = np.linalg.solve(A, b)
print("方程组的解:", x)

# 特征分解（第一章 1.9 节 PCA 的地基）
eigvals, eigvecs = np.linalg.eig(np.array([[2, 1], [1, 2]]))
print("特征值:", eigvals, "特征向量:\n", eigvecs)
```

> ⚠️ 新手第一大坑：`A * B` 是**逐元素相乘**（Hadamard 积），`A @ B` 才是**矩阵乘法**。在深度学习中，几乎所有地方都要用 `@` 或 `torch.matmul`。

### 2.7 随机数：造数据、做抽样

强烈建议使用 `np.random.default_rng(seed)`——**同一个 seed 永远得到同一串随机数**，训练/实验可复现。

```python
rng = np.random.default_rng(42)

uniform = rng.uniform(0, 1, 5)         # [0,1) 均匀分布
normal  = rng.normal(170, 6, 5)        # 正态分布（均值 170，标准差 6）→ 身高
integers = rng.integers(0, 10, 5)      # 随机整数
choice  = rng.choice(["猫", "狗", "兔"], 5, p=[0.5, 0.3, 0.2])  # 带权重采样

print("身高样本:", np.round(normal, 1))
print("宠物采样:", choice)
```

> 📌 关联：第一章里说的"从多项分布采样"、LLM 的"按概率选下一个词"，在 NumPy 里就是 `rng.choice(..., p=概率)`。预训练时打乱数据顺序、随机 masking，都靠这套随机数。

---

## 3. Pandas：让表格数据"听话"

### 3.1 起源故事：一个量化交易员的"起床气"

- **2008 年（美国）**：量化基金 AQR 的分析师 **韦斯·麦金尼（Wes McKinney）** 受够了在 Python 里笨拙地处理金融时间序列数据，于是动手写了 Pandas，目标是"**数据处理界的 Excel**"。
- 取名 **Pandas** = "Panel Data"（面板数据，经济学的术语）的谐音；也致敬了 Python 社区"让分析更优雅"的期许。
- 2012 年他出版《利用 Python 进行数据分析》，把 Pandas 的使用范式（DataFrame、groupby、merge）普及给全世界。2015 年起 Pandas 成为 Python 数据科学的事实标准。

> 🧠 一句话记住：**Pandas = 用 Python 代码操作的 Excel，而且功能更强、可复现、能处理上亿行数据。**

### 3.2 两个核心对象：Series 与 DataFrame

- **Series**：带索引的一列数据（≈ Excel 的一列 + 行号）。
- **DataFrame**：多个 Series 拼成的二维表（≈ 一张 Excel 工作表）。

```python
import pandas as pd

# 构造一个"水果店每日销售表"
df = pd.DataFrame({
    "水果":   ["苹果", "梨", "香蕉", "苹果", "梨"],
    "日期":   ["周一", "周一", "周一", "周二", "周二"],
    "销量":   [20, 15, 30, 24, 12],
    "单价":   [5.0, 4.0, 3.0, 5.0, 4.5],
})

print(df)
print("列名:", list(df.columns))
print("每列类型:\n", df.dtypes)      # 自动推断数据类型
print("快速统计:\n", df.describe())   # 计数/均值/标准差/分位数（第一课统计学！）
```

> 🌰 生动例子：把 DataFrame 想成**一张会算数的 Excel 表**：`describe()` 一秒给出均值、方差、分位数——第一课"期望与方差"的工具化。

### 3.3 读取数据：从文件到内存

```python
# 最常用的几种：统一变成 DataFrame
# df = pd.read_csv("data.csv")                 # CSV
# df = pd.read_excel("data.xlsx")              # Excel
# df = pd.read_json("data.json")               # JSON
```

> 📌 实际项目中 80% 的时间都花在"把各种格式的脏数据变成干净的 DataFrame"上——**数据分析师真正的工作是"洗数据"**。

### 3.4 筛选、排序与条件查询

```python
df = pd.DataFrame({
    "水果": ["苹果", "梨", "香蕉", "苹果", "梨"],
    "日期": ["周一", "周一", "周一", "周二", "周二"],
    "销量": [20, 15, 30, 24, 12],
    "单价": [5.0, 4.0, 3.0, 5.0, 4.5],
})

print("销量 >= 20 的行:\n", df[df["销量"] >= 20])

# 复合条件：苹果 且 销量高
query = df[(df["水果"] == "苹果") & (df["销量"] >= 20)]
print("苹果且畅销:\n", query)

# 排序
print("按销量降序:\n", df.sort_values("销量", ascending=False))

# 新增一列：计算营收 = 销量 × 单价
df["营收"] = df["销量"] * df["单价"]
print("新增营收列:\n", df)
```

> ⚠️ 新手两大坑：条件之间要用 `&`（不是 `and`）；括号别省。`df["列"]` 是 Series，`df[["列1","列2"]]` 才是 DataFrame。

### 3.5 处理缺失值：脏数据怎么救

```python
import numpy as np

df = pd.DataFrame({
    "姓名": ["张三", "李四", "王五", "赵六"],
    "成绩": [88, np.nan, 75, np.nan],    # NaN = 缺失值
})

print("缺失情况:\n", df.isna().sum())
print("删除含缺失的行:\n", df.dropna())
print("用 0 填充:\n", df.fillna(0))
print("用均值填充（更常用）:\n", df.fillna(df["成绩"].mean()))
```

> 🌰 生动例子：缺失值就像**表格上的墨渍**。选项有三：删掉整行（`dropna`）、当 0 处理（`fillna(0)`）、用整列均值"脑补"（`fillna(mean)`）。选哪种取决于"墨渍"是随机还是系统性的。

### 3.6 groupby：分组聚合

```python
df = pd.DataFrame({
    "班级": ["A", "A", "B", "B", "C"],
    "姓名": ["张三", "李四", "王五", "赵六", "孙七"],
    "分数": [88, 92, 75, 60, 80],
})

grouped = df.groupby("班级")["分数"].agg(["mean", "std", "count"])
print("每班统计:\n", grouped)
```

> ⚠️ 记一个经典报错：`df.groupby("班级")["分数"].mean()` 返回 Series；想要表格输出，用 `agg(...)` 或 `.mean().reset_index()`。

> 🌰 生动例子：groupby = "先把人按班分堆，再对每一堆做统计"——Excel 的"数据透视表"，SQL 的 `GROUP BY`，都是同一个概念。

### 3.7 合并与连接：把多张表拼起来

```python
students = pd.DataFrame({"学号": [1, 2, 3], "姓名": ["张三", "李四", "王五"]})
scores   = pd.DataFrame({"学号": [1, 2, 3], "分数": [88, 92, 75]})

merged = pd.merge(students, scores, on="学号")   # 按公共列横向拼接
print("内连接:\n", merged)
```

Pandas 与 SQL 的对照（会 SQL 的同学秒懂）：

| Pandas | SQL |
| --- | --- |
| `df[df["列"]>10]` | `SELECT * FROM df WHERE 列>10` |
| `df.groupby("k").agg(...)` | `SELECT k, ... FROM df GROUP BY k` |
| `pd.merge(a, b, on="k")` | `JOIN` |
| `df.sort_values("c")` | `ORDER BY c` |

### 3.8 高频技巧速查

```python
df = pd.DataFrame({
    "水果": ["苹果", "梨", "香蕉", "苹果", "梨"],
    "销量": [20, 15, 30, 24, 12],
})

print(df["水果"].value_counts())          # 每种水果出现次数
print(df.head(2))                         # 看前 2 行（先"探个头"）
print(df["销量"].max(), df["销量"].min()) # 极值

# apply：对整列应用函数（自定义逻辑）
df["销量标签"] = df["销量"].apply(lambda x: "高" if x >= 20 else "低")
print(df)
```

> 📌 关联：LLM 的数据管线里，**清洗、去重、过滤垃圾文本**就是用 Pandas 处理千万行的文本表格；`Dataset`/`DataLoader` 等工具只是这个思路在大规模下的延伸。

---

## 4. Matplotlib：把数据"画"出来

### 4.1 起源故事：一个神经科学家的"职业病"

- **2002 年（美国）**：神经科学家 **John D. Hunter** 需要在研究里快速画脑电波形图，却发现 Python 没有像 MATLAB 那样好用的绘图库，于是自己写了一个，取名 **Matplotlib**（MATLAB + plotlib）。
- 1960 年代 **John Tukey** 提出"**探索性数据分析（EDA）**"理念：**看图往往比看数字更快发现问题**——Matplotlib 就是这套理念的工具化。
- 今天它既是 pandas 的默认绘图引擎（`df.plot()`），也是 seaborn（更美观的高级封装）和 PyTorch 可视化生态的地基。

> 🧠 一句话记住：**先画图，再建模。图能省下你一半的错误建模时间。**

### 4.2 四种最常用的图

先做一步"中文美化"配置，否则图表标题的中文会显示成方块：

```python
import matplotlib.pyplot as plt
# Windows 推荐微软雅黑/黑体；macOS 可以换成 "Arial Unicode MS" 或 "Heiti TC"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False   # 修复负号 "-" 显示为方块的问题
```

```python
import numpy as np
import matplotlib.pyplot as plt   # 无显示环境可先 export MPLBACKEND=Agg

rng = np.random.default_rng(1)

# 1) 折线图：看趋势
days = np.arange(1, 31)
temps = 18 + 8 * np.sin(days / 5) + rng.normal(0, 1, 30)
plt.figure(figsize=(6, 3))
plt.plot(days, temps, "o-")
plt.title("30 天温度趋势")
plt.xlabel("天"); plt.ylabel("温度(°C)")
plt.tight_layout()

# 2) 散点图：看两个变量的关系
plt.figure(figsize=(6, 3))
x = rng.normal(170, 8, 200)
y = 0.6 * (x - 170) + 60 + rng.normal(0, 3, 200)
plt.scatter(x, y, alpha=0.5)
plt.title("身高 vs 体重（正相关）")
plt.xlabel("身高(cm)"); plt.ylabel("体重(kg)")
plt.tight_layout()

# 3) 直方图：看分布形状（第一章的正态分布！）
plt.figure(figsize=(6, 3))
samples = rng.normal(170, 6, 2000)
plt.hist(samples, bins=40, edgecolor="white")
plt.title("身高分布（钟形曲线）")
plt.xlabel("身高(cm)"); plt.ylabel("人数")
plt.tight_layout()

# 4) 柱状图：看类别大小
categories = ["苹果", "梨", "香蕉", "橙子"]
sales = [450, 310, 520, 380]
plt.figure(figsize=(6, 3))
plt.bar(categories, sales, color=["red", "green", "yellow", "orange"])
plt.title("各水果销量")
plt.ylabel("销量(斤)")
plt.tight_layout()

print("已生成 4 张图（在 Jupyter 中会直接显示，本示例只演示绘图代码）")
```

> 🌰 生动例子：折线图看"趋势"、散点图看"关系"、直方图看"分布"、柱状图看"多少"——先想清楚**你要回答什么问题**，再选图，而不是每张图都画一遍。

### 4.3 一张图读懂"68-95-99.7 法则"

把第一章的正态分布画出来，你会"看到"标准差的意义：

```python
mu, sigma = 170, 6
x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

plt.figure(figsize=(7, 4))
plt.plot(x, pdf, "b-")
for k, color in [(1, "red"), (2, "orange"), (3, "yellow")]:
    mask = np.abs(x - mu) <= k * sigma
    plt.fill_between(x[mask], pdf[mask], alpha=0.3, color=color)
plt.title("正态分布：±1σ=68%，±2σ=95%，±3σ=99.7%")
plt.xlabel("身高(cm)"); plt.ylabel("概率密度")
plt.tight_layout()
```

> 📌 关联：你会自然地想到——"如果某个样本落在 3σ 之外，它很可能是个**异常值**"。异常检测、白化（whitening）、标准化（z-score）全都是这套正态语言。

### 4.4 可视化"看走眼"的陷阱

- **相关 ≠ 因果**（第一章 3.6 节）：散点图只能说"一起变"，不能证明"谁导致谁"。
- **坐标轴起点故意不归零**：柱状图 4 万 vs 4.5 万会被画成巨大差异——看轴！
- **颜色误导**：热力图用彩虹色会让人误判中间值的大小。
- **数据太少强行"插值"成平滑曲线**：会画出不存在的规律。
- 记住一句话：**"一张图说明一切"的时候，恰恰最该警惕。**

> 💡 现代替代：需要更漂亮的图时，用 **Seaborn**（基于 Matplotlib 的高层封装）或 **Plotly**（交互式）。

---

## 5. scikit-learn：机器学习工具箱

### 5.1 起源故事：学术界共建的"工具箱"

- **2007 年（法国）**：法国国家信息与自动化研究所（INRIA）的 **David Cournapeau** 在 Google Summer of Code 中启动项目，2010 年发布了 1.0 版。scikit-learn（写作 `sklearn`）是**开放学术社区**长期共建的成果，也是 Python 机器学习的地基。
- 古希腊/工程师的智慧：sklearn 最伟大的贡献不是某个算法，而是**一套统一、一致的接口（API）**——所有模型都长一个样：

```text
model.fit(X, y)     # 训练：学习数据里的规律
model.predict(X)    # 预测：用规律给新数据下结论
model.score(X, y)   # 评估：模型表现如何
```

> 🧠 一句话记住：**scikit-learn = "插拔式"机器学习工具箱：想换算法？只改一行，其它代码照旧。**

### 5.2 数据划分：train_test_split

**绝不能拿训练用的数据来"考试"**——那等于开卷考试还带答案。标准做法是把数据分成：

- **训练集（train）**：模型学习的教材。
- **测试集（test）**：从没见过的考卷，用来诚实评估。
- （更严谨时再加**验证集**：调参用，最后才碰测试集。）

```python
import numpy as np
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 4))          # 200 个样本，4 个特征
y = rng.integers(0, 2, size=200)       # 二分类标签 0/1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape[0]} 样本 | 测试集: {X_test.shape[0]} 样本")
```

> 🔑 三个参数解读：`test_size=0.2` 留 20% 当考卷；`random_state=42` 固定随机切分（可复现）；`stratify=y` 保证训练/测试里"正负样本比例一致"（分层抽样，防止恰好把某类全切走）。

### 5.3 评估指标：别被"准确率"骗了

以"医疗检测"为例：1000 人里只有 10 个病人（正类）。

- **准确率（Accuracy）** = 全对 / 总数。如果一个模型**永远回答"没病"**，准确率高达 99%——但它毫无用处！
- 所以要看：
  - **精确率（Precision）** = 被诊断为"有病"的人里，真病比例。衡量"说了就要准"。
  - **召回率（Recall）** = 真病人里，被咱找出来的比例。衡量"不能漏掉"。
  - **F1** = 两者的调和平均，兼顾二者。
  - **混淆矩阵（Confusion Matrix）**：四种情况一目了然（真阳/假阳/真阴/假阴）。

```python
from sklearn.metrics import confusion_matrix, classification_report

y_true = np.array([1, 0, 1, 1, 0, 0, 0, 1])
y_pred = np.array([1, 0, 0, 1, 0, 1, 0, 0])

print("混淆矩阵:\n", confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, zero_division=0))
```

> 🌰 生动例子：肿瘤筛查要**高召回**（宁可错报、不可漏报）；垃圾邮件过滤要**高精确**（误删正常邮件更恼火）。**指标的选择由业务场景决定，不是技术问题。**

### 5.4 回归实战：线性回归

线性回归 = 第一章最小二乘的 sklearn 版：找直线（超平面）$y = Xw + b$ 使均方误差最小。

```python
from sklearn.datasets import load_diabetes        # 内置糖尿病数据集（无网络依赖）
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)                        # 内部解 (XᵀX)⁻¹Xᵀy

y_pred = model.predict(X_test)

print("R²（解释方差比例，最大 1）:", round(r2_score(y_test, y_pred), 3))
print("RMSE（误差的量纲化）:", round(mean_squared_error(y_test, y_pred) ** 0.5, 2))
print("特征重要性（回归系数）前 3:\n", sorted(zip(data.feature_names, model.coef_),
                                            key=lambda t: abs(t[1]), reverse=True)[:3])
```

> 📌 数学链接：`LinearRegression` 求的就是最小化 $\lVert Xw - y\rVert^2$ 的解析解（第一章 5.5 节正规方程）；在**高斯噪声假设**下它又等价于最大似然估计（第三章 3.8 节）。

### 5.5 分类实战：逻辑回归（softmax 的先导）

逻辑回归名字带"回归"，其实是**分类**器：它输出"属于每个类别的概率"。二分类用 sigmoid，多分类用 softmax（第一章 3.4 节）。

```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target          # 150 朵花，4 个特征，3 个品种

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=y)

clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
probs = clf.predict_proba(X_test)      # 每朵花属于 3 类的概率（和为 1）

print("准确率:", round(accuracy_score(y_test, y_pred), 3))
print("第一朵测试花的类别概率:", np.round(probs[0], 3))
print("预测类别:", y_pred[0], "| 真实类别:", y_test[0])
```

> 🧠 直觉：逻辑回归 = "先算每个类别的得分，再用 softmax 变成概率，最后取概率最大者"。**LLM 的最后一层干的就是完全一样的事**，只是把"4 个特征"换成了"几千维的隐藏向量"。

### 5.6 决策树与随机森林："二十问"游戏的大规模版

#### 决策树

决策树把"二十问"游戏自动化：每次问一个**最能区分数据**的问题，不断把数据切成更纯的子集。

- **起源**：1963 年 **Morgan & Sonquist** 提出自动交互检测；1984 年 **Breiman 等人**的 CART 算法让决策树可规模化应用。
- **分裂标准**：每次选择让"子节点纯度提升最多"的特征和阈值。常用 Gini 系数或信息增益（第一章的熵！）。

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

tree = DecisionTreeClassifier(max_depth=3, random_state=0)   # max_depth 限制深度，防过拟合
tree.fit(X_train, y_train)
print("决策树准确率:", round(tree.score(X_test, y_test), 3))
print("最重要的特征:", iris.feature_names[tree.feature_importances_.argmax()])
```

#### 随机森林：一群树的"投票民主"

一棵树容易**过拟合**（背答案）；随机森林训练**很多棵树**，每棵只用**随机抽样的一部分数据 + 随机挑选一部分特征**，最后投票。这就是 **Bagging（自助聚合）**。

- 名字里有"随机"四个来源：数据随机、特征随机、树不同、彼此独立。
- 效果：方差大幅下降、几乎不增加偏差 —— "**三个臭皮匠顶个诸葛亮**"的机器学习版。

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(X_train, y_train)
print("随机森林准确率:", round(rf.score(X_test, y_test), 3))

# 特征重要性：森林的"共识"比单棵树更可靠
for name, imp in sorted(zip(iris.feature_names, rf.feature_importances_),
                        key=lambda t: t[1], reverse=True):
    print(f"  {name}: {imp:.3f}")
```

> 🌰 生动例子：决策树 = 一个经验老到的医生问诊（容易有个人偏见）；随机森林 = 请 200 位医生各自独立问诊后**投票表决**——整体判断更稳。

### 5.7 K 近邻（KNN）：物以类聚

- **起源**：1951 年 **Fix & Hodges** 提出；1967 年 **Cover & Hart** 完善。它是**最古老也最容易理解**的机器学习算法。
- **思想**：新样本的分类 = 看它**最近的 K 个邻居**投什么票。
- 特点：**懒惰学习**——训练时不学任何参数，预测时才翻旧账；因此"慢"在预测、快在训练。

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("KNN(5) 准确率:", round(knn.score(X_test, y_test), 3))
```

> 🌰 生动例子：判断一个人是否爱运动，看 TA 常混的 5 个朋友是不是都爱运动——**"你不是你的圈子，但你很像你的圈子"。** K 越小越敏感（易受噪声影响），K 越大越平滑。

### 5.8 朴素贝叶斯（Naive Bayes）

直接在第一章的**贝叶斯定理**上做分类：

$$
P(类别 \mid 特征) \propto P(特征 \mid 类别) \times P(类别)
$$

"朴素" = 假设特征之间**相互独立**（现实中常不成立，但往往够用且极快）。

```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print("朴素贝叶斯准确率:", round(gnb.score(X_test, y_test), 3))
```

> 📌 关联：文本分类（垃圾邮件过滤）的经典实现就是它——"免费"和"中奖"同时出现的概率 ≈ 各自概率相乘（独立性假设）。

### 5.9 聚类：K-Means（无监督学习）

前面都是"有标签"的监督学习；聚类面对的是**没有答案**的数据：把相似的样本自动分成 K 组。

- **起源**：1957 年 **Stuart Lloyd** 在贝尔实验室为"脉冲编码调制（PCM）"设计量化算法，是 K-Means 原始版本；它至今是最常用的聚类算法。
- **思想**：反复两步骤——把每个点分给**最近的质心**；再把每个组的**质心更新为组内均值**。直到质心不再移动。
- 损失目标：最小化"每个点到所属质心的距离平方和"（就和最小二乘一脉相承）。

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X_blob, _ = make_blobs(n_samples=300, centers=4, random_state=0)   # 生成 4 团数据

kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)
labels = kmeans.fit_predict(X_blob)

print("找到的质心:\n", np.round(kmeans.cluster_centers_, 2))
print("各组样本数:", np.bincount(labels))
```

> 🌰 生动例子：超市想把顾客分成几类来精准营销——没有标准答案，全靠"**看谁和谁抱团**"。K 的选择常用**肘部法则**：画"损失 vs K"曲线，找拐点。

### 5.10 降维：PCA 与 t-SNE

高维数据（如图像、词向量）没法直接画图。**PCA**（第一章 1.9 节）把数据投影到"方差最大"的前几个方向上：

```python
from sklearn.decomposition import PCA

X_reduced = PCA(n_components=2).fit_transform(iris.data)   # 4 维 → 2 维
print("降维后形状:", X_reduced.shape)                       # (150, 2)
print("前两主成分解释方差比例:",
      np.round(PCA(n_components=2).fit(iris.data).explained_variance_ratio_, 3))
```

> 📌 关联：PCA 的数学就是特征分解（第一章）；**t-SNE / UMAP** 是更灵活的可视化降维方法，常用于把 LLM 的嵌入向量画在 2D 平面上看"语义聚类"。

### 5.11 预处理、管线与交叉验证

#### 特征标准化：让不同尺度的特征公平竞争

身高（cm，100 级别）和体重（kg，10 级别）量纲不同，会让"距离"被大数主导。标准化把每个特征变成均值 0、方差 1：

$$
z = \frac{x - \mu}{\sigma}
$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)   # 用"训练集的 μ 和 σ"变换
X_test_scaled = scaler.transform(X_test)   # 测试集必须复用同一套统计量！
print("缩放后均值≈0:", np.round(X_scaled.mean(axis=0), 6))
print("缩放后方差≈1:", np.round(X_scaled.var(axis=0), 3))
```

> ⚠️ 防泄漏铁律：`fit` 永远只在**训练集**上做；测试集只 `transform`。用整份数据算 μ/σ 再切分 = "作弊"（数据泄漏）。

#### Pipeline：把几步串成一条流水线

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
pipe.fit(X_train, y_train)
print("Pipeline KNN 准确率:", round(pipe.score(X_test, y_test), 3))
```

#### 交叉验证（Cross-Validation）：更诚实的成绩单

一次"训练/测试划分"有运气成分；**K 折交叉验证**把数据切成 K 份，轮流拿 1 份当考卷、其余当教材，K 次平均——每个样本都被考过一次。

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, X_train, y_train, cv=5)
print("5 折交叉验证准确率:", np.round(scores, 3))
print("平均:", round(scores.mean(), 3), "±", round(scores.std(), 3))
```

> 🧠 一句话记 Pipeline + CV：**"把洗数据、缩放、建模、评估打包，反复交叉验证取平均"**——这才是能上线的规范姿势。

---

## 6. 端到端实战：从数据到模型

把前面所有积木拼成一个完整项目。任务：**用鸢尾花（Iris）数据训练一个可比较的智能分类器**，体会"真实工作流"长什么样。

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# ① 数据：读进来，看一眼
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["品种"] = [iris.target_names[i] for i in iris.target]
print("数据规模:", df.shape)
print("每类数量:\n", df["品种"].value_counts())

# ② 划分
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

# ③ 统一流水线（缩放 + 模型），对比 4 个算法
models = {
    "逻辑回归":  LogisticRegression(max_iter=500),
    "K近邻":     KNeighborsClassifier(n_neighbors=5),
    "随机森林":  RandomForestClassifier(n_estimators=200, random_state=0),
    "支持向量机": SVC(random_state=0),
}

pipes = {name: Pipeline([("scale", StandardScaler()), ("model", model)])
         for name, model in models.items()}

# ④ 交叉验证打分（诚实成绩单）
print("\n=== 5 折交叉验证 ===")
results = {}
for name, pipe in pipes.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=5)
    results[name] = scores
    print(f"{name:8s} 平均准确率 {scores.mean():.3f} ± {scores.std():.3f}")

# ⑤ 用最优模型在"从未见过的测试集"上做最终评估
best_name = max(results, key=lambda k: results[k].mean())
best_pipe = pipes[best_name]
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)

print(f"\n最优模型: {best_name}")
print("测试集准确率:", round(best_pipe.score(X_test, y_test), 3))
print(classification_report(y_test, y_pred,
                            target_names=iris.target_names, zero_division=0))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
```

> 📋 这个 6 步流程（读数据 → 看分布 → 划分 → 多模型对比 → 交叉验证 → 最终测试）就是**你以后做任何 sklearn 项目的模板**。没有魔法，全是本课学过的积木。

## 7. 与 LLM 的关联

### 7.1 数据处理：思想完全一致，规模不同

| sklearn 时代的你会做 | LLM 训练里对应的事 |
| --- | --- |
| `pd.read_csv()` 读数据 | `datasets.load_dataset()` 读文本语料 |
| 清洗缺失值、去重 | 清洗 HTML、去重、按质量过滤语料 |
| `train_test_split` 划分 | 训练/验证/测试集划分 |
| `StandardScaler` 标准化 | `tokenizer` 把文本变成统一格式的数字序列 |
| `Pipeline` 串步骤 | `Dataset.map()` 批量预处理 |
| `cross_val_score` 交叉验证 | 定期在验证集上算 loss / 指标 |
| 随机种子复现 | 一切 `seed` 可复现实验 |

### 7.2 从 sklearn 到 PyTorch：思维平移

- sklearn：`model.fit()` → 内部帮你完成"前向 + 求梯度 + 更新"。
- PyTorch：每一步**显式可见**——前向 `outputs = model(x)`、损失 `loss = criterion(outputs, y)`、清零梯度 `opt.zero_grad()`、反向 `loss.backward()`、更新 `opt.step()`。
- 你本课在 sklearn 里建立的所有概念（过拟合、数据泄漏、交叉验证、学习率思想）到 LLM 训练中全部适用，只是**数据量、参数规模、算力**上升了几个数量级。

### 7.3 学完本课，你应该掌握的技能清单

- [ ] 能用 NumPy 完成矩阵运算、向量化、随机抽样
- [ ] 能用 Pandas 读数据、筛选、分组、合并、处理缺失
- [ ] 能用 Matplotlib 画出 4 种基本图表和一目了然的分布图
- [ ] 能熟练用 sklearn 的 `fit / predict / score` 训练并评估 4+ 种模型
- [ ] 能解释为什么需要 train/test 划分、为什么要交叉验证
- [ ] 能讲清楚"数据泄漏""过拟合""相关≠因果"三个大坑

---

## 8. 学习路线与推荐资源

### 建议学习路径

1. **第一遍（3-5 天）**：按本课顺序把代码全部跑通，参考下图"每节对应哪步"。
2. **第二遍（1 周）**：换一个真实数据集（如 Kaggle 的泰坦尼克号），完整走一遍第 6 节流程。
3. **进阶**：学完本课后进入《05-深度学习-神经网络》，把 `sklearn.fit` 换成 `PyTorch` 的自定义训练循环。

### 📚 推荐课程（保留原推荐，按需补充）

**Python 语言基础**
- [Real Python](https://realpython.com/)：全面的 Python 教程与文章库，非常友好
- [freeCodeCamp - 学习 Python](https://www.youtube.com/watch?v=rfscVS0vtbw)：两小时视频讲完核心语法
- [Python 官方教程](https://docs.python.org/3/tutorial/)：最权威的"说明书"

**数据科学四大件**
- [Python Data Science Handbook（Jake VanderPlas）](https://jakevdp.github.io/PythonDataScienceHandbook/)：免费电子书，NumPy/Pandas/Matplotlib 的黄金入门
- [NumPy 官方快速入门](https://numpy.org/doc/stable/user/absolute_beginners.html)：边做边学
- [Pandas 官方 10 分钟入门](https://pandas.pydata.org/docs/user_guide/10min.html)：最快上手路径
- [scikit-learn 官方教程](https://scikit-learn.org/stable/tutorial/index.html)：每个模型都有 live 示例

**机器学习系统课**
- [freeCodeCamp - 机器学习](https://youtu.be/i_LwzRVP7bg)：为初学者实测讲解多种算法
- [Udacity - 机器学习简介](https://www.udacity.com/course/intro-to-machine-learning--ud120)：涵盖 PCA 等多种概念，课程免费
- [吴恩达 - Machine Learning Specialization（Coursera）](https://www.coursera.org/specializations/machine-learning-introduction)：全球最经典的 ML 入门课，适合配合本课查漏
- [Kaggle Learn - Python/机器学习/可视化 微课](https://www.kaggle.com/learn)：每节 15 分钟，带真实数据集练习

> 💡 工具建议：装好 [Jupyter](https://jupyter.org/install) 或直接用 [Google Colab](https://colab.research.google.com/)，把每个代码块改成"自己的数据"再跑一遍，这是把知识转成技能最快的方法。

---

## 🎓 本节小结

| 你想做的事 | 交给谁 |
| --- | --- |
| 高性能数值/矩阵计算 | NumPy（§2） |
| 表格数据处理与清洗 | Pandas（§3） |
| 数据可视化发现规律 | Matplotlib（§4） |
| 训练/评估标准 ML 模型 | scikit-learn（§5） |
| 端到端项目流程 | 第 6 节的六步模板 |

**本课与 `02-机器学习中的数学` 的关系**：数学篇给你"为什么"（损失函数=最大似然、梯度下降原理），本课给你"怎么做"（sklearn 生态）。下一课《05-深度学习-神经网络》将把 torch 引入，从零搭出真正的深度学习模型。准备好，我们继续。😀
