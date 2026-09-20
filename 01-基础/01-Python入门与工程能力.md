# 🐍 Python 入门与工程能力：先让"代码能跑"这件事站稳

> 对应知识点：地图 §17.1.1（Python 工程能力，🔴 深入）、§1.1（Python / Linux / Git）。
> 前置：无——这是 `01-基础` 的**第一间内容房**，也是两条路线的共同起跑线。
> 一句话：**大模型工程师的"母语"是 Python，但"会写 Python"≠"能干活"——环境隔离、命令行、Git 版本管理、数据处理与结构化代码，才叫工程能力。这一课把后者补齐，后面每一篇的代码在你机器上都能同样跑起来。**
> 可复现：本文数字来自 `code/scripts/python_basics_demo.py`（Windows 中文 stdout 已处理；本机 CPU 环境一键复制）。

---

## 📑 本章目录

1. [为什么把它当第一课：先扫清"跑不起来"](#0-为什么把它当第一课先扫清跑不起来)
2. [环境：uv / conda / venv，先把"家"安好](#1-环境uv--conda--venv先把家安好)
3. [Python 十分钟上手：从任意语言跳到 Python](#2-python-十分钟上手从任意语言跳到-python)
4. [Linux 命令行 + Git：逃不掉的日常](#3-linux-命令行--git逃不掉的日常)
5. [numpy / pandas 起步：让数据"听话"（含实测）](#4-numpy--pandas-起步让数据听话含实测)
6. [dataclass：给数据行一个"类型"（含实测）](#5-dataclass给数据行一个类型含实测)
7. [熟悉工程性写法：做什么都像管线](#6-熟悉工程性写法做什么都像管线)
8. [常见坑](#7-常见坑)
9. [衔接下一章](#8-衔接下一章)
10. [参考资料](#9-参考资料)

---

## 0. 为什么把它当第一课：先扫清"跑不起来"

很多大模型课程的"第一课"是数学公式或"GPT 是什么"——你听着热血，回来打开教程，`import torch` 报错，热情瞬间归零。

本仓库反着来：**先让一切在你机器上真的能跑**，再谈原理。原因很现实：

- 后面 `03-机器学习-范式与评估`、`04-sklearn机器学习实战`、`05-深度学习-神经网络` 每篇都有必跑脚本/notebook——**如果连 numpy/sklearn 都装不明白，你在每一篇门口都会卡住 30 分钟起步**。
- 大模型从业者的日常工作不是"写一段诗意的代码"，而是**反复地：改数据 → 跑脚本 → 看日志 → 调参数 → 提交代码**。这四个动作的每一环都是 Python 工程能力。

所以这一课不教你"ChatGPT 原理"，教的是四样"地基"：**环境（安 10 分钟）→ 语言（读 10 分钟）→ 命令行/Git（每日用）→ 数据处理入门（配实测）**。学完你就能直接进入真正的"三条 AI 主线"。

> 完全零编程基础？也没关系——第 2 节是"从零跳进 Python"，不需要你先学过别的语言。

---

## 1. 环境：uv / conda / venv，先把"家"安好

### 1.1 为什么第一步是"环境"而不是"代码"

Python 的依赖地狱是真实的：项目 A 要 `numpy==1.26`，项目 B 要 `numpy==2.x`。直接 `pip install` 到全局，装 A 破坏 B、装 B 破坏 A。为解决这个，"**每个项目一个独立的环境**"是 2026 年的基本常识。

三个词先分清：

| 工具 | 是什么 | 一句话 |
|---|---|---|
| `venv` | Python 自带的最小隔离工具 | 造一个文件夹当"独立 Python 家"，`pip install` 只装进它 |
| `conda` | 环境 + 包管理的全家桶（Anaconda/Miniconda） | 不只能管 Python，还能装非 Python 依赖（CUDA、MKL），环境切换 `conda activate xxx` |
| `uv` | 新一代极速包管理（Rust 写的） | `uv pip install` / `uv venv`，速度约为 pip/conda 数倍，2025 起越来越多仓库官方推荐 |

**你实际最常遇到的差别**：Colab / Kaggle / 服务器里常有 `conda activate`；而不少 2026 年的 LLM 项目（尤其 FastAI、某些微调 repo）直接用 `uv`。两者都认识、知道"这是换环境"即可，不必死记命令。

> 🧠 一句话记住：**环境 = "每个项目私有的依赖全家桶"；`venv`/`conda`/`uv` 只是三种换肩膀的方式。**

### 1.2 装环境的最小闭环（任何人此刻都能做完）

```bash
# 方式一：用 uv（约 10 秒装好一个项目环境）
uv venv .venv                # 在本目录建一个 .venv 环境
source .venv/bin/activate    # Windows 用 .venv\Scripts\activate
uv pip install numpy pandas scikit-learn

# 方式二：用 conda（老牌，带完整工具链）
conda create -n llm python=3.11 numpy pandas scikit-learn
conda activate llm

# 装完验证（两条都认识就行）
python -c "import numpy; print(numpy.__version__)"
```

本仓库的所有脚本/notebook 都在 **numpy 2.3 / pandas 3.0 / sklearn 1.9** 上实测通过；你的版本只要在上面这个"任意独立环境"里，就不会互相踩踏。

---

## 2. Python 十分钟上手：从任意语言跳到 Python

不管你是从 C/Java/没写过程序来，Python 的哲学只有一句话：**读起来像大白话**。给你 90% 的场景够用的"速记卡"：

```python
# ── 基础四件套 ──
x = 5                # 不用声明类型：'int'
s = "hello"          # 字符串（引号单双都行）
lst = [1, 2, 3]      # 列表（可变、有序）
d = {"a": 1, "b": 2} # 字典（键值对，大模型无处不在：参数、配置、JSON）

# ── 控制流（缩进就是代码块，别用花括号）──
if x > 3:
    print("大")          # 缩进的这行属于 if
else:
    print("小")

# ── 列表推导式：造列表的一句话写法 ──
squares = [i * i for i in range(5)]      # [0, 1, 4, 9, 16]

# ── 函数 / 类（类不用 self 之外的魔法）──
def mean(xs):
    return sum(xs) / len(xs)

class Record:
    def __init__(self, name, score):  # self = "这个对象自己"
        self.name  = name
        self.score = score
    def passed(self):
        return self.score >= 60
```

**你真的只需要会这么多就去下一课**——`numpy/pandas/sklearn/torch` 的 API 不会被"语言本身"再挡住。遇到看不懂的语法，Google/搜索即可，这是所有人（包括作者）的日常。

> 想深一层：大模型时代，"写代码"越来越多变成"读代码 + 让模型改代码"。你的任务从"背语法"变成"**能看懂 100 行脚本在干什么、它跑挂了日志哪行报错**"——这就是下面第 4 节"工程能力"的意义。

---

## 3. Linux 命令行 + Git：逃不掉的日常

§17.1.1 把 Python **工程能力**（而不是"Python 语法"）当知识点，因为它还包括两样同级的工具：**命令行**与 **Git**。理由直白：训练/评测/部署都在 Linux 服务器与集群上，而你的代码从第一天就该进版本库。

### 3.1 命令行只记 8 条（够覆盖 95% 服务器操作）

```bash
pwd              # 我在哪          cd  myproj   # 进目录
ls               # 这里有什么       ls -lh *.py  # 只看 py，带大小
cat train.py     # 看文件全部      head -20 train.py  # 只看前 20 行
mkdir logs       # 建目录          mv a.py b.py  # 重命名/移动
grep "error" log.txt         # 在日志里搜"error"
python train.py              # 跑脚本
nohup python train.py > train.log &   # 后台跑并记日志（服务器日常）
```

> Windows 用户别慌：这些命令在你的 Git Bash / WSL / 云主机的 Linux 环境里长这样；日常看日志、改配置、跑脚本，三条起手式 `cd / ls / python xxx.py` 足够。

### 3.2 Git 只记 5 条（本仓库每篇提交都在用）

```bash
git status            # 改了啥（红=没暂存，绿=已暂存）
git add -A            # 全加进暂存区
git commit -m "完整描述"   # 打一个提交
git pull && git push  # 拉到最新 / 推上去
git log --oneline | head -10    # 看最近 10 次提交
```

你在这个仓库看到的每一次 commit（`29fa7c7`「01-基础补齐三篇」、`167a7c8`「PLAN v2」……）都是用这 5 条完成的。**能持续提交，你的实验和别人的协作就都可回退、可追溯**——这正是"工程能力"和"随手写文件"的分界线。

---

## 4. numpy / pandas 起步：让数据"听话"（含实测）

`numpy`（多维数组/矩阵运算）与 `pandas`（表格数据）是 Python 数据科学的左膀右臂，也是 `sklearn`、`torch` 的地板。两种情况都会持续遇到：**读表格 → 清洗 → 分组统计**。

直接看一份**真跑出来的**脚本输出（`python_basics_demo.py` 实验 B，作者本机）：

```text
总行数 100 · score 缺失 3 行 · 班级 ['B', 'A', 'C']
score 描述统计（describe）:
count     97.00
mean      80.92
std        8.23
min       59.78
25%       76.48
50%       81.60
75%       86.98
max      101.86
按班级分组均值（groupby）:
class
A    83.27
B    80.06
C    79.27
```

三行必背 API：

```python
import numpy as np, pandas as pd

df = pd.DataFrame({"class": ["A", "A", "B"],
                   "score": [88.0, 72.5, 91.0]})
df.describe()                    # 快速统计（上面那个"count/mean/std/…"就是它）
df["score"].isna().sum()         # 数缺失值（真实数据里第一眼要看这个）
df.groupby("class")["score"].mean()   # 按组聚合
arr = df["score"].to_numpy()     # 表格 → numpy 数组（给 sklearn/torch 用）
```

**为什么要优先学处理表格？** 后面 `04-sklearn机器学习实战` 的每一步（读数据、清洗、训练集划分、看指标）都长在 pandas 上；`08-RAG` 里的文档元数据、`09-Agent` 里的工具调用日志，本质也全是"表格思维"。

> 📌 本课只给"起步四件套"；深挖在 `04-sklearn机器学习实战` 的 §3–4。你现在只要"能读一份表格、能数缺失、能分组"，就够了。

---

## 5. dataclass：给数据行一个"类型"（含实测）

从第 4 节开始，你反复看到"一个数据行"的概念：一个学生（名字/成绩/是否及格）、一条订单（城市/数量/金额）。**用裸 `dict` 存行的最大问题是没有"类型"**——字典里是 `"score"` 还是 `"Score"`？是字符串还是数字？写错只有跑起来才爆炸。

Python 的标准答案（2017 起内置）：**`dataclass`**——三行定义一个"自带类型的行"，构造时核对字段，还能直接转成 JSON/dict。看真跑输出（实验 C）：

```text
Alice      score= 92.5  passed=True
Bob        score= 74.0  passed=True
Chen       score= 58.5  passed=False
asdict → 直接喂给 json/pandas:
   {'name': 'Alice', 'score': 92.5, 'passed': True}
```

对应代码（全仓库的数据行风格，`09-Agent`、`13-工程化` 也会见到）：

```python
from dataclasses import dataclass, field, asdict

@dataclass(order=True)
class Record:
    name: str
    score: float
    passed: bool = field(default=False)

recs = [Record("Alice", 92.5), Record("Bob", 74.0), Record("Chen", 58.5)]
for r in recs:
    r.passed = r.score >= 60
print(asdict(recs[0]))   # {'name': 'Alice', 'score': 92.5, 'passed': True}
```

招式的价值在**出管线**时炸响——把 100 万行裸 dict 变成 `Record` 这种"带类型的行"，任何字段写错在构造那一下立刻报类型错，而不是跑 1 小时后才在第五节崩。

---

## 6. 熟悉工程性写法：做什么都像管线

「数据处理 → 建模 → 评测」在真实项目里都是一条**流水线**（pipeline）：一步步变换、每一段可复现、最后有个可打印的汇总。社区里这被称为 **"脚本式开发"**：**一段脚本，把输入变成输出，中间每个环节都有日志**。

看一眼 `python_basics_demo.py` 实验 D 的完整闭环（真跑输出）：

```text
清洗后保留 5/6 行（丢 1 行空 city）
按城市汇总（sum/mean）:
city
上海      155  77.5  18900.00   9450.00
北京      230  115.0  20000.50  10000.25
深圳       80   80.0   7800.25   7800.25
→ 三段式：pandas 接数据 → 类型化(dataclass) → 分组聚合
```

它背后的 4 行，就是"管线思维"的最小样板：

```python
# 1. 接  raw_rows = pd.read_csv(...)                       # 接原样数据
# 2. 清  df2 = raw_rows.dropna(subset=["city"])            # 去脏行
# 3. 型  recs = [Record(str(c), float(g)) for c, g in zip(...)]  # 类型化
# 4. 出  df2.groupby("city")[["orders","gmv"]].agg(["sum","mean"])
```

之后你读 `04-sklearn` 的完整建模、`08-RAG` 的检索管线，会发现全都是这个骨架的放大版：**接 → 清 → 型 → 算 → 看**。这就是第 4 节说的"做什么都像管线"。

---

## 7. 常见坑

| 坑 | 症状 | 修法 |
|---|---|---|
| 全局 `pip install`，项目互相污染 | 装 A 后 B 报 numpy 版本错 | 每个项目用 `venv/conda/uv` 独立环境 |
| 同一个脚本跑出不同版本结果 | 今天 `import` 的库和昨天不是一套 | 把 `pip freeze > requirements.txt` 或 `uv.lock` 提交，别裸 `install` |
| 中文 print 报 GBK 编码错 | Windows 下 `print("你好")` 崩 | 脚本开头 `sys.stdout.reconfigure(encoding='utf-8', errors='replace')`（本仓库所有脚本都已内置） |
| 不知道改了哪些文件/回不去 | 改坏一版不可逆 | 每完成一小步就 `git add -A && git commit -m "描述"` |
| 在 Colab 与本地结果不一致 | 本地装了新库本地对、云端旧 | 数据/代码/库版本三样都锁住再复现；先把 notebook 导出成脚本跑 |
| `list.index()` / 下标越界 | index 可能不存在 | 改用 `dict` 或先判 `x in lst`；工程代码默认"数据是脏的" |

---

## 8. 衔接下一章

这一课把你从"会写 Python 的人"拉成"**能在这门课里真正动起来的人**"：环境能装、代码能跑、数据能读、行有类型、能提交版本。你的工具箱现在是：

- **环境** → `02-机器学习中的数学` 的每个公式验证脚本、`04-sklearn` 的每段建模代码，都在它的上面跑；
- **命令行/Git** → 后面的每篇提交都靠它；看到 `0affd6c`、`29fa7c7` 这样的提交号先想到"能追溯、能回退"；
- **numpy/pandas** → `03-机器学习-范式与评估` 的「不平衡陷阱」、`04-sklearn` 的端到端，第一个动作全是 `pandas.read_*`。

按推荐顺序，下一篇是 [02-机器学习中的数学](02-机器学习中的数学.md)——它会带你进入"**搞懂 model 在算什么**"的武器库：线性代数、概率、最优化。公式不要怕，到时候你已经能用 numpy 亲手验证每一个。

> 一句话带走：**工程能力 = 环境跑得起来 + 数据看得懂 + 行有类型 + 版本可回滚。四样都不难，但缺一样，后面的路都会踩坑。**

---

## 9. 参考资料

- 知识地图 §17.1.1（Python 工程能力）、§1.1（Python/Linux/Git）
- 本仓库：`code/scripts/python_basics_demo.py`（本文全部实测数字来源）；`04-sklearn机器学习实战.md`（下一个用 Python 深挖的地方）；`09-Agent体系`、`13-工程化与基础设施`（管线思维延伸）
- 官方入门：Python 官方 tutorial（前三章足矣）；《Hands-On Machine Learning》第 2 章"端到端机器学习项目"（本课管线的放大版）
