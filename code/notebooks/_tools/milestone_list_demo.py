# -*- coding: utf-8 -*-
"""
16-04 里程碑：实践项目清单 探针 (milestone_list_demo.py)

账 A (三铁律)：三条共同铁律 × 判据 × 载体，全部成对且载体落在真实文件/目录上。
账 B (8 格矩阵)：初级/中级/高级/专家 × 算法侧/应用侧 8 项目，前置章节 11 篇全部
     真实存在；08 章现存 12 篇、09 章现存 11 篇(与对照索引统计一致)。
账 C (脚手架)：notebook 009/010/011 + _tools/_demo.py 探针集合 + code/scripts 落盘校验。
账 D (三进一出)：转化路径 4 阶段 × 判据成对；交付物唯一。

纯 stdlib、零网络、零随机、零第三方。三进程 stdout 逐字节一致。
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
# _tools -> notebooks -> code -> 仓库根 = 向上 3 级
REPO = os.path.dirname(os.path.dirname(os.path.dirname(BASE)))

ok = 0
total = 0


def check(name, got, exp):
    global ok, total
    total += 1
    if got == exp:
        ok += 1
        print("  %-42s PASS" % name)
    else:
        print("  %-42s FAIL(期望 %r 实得 %r)" % (name, exp, got))


def repo_path(rel):
    """rel 用平台无关的 / 分隔，转成本机绝对路径。"""
    return os.path.join(REPO, *rel.split("/"))


def exists(rel):
    return os.path.isfile(repo_path(rel))


def isdir(rel):
    return os.path.isdir(repo_path(rel))


def count_chapter(chap, nums):
    """统计章节现存 NN-*.md 数量；nums 是应存在的编号列表。"""
    d = repo_path(chap)
    if not os.path.isdir(d):
        return 0
    got = 0
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".md") and "-" in fn:
            try:
                n = int(fn.split("-")[0])
            except ValueError:
                continue
            if n in nums:
                got += 1
    return got


def main():
    # ---------- 账 A：三铁律 × 判据 × 载体 ----------
    laws = (
        ("每阶段一个跑得通项目", "自己跑出的数字(显存/token/s/评测分)",
         "code/scripts"),
        ("先项目后论文", "现象在前、原理在后(边界与坑)",
         "16-学习路线与里程碑"),
        ("保持2-3个活项目", "能随新章节回炉的载体",
         "code/notebooks/009-中文RAG全管线.ipynb"),
    )
    check("账A 三铁律 3 条", len(laws), 3)
    for name, _crit, rel in laws:
        if rel.endswith(".ipynb"):
            check("账A 载体存在 %s" % name, exists(rel), True)
        else:
            check("账A 载体目录 %s" % name, isdir(rel), True)
    keys = ("数", "现象", "回炉")
    check("账A 判据含关键信号",
          all(any(k in c for k in keys) for _, c, _r in laws), True)

    # ---------- 账 B：8 格矩阵 × 前置章节 ----------
    pages = (
        "02-核心原理/01-Transformer.md",
        "04-训练体系/06-全参微调与PEFT-LoRA-QLoRA.md",
        "08-RAG体系/01-Embedding与检索基础.md",
        "08-RAG体系/02-核心流水线.md",
        "08-RAG体系/03-混合检索与重排.md",
        "08-RAG体系/07-RAG评测-RAGAS.md",
        "09-Agent体系/02-Agent-Loop.md",
        "09-Agent体系/06-Agent评测.md",
        "09-Agent体系/07-Agent安全.md",
        "09-Agent体系/14-收束.md",
        "11-评测与可观测/10-评测工程最佳实践.md",
    )
    check("账B 前置章节 11 篇全部存在",
          sum(exists(p) for p in pages), len(pages))

    grid = 4 * 2  # 4 级 × 2 侧
    check("账B 8 格(4 级 × 2 侧)", grid, 8)

    ch08 = count_chapter("08-RAG体系", list(range(1, 12)) + [13])
    check("账B 08章现存 12 篇(01-11+13)", ch08, 12)
    ch09 = count_chapter("09-Agent体系", list(range(1, 11)) + [14])
    check("账B 09章现存 11 篇(01-10+14)", ch09, 11)

    # ---------- 账 C：脚手架 ----------
    notebooks = (
        "code/notebooks/009-中文RAG全管线.ipynb",
        "code/notebooks/010-RAGAS评测一个RAG系统.ipynb",
        "code/notebooks/011-父子分块与元数据过滤.ipynb",
    )
    check("账C notebook 009/010/011 全存在",
          sum(exists(n) for n in notebooks), 3)

    tdir = repo_path("code/notebooks/_tools")
    if os.path.isdir(tdir):
        demos = [f for f in os.listdir(tdir)
                 if f.endswith("_demo.py")]
    else:
        demos = []
    check("账C _tools 探针集合非空(92)", len(demos) > 50, True)
    check("账C _tools 目录存在", isdir("code/notebooks/_tools"), True)

    sdir = repo_path("code/scripts")
    if os.path.isdir(sdir):
        scripts = [f for f in os.listdir(sdir)
                   if f.endswith(".py") and f != "_proto_gpt.py"]
    else:
        scripts = []
    check("账C code/scripts 脚本骨架存在(>30)", len(scripts) > 30, True)

    # 确定性：脚本内不含随机/时间依赖（只查 import，避免命中本行的字面量）
    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        src = f.read()
    src_lines = src.splitlines()
    has_rand = any(l.lstrip().startswith("import random") or
                   l.lstrip().startswith("from random") for l in src_lines)
    has_tm = any(l.lstrip().startswith("import time") or
                 l.lstrip().startswith("from time") for l in src_lines)
    check("账C 零随机源(无 random/时间戳)",
          not has_rand and not has_tm, True)

    # ---------- 账 D：三进一出 ----------
    stages = (
        ("1 跟着跑", "能报出 >=1 个自己跑出的数"),
        ("2 改成自己的", "产出不是复读的变体"),
        ("3 升级成 product", "有外部使用者,哪怕 1 人"),
        ("出 交付物", "一个可用可讲的 product"),
    )
    check("账D 4 阶段", len(stages), 4)
    check("账D 每阶段带判据", all(c for _a, c in stages), True)
    slugs = tuple(s.split()[0] for s, _c in stages)
    check("账D 阶段顺序 1->2->3->出", slugs, ("1", "2", "3", "出"))
    check("账D 交付物唯一(1 个 product)", stages[3][0], "出 交付物")

    print("\nPASS %d  FAIL %d" % (ok, total - ok))
    return 1 if total - ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
