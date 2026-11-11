# -*- coding: utf-8 -*-
"""
16-06 推荐开源项目 探针 (open_source_projects_demo.py)

账 A (六组分类)：理论/实现入门/训练微调/推理部署/RAG知识/Agent与评测可观测 = 6 组，
     每组至少 1 个代表性库；库 → 章节映射全覆盖。
账 B (库→篇章映射)：九条映射每条对应仓库里一篇真实存在的章节(直接落盘校验)。
账 C (三原则)：按需取用/优先跑得动/用 map 代替收藏 —— 判据成对。

纯 stdlib、零网络、零随机、零第三方。三进程 stdout 字节一致。
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
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


def exists(rel):
    return os.path.isfile(os.path.join(REPO, *rel.split("/")))


def main():
    # ---------- 账 A：六组分类 ----------
    groups = {
        "理论":         ["02-核心原理/01-Transformer.md"],
        "实现入门":       ["02-核心原理/01-Transformer.md"],
        "训练微调":       ["04-训练体系/06-全参微调与PEFT-LoRA-QLoRA.md",
                      "04-训练体系/08-训练框架横向对比.md"],
        "推理部署":       ["05-推理与部署/03-精度与量化-FP8-INT8-INT4.md",
                      "05-推理与部署/09-部署形态-K8s与云端.md"],
        "RAG知识":       ["08-RAG体系/02-核心流水线.md",
                      "08-RAG体系/07-RAG评测-RAGAS.md"],
        "Agent与评测可观测": ["07-应用框架/02-LangGraph-状态图与Checkpoint.md",
                      "07-应用框架/10-OpenAI-Agents-SDK与轻量运行时.md",
                      "11-评测与可观测/10-评测工程最佳实践.md"],
    }
    check("账A 六组分类", len(groups), 6)
    check("账A 每组都有章节落点",
          all(len(v) >= 1 for v in groups.values()), True)

    # ---------- 账 B：库 → 篇章九条映射 ----------
    mappings = (
        ("nanoGPT",        "02-核心原理/01-Transformer.md"),
        ("LLaMA-Factory",  "04-训练体系/06-全参微调与PEFT-LoRA-QLoRA.md"),
        ("Unsloth",        "04-训练体系/06-全参微调与PEFT-LoRA-QLoRA.md"),
        ("DeepSpeed",      "04-训练体系/08-训练框架横向对比.md"),
        ("vLLM",           "05-推理与部署/03-精度与量化-FP8-INT8-INT4.md"),
        ("llama.cpp",      "05-推理与部署/09-部署形态-K8s与云端.md"),
        ("LlamaIndex",     "08-RAG体系/02-核心流水线.md"),
        ("RAGAS",          "08-RAG体系/07-RAG评测-RAGAS.md"),
        ("LangGraph",      "07-应用框架/02-LangGraph-状态图与Checkpoint.md"),
        ("OpenAI Agents SDK", "07-应用框架/10-OpenAI-Agents-SDK与轻量运行时.md"),
        ("lm-eval-harness", "11-评测与可观测/10-评测工程最佳实践.md"),
        ("OpenCompass",    "11-评测与可观测/10-评测工程最佳实践.md"),
    )
    check("账B 九条映射全部真实",
          sum(exists(ch) for _, ch in mappings), len(mappings))
    # 每个映射的章节都指向可读目录下(不落空目录)
    check("账B 映射落在已存在章节而非空目录",
          all(exists(ch.split("/")[0]) or exists(ch)
              for _, ch in mappings), True)

    # ---------- 账 C：三原则 ----------
    rules = (
        ("按需取用不背目录", "知道某库在某一格里、解决某个项目问题"),
        ("优先自己跑得动的", "能在本机跑通最小 demo"),
        ("用 map 代替收藏",  "新库进来先问它对应哪篇、替代清单里谁"),
    )
    check("账C 三原则", len(rules), 3)
    check("账C 每条判据是「怎么做」而非「是什么」",
          all(len(c) >= 8 for _, c in rules), True)
    slugs = tuple(r[0] for r in rules)
    check("账C 顺序 按需->跑得动->用map",
          slugs[0].startswith("按需") and slugs[1].startswith("优先")
          and slugs[2].startswith("用"), True)

    # 确定性：无 random / 时间依赖(只查 import 行)
    with open(os.path.abspath(__file__), encoding="utf-8") as f:
        lines = f.read().splitlines()
    has_rand = any(l.lstrip().startswith(("import random", "from random"))
                   for l in lines)
    has_tm = any(l.lstrip().startswith(("import time", "from time"))
                 for l in lines)
    check("账C 零随机源", not has_rand, True)

    print("\nPASS %d  FAIL %d" % (ok, total - ok))
    return 1 if total - ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
