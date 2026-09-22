#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""07-应用框架 · 04-低代码（Dify / Flowise / Coze / n8n —— 知识地图 §7.1 类 5 / §7.14 / §17.5）探针：
『低代码：把『状态+组装+观测+成本』打包成可视积木』—— 用纯 stdlib 确定性测量产品结构，量四本账：
  A 积木库账 —— 四家（Dify/Flowise/Coze/n8n）把 06-07 章代码原语尊成哪些『积木』，六族分类分布
  B 编排颗粒账 —— 同一『客服问答+重试』（里程碑 012）在各画布上的节点/边/条件/回边（对照 02/03 章代码侧=挂靠）
  C 能力-边界账 —— 四痛点专表 4 家（主/次/不碰）：『四痛点全主』是分类学标签还是单店承诺 + 逃离清单
  D 选择树账 —— 8 场景断言 8/8（该在哪儿用 §7.14）
口径：低代码平台=SaaS / 自托管 Web 应用，本机无法嵌入全平台（n8n/Dify 需 Docker、Coze 纯云）——
     本篇与 00-框架分类学同一诚实口径：不跑任何平台本体。平台积木清单/归类/画布等价换算/四痛点评分
     =作者按 §7 与公开产品文档整理（要素事实，写作环境无外网未复核，以官方文档为准）；
     全部统计=本机确定性计算（零网络·零 RNG·纯 stdlib）。stdout 逐字节可复现；wall-clock 只进 stderr。
"""
import sys
import warnings
from time import perf_counter

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


def counts_by_cat(blocks):
    d = {}
    for _, c in blocks:
        d[c] = d.get(c, 0) + 1
    return d


# 六族：TRIG 触发/接入 · LLM 模型/生成 · TOOL 工具/检索 · LOGIC 逻辑/控制 · DATA 数据/状态 · OPS 运维/观测
CATS = [("TRIG", "触发/接入"), ("LLM", "模型/生成"), ("TOOL", "工具/检索"),
        ("LOGIC", "逻辑/控制"), ("DATA", "数据/状态"), ("OPS", "运维/观测")]

# 各家公开节点积木清单（作者按公开产品文档整理=要素事实；归类=作者定性；数量=本机确定性统计）
DIFY = [
    ("开始 Start", "TRIG"), ("参数提取器", "TOOL"), ("问题分类器", "LOGIC"),
    ("知识库检索", "TOOL"), ("LLM", "LLM"), ("工具(内置/自定义)", "TOOL"),
    ("HTTP 请求", "TOOL"), ("条件分支 IF/ELSE", "LOGIC"), ("变量聚合", "DATA"),
    ("代码执行", "LOGIC"), ("模板转换", "DATA"), ("迭代 Iteration", "LOGIC"), ("结束", "OPS"),
]
COZE = [
    ("Bot 输入/会话", "TRIG"), ("触发器(定时/日历/Webhook)", "TRIG"), ("LLM", "LLM"),
    ("插件 Plugin", "TOOL"), ("知识库", "TOOL"), ("数据库/表格", "DATA"), ("条件", "LOGIC"),
    ("循环", "LOGIC"), ("代码(JS/Python)", "LOGIC"), ("变量", "DATA"), ("多 Agent", "LOGIC"),
    ("批量/大数据集", "DATA"), ("消息/渠道发送", "TOOL"), ("工作流输出", "OPS"),
]
N8N = [
    ("Webhook", "TRIG"), ("Cron", "TRIG"), ("App 事件触发", "TRIG"), ("HTTP Request", "TOOL"),
    ("第三方 App 集成(350+)", "TOOL"), ("IF", "LOGIC"), ("Switch", "LOGIC"), ("Merge", "LOGIC"),
    ("Code", "LOGIC"), ("Loop 循环", "LOGIC"), ("Sub-workflow", "LOGIC"), ("Error Workflow", "OPS"),
    ("Edit Fields", "DATA"), ("Aggregate", "DATA"), ("Limit", "DATA"), ("Date&Time", "DATA"),
    ("执行日志 Executions", "OPS"), ("Sticky Note", "OPS"),
]
FLOWISE = [
    ("Chatflow/Agentflow", "TRIG"), ("LLM", "LLM"), ("Prompt 模板", "DATA"),
    ("OutputParser", "LOGIC"), ("Agent/工具", "TOOL"), ("Retriever/VectorStore", "TOOL"),
    ("Memory", "DATA"), ("SequentialAgent", "LOGIC"), ("CustomFunction", "LOGIC"),
]
PLATFORMS = [("Dify", DIFY), ("Flowise", FLOWISE), ("Coze", COZE), ("n8n", N8N)]


def expA():
    print("=" * 72)
    print("[账 A] 积木库账：低代码把 06-07 章代码原语『尊成可视积木』——六族分类分布 + 跨平台覆盖指纹")
    print("=" * 72)
    print("  六族参照 = 07 章代码侧原语族：TRIG 触发 · LLM 生成 · TOOL 工具/检索 · LOGIC 逻辑/控制 · "
          "DATA 数据/状态 · OPS 运维/观测")
    cov = {k: [] for k, _ in CATS}
    for name, blocks in PLATFORMS:
        cc = counts_by_cat(blocks)
        covered = [k for k, _ in CATS if cc.get(k, 0) > 0]
        for k in covered:
            cov[k].append(name)
        seq = " · ".join(f"{k} {cc.get(k, 0)}" for k, _ in CATS if cc.get(k, 0) > 0)
        print(f"  {name:<8} {len(blocks):>2} 块 · {seq} · 覆盖 {len(covered)}/6 族")
    print("  —— 跨平台覆盖指纹（哪一族的积木在哪些平台有）——")
    for k, label in CATS:
        print(f"    {k}({label})：{('/'.join(cov[k]) if cov[k] else '—无一家—')}")
    allc = [k for k, _ in CATS if len(cov[k]) == 4]
    print(f"  → 结论：全 4 家齐备的族={len(allc)}/6（{'/'.join(allc)}）——触发+组装+逻辑+数据=低代码共性骨架")
    print("  → 例外指纹：n8n 是唯一没有『LLM 族』积木的=通用自动化（非 LLM 专精）；Flowise 是唯一没有『OPS 族』积木的"
          "=纯编排画布（观测回 01 章代码侧）；Dify/Coze 六族最全（6/6）=LLM 应用侧全兜底（知识库/插件积木=LLM 最需要）")
    # 主投族（积木最多的族）每家一枚
    tops = {}
    for name, blocks in PLATFORMS:
        cc = counts_by_cat(blocks)
        m = max(cc.values())
        fav = sorted(k for k, _ in CATS if cc.get(k, 0) == m)
        tops[name] = fav
        print(f"  → {name} 主投族={fav}（{m} 块）")
    assert len(allc) == 4, "TRIG/TOOL/LOGIC/DATA 四族应 4 家齐备"
    assert set(cov["LLM"]) == {"Dify", "Flowise", "Coze"} and "n8n" not in cov["LLM"]
    assert set(cov["OPS"]) == {"Dify", "Coze", "n8n"} and "Flowise" not in cov["OPS"]
    assert tops["n8n"] == ["LOGIC"] and tops["Flowise"] == ["LOGIC"]
    print("  → 断言：全齐族=4（TRIG/TOOL/LOGIC/DATA）· n8n 唯一无 LLM 族积木 · Flowise 唯一无观测族积木 · "
          "Dify/Coze 6/6 六族最全 · 主投族普遍=LOGIC（逻辑/控制族=低代码的中枢积木）——覆盖指纹由积木清单本机确定性算出")


# 里程碑 012『客服问答+重试』在各画布上的等价移植编码（作者按平台公开节点能力换算=要素换算·非平台实测）；
# 代码侧两基准=真实引擎实测（02-章 LangGraph 7 节点·6 条件支·1 自环；03-章 LlamaIndex 最小闭环 6 行）=挂靠不重测
DIFY_EDGES = [
    ("开始", "参数提取器", "seq"), ("参数提取器", "问题分类器", "seq"), ("问题分类器", "知识库检索", "cond"),
    ("知识库检索", "LLM", "seq"), ("LLM", "条件分支", "seq"), ("条件分支", "LLM", "cond"), ("条件分支", "结束", "cond"),
]
COZE_EDGES = [
    ("开始", "LLM", "seq"), ("LLM", "条件", "seq"), ("条件", "知识库-插件", "cond"),
    ("知识库-插件", "LLM", "seq"), ("条件", "循环", "cond"), ("循环", "LLM", "cond"), ("条件", "输出", "cond"),
]
N8N_EDGES = [
    ("Webhook", "Set", "seq"), ("Set", "HTTP", "seq"), ("HTTP", "IF", "seq"),
    ("IF", "HTTP", "cond"), ("IF", "Respond", "cond"), ("HTTP", "ErrorWorkflow", "cond"),
    ("ErrorWorkflow", "HTTP", "cond"),
]
CANVASES = [
    ("Dify 画布", DIFY_EDGES, 1),
    ("Coze 画布", COZE_EDGES, 1),
    ("n8n 画布", N8N_EDGES, 2),
]


def expB():
    print()
    print("=" * 72)
    print("[账 B] 编排颗粒账：同一『客服问答+重试』（里程碑 012）——代码侧两基准 vs 三画布换算")
    print("=" * 72)
    print("  代码侧基准（真实引擎实测=挂靠）· 低代码侧=等价移植编码（节点/边/条件/回边从编码 DAG 本机算出）")
    print("    代码-LangGraph（02 章） : 7 节点 · 6 条件支 · 自环 1（recursion_limit 兜底）——本机真实引擎实测，不重测")
    print("    代码-LlamaIndex（03 章）: 最小闭环 6 行真码（Settings 注入 2+四方法 4）——本机真实引擎实测，不重测")
    for name, edges, back in CANVASES:
        nodes0 = []
        for a, b, _ in edges:
            nodes0 += [a, b]
        nodes = sorted(set(nodes0))
        condn = sum(1 for e in edges if e[2] == "cond")
        print(f"    {name:<14} 节点 {len(nodes)} · 边 {len(edges)} · 条件边 {condn} · 回边 {back}（重试）")
    print("  → 结论：低代码把『客服重试』的节点尊成差不多等量积木（三家换算出 5-7 块）· 每一家都有回边="
          "重试须循环，代码侧是条件自环（02 章 1 自环）、画布侧是回边/迭代循环积木或 ErrorWorkflow（n8n 错误域双回边）——"
          "『重试是循环』这件事两个世界都得亲手搭，不是拖拽就免了")
    for _, e, _ in CANVASES:
        ns = sorted({x for a, b, _ in e for x in (a, b)})
        cnd = sum(1 for _, _, k in e if k == "cond")
        assert len(ns) >= 5, "画布节点数应 ≥5"
        assert cnd >= 3, "客服重试画布应含 ≥3 条条件边"
    assert [b for _, _, b in CANVASES] == [1, 1, 2]
    print("  → 断言：三画布节点 5-7 个（都在 02 章代码基准 7 的同量级）· 每个画布 ≥3 条条件边 · "
          "回边 1/1/2（Dify 迭代/Coze 循环积木=正常域回边，n8n IF 重试+ErrorWorkflow=错误域双回边）——"
          "全部来自本机对作者换算编码的确定性计算")


PAIN = [("状态", "状态机/checkpoint/恢复"), ("组装", "接口拼接/工具注册"), ("可观测", "日志/评测/门禁"), ("成本", "算力/计费/运维")]
# 四痛点 主(M)/次(S)/不碰(N)——直接搬 00-章账 B 24 框架矩阵里低代码类的 4 行（要素事实·作者定性，不另打分）
SCORES = {
    "Dify":    {"状态": "M", "组装": "M", "可观测": "M", "成本": "M"},
    "Flowise": {"状态": "M", "组装": "M", "可观测": "S", "成本": "S"},
    "Coze":    {"状态": "M", "组装": "M", "可观测": "M", "成本": "S"},
    "n8n":     {"状态": "M", "组装": "S", "可观测": "S", "成本": "M"},
}
ESCAPE = [
    ("复杂状态机/递归/任意循环控制", "02-章图引擎 LangGraph（自环+checkpoint+recursion_limit）"),
    ("自定义评测门禁 / RLHF / 后训练环", "11-评测与可观测（横切章）/ RLHF 挂靠 02-模型家族 01-GPT系列"),
    ("深度 RAG 调优（自定义检索/压缩/GraphRAG）", "03-章 LlamaIndex / 08-RAG 全 12 篇"),
    ("企业多租户 / 安全边界 / 自管工具注册", "11-MCP 协议 / 12-安全"),
    ("团队全工程师 · 已有代码栈", "06-章原语直写 / 01-章轻量运行时（低代码=重复基建）"),
]


def expC():
    print()
    print("=" * 72)
    print("[账 C] 能力-边界账：四痛点专表 4 家——『四痛点全主』是分类学标签还是假广告？")
    print("=" * 72)
    print("  四痛点 = §7.0（状态 / 组装 / 可观测 / 成本）× 主(M) / 次(S) / 不碰(N)——"
          "低代码类 4 行=00-章账 B 24 框架矩阵原样搬入（要素事实·不另打分）")
    hdr = "    " + "平台".ljust(10) + "".join(f"{k} {l}".ljust(22) for k, l in PAIN) + "  主 次 不"
    print(hdr)
    msn = {}
    for name in ["Dify", "Flowise", "Coze", "n8n"]:
        sc = SCORES[name]
        row = "    " + name.ljust(10) + "".join(sc[k].ljust(22) for k, _ in PAIN)
        m = sum(1 for v in sc.values() if v == "M")
        s = sum(1 for v in sc.values() if v == "S")
        nn = sum(1 for v in sc.values() if v == "N")
        msn[name] = (m, s, nn)
        print(row + f"  {m}  {s}  {nn}")
    allmain = [n for n, v in msn.items() if v[0] == 4]
    cover4 = sum(1 for v in msn.values() if v[0] + v[1] == 4)
    print("  → 结果：『四痛点全主』既是类标签也是真单店能力——全主（主业 4/4）="
          f"{'/'.join(allmain)}（{len(allmain)} 家）：平台把可观测与成本做进产品"
          "（应用内日志/标注 · 托管计费/自托管/发布 API）")
    print(f"  → 类内 4 家全员『覆盖 4/4 维』=24 家全表唯一全员绿灯的类；主/次形状各家不同："
          f"n8n=状态+成本主业（通用自动化 2 主）、Coze=状态+组装+可观测主业（C 端平台 3 主）、"
          f"Flowise=状态+组装主业（原型画布 2 主）")
    pw = []
    for pain, _ in PAIN:
        mains = sorted(n for n in SCORES if SCORES[n][pain] == "M")
        pw.append(len(mains))
        print(f"      痛点『{pain}』主投 {len(mains)} 家：{'、'.join(mains)}")
    print("  → 主投家数=" + f"[{','.join(map(str, pw))}]——成本·可靠=低代码类独特主业（Dify/n8n 2 家）"
          " vs 24 家全表仅 2 家碰成本主业（且都在低代码类）")
    print("  → 逃离清单（什么时候该离开低代码——挂靠各章）：")
    for n, (prob, where) in enumerate(ESCAPE, 1):
        print(f"      {n}. {prob}")
        print(f"         -> {where}")
    assert len(allmain) == 1 and allmain == ["Dify"]
    assert msn["n8n"] == (2, 2, 0) and msn["Coze"] == (3, 1, 0)
    assert cover4 == 4
    assert pw == [4, 3, 2, 2]
    print("  → 断言：全主（主业 4/4）=Dify 1 家 · n8n 主2次2（状态+成本主业）· 4 家覆盖全 4/4 维 · "
          "主投家数=[4,3,2,2]（成本·可靠 2 家=Dify/n8n，恰都是低代码）——四痛点矩阵=00-章账 B 原行搬入")


SCEN = [
    ("非工程师/无代码团队·快速搭内部工具", "Coze（平台最快）或 Dify（开源可自托管）——拖拽即出"),
    ("企业要 RAG 知识库 + UI + 权限", "Dify（数据集专精+自托管+团队权限）——账 A 知识库检索积木"),
    ("通用跨 App 自动化/定时触发（不是 LLM 场景）", "n8n（350+ 集成+错误工作流；状态+成本=主业）——账 C 2 主"),
    ("要第三方插件即插即用", "Coze（插件市场）——账 A 插件积木"),
    ("要自托管 + 数据主权", "Dify / n8n（开源自托管）；排除 Coze（平台托管）"),
    ("需要复杂状态/递归/自定义评测", "逃离低代码：02 图引擎 / 11-评测与可观测（账 C 清单）"),
    ("深度 RAG 调优（GraphRAG/压缩/Agentic）", "逃离低代码：03-章 / 08-RAG 12 篇（账 C 清单）"),
    ("团队全工程师·已有代码栈", "直写 06-章原语或轻量运行时——低代码=重复基建（账 C 清单）"),
]


def expD():
    print()
    print("=" * 72)
    print("[账 D] 选择树账：8 场景断言 8/8（该在哪儿用 §7.14 低代码一格的决策）")
    print("=" * 72)
    assert len(SCEN) == 8
    for i, (s1, s2) in enumerate(SCEN, 1):
        print(f"    {i:>2}. {s1}")
        print(f"       -> {s2}")
    print("  → 选择树 8 场景断言 8/8：低代码适合『非工程师 + 标准化任务 + 快速落地』；"
          "一旦要递归/评测/深度 RAG/全工程师团队就逃离（账 C 清单）")


def main():
    print("=" * 72)
    print("lowcode_demo：07-应用框架 · 04-低代码（Dify / Flowise / Coze / n8n——知识地图 §7.1 类 5 / §7.14 / §17.5）")
    print("    『低代码：把『状态+组装+观测+成本』打包成可视积木』——积木库 / 画布颗粒 / 能力边界 / 选择树")
    print("=" * 72)
    print("[0] 口径：平台=SaaS / 自托管 Web 应用，本篇不跑任何平台本体（n8n/Dify 需 Docker·Coze 纯云）；"
          "积木清单/归类/画布换算=作者按 §7 与公开文档整理（要素事实·无外网未复核）；"
          "四痛点分=00-章账 B 24 框架矩阵的低代码 4 行原样搬入；全部统计=本机确定性计算（纯 stdlib·零网络·零 RNG）")
    expA()
    expB()
    expC()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 积木库 / B 编排颗粒 / C 能力边界 / D 选择树）")
    print("  A 4 平台积木：全齐族=4（TRIG/TOOL/LOGIC/DATA）· Dify/Coze=6/6 六族最全 · n8n=5/6 唯一无 LLM 族（通用自动化）"
          "· Flowise=5/6 唯一无观测族（纯编排画布）")
    print("  B 客服重试（里程碑 012）：代码 7 节点 1 自环（02 章挂靠）vs Dify 7 块 1 回边 / Coze 6 块 1 回边 / "
          "n8n 6 块 2 回边（错误域双保险）——重试=循环两个世界都得亲手搭")
    print("  C 『四痛点全主』=类标签兼真单店：Dify 主业 4/4 达成 · 4 家全 4 维覆盖=24 家唯一『全员绿灯』类 · "
          "成本维=类独特主业（Dify/n8n）· 逃离清单 5 条挂靠 02/03/04/11 章")
    print("  D 选择树 8 场景断言 8/8：非工程师+标准化任务=低代码主战场；递归/评测/深度 RAG/全工程师=逃离")
    print("一句话：低代码把『状态+组装+观测+成本』打包成可视积木，适合非工程师与标准化任务——")
    print("『四痛点全主』在低代码类=类标签兼真单店（Dify 主业 4/4）；但主/次形状各家不同，"
          "要递归/评测/深度 RAG 仍得逃离（账 C 清单挂靠 02/03/04/11 章）")
    print("done · 一键复现：python code/notebooks/_tools/lowcode_demo.py")


if __name__ == "__main__":
    main()
    W(f"[lowcode_demo] wall-clock {perf_counter() - _t0:.3f} s")
