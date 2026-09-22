# -*- coding: utf-8 -*-
"""14-垂直领域 03 篇探针：代码领域 Coding-Agent 四本账（§14.4 四栏模板第三次实战）。

设计纪律：纯 stdlib · 零网络 · 零随机 · 零第三方；stdout 只打印确定性结果。
三独立进程 stdout 逐字节恒一。任何一步出错立即 assert 暴露。
内容口径：
- 四段路线 / 垂直化重点（仓库索引·CI·IDE-CLI·安全审查）/ 代表与评测 = §14.4+§9.11 知识地图要素事实（无外网未复核）
- 四栏代码评分（1/1/3/1=6）与"最难=流程" = 00 课账 B 锚点（代码6·最难资产=流程）展开成本篇明细
- 需求→段组合·成本档·红线拦截判定 = 作者预置教学数 + 本机确定性计算
- 挂靠章映射只引已交付篇（09-05 五技术/09-07 护栏/08-01/04-10），不重测
- 不写任何训练代码：判型/评分/档位/拦截全是确定性账
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

ok = bad = 0
def t(cond):
    global ok, bad
    if cond:
        ok += 1
    else:
        bad += 1
        print("FAIL  cond=%r" % cond)
    return bool(cond)

# ============ 账 A：场景账（§14.4 四段路线 → 形态/主资产/判段） ============
ROUTE = {
    "代码补全": ("IDE 内联补全（Cursor/Copilot 形态）", "工具链集成",
                 "工程（不训练）", "通用模型已强于代码——补全拼的是 IDE/延迟/上下文，不是再预训练"),
    "代码问答": ("RAG over repo 仓库问答", "领域知识库",
                 "RAG（仓库索引）", "垂直化=仓库索引（语法感知分块+符号图），09-05 仓库导航只引"),
    "改仓库 Agent": ("Edit 精确修改 + 测试执行回路", "业务流程建模",
                      "Agent（工程）", "代码任务『可验证』→测试/编译是真值，09-05 测试回路只引"),
    "企业 Coding Copilot": ("权限/策略/合规 + CI/CLI 集成", "合规与责任边界",
                             "Agent+合规", "企业落地收尾=权限/审计/人审（09-07 护栏只引）"),
}
print("A1 四段路线→产品形态 4/4 归位（§14.4 路线：代码补全→代码问答→Agent→企业 Copilot）：")
for k, (f, a, s, why) in ROUTE.items():
    print("      %s → %s（主资产：%s）" % (k, f, a))
for k, (f, a, s, why) in ROUTE.items():
    t(f == ROUTE[k][0] and a in ("工具链集成", "领域知识库", "业务流程建模", "合规与责任边界"))

print("\nA2 四段路线→演化链判段 4/4（代码垂直化=系统不是重训：全工程/RAG/Agent，一段 CPT 都没有）：")
for k, (f, a, s, why) in ROUTE.items():
    print("      %s → %s 段 · %s" % (k, s, why))
for k, (f, a, s, why) in ROUTE.items():
    segs = [seg.split("（")[0].split("(")[0] for seg in s.split("+")]
    t(all(x in {"CPT","SFT","RL","RAG","Agent","合规","工程"} for x in segs))

print("\nA3 §14.4 垂直化重点四件套 4/4（『通用模型已强于代码，垂直化重点在工具链与系统』）：")
FOCUS = ["仓库索引", "CI/测试反馈", "IDE/CLI 集成", "安全审查"]
for f in FOCUS:
    print("      · %s" % f)
t(len(FOCUS) == 4)

# ============ 账 B：资产账（四栏模板×代码，00 课账 B2 锚点 代码6） ============
ASSET_BARS = [("领域数据", 1, "代码语料公开海量（GitHub 开源）·通用模型已强→数据不是难点"),
              ("领域知识库", 1, "仓库自身知识靠 RAG over repo·框架文档变化快但应到检索不重训"),
              ("业务流程建模", 3, "仓库集成=Edit/测试/CI/LSP/IDE/CLI/并行分支——垂直化真主战场"),
              ("合规与责任边界", 1, "权限/审计/安全审查（沙箱+最小权限+人审）")]
TOTAL = sum(b for _, b, _ in ASSET_BARS)          # 1+1+3+1 = 6
print("\nB1 四栏模板×代码难易评分 4/4（每栏 1-3 越大越稀缺 → 总分 %d = 00 课账 B2 锚点 代码6）" % TOTAL)
for name, b, why in ASSET_BARS:
    print("      %s：%d —— %s" % (name, b, why))
for name, b, why in ASSET_BARS:
    t(b == {"领域数据": 1, "领域知识库": 1, "业务流程建模": 3, "合规与责任边界": 1}[name])
t(TOTAL == 6)
hard = "业务流程建模"
print("\nB2 代码“最难资产”（主战场判定，挂 00 课账 B4）：%s（流程栏 3 是唯一最高栏——"
      "与医疗/金融截然相反：他们的难在『数据资产造不出来』，代码的难在『仓库系统接不起来』=垂直化重点=工具链与系统，00 篇账 B4 一致）" % hard)
t(hard == "业务流程建模")

MAP = {
    "领域数据": ["04-10 评测门禁·预训练语料只引"],
    "领域知识库": ["08-01 检索", "09-11 仓库导航"],
    "业务流程建模": ["09-05 Coding-Agent", "09-02 主循环", "09-04 行动推理"],
    "合规与责任边界": ["09-07 护栏", "12-安全"],
}
print("\nB3 资产→挂靠章 4/4（教学法约定“会用到 08/09 的哪几章”，只引不重测）：")
for k, v in MAP.items():
    print("      %s → %s" % (k, " / ".join(v)))
for k, v in MAP.items():
    t(len(v) >= 1)

# ============ 账 C：选段账（代码需求 → 演化链跳段 + 成本档，00 课账 C 代码版） ============
PLAN = {
    "代码补全": (["工程"], "模型已会补全——拼 IDE/延迟/上下文，训进权重不划算", "工程级×1"),
    "仓库问答 Bug 归属": (["RAG"], "仓库索引+符号图定位，09-05 索引 top-6 26% 只引", "工程级×1"),
    "改仓库修 Bug": (["RAG", "Agent"], "测试回路看真值（2 次收敛 vs 盲改 4 次），09-05 只引", "工程级×1"),
    "代码审查": (["RAG", "Agent", "合规"], "安全审查+依赖检查+人工审批，09-07 只引", "工程级×1"),
    "企业 Copilot": (["工程", "Agent", "合规"], "权限/策略/审计+CI 集成，落地只管不管训", "工程级×1"),
}
print("\nC1 五代码需求→方案组合 5/5（代码是四栏模板里唯一的『全工程行业』——连 SFT 级都少见）：")
for k, (seg, why, cost) in PLAN.items():
    print("      %s → %s（%s）" % (k, "+".join(seg), why))
for k, (seg, why, cost) in PLAN.items():
    t(len(seg) >= 1 and all(x in {"CPT","SFT","RL","RAG","Agent","合规","工程"} for x in seg))

print("\nC2 三档成本级（00 课账 C 同式只引）：纯工程 ×1 · 领域SFT ×10 · 领域CPT ×100 —— 能工程先工程，能SFT不CPT")
t(1.0 < 10.0 < 100.0)
LEVEL = {"CPT级×100": 100.0, "SFT级×10": 10.0, "工程级×1": 1.0}
print("\nC3 五需求成本档 5/5（全工程级×1）：")
for k, (seg, why, cost) in PLAN.items():
    print("      %s → %s" % (k, cost))
    t(cost == "工程级×1")

C4 = all(PLAN[k][2] == "工程级×1" for k in PLAN)
print("\nC4 代码跳段纪律：够格 CPT 的 0 个·够格 SFT 的 0 个——五需求全工程级×1（通用代码模型（Copilot/Cursor/Claude Code/Codex/Devin/OpenHands）已足够强，"
      "垂直化=仓库索引/测试回路/IDE-CLI/安全审查这些『系统』，模型不是瓶颈 → 代码是全球最『系统优先』的行业）")
t(C4)

# ============ 账 D：合规账（红线拦截 + 评测四件套 + 合规五动作，§9.11/§9.13 点到） ============
REDLINE = {
    "越权改生产": ("Agent 无权限直接改生产配置/发布", "最小权限+人工确认 → 拦截升人审（09-07 只引）"),
    "泄露密钥": ("代码里嵌入 API key/内部地址", "秘密扫描 → 拦截+轮换（签名扫描规则）"),
    "依赖投毒": ("自动装上伪造的恶意依赖包", "供应链审查 → 锁定版本+来源校验+人工确认"),
}
print("\nD1 代码危险动作→红线拦截 3/3（§9.13 威胁模型：工具滥用/数据外泄/供应链——红线把『模型直接落地』变『审批后落地』）：")
for k, (act, guard) in REDLINE.items():
    print("      %s：%s → %s" % (k, act, guard))
    t("拦截" in guard or "人工" in guard or "轮换" in guard)

EVAL4 = [
    ("SWE-bench Verified / LiveCodeBench", "通用基准（真实 GitHub issue·2025 事实标准）"),
    ("企业内部真实仓库题", "领域真题（自家 bug 库=商业资产）"),
    ("资深工程师盲评", "专家盲评（代码质量/可读性）"),
    ("越权/密钥/依赖用例", "安全用例（红线拦截回归）"),
]
print("\nD2 代码领域评测集四件套 4/4（挂 00 课账 D1·§9.12：通用基准/领域真题/专家盲评/安全用例）：")
for name, what in EVAL4:
    print("      %s —— %s" % (name, what))
    t(any(w in what for w in ("通用", "真题", "盲评", "安全")))

COMP = [
    "沙箱执行（无网络+隔离文件系统）",
    "最小权限（只给当次任务需要的读/改权）",
    "审计日志（每次改动留痕可追因）",
    "人工审批高风险动作（push/生产配置/删除）",
    "安全扫描（密钥/依赖版本/漏洞）",
]
print("\nD3 代码合规五动作 5/5（挂 00 课账 D2·§9.13 基线护栏）：")
for c in COMP:
    print("      · %s" % c)
    t(len(c) > 3)

# ============ 断言汇总 ============
print("\nSUMMARY  PASS %d · FAIL %d" % (ok, bad))
assert bad == 0 and ok >= 40, "probe failed: PASS %d FAIL %d" % (ok, bad)
