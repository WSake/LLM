#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""09-Agent体系 · 02-Agent-Loop（知识地图 §9.2/§9.3/§9.7/§9.12）探针：
『主循环』——把 09-01 账 A 的 LOOP 原语画成完整的六状态循环，
并把 09-01 账 B 判为 Agent 的 4 个模板样本逐一跑起来。
量四本账：
  A 六状态机账 —— 目标→感知→计划→行动→观察→评估→(循环→计划)，把 4 个
              Agent 模板任务（ReAct 带工具 / LangGraph 客服重试 / 代码解释器 /
              Operator 订航班）各跑一遍，输出循环轨迹与终局原因（完成/放弃/转人工）
  B 护栏三件套账 —— §9.2「工程重点=给循环加护栏」：① 步骤上限（失控循环 7 步
              才收敛，cap=4 触顶→放弃/转人工） ② 错误重试（2 次瞬时错→第 3 次
              成功 自愈；预算 1 耗尽→转人工） ③ 人工确认（低风险自动放行、
              高风险升级人工）
  C 可失败点与验证回路账 —— §9.2 的 4 个可失败点各配一条护栏：
              计划太抽象→步骤上限 / 工具用错→错误回喂 / 观察被噪声淹没→证据
              纯度过滤 / 评估放过幻觉→规则验证器（算术 3+5=8，agent 初答 7
              被验证器抓住→重规划→修正 8）
  D 计划形态与停止判据账 —— §9.3 三种计划形态（Plan-and-Execute 一次性 /
              ReAct 逐步 / 分层计划）计划调用次数（1<3<4）；§9.2/§9.12 停止
              判据三态（完成=验证通过 / 放弃=触顶·预算耗尽 / 转人工=高风险）
              —— 好的 Agent 评测=固定环境+可验证终结+成本指标
口径：任务场景/循环轨迹/失败注入/计划调用数 = 作者按地图 §9 预置（仿真=非真实
     LLM 运行的 Agent），A-D 全部统计 = 本机确定性计算（纯 stdlib · 零随机 ·
     零网络 · stdout 逐字节可复现；wall-clock 只进 stderr）；护栏的"真实引擎
     版本"挂靠 07-02 recursion_limit 真抛 / 06-05 自愈 3/3 / 08-06 证据纯度。
"""
import sys
from time import perf_counter

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ================= 账 A 数据：六状态机 =================
# §9.2 主循环六状态；边：EVAL→(完成/放弃/转人工)/→PLAN 再循环
NODES = ["GOAL 目标", "PERCEIVE 感知", "PLAN 计划", "ACT 行动", "OBSERVE 观察", "EVAL 评估决策"]
NODE_SORT = ["GOAL", "PERCEIVE", "PLAN", "ACT", "OBSERVE", "EVAL"]
FIRST = ["GOAL", "PERCEIVE", "PLAN", "ACT", "OBSERVE", "EVAL"]   # 首遍走全环节
LOOPI = ["PLAN", "ACT", "OBSERVE", "EVAL"]                        # 回环一遍（计划→行动→观察→评估）


def expA():
    print("=" * 72)
    print("[账 A] 六状态机账：把 §9.2 主循环画成可执行状态机，跑 09-01 账 B 判为 Agent 的 4 个模板样本")
    print("=" * 72)
    # (任务, 循环遍数, 终局原因, 一句话终局依据)
    TASKS = [
        ("ReAct 带工具·查天气回答", 1, "完成", "检索结果信息够→EVAL 判『够了吗=够了』"),
        ("LangGraph 客服·故障重试(012)", 2, "完成", "第一遍 OBSERVE=失败→EVAL 判『重试』→第二遍 OBSERVE=恢复→完成"),
        ("代码解释器·写函数并测试", 2, "完成", "第一遍测试失败(fail#1)→EVAL 判『修正』→第二遍测试通过→完成"),
        ("OpenAI Operator·订航班", 3, "转人工", "查/比/选三遍，付款属高风险动作→EVAL 判『升级人工』"),
    ]
    ledger = {}
    for name, loops, exit_, why in TASKS:
        trace = [FIRST] + [LOOPI] * (loops - 1)
        counts = {k: 0 for k in NODE_SORT}
        for it in trace:
            for n in it:
                counts[n] += 1
        ledger[name] = counts
        print(f"  ● {name}")
        for i, it in enumerate(trace, 1):
            print(f"      第{i}遍  {'→'.join(it)}")
        visited = [k for k in NODE_SORT if counts[k] > 0]
        print(f"      → 终局【{exit_}】{why}｜访问 {len(visited)}/6 节点 · 共 {sum(counts.values())} 次节点访问")
    done = sum(1 for _, _, e, _ in TASKS if e == "完成")
    human = sum(1 for _, _, e, _ in TASKS if e == "转人工")
    give = sum(1 for _, _, e, _ in TASKS if e == "放弃")
    ok_exits = {"完成", "放弃", "转人工"}
    all_nodes = set()
    for c in ledger.values():
        all_nodes |= set(k for k in NODE_SORT if c[k] > 0)
    print(f"  → 终局台账：完成 {done} · 放弃 {give} · 转人工 {human}（4/4 都在三态内）")
    print(f"  → 六节点每节点至少被访问一次：{'是' if len(all_nodes) == 6 else '否'}（{len(all_nodes)}/6）")
    assert len(TASKS) == 4
    assert all(e in ok_exits for _, _, e, _ in TASKS)
    assert (done, give, human) == (3, 0, 1)
    assert len(all_nodes) == 6
    print("  → 断言：4/4 终止 · 终局三态 {完成3 放弃0 转人工1} · 六节点全动")


# ================= 账 B 数据：护栏三件套 =================
def run_with_cap(need_steps, cap):
    """步骤上限护栏：看似正常却失控的循环（需 need_steps 步才收敛），cap 步内必须停。"""
    used = 0
    while used != need_steps:
        if used >= cap:
            return used, "放弃·转人工", "步骤上限（护栏①）触顶，不再白烧 token"
        used += 1
    return used, "完成", "在 cap 内收敛"


def run_with_retry(attempts, budget):
    """错误重试护栏：attempts=[失败,失败,成功] 瞬时错，budget 次数内重试。"""
    used = 0
    for a in attempts:
        if a == "失败":
            used += 1
            if used > budget:
                return used, "转人工", "重试预算（护栏②）耗尽，交人兜底"
        else:
            return used, "完成", f"第 {used + 1} 次尝试成功（此前误伤 {used} 次）"
    return used, "转人工", "用尽全部尝试仍失败"


def expB():
    print()
    print("=" * 72)
    print("[账 B] 护栏三件套账：§9.2『工程重点=给循环加护栏』——①步骤上限 ②错误重试 ③人工确认")
    print("=" * 72)
    # ① 步骤上限
    s, r, why = run_with_cap(need_steps=7, cap=4)
    print(f"  ①步骤上限  失控循环需 7 步才收敛 · cap=4 → 第 {s} 步触顶 · 终局【{r}】{why}")
    assert (s, r) == (4, "放弃·转人工")
    s2, r2, _ = run_with_cap(need_steps=3, cap=6)
    print(f"             健康循环 3 步收敛 · cap=6 → 第 {s2} 步完成 · 终局【{r2}】（cap 越界才触发=不误伤好循环）")
    assert (s2, r2) == (3, "完成")
    # ② 错误重试
    u, r, why = run_with_retry(["失败", "失败", "成功"], budget=3)
    print(f"  ②错误重试  2 次瞬时错 · budget=3 → 重试 {u} 次后成功 · 终局【{r}】{why}")
    assert (u, r) == (2, "完成")
    u2, r2, why2 = run_with_retry(["失败", "失败", "成功"], budget=1)
    print(f"             budget=1 → 第 2 次失败越界→送人工 · 重试 {u2} 次耗尽 · 终局【{r2}】")
    assert (u2, r2) == (2, "转人工")  # 第 2 次失败超出 budget=1
    # ③ 人工确认
    ACTIONS = [("读项目文件", "low"), ("向对公账户付款 ¥5000", "high")]
    approved = [a for a, lv in ACTIONS if lv == "low"]
    manual = [a for a, lv in ACTIONS if lv == "high"]
    print(f"  ③人工确认  {len(approved)} 个低风险动作自动放行：{'｜'.join(approved)}")
    print(f"             {len(manual)} 个高风险动作升级人工审批：{'｜'.join(manual)} → 人工确认后才执行")
    assert len(approved) == 1 and len(manual) == 1
    print("  → 断言：capped 循环 4 步触顶 · 自愈循环重试 2 次成功 · 预算 1 耗尽转人工 · 高低风险分诊 1/1")
    print("  → 真实引擎版本挂靠：07-02 recursion_limit 真抛 GraphRecursionError=cap 真兜·原生护栏；06-05 报错自愈 3/3")


# ================= 账 C 数据：可失败点 × 护栏 × 验证回路 =================
FAILPOINTS = [
    # (节点, 可失败点, 护栏, 挂靠锚点, 锚点数字)
    ("PLAN 计划", "计划太抽象（列出的是目标不是步骤）", "步骤上限 + 计划可执行性检查",
     "本账 B·cap / 账 D·P&E", "cap=4 触顶不白烧 · P&E 计划调用 1"),
    ("ACT 行动", "工具用错（参数/工具名不匹配）", "错误回喂·自愈重试",
     "06-05 工具调用", "报错自愈 3/3（重试 1·1·2）"),
    ("OBSERVE 观察", "观察被噪声淹没（陪跑行盖住目标行）", "证据纯度过滤·只留命中行",
     "08-06 上下文压缩", "证据纯度 0.226→1.000"),
    ("EVAL 评估", "评估放过幻觉（『看起来像』就当对）", "规则验证器（真实验收条件）",
     "02-20 推理模型", "best-of-k≈覆盖率 78/80%"),
]
VERIFIER_CASE = {  # 3+5=8；agent 初答 7 是幻觉，验证器按真值抓住
    "expr": "3+5", "agent_first": 7, "truth": 8,
}


def expC():
    print()
    print("=" * 72)
    print("[账 C] 可失败点与验证回路账：§9.2 的 4 个可失败点 × 4 条护栏（『验证回路 > 更好的提示词』）")
    print("=" * 72)
    for node, fail, rail, anchor, num in FAILPOINTS:
        print(f"  {node:<10}·{fail}")
        print(f"      护栏→{rail}｜挂靠 {anchor}＝{num}")
    assert len(FAILPOINTS) == 4 and len({f[0] for f in FAILPOINTS}) == 4
    # 评估放过幻觉的对冲：规则验证器
    expr, first, truth = VERIFIER_CASE["expr"], VERIFIER_CASE["agent_first"], VERIFIER_CASE["truth"]
    ans = first
    caught = False
    steps = 0
    while True:
        steps += 1
        if ans == truth:
            break
        caught = True  # 验证器判决： ans 与真值不符 → 重规划
        ans = truth     # 重规划后修正（仿真：给出正确执行的产物）
    print(f"  → 幻觉对冲演示（{expr}）：agent 初答 {first} → 规则验证器判决 ≠ {truth} → 抓住 → 重规划 → 修正为 {truth}")
    print(f"     用了 {steps} 遍循环（比不校验多付 1 遍，但把『放过幻觉』拒在门外）· 真值来源=可执行计算非『看起来像』")
    assert caught and ans == truth and steps == 2
    # 观察噪声对冲：证据纯度（借用 08-06 口径 0.226→1.000 的最小再现）
    obs = ["费用说明无关行", "命中行：明天下雨记得带伞", "无关行", "无关行", "无关行"]
    hit = sum(1 for l in obs if l.startswith("命中行"))
    raw = hit / len(obs)
    top1 = 1.0  # 过滤后只保留命中行
    print(f"  → 噪声对冲（观察·命中行 1/5）：raw 纯度 {raw:.3f} → top-1 过滤后纯度 {top1:.3f}")
    assert round(raw, 3) == 0.200 and top1 == 1.0
    print("  → 断言：四可失败点各配一条护栏 · 幻觉被验证器抓住并修正（steps=2）· 噪声纯度 0.200→1.000")


# ================= 账 D 数据：计划形态 × 停止判据 =================
PLAN_FORMS = [
    # (形态, 计划调用, 循环遍数, 适用, 一句话机制)
    ("Plan-and-Execute（一次性）", 1, 3, "确定性任务", "1 次计划列出全部 3 步清单 → 顺序执行，计划不重算"),
    ("ReAct（逐步）", 3, 3, "开放任务", "每子任务前现想一步（思考=计划调用），task1/task2/task3 各 1 次"),
    ("分层计划（Task树）", 4, 3, "复杂大任务", "1 次根计划 + 3 次子任务细化 = 1+3 次计划调用"),
]
TERMINAL = [
    # (终局, 触发条件, 成本, 台账来源)
    ("完成", "验证通过（真实验收/测试/规则校验）", "最低·算到确认对为止",
     "账A 完成×3 · 账B 自愈循环"),
    ("放弃", "步骤上限/预算耗尽（护栏①触顶）", "已烧步数白付（比继续烧省）",
     "账B capped 循环 · 账B 重试预算耗尽"),
    ("转人工", "高风险动作 / 不确定且反复失败", "半途交人·保全风险",
     "账A Operator · 账A/B 触顶交人"),
]


def expD():
    print()
    print("=" * 72)
    print("[账 D] 计划形态与停止判据账：§9.3 三种计划形态(N=3 子任务)· §9.2/§9.12 停止判据三态")
    print("=" * 72)
    print("  计划形态账（同一 3 子任务，计划调用次数 = 想几次才开始做）：")
    for form, pc, ps, fit, mech in PLAN_FORMS:
        print(f"    {form:<24}计划调用 {pc} · 循环遍数 {ps} · 适用={fit}｜{mech}")
    pcs = [pc for _, pc, _, _, _ in PLAN_FORMS]
    assert pcs == sorted(pcs) and tuple(pcs) == (1, 3, 4)
    print(f"  → 计划调用排序断言：1（P&E）< 3（ReAct 逐步）< 4（分层）——计划越细=想得越多")
    print("  停止判据三态（好的 Agent 评测=固定环境 + 可验证终结 + 成本指标 §9.12）：")
    for term, trig, cost, src in TERMINAL:
        print(f"    【{term}】触发={trig}")
        print(f"             成本={cost}｜台账来源={src}")
    assert len(TERMINAL) == 3 and {t for t, _, _, _ in TERMINAL} == {"完成", "放弃", "转人工"}
    print("  → 断言：停止判据三态互斥 · 每态有明确触发条件（验证通过/触顶/高风险）——不靠『看起来像』")
    print(f"  合并台账：账A 完成3·转人工1·放弃0；账B 追加 放弃1（cap 触顶）+ 转人工1（预算耗尽）")
    print(f"            → 全探针终局口径 完成3 · 放弃1 · 转人工2（6 个终局样本全在三态内）")


def main():
    print("=" * 72)
    print("agent_loop_demo：09-Agent体系 · 02-Agent-Loop（主循环：画出来 → 跑起来 → 护栏兜住）")
    print("    关联：地图 §9.2/§9.3/§9.7/§9.12 ｜ 前提 09-01 判别器（账 B 判为 Agent 的 4 样本）")
    print("=" * 72)
    print("[0] 口径：任务场景/循环轨迹/失败注入/计划调用数 = 作者按地图 §9 预置（仿真=非真实 LLM 运行的 Agent）；")
    print("    A-D 全部统计 = 本机确定性计算（纯 stdlib·零随机·零网络）")
    expA()
    expB()
    expC()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 六状态机 / B 护栏三件套 / C 可失败点×护栏 / D 计划形态×停止判据）")
    print("  A 4/4 模板任务三态终止（完成3 放弃0 转人工1）· 六节点全动 · 循环轨迹逐遍可见")
    print("  B 步骤上限 cap=4 触顶不白烧 · 错误重试预算内自愈 2 次（耗尽则转人工）· 人工确认高低风险分诊 1/1")
    print("  C 4 可失败点 × 4 条护栏 · 规则验证器抓住幻觉 7→8（真值=可执行计算）· 证据纯度 0.200→1.000")
    print("  D 计划调用 P&E 1 < ReAct 3 < 分层 4 · 停止判据三态互斥（验证通过/触顶/高风险）")
    print("一句话：主循环=目标→感知→计划→行动→观察→评估→(循环)，核心不是节点而是停止判据与护栏——")
    print("         验证回路的 Agent 才叫 Agent（§9.7），没有 steps cap / retry / approve 的循环只是死循环＋幻觉")
    print("done · 一键复现：python code/notebooks/_tools/agent_loop_demo.py")


if __name__ == "__main__":
    main()
    W(f"[agent_loop_demo] wall-clock {perf_counter() - _t0:.3f} s")
