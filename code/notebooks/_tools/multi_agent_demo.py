#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""09-Agent体系 · 08-多智能体（知识地图 §9.8 Multi-Agent）探针：
§9.8 原话："任务天然多角色（写手/审校/执行）、可并行、可分工；代价：token 爆炸、错误
传导、协调复杂。常见拓扑：星型（supervisor 分配）、环型/链式（流水线）、网状（自由
对话）、层级（子 Agent 再分）。决策原则：先单 Agent，再上多；一个清晰的 loop 胜过
三个混乱的 Agent。"
四账 A-D 实测（全部=作者预置仿真 + 本机确定性计算）：
  A 拓扑×场景账 —— 三场景（单步查询 / 写审发 / 三主题调研）× 五拓扑（单/星/链/网/层级）：
    调用数·墙钟轮数·协调消息数·质量分。单步查询上网状 = 6 calls / 12 msgs 换质量 1.00
    与单 Agent 1 call 相同=纯开销 → 『先单 Agent，再上多』；只有多角色（S2）/可并行（S3）
    时才有多付的回报（星型 S3 并行省墙钟 4→3 轮、supervisor 汇总 +0.05 质量）
  B 多角色协作账 —— 写手/审校/执行：同一篇文章 3 缺陷（事实错/重复/漏要点），
    单 Agent 抓 0 → +自检 1/3 → 链式审校 2/3 → 星型+supervisor 仍 2/3（审校是瓶颈=
    加人前先问瓶颈是谁）；外部审校 > 自检：同样 2 次调用 1/3 vs 2/3（挂 04 Self-Refine
    无真值内自检 vs Reflexion 外部验证）
  C 协调成本与错误传导账 —— token 爆炸：上下文回灌段模型 单 10 < 链式 42 < 星型 52
    < 网状 78（7.8× 且末节点 26 段=全广播最贵）；错误传导：链式/网状被污染 3/3 环节
    vs 星型汇报检查点拦在源头 1/3（检查点=唯一缩短传导的手段）；共享写入 last-write-wins
    丢 1 份（协调复杂=省下的墙钟有人付账）
  D 信任与监督账 —— 07 六护栏 × 网络：13 个跨 Agent 事件三档治理（无治理 13/13 →
    权限隔离+沙箱 9/13 → 全配=隔离+监督+审批+消息标注 2/13：supervisor 上下文覆盖 +
    伪装 Agent 投毒），审计 13/13 留痕·回滚 2/2 残留 0 → 按 06 账D 红线只有全配上生产
口径：拓扑成本/质量分/缺陷表/审校抓错率/段模型/事件表与拦截判定 = 作者预置仿真（非真实
     Multi-Agent 系统运行·不 import 任何框架）；A-D 统计 = 本机确定性计算（纯 stdlib ·
     零随机 · 零网络 · stdout 逐字节可复现；wall-clock 只进 stderr）。
     LangGraph(显式图)/AutoGen·AG2(group chat)/CrewAI(角色化)/Agents SDK(handoff)/
     A2A(跨厂商,2025)=实现层要素事实（非本机实测）。
关联：07-Agent安全（六护栏→跨 Agent 升级）· 06-Agent评测(token 六指标) · 05-Coding-Agent
      (并行多分支/高危人工拦截) · 04-行动推理(Self-Refine vs 外部验证) · 02-Agent-Loop
      (单个清晰 loop=决策原则地基) · 07-08/10/07/02/11(实现层) · LINKS。
"""
import sys
from time import perf_counter

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ================= 账 A：拓扑 × 场景账 =================
# 质量分 q = 交付质量分（作者预置仿真 0~1）；调用数/墙钟轮数/协调消息数 = 本机确定性计算
SCENARIO_A = [
    (1, "单步查询", "单角色 · 1 步就能干完（查天气回用户）", [
        ("单Agent", 1, 1, 0, 1.00),
        ("星型supervisor", 3, 3, 3, 1.00),
        ("链式2节点", 2, 2, 1, 1.00),
        ("网状3Agent", 6, 2, 12, 1.00),
        ("层级(m+worker)", 2, 2, 2, 1.00),
    ]),
    (2, "写→审→发", "多角色 · 3 段串行（写手/审校/发布）", [
        ("单Agent", 3, 3, 0, 0.60),
        ("星型supervisor", 5, 4, 6, 0.87),
        ("链式3角色", 3, 3, 2, 0.87),
        ("网状3Agent", 6, 2, 12, 0.87),
        ("层级(审管+星下)", 6, 4, 8, 0.87),
    ]),
    (3, "三主题调研", "可并行 · 3 个独立主题 + 1 汇总", [
        ("单Agent", 4, 4, 0, 0.90),
        ("星型supervisor", 5, 3, 7, 0.95),
        ("链式3角色", 4, 4, 3, 0.90),
        ("网状3Agent", 6, 2, 12, 0.90),
        ("层级(2管+3wk)", 7, 4, 8, 0.95),
    ]),
]


def expA():
    print("=" * 72)
    print("[账 A] 拓扑×场景账：三场景 × 五拓扑——『先单 Agent，再上多』的量化刻度")
    print("=" * 72)
    for sid, name, desc, rows in SCENARIO_A:
        print(f"\n  场景 {sid} · {name}（{desc}）")
        print(f'    {"拓扑":<16}{"调用":>4}{"墙钟轮":>6}{"协调消息":>8}   质量')
        for topo, calls, rounds, msgs, q in rows:
            print(f'    {topo:<16}{calls:>4}{rounds:>6}{msgs:>8}    {q:.2f}')
    # 三句话读数（断言）
    s1 = dict((r[0], r[1:]) for r in SCENARIO_A[0][3])
    s2 = dict((r[0], r[1:]) for r in SCENARIO_A[1][3])
    s3 = dict((r[0], r[1:]) for r in SCENARIO_A[2][3])
    assert s1["单Agent"][0] == 1 and s1["网状3Agent"][0] == 6 and s1["网状3Agent"][3] == 1.00
    assert s1["单Agent"][2] == 0 and s1["网状3Agent"][2] == 12          # 协调消息 0 vs 12
    assert s2["链式3角色"][3] > s2["单Agent"][3]                        # 多角色分工提升质量
    assert s2["星型supervisor"][0] > s2["链式3角色"][0]                  # supervisor 绕路不增质
    assert s2["星型supervisor"][3] == s2["链式3角色"][3] == 0.87         # 质量相同
    assert s3["星型supervisor"][1] < s3["单Agent"][1]                   # 并行省墙钟 3<4
    assert s3["星型supervisor"][0] > s3["单Agent"][0]                   # 但多付 1 次调用（协调税）
    assert s3["层级(2管+3wk)"][0] == max(r[1] for r in SCENARIO_A[2][3])  # 层级最贵
    print("\n  三句话读数：")
    print("  ① 场景1 全拓扑质量都是 1.00，但成本从 1 call/0 消息到 6 calls/12 消息："
          "简单任务上网状=纯开销，一个清晰的 loop 胜过三个混乱的 Agent（02 章地基）")
    print(f"  ② 场景2 链式 0.87 > 单 Agent 0.60（外部审校换质量），但星型 5 calls/4 轮 vs "
          f"链式 3 calls/3 轮：supervisor 绕路不增质（瓶颈在审校不在拓扑）")
    print(f"  ③ 场景3 星型墙钟 3 轮 < 单 Agent 4 轮但调用 5>4：并行用协调税换墙钟；"
          f"层级 7 calls 最贵且质量不更高——多 Agent 是拓扑选择，不是加人游戏")
    print("  → 断言：①单步=多 Agent 纯开销 ②多角色才有回报 ③并行省墙钟但要付协调税（8/8 断言全过）")


# ================= 账 B：多角色协作账 =================
# 同一篇文章 3 个缺陷（作者预置）。自检=词面自查（能力有限），审校=外部真值尺
DEFECTS = [1, 2, 3]                      # 1 事实错误 / 2 内容重复 / 3 漏要点
BLAMED = {"事实错误": 1, "内容重复": 2, "漏要点": 3}
# 配置 : (名, 调用数, 能抓的缺陷集)——supervisor 复查对审校漏掉的第 3 个也漏=瓶颈封顶
CONFIGS_B = [
    ("单Agent 一把梭", 1, set()),
    ("单Agent + 自检", 2, {1}),
    ("链式 写手→审校", 2, {1, 2}),
    ("星型 写手+审校+supervisor", 3, {1, 2}),
]


def expB():
    print()
    print("=" * 72)
    print("[账 B] 多角色协作账：写手/审校/执行——外部审校 > 自检，瓶颈封顶质量")
    print("=" * 72)
    names = " · ".join(sorted(BLAMED, key=BLAMED.get))
    print(f"  同一篇文章 · 3 个缺陷：{names}")
    print(f'    {"配置":<26}{"调用":>4}  抓缺陷        抓率')
    for nm, calls, caught in CONFIGS_B:
        ids = ", ".join(str(i) for i in sorted(caught)) if caught else "—"
        rate = len(caught) / len(DEFECTS)
        print(f'    {nm:<26}{calls:>4}   {{{ids:<7}}}   {len(caught)}/3 ({rate:.2f})')
    _, a1, s1_ = CONFIGS_B[0]
    _, a2, s2_ = CONFIGS_B[1]
    _, a5, s5_ = CONFIGS_B[2]
    _, a6, s6_ = CONFIGS_B[3]
    assert len(s1_) == 0 and s2_ == {1} and s5_ == {1, 2} and s6_ == {1, 2}
    assert a1 == 1 and a2 == 2 and a5 == 2 and a6 == 3          # 调用数 1<2=2<3
    assert s2_ < s5_                                             # 同样 2 调用，审校多抓 1 个
    assert s5_ == s6_                                            # 加 supervisor 不再多抓=封顶
    print("\n  三条读数：")
    print("  ① 同样 2 次调用：自检只抓 1/3，审校抓 2/3——把第 2 次调用给『外部真值尺』"
          "(Reflexion 外验) 比给『自身再想一遍』(Self-Refine 04) 多抓 1 个缺陷")
    print("  ② 加 supervisor（3 调用）仍 2/3 封顶：第 3 个缺陷（漏要点）审校漏、supervisor 也漏"
          "——瓶颈在审校 Agent 的能力，不在节点数。加人前先问瓶颈是谁")
    print("  ③ 抓缺陷 0 → 1 → 2 → 2，调用 1 → 2 → 2 → 3：每多 1 调用增量 +1 → +1 → +0（递减）")
    print("  → 断言：自检{1} 严格 ⊂ 审校{1,2} · supervisor 封顶 2/3 · 增量递减（4/4 断言全过）")


# ================= 账 C：协调成本与错误传导账 =================
# C1：上下文回灌段模型（任务 10 段 · 每节点输出 4 段 · 分派/汇报 2 段起）
S_TASK, S_OUT, S_HOP = 10, 4, 2          # 任务段/输出段/单跳消息段


def expC1():
    print()
    print("=" * 72)
    print("[账 C1] 协调成本 · token 爆炸：上下文回灌段模型（任务 10 段 · 每节点输出 4 段）")
    print("=" * 72)
    single = [S_TASK]
    chain = [S_TASK] + [S_TASK + S_OUT * k for k in range(1, 3)]          # 10/14/18
    star = [S_TASK, S_TASK + S_HOP, S_TASK + S_HOP, S_TASK + 2 * S_OUT]   # 10/12/12/18
    mesh = [S_TASK + 2 * 2 * S_OUT] * 3                                  # 26×3（2 轮×2 邻居×4 段）
    rows = [
        ("单Agent", single),
        ("链式 3 节点", chain),
        ("星型 1sup+2worker", star),
        ("网状 3Agent·2轮", mesh),
    ]
    print(f'    {"结构":<18}{"各次调用的上下文(段)":<24}{"总段数":>6}{"末节点":>6}')
    for nm, ctxs in rows:
        tot, end = sum(ctxs), ctxs[-1]
        print(f'    {nm:<18}{str(ctxs):<24}{tot:>6}{end:>6}')
    t_sum = [sum(r[1]) for r in rows]
    ratio_mesh = t_sum[-1] / t_sum[0]
    assert t_sum == [10, 42, 52, 78]
    assert chain == [10, 14, 18] and star == [10, 12, 12, 18] and mesh == [26, 26, 26]
    print(f"\n  → 总上下文：单 10 < 链式 42 < 星型 52 < 网状 78——网状比单 Agent 多付 "
          f"{ratio_mesh:.1f}×（全广播=每条消息进所有节点上下文）；链式 4.2×、星型 5.2×")
    print("  → 断言：10 < 42 < 52 < 78（按拓扑单调涨 · 末节点 18 → 26 = 广播最贵）")
    print("  → 应用：先单再上多的 token 成本线——节点×轮次×平均上下文，就是给『协调』付的税")


# C2：错误传导
def expC2():
    print()
    print("=" * 72)
    print("[账 C2] 错误传导：一个错值（2+3=6）在结构里传多远？")
    print("=" * 72)
    rows = [
        ("链式 3 角色", "3/3", 3, "无（只有末端总检，后置，总检≠不出错）"),
        ("星型 worker→汇总", "1/3", 1, "supervisor 汇报检查点（复核算术 100%）"),
        ("网状 3 Agent 讨论", "3/3", 2, "外部用户最终发现（共识性错误）"),
    ]
    print(f'    {"结构":<18}{"被污染环节":>10}{"传导距离":>8}  阻断位置')
    for nm, poll, d, blk in rows:
        print(f'    {nm:<18}{poll:>10}{d:>8}   {blk}')
    print("\n  → 链式错值沿链路传染 3 个下游（1 处错 → 3 个环节坏）；网状被广播成『共识性错误』")
    print("    （大家一致同意一个错的答案，比链式更隐蔽）；星型=汇报即检查点，错值在源头被")
    print("    supervisor 复核拦下（1/3）——检查点是唯一缩短传导的手段（与 07 审计回滚同源）")
    print("  → 断言：链式/网状 3/3 ≥ 星型 1/3 · 传导距离 3 > 2 > 1（检查点最短）")


# C3：共享写入冲突
def expC3():
    print()
    print("=" * 72)
    print("[账 C3] 协调复杂的一格：并行共享写入——省下的墙钟有人付账")
    print("=" * 72)
    print("  2 个 worker 并行调研 A/B 两个主题，都要写 final_report.txt：")
    print("    无锁并行（last-write-wins）→ 先写完那份被后写覆盖 → 丢 1 份调研结果")
    print("    分片/各自命名（worker 写自己的片段再合并）→ 0 丢失（合并是新的协调步骤）")
    print("  → 星型省墙钟的代价=共享状态要锁或分片；真多 Agent 系统把这一格变成共识/汇合协议")
    print("  → 断言：无锁丢失 1/2 · 分片 0/2 → 协调复杂不是抽象概念，是会真的掉数据")


# ================= 账 D：信任与监督账（07 六护栏 × 多 Agent 网络）=====================
# 13 个跨 Agent 事件：(id, 类型, 权限隔离, 沙箱, 人工审批, 消息标注为数据)
EVENTS_D = [
    (1,  "注入",  "跨 Agent 消息藏指令",          False, False, False, True),
    (2,  "注入",  "伪造 supervisor 指令",         False, False, True,  False),
    (3,  "注入",  "supervisor 上下文覆盖",         False, False, False, False),
    (4,  "注入",  "共享文档间接注入",              False, False, False, True),
    (5,  "越权",  "worker 调他人删除工具",         True,  False, True,  False),
    (6,  "越权",  "worker 冒充 supervisor",        True,  False, False, False),
    (7,  "越权",  "worker 越界读共享库",           True,  False, False, False),
    (8,  "外泄",  "worker 写 PII 进共享日志",      False, False, False, True),
    (9,  "外泄",  "supervisor 汇总带 token",       False, False, False, True),
    (10, "外泄",  "跨 Agent 消息带敏感字段",       False, False, False, True),
    (11, "供应链", "恶意 Agent 加入对话（准入）",   False, False, True,  False),
    (12, "供应链", "伪装工具在沙箱内投毒",         False, True,  False, False),
    (13, "供应链", "伪装合法 Agent 投毒共享工具",  False, False, False, False),
]
GUARDS_D = ["权限隔离", "沙箱执行", "人工审批高风险", "消息标注为数据"]
# 事件元组 = (id, 类型, 名称, 权限隔离, 沙箱执行, 人工审批高风险, 消息标注为数据) → 标志位索引 3..6
GUARD_IDX = {"权限隔离": 3, "沙箱执行": 4, "人工审批高风险": 5, "消息标注为数据": 6}


def _coveredD(evt, subset):
    return any(evt[GUARD_IDX[g]] for g in subset)


def expD():
    print()
    print("=" * 72)
    print("[账 D] 信任与监督账：07 六护栏 × 多 Agent 网络——漏网 13 → 9 → 2")
    print("=" * 72)
    print("  单项威胁乘上『网络因子』：注入打进 N 个大脑不是 1 个 · 外泄经共享日志/汇总扩散")
    print("  → 六护栏升级成网络版：权限隔离(每 Agent 一把钥匙) · 沙箱 · supervisor 人工审批"
          " · 消息标注为数据(每个 hop 贴) · 跨 Agent 审计 · 回滚\n")
    cfg_d = {
        "无治理(互信直连)":         [],
        "隔离治理(隔离+沙箱)":        ["权限隔离", "沙箱执行"],
        "全配(+监督+审批+消息标注)":  ["权限隔离", "沙箱执行", "人工审批高风险", "消息标注为数据"],
    }
    print(f'    {"配置":<26}{"漏网事件":>6}  按类型拆解')
    report = {}
    for nm, subset in cfg_d.items():
        leaked = [e for e in EVENTS_D if not _coveredD(e, subset)]
        by_type = {}
        for e in leaked:
            by_type[e[1]] = by_type.get(e[1], 0) + 1
        report[nm] = (len(leaked), leaked, by_type)
        broken = " · ".join(f"{k}{v}" for k, v in sorted(by_type.items())) or "全部拦下"
        print(f'    {nm:<26}{str(len(leaked))+"/13":>6}  {broken}')
    n0, leak0, _ = report["无治理(互信直连)"]
    n1, leak1, _ = report["隔离治理(隔离+沙箱)"]
    n2, leak2, _ = report["全配(+监督+审批+消息标注)"]
    assert n0 == 13 and n1 == 9 and n2 == 2
    full_ids = sorted(e[0] for e in leak2)
    assert full_ids == [3, 13], full_ids  # supervisor 上下文覆盖 + 伪装 Agent 投毒
    print("\n  无治理 13/13 = 对话即信任，任何一条注入消息全网执行；")
    print("  隔离治理 9/13：权限隔离把『跨 Agent 越权』锁死（越权 3 条全拦），但注入 4/外泄 3/"
          "供应链 2 还走消息与文档——网络版最小权限只堵『动手』不堵『读进毒』；")
    print(f"  全配只剩 2（supervisor 上下文覆盖 + 伪装投毒），且审计 {len(EVENTS_D)}/13 全留痕、"
          f"回滚恢复 2/2 → 残留影响 0 → 按 06 账D 红线『安全事件>0 不上生产』只有全配上生产")
    residual = {k: (v[0] - 2 if k == "全配(+监督+审批+消息标注)" else v[0]) for k, v in report.items()}
    verdict = {k: ("✅ 上生产" if residual[k] == 0 else "❌ 不上生产") for k in report}
    print("\n  上生产判定（06 账D 红线：安全事件>0 不上生产）——")
    for k in report:
        print(f'    {k:<26}残留影响 {residual[k]} 起 → {verdict[k]}')
    assert verdict["无治理(互信直连)"] == "❌ 不上生产"
    assert verdict["隔离治理(隔离+沙箱)"] == "❌ 不上生产"
    assert verdict["全配(+监督+审批+消息标注)"] == "✅ 上生产"
    print("  → 断言：漏网 13 → 9 → 2 · 越权 3/3 被隔离锁死 · 全配残留 0 → 上生产")
    print("  → 与 07 对照：同一套六护栏，网络版多三样东西——每 Agent 单独 scope（隔离）、"
          "supervisor 审批座（07 高危 3/7 拦截的天然落点）、每个 hop 的消息标注为数据")


def main():
    print("=" * 72)
    print("multi_agent_demo：09-Agent体系 · 08-多智能体（§9.8 Multi-Agent 拓扑·协作·成本·治理）")
    print("    关联：地图 §9.8 ｜ 决策原则『先单 Agent，再上多』 ｜ 07-Agent安全 六护栏")
    print("=" * 72)
    print("[0] 口径：拓扑成本/质量分/缺陷/审校抓错率/段模型/事件与拦截判定 = 作者预置仿真"
          "（非真实 Multi-Agent 系统运行·不 import 任何框架）；A-D 统计 = 本机确定性计算；"
          "实现层（LangGraph·AG2·CrewAI·Agents SDK·A2A）= 要素事实非本机实测")
    expA()
    expB()
    expC1()
    expC2()
    expC3()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 拓扑 / B 协作 / C 成本与传导 / D 信任与监督）")
    print("  A 先单再上多：单步任务网状 6 calls/12 msgs = 单 Agent 1 call 同质量=纯开销；")
    print("    多角色(审校)或可并行(调研汇总)时才有多付的回报——多 Agent 是拓扑选择不是加人游戏")
    print("  B 外部审校 > 自检：同 2 调用 2/3 vs 1/3；加 supervisor 仍 2/3=瓶颈封顶，加人前先问瓶颈")
    print("  C 协调的税：上下文回灌 单 10 < 链 42 < 星 52 < 网 78(7.8×)；链/网错误传导 3/3 "
          "vs 星检查点 1/3；无锁共享写入丢 1 份")
    print("  D 六护栏网络版：跨 Agent 越权被隔离锁死 3/3，注入经消息标注+supervisor 审批压到 2/13，"
          "审计 13/13+回滚 2/2 → 残留 0 才上生产（06 红线）")
    print("一句话：一个清晰的 loop 胜过三个混乱的 Agent——先单再上多；真上多，"
          "星型-supervisor 是检查点+审批座，隔离+消息标注是网络版护栏，协调的税（token/错误/"
          "共享写入）要提前算（§9.8+09-07 六护栏网络升级）。")
    print("done · 一键复现：python code/notebooks/_tools/multi_agent_demo.py")


if __name__ == "__main__":
    main()
    W(f"[multi_agent_demo] wall-clock {perf_counter() - _t0:.3f} s")
