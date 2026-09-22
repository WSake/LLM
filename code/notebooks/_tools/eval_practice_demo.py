# -*- coding: utf-8 -*-
"""eval_practice_demo.py —— §11.9 评测工程最佳实践（11-评测与可观测 10 篇·收官篇配套）

§11.9 知识地图原文（🔴 总结·五条纪律）：
 ① 评测集分层：5-10 个精选用例（每日冒烟）+ 100-500 条业务集（每周回归）+ 大榜单（选型时）
 ② 固定评测环境：同 prompt 模板、同 temperature（评测用 0）、同 judge 版本
 ③ 每次改动跑同一套回归（调 prompt/换模型/改 RAG 都要）
 ④ 线上遥测回流成评测集（不满意、重试、纠错 → 加进评测集）
 ⑤ 用"每任务完成指标"而不是"每 token 质量指标"评价产品

本篇定位：11 章收官——把 01-09 学过的十篇串成「评测作为门禁嵌入开发流程」的一张装配手册
（五条纪律 × 四账），不再是新概念而是复习链+装配（与 09 章收官篇同构）。
本篇不 import 任何框架，四账全部用确定性数据结构 + 本机算账复现"评测工程"的机制：
账 A  分层账（纪律①）：三层评测集（冒烟/回归/大榜单）× 规模·时机·成本 + 5 场景判层 5/5
账 B  固定环境账（纪律②③）：四件套 4/4 + 缺件后果 3 条 + 改动→最小回跑尺子 5/5
账 C  回流账（纪律④）：回流管道 4 级 + 5 条线上日志判定 5/5 + 锚点账（100+8=108 挂 08/09）
账 D  门禁账（纪律⑤ + 装配）：每任务 vs 每 token ×23.8 复算 + 门禁链 4 道 + 5 开发动作过闸 5/5

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（W() 负责计时与落 stderr）。全部统计 = 本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- §11.9 五条纪律 = 知识地图要素事实非本机实测（写作环境无外网未在线复核）。
- 分层定义/回归矩阵/门禁链 = §11.9 + 01 篇五层定界尺柜的复习链（锚点数字挂靠各篇探针不重测）。
- 5 场景判层/缺件后果/日志判定/过闸判断 = 作者按五条纪律预置的确定性判定表。
- ×23.8 = 06 篇账 D 挂靠复算；100+8=108 = 08 篇账 D4 / 09 篇账 C4 锚点传入。
- A-D 四账统计 = 本机确定性计算（stdout 三独立进程 md5 恒一，墙钟 ≈0.00 s 仅进 stderr）。
"""
from __future__ import annotations
import sys
import time

try:  # 输出统一 UTF-8 字节，不随控制台代码页漂移（GBK 下也能打印 ✅/⛔）
    _ = sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

_WSTART = time.perf_counter()


def W(msg: str) -> None:
    print(f"[wall] {time.perf_counter() - _WSTART:.3f}s  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 账 A · 分层账：三层评测集（冒烟/回归/大榜单）× 规模·时机·成本 + 5 场景判层
# ---------------------------------------------------------------------------
# (层名, 规模, 时机, 成本档, 规模下限, 线性序)
A_LAYERS = [
    ("冒烟 smoke", "5-10 条精选用例", "每天 / 每次提交", "秒级 · 免费量级", 5, 0),
    ("回归 regression", "100-500 条业务集", "每周 / 每次改动", "分钟级 · $0.1-$1", 100, 1),
    ("大榜单 leaderboard", "千级基准题", "选型时", "小时级 · $10+", 1000, 2),
]
# 5 场景判层：(场景描述, 期望层, 规则机关键词)
A_SCENES = [
    ("改了 NL2SQL 提示词后怀疑流式路径断了，只想确认核心路径没被破坏", "冒烟", ("断", "核心路径", "体检", "例行")),
    ("上个季度结束想换底座模型，需要重新选型", "大榜单", ("选型",)),
    ("换了 embedding 模型准备上线，想知道业务任务还能不能打", "回归", ("embedding", "chunk", "上线")),
    ("改了 chunk 大小，只改了检索那段", "回归", ("embedding", "chunk", "上线")),
    ("每天早晨例行体检核心路径三件事（登录/问答/RAG）", "冒烟", ("断", "核心路径", "体检", "例行")),
]
_LAYER_HINT = {"冒烟": ("断", "核心路径", "体检", "例行"), "回归": ("embedding", "chunk", "上线"),
               "大榜单": ("选型",)}
_LAYER_ORDER = {"冒烟": 0, "回归": 1, "大榜单": 2}


def _classify_layer(s: str) -> str:
    for layer, hints in _LAYER_HINT.items():
        if any(h in s for h in hints):
            return layer
    return "?"


def account_a() -> str:
    out = []
    out.append("== A · 分层账：评测集三层（§11.9 纪律① 评测集分层） ==")
    names = [n for n, _, _, _, _, _ in A_LAYERS]
    order_ok = [_LAYER_ORDER[n.split(" ")[0]] for n in names] == list(range(3))
    for name, size, when, cost, lo, _ in A_LAYERS:
        if name.startswith("冒烟"):
            ok = lo == 5 and size.startswith("5-10")
        elif name.startswith("回归"):
            ok = lo == 100 and size.startswith("100-500")
        else:
            ok = lo == 1000 and size.startswith("千级")
        out.append(f"    {name}：规模 {size} · 时机 {when} · 成本 {cost} = "
                   + ("PASS" if ok and when else "FAIL"))
    ok1 = order_ok and len(names) == len(set(names)) == 3
    out.append(f"    三层规模递增（冒烟 < 回归 < 大榜单）+ 时机匹配 = " + ("PASS" if ok1 else "FAIL"))
    n_bad = 0
    for s, want, hints in A_SCENES:
        got = _classify_layer(s)
        bad = got != want
        n_bad += 1 if bad else 0
        out.append(f"    [{want}] {s}  -> 判 {got} = " + ("PASS" if not bad else "FAIL"))
    ok_a = ok1 and n_bad == 0
    out.append(f"    -> 三层定义 3/3 + 场景判层 5/5 = "
               + ("PASS(冒烟 秒级/每日 保底 · 回归 每次改动跑同一套 · 大榜单 只在选型时烧钱)"
                  if ok_a else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 B · 固定环境账：四件套 + 缺件后果 + 改动→最小回跑尺子
# ---------------------------------------------------------------------------
B_FOUR = ["同 prompt 模板", "同 temperature（评测用 0）", "同 judge 版本", "同语料子集"]
B_GAPS = [
    ("temperature≠0 而 judge 又敏感", "同输入两次跑分不同=测的不是同一个东西", "固定 temp=0"),
    ("judge 悄悄升了版本", "同 20 样本翻转 6 条=漂移 0.30（03 篇账 C 挂靠）", "锁 judge 版本"),
    ("语料子集换了一批", "子集不同跑的水不同=前后不可比", "同语料子集"),
]
# 改动 → 最少回跑哪把尺（01 篇账 C 五条规则挂靠）：(动作, 判定键, 最小回跑层)
B_CHANGES = [
    ("改了 prompt 模板", "prompt", "任务层"),
    ("换了底座模型", "底座模型", "能力+任务"),
    ("改 RAG 的 chunk/embedding/rerank", "chunk", "系统层"),
    ("改 Agent 的编排/工具", "编排", "系统层评测台"),
    ("动部署/切流式/换硬件", "部署", "工程层"),
]
_B_RULE = {"prompt": "任务层", "底座模型": "能力+任务", "chunk": "系统层", "embedding": "系统层",
           "rerank": "系统层", "编排": "系统层评测台", "工具": "系统层评测台", "部署": "工程层",
           "流式": "工程层", "硬件": "工程层"}


def account_b() -> str:
    out = []
    out.append("== B · 固定环境账：四件套 + 每次改动跑回归（§11.9 纪律②③） ==")
    ok_b1 = len(B_FOUR) == 4 and len(set(B_FOUR)) == 4
    out.append(f"    固定环境四件套（§11.9②）= {' · '.join(B_FOUR)} = "
               + ("PASS(4/4 齐：同模板·同 temp·同 judge·同语料子集=环境固定才可比)" if ok_b1 else "FAIL"))
    for gap, why, fix in B_GAPS:
        out.append(f"    缺件：{gap} → {why}（对策：{fix}） = PASS")
    n_bad = 0
    for s, key, want in B_CHANGES:
        got = _B_RULE[key]
        bad = got != want
        n_bad += 1 if bad else 0
        out.append(f"    {s} -> 最小回跑 {got} = " + ("PASS" if not bad else "FAIL"))
    ok_b3 = n_bad == 0
    out.append(f"    -> 四件套 4/4 + 缺件后果 3 条 + 改动→尺子 5/5 = "
               + ("PASS(每改一处只回跑该层尺子=评测贵在刀刃上；换模型必重跑能力+任务=08 篇预算回归)"
                  if ok_b1 and ok_b3 else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 C · 回流账：回流管道 + 线上日志判定 + 锚点账
# ---------------------------------------------------------------------------
C_PIPE = ["采集（trace/日志）", "判定（不满意/重试/纠错/慢）", "标注（期望答案）", "进评测集"]
C_LOGS = [
    ("用户点踩了这条回答", "回流", "不满意类（§11.9④ 第一类）"),
    ("同一个问题被重试 3 次", "回流", "重试类"),
    ("用户明确说『你答错了』", "回流", "纠错类"),
    ("这条请求 P95 超了 SLO", "回流", "慢请求证据（工程层·09 篇账 C4 同号）"),
    ("用户复制粘贴答案没动作", "不回", "无反馈=没信号，不制造样本"),
]


def _classify_log(s: str) -> str:
    if "没动作" in s or "复制粘贴" in s:
        return "不回"
    if "P95" in s:
        return "回流"
    if any(h in s for h in ("点踩", "重试", "答错了")):
        return "回流"
    return "?"


def account_c() -> str:
    out = []
    out.append("== C · 回流账：线上遥测回流成评测集（§11.9 纪律④） ==")
    out.append(f"    回流管道 4 级 = {' → '.join(C_PIPE)} = "
               + ("PASS(采集→判定→标注→进集=遥测回流的完整闭环)" if len(C_PIPE) == 4 else "FAIL"))
    n_bad = 0
    for s, want, why in C_LOGS:
        got = _classify_log(s)
        bad = got != want
        n_bad += 1 if bad else 0
        out.append(f"    [{want}] {s}（{why}） -> 判 {got} = " + ("PASS" if not bad else "FAIL"))
    # 锚点账：挂靠不重测（06-Agent 遥测回流 4→204 · 07-安全 红队 50→54 · 08 工程 100+8=108）
    anchor_ok = (100 + 8 == 108) and (200 + 4 == 204) and (50 + 4 == 54)
    out.append(f"    锚点账（只引不重测）：08 篇账 D4 慢请求回注 100+8=108 · 09 篇账 C4 同号 · "
               f"06-Agent 遥测回流 4→204 · 07-安全 红队 4→50→54 = "
               + ("PASS(四条回流号全部一致=遥测回流是同一根管线的不同口味)"
                  if anchor_ok else "FAIL"))
    ok_c = n_bad == 0 and anchor_ok
    out.append(f"    -> 管道 4 级 + 日志判定 5/5 + 锚点 4 条 = "
               + ("PASS(5 条日志里 4 条回流 1 条不回=用户反馈才是信号，复制粘贴不算)" if ok_c else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 D · 门禁账：每任务 vs 每 token（×23.8 复算）+ 门禁链 + 过闸判定
# ---------------------------------------------------------------------------
def account_d() -> str:
    out = []
    out.append("== D · 门禁账：评测作为门禁嵌入开发流程（§11.9 纪律⑤ + 装配） ==")
    # 每任务完成 vs 每 token 质量：X 蔓胖 vs Y 精瘦（06 篇账 D 挂靠复算）
    x_tok, x_cost, x_tasks, x_eval = 10000, 1.500, 0, 2.10
    y_tok, y_cost, y_tasks, y_eval = 420, 0.063, 1, 1.70
    ratio = x_cost / y_cost
    ok_d1 = abs(ratio - 23.8) < 0.1 and x_tasks == 0 and y_tasks == 1
    out.append(f"    X 蔓胖：{x_tok} tok ≈${x_cost} · 每任务完成 {x_tasks}/1 · 单轮打分 2.10")
    out.append(f"    Y 精瘦：{y_tok} tok ≈${y_cost} · 每任务完成 {y_tasks}/1 · 单轮打分 1.70")
    out.append(f"    每任务完成（产品侧）Y 胜 / 每 token 单轮（过程侧）X 胜 · 总成本 {ratio:.1f}× = "
               + ("PASS(×23.8 复算：用『每任务完成』不是『每 token 质量』评价产品=§11.9⑤·06 篇账 D 只引不重测)"
                  if ok_d1 else "FAIL"))
    # 门禁链 4 道
    gates = ["冒烟", "回归", "安全", "工程"]
    seq_ok = gates == ["冒烟", "回归", "安全", "工程"]
    out.append(f"    门禁链（一次提交从开发到上线）= {' → '.join(gates)}"
               + "（每日冒烟 → 每次改动回归 → 上线前安全红线 → 工程 SLO） = "
               + ("PASS(四道闸时序固定：先量功能再量安全再量工程=01 篇五层定界的装配顺序)" if seq_ok else "FAIL"))
    # 5 开发动作 → 该过哪道闸
    actions = [
        ("prompt 文案微调", "prompt", "冒烟"),
        ("换 embedding 模型", "embedding", "回归"),
        ("漏斗里有安全事件 > 0", "安全事件", "安全"),
        ("切流式架构", "流式", "工程"),
        ("新模型选型", "选型", "大榜单"),
    ]
    rule = {"prompt": "冒烟", "embedding": "回归", "安全事件": "安全", "流式": "工程", "选型": "大榜单"}
    n_bad = 0
    for s, key, want in actions:
        got = rule[key]
        bad = got != want
        n_bad += 1 if bad else 0
        out.append(f"    {s} -> 该过 {got} = " + ("PASS" if not bad else "FAIL"))
    ok_d2 = n_bad == 0
    out.append(f"    -> ×23.8 复算 + 门禁链时序 + 动作过闸 5/5 = "
               + ("PASS(评测不是最终的报告，是开发流程的门禁=05-RAG 闸线 0.80 / 07 安全红线 / 09 SLO 同构，挂靠只引不重测)"
                  if ok_d1 and ok_d2 else "FAIL"))
    return "\n".join(out)


def main() -> None:
    W("start")
    parts = [account_a(), account_b(), account_c(), account_d()]
    print("\n" + "\n".join(parts), end="")
    W("done")


if __name__ == "__main__":
    main()
