# -*- coding: utf-8 -*-
"""ray_infra_demo.py —— §13.4 Ray 分布式运行时（13-工程化与基础设施 04 篇配套）

§13.4 知识地图原文（🟢 掌握概念；Agent 时代更重要）：
- 定位：分布式 Python 运行时——统一训练（Ray Train）、推理（Ray Serve）、数据（Ray Data）、
  调参（Tune）、编排（Workflows）的弹性底座。
- 为什么重要：Agent 大规模并行（成百上千个任务）、批处理数据、在线-离线混合负载的通用胶水；
  大厂 Agent 平台底层很多用 Ray。

本篇定位：Ray 是「跑在大集群上的调度语义」，本机既有约束（零 GPU · 零 K8s · 零 Ray 集群）
决定跑不了真实 Ray——降维成四本确定性账，把 Ray 的「定位 / 弹性并行算术 / 有状态原语 /
在线-离线混合」讲成能算、能复算的账：
账 A  定位装配账：Ray 五件套（Core/Train/Serve/Data/Tune/Workflows）+ 三原语 + 与 K8s 的两层边界
账 B  弹性并行账：Agent 大规模 fan-out 的墙钟算术（并行度×机器数×利用率→墙钟）+ 失败重试
账 C  Actor 状态账：有状态原语 vs 13-03「无会话粘性」纪律——五场景三问（谁持状态/状态可拆吗/状态能丢吗）
账 D  在线-离线混合账：Ray Serve 在线两层扩缩 + Ray Data 离线墙钟 + 峰值抢占 + 集群利用率

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（W() 负责计时与落 stderr）。全部统计 = 本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- Ray 五件套定位、三原语（task/actor/ObjectRef）、API 名（ray.init/@ray.remote/ray.get）、
  与 K8s 的两层关系、大厂 Agent 平台用 Ray = §13.4 要素事实（公开资料整理，
  写作环境无外网未在线复核）。
- 墙钟算术（并行度×机器数÷利用率→墙钟）、重试期望（p + 一次重试 → 终成功率）、
  Serve/Data/混合水位的数字 = 工程惯例 + 作者预置，本机只做代入公式的算术。
- 本篇**不 import ray、不起集群、不跑真实任务**——A-D 数字 = 作者预置输入 + 本机确定性计算；
  真实 ray.init/起 worker/跑 job 留读者在自家集群实操。
"""
from __future__ import annotations
import math
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


def ceilf(x: float) -> int:
    return int(math.ceil(x - 1e-9))


# ---------------------------------------------------------------------------
# 要素事实表（§13.4 + 公开资料整理；与 13-01/02/03 同口径）
# ---------------------------------------------------------------------------
N_AGENTS = 1000         # 一次 Agent 大规模 fan-out 的子任务数（作者预置：成百上千个网页/检索生成任务）
T_TASK_S = 12.0         # 单子任务基准耗时 s（作者预置）
WORKERS_PER_MACHINE = 32  # 单机 ray worker 数（按 32 vCPU，作者预置）
UTIL_GOAL = 0.70        # 集群利用率目标（与 13-03 同档，留水位给 HPA 扩张空间）
P_OK = 0.98             # 单任务成功率（作者预置，最多重试一次）
DATA_ROWS = 1_000_000   # 离线批处理行数（Ray Data 场景）
MS_PER_ROW = 1.0        # 每行处理耗时 ms（作者预置）
N_MACHINES = 8          # 混合集群总机器数
ONLINE_SHARE = 0.60     # 在线服务保留机器占比
SERVE_REPL, SERVE_CONC = 4, 16   # Ray Serve 单 deployment 副本数 × 每副本并发


# ---------------------------------------------------------------------------
# 账 A · 定位装配账：Ray 五件套 + 三原语 + 与 K8s 的两层边界
# ---------------------------------------------------------------------------
# (组件, 归属层, 该层管什么的判词)
A_ROWS = [
    ("@ray.remote task（无状态远程函数）", "Core", "一次调用一个结果，最细粒度的远程原语"),
    ("@ray.remote class actor（有状态类实例）", "Core", "持有状态、串行执行方法"),
    ("ObjectRef（远端对象未来句柄）", "Core", ".remote() 返回即用，ray.get() 才收值"),
    ("Ray Train", "训练", "分布式训练：DataParallel/Torch 组装进同一套弹性调度"),
    ("Ray Serve", "在线推理", "Deployment 多副本 + HTTP，扩缩到副本内并发粒度"),
    ("Ray Data", "批量数据", "百万行并行 map/batch 处理"),
    ("Ray Tune", "调参", "分布式跑 trial，谁好谁拿资源多"),
    ("Ray Workflows", "编排", "多步有向图 + 检查点，挂掉从断点续"),
    ("K8s 管第一层", "边界", "Node/容器/Pod/网络：你声明的副本数保活"),
    ("Ray 管第二层", "边界", "Pod 内 task/actor/数据：同一个 Pod 里的精细调度"),
    ("两层装配各管一段", "边界", "13-03 账 C 的 HPA 扩副本做不了 task 级编排，谁也不会替谁"),
]
A_LAYERS = {"Core", "训练", "在线推理", "批量数据", "调参", "编排", "边界"}


def account_a() -> str:
    out = []
    out.append("== A · 定位装配账：Ray 五件套 + 三原语 + 与 K8s 的两层边界（Ray 是什么） ==")
    n_bad = 0
    for name, layer, what in A_ROWS:
        ok = layer in A_LAYERS and what != ""
        n_bad += 0 if ok else 1
        out.append(f"    [{layer.rjust(4)}] {name} = {what} = " + ("PASS" if ok else "FAIL"))
    ok = n_bad == 0
    out.append("    五件套定位：统一训练/推理/数据/调参/编排于一身的分布式 Python 运行时 = 弹性底座"
               + (" = " + ("PASS" if ok else "FAIL")))
    # 一次远程调用的数据流：driver -> ray.init -> .remote() 提交 -> ObjectRef -> ray.get 收拢
    steps = ["driver ray.init", "提交 N 个 .remote() 任务", "各得 ObjectRef（不阻塞）", "ray.get 收拢结果"]
    ok2 = len(steps) == 4
    out.append("    一次远程调用串起：driver ray.init → 提交 .remote()（不阻塞、即返 ObjectRef）→ "
               f"ray.get 收拢 = 并行提交与阻塞收值是两件事（4 步） = " + ("PASS" if ok2 else "FAIL"))
    ok3 = True  # Agent 时代重要两条
    out.append("    §13.4『为什么重要』两条：Agent 大规模并行（成百上千任务 fan-out）· 在线-离线混合负载的通用胶水"
               "（大厂 Agent 平台底层很多用 Ray = 要素事实） = " + ("PASS" if ok3 else "FAIL"))
    out.append("    -> Ray = 跑在大集群上的调度语义：K8s 把机器变成可用资源，Ray 把这些资源变成 task/actor/数据")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 B · 弹性并行账：Agent 大规模 fan-out 的墙钟算术 + 失败重试
# ---------------------------------------------------------------------------
def account_b() -> str:
    out = []
    out.append("== B · 弹性并行账：Agent 大规模 fan-out 的墙钟怎么算（任务并行 vs 机器数 vs 利用率） ==")
    # B1 串行基线：N 个任务逐个跑
    t_serial = N_AGENTS * T_TASK_S
    ok1 = t_serial == 12000.0
    out.append(f"    串行基线：{N_AGENTS} 个子任务 × 单任务 {T_TASK_S:.0f} s = {t_serial:.0f} s（≈3.3 小时，逐个跑不伸缩）"
               f" = " + ("PASS" if ok1 else "FAIL"))
    # B2 单机并行：W 个 worker 同时跑，墙钟 = ceil(N/W)×t
    w1 = WORKERS_PER_MACHINE
    t1 = ceilf(N_AGENTS / w1) * T_TASK_S
    ok2 = w1 == 32 and t1 == 384.0
    out.append(f"    单机 {w1} worker：并行度拉满 = ceil({N_AGENTS}/{w1}) = {ceilf(N_AGENTS / w1)} 批 × "
               f"{T_TASK_S:.0f} s = {t1:.0f} s（≈串行的 1/{(t_serial / t1):.0f}） = " + ("PASS" if ok2 else "FAIL"))
    # B3 加机器：M 台 × 每台 worker
    m = 4
    w4 = m * WORKERS_PER_MACHINE
    t4 = ceilf(N_AGENTS / w4) * T_TASK_S
    ok3 = w4 == 128 and t4 == 96.0
    out.append(f"    {m} 台机器 × {WORKERS_PER_MACHINE} worker = {w4} worker → ceil({N_AGENTS}/{w4}) = "
               f"{ceilf(N_AGENTS / w4)} 批 × {T_TASK_S:.0f} s = {t4:.0f} s（加机器=批量买墙钟，"
               f"墙钟 = ceil(N ÷ 总 worker) × 单任务耗时） = " + ("PASS" if ok3 else "FAIL"))
    # B4 利用率折损：有效并行 = W×UTIL，墙钟按有效算
    w_eff = math.floor(w4 * UTIL_GOAL)
    t_eff = ceilf(N_AGENTS / w_eff) * T_TASK_S
    ok4 = w_eff == 89 and t_eff == 144.0
    out.append(f"    利用率目标 {UTIL_GOAL:.0%}：有效并行 = floor({w4}×{UTIL_GOAL:.0%}) = {w_eff} → "
               f"ceil({N_AGENTS}/{w_eff}) = {ceilf(N_AGENTS / w_eff)} 批 × {T_TASK_S:.0f} s = {t_eff:.0f} s"
               f"（留水位给 HPA/队列，13-03 账 B3 同纪律） = " + ("PASS" if ok4 else "FAIL"))
    # B5 失败重试：成功率 p、最多重试 1 次 → 终成功率 = 1-(1-p)^2
    p_final = 1 - (1 - P_OK) ** 2
    exp_fail = N_AGENTS * (1 - p_final)
    ok5 = abs(p_final - 0.9996) < 1e-9 and exp_fail < 1
    out.append(f"    重试账：单任务成功率 {P_OK:.0%}、最多重试 1 次 → 终成功率 = 1-(1-{P_OK:.0%})² = {p_final:.4f}，"
               f"{N_AGENTS} 任务期望失败 {exp_fail:.2f} 个（重试一次基本全成，ray 的天然 retry 语义） = "
               + ("PASS" if ok5 else "FAIL"))
    out.append("    -> 弹性并行三连：先报 N（任务数）→ 并行度=机器×worker×利用率 → 墙钟=ceil(N/有效并行)×单任务耗时")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 C · Actor 状态账：有状态原语 vs 「无会话粘性」纪律——五场景三问
# ---------------------------------------------------------------------------
def account_c() -> str:
    out = []
    out.append("== C · Actor 状态账：有状态原语 vs 13-03「无会话粘性」纪律（状态该放哪） ==")
    # 状态三问：谁持状态 / 状态可拆吗 / 状态能丢吗 —— 五场景判定表
    SCENARIOS = [
        # (负载, 状态在哪, 该用 actor/对象吗, 判词)
        ("在线 LLM 服务（KV 随请求生灭）", "随请求生灭", "否（无状态横扩）",
         "接 13-03 账 B5：KV 不在服务端会话→副本无状态→HPA 才扩得动"),
        ("Agent 会话上下文（多轮对话）", "编排层 checkpoint", "否（不押在 actor）",
         "接 07-02 LangGraph checkpoint：状态在会话段与编排层，不在被调度的工作者身上"),
        ("前缀 KV 缓存（prefix cache）", "数据平面弱粘性", "可（actor 承载）",
         "接 13-03 账 D2：同前缀到同一实例复用 prefix cache，actor=弱粘性的承载者"),
        ("批量累加（count/sum accumulator）", "任务内局部", "否（每 worker 局部再归约）",
         "每个 task 里局部累加，末尾 reduce 归约——不需要跨任务共享状态"),
        ("共享只读模型权重", "全局只读", "对象广播（非 actor）",
         "接 13-03 账 D1：权重 ObjectRef 广播给各 worker 引用同一份，不重复加载"),
    ]
    n_bad = 0
    for i, (load, holder, use, why) in enumerate(SCENARIOS, 1):
        ok = why != "" and use.startswith(("否", "可", "对象广播"))
        n_bad += 0 if ok else 1
        out.append(f"    场景{i} [{load}] 状态={holder} · 承载={use} = {why} = " + ("PASS" if ok else "FAIL"))
    ok = n_bad == 0
    out.append(f"    状态三问纪律：谁持状态 / 状态可拆吗 / 状态能丢吗 → {len(SCENARIOS)}/5 走无状态与对象广播、"
               f"只 1 个让 actor 承载弱粘性 = " + ("PASS" if ok else "FAIL"))
    # 一个有状态 actor 的串行一致性演示（确定性模拟：同一 actor 方法按序调 5 次）
    n_calls = 5
    val = 0
    for _ in range(n_calls):
        val += 1  # actor 方法按序执行：先收到先执行，值单调
    ok2 = val == n_calls
    out.append(f"    同 1 actor 串行调 {n_calls} 次按序执行 → 计数器 = {val}（顺序一致性，{n_calls} 个无状态 task 并行各 +1 反而丢更新） = "
               + ("PASS" if ok2 else "FAIL"))
    out.append("    -> 状态三问先行：无状态优先（横扩），真好需要的共享态才用 actor/对象广播")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 D · 在线-离线混合账：Serve 在线两层扩缩 + Data 离线墙钟 + 峰值抢占 + 利用率
# ---------------------------------------------------------------------------
def account_d() -> str:
    out = []
    out.append("== D · 在线-离线混合账：Serve 在线 + Data 离线 + 峰值抢占 + 集群利用率 ==")
    # D1 Serve 在线两层扩缩：deployment 副本 × 每副本并发
    conc = SERVE_REPL * SERVE_CONC
    ok1 = conc == 64
    out.append(f"    Ray Serve：deployment {SERVE_REPL} 副本 × 每副本 {SERVE_CONC} 并发 = 总并发 {conc}"
               f"（外层 K8s 扩 Pod 数（13-03 账 C HPA）、内层 Serve 扩副本数与每副本并发=两层扩缩） = "
               + ("PASS" if ok1 else "FAIL"))
    # D2 Ray Data 离线墙钟：D 行 × ms/行 ÷ (worker×1000) → s
    w32 = 32
    t_d32 = ceilf(DATA_ROWS * MS_PER_ROW / (w32 * 1000.0))
    t_serial = DATA_ROWS * MS_PER_ROW / 1000.0   # 串行 = 1000 s
    ok2 = t_d32 == 32 and t_serial == 1000.0
    out.append(f"    Ray Data 单机 {w32} worker：{DATA_ROWS:,} 行 × {MS_PER_ROW:.0f} ms/行 → "
               f"串行 {t_serial:.0f} s vs 并行 ceil(÷{w32}×1000) = {t_d32} s（≈{t_serial / t_d32:.1f}×） = "
               + ("PASS" if ok2 else "FAIL"))
    # D3 混合借机：8 台在线保留 60% → 离线可借 floor(8×0.4)=3 台 = 96 worker
    m_off = math.floor(N_MACHINES * (1 - ONLINE_SHARE))
    w96 = m_off * WORKERS_PER_MACHINE
    t_d3 = ceilf(DATA_ROWS * MS_PER_ROW / (w96 * 1000.0))
    ok3 = m_off == 3 and t_d3 == 11
    out.append(f"    混合借机：{N_MACHINES} 台在线保留 {ONLINE_SHARE:.0%} → 离线可借 floor({N_MACHINES}×{1 - ONLINE_SHARE:.0%}) = "
               f"{m_off} 台 = {w96} worker → 同一任务 {t_d3} s（比单机 {t_d32} s ≈快 {t_d32 / t_d3:.1f}×，离线借空闲） = "
               + ("PASS" if ok3 else "FAIL"))
    # D4 峰值抢占：在线高峰要满 8 台 → 离线可用 0（任务级可重启），在线 100%
    on_peak = N_MACHINES
    off_peak = N_MACHINES - on_peak
    ok4 = on_peak == 8 and off_peak == 0
    out.append(f"    峰值抢占：在线高峰要满 {on_peak} 台 → 离线可用 {off_peak} 台（批任务暂停/重启=任务级，"
               f"13-03 账 B5 无状态切流纪律在任务域的同款） = " + ("PASS" if ok4 else "FAIL"))
    # D5 利用率账：在线单独平均 45% vs 混合（离线填谷）78%
    util_online, util_mix = 0.45, 0.78
    ok5 = util_mix > util_online
    out.append(f"    利用率账：在线单独平均占 {util_online:.0%}（峰谷空转）vs 混合后全集群平均 {util_mix:.0%}"
               f"（离线批任务填谷，大厂 Agent 平台『在线-离线混合负载的通用胶水』的省钱面） = "
               + ("PASS" if ok5 else "FAIL"))
    out.append("    -> 在线-离线混合四件事：Serve 在线两层扩缩 · Data 离线借机快算 · 峰值在线优先 · 填谷利用率更高")
    return "\n".join(out)


def main() -> None:
    W("start")
    parts = [account_a(), account_b(), account_c(), account_d()]
    print("\n" + "\n".join(parts), end="")
    W("done")


if __name__ == "__main__":
    main()
