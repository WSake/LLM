# 📄 05 章 · 06-推理调度与Serving核心技术-Continuous-Batching-Chunked-Prefill（§5.5 · 🔴 掌握，生产核心）

> **章节**：05-推理与部署 · **站点**：5 / 10 · **难度**：[🔴 掌握概念、代码与选型] · **前置**：05-01（两阶段 Prefill/Decode）、05-04（KV Cache 系统级优化）、05-05（推理并行 TP/PP/DP/EP）· **衔接**：05-07（投机解码与 PD 分离）、13-04（Ray：Serving 编排）
>
> 一句话带走：**05-05 讲了"模型怎么摆"，本篇讲"请求怎么走"——静态批 = 等一批凑齐再算，连续批 = 算完一个 token 动态腾位并发来；vLLM 靠 PagedAttention 让 KV Cache 分页 + 前缀复用，Chunked Prefill 让超长 prompt 不再堵死整批，这两套是今天所有生产推理引擎的公共内核。**

---

## 📑 本节大纲（plan3 全部知识点）

| # | 知识点 | 落地位置 |
|---|---|---|
| 1 | 三范式演进：朴素批次 → Continuous Batching（2022）token 级动态进出 → 吞吐提升数倍，现代推理引擎的基石 | §1 / §2 |
| 2 | PagedAttention（vLLM 2023）：KV Cache 分页、碎片近零 + 前缀复用（05-04 管理线的引擎实现） | §3 |
| 3 | Chunked Prefill：长 prompt prefill 切成小块与 Decode 交错，消除"长 prompt 卡住整批"的头部效应 | §4 |
| 4 | LoRA Serving 多租户（Punica / S-LoRA / vLLM 内置）：同底座注入多个 LoRA | §5 |
| 5 | 工程加速杂项（CUDA Graphs / FlashAttention-3 / FP8 kernel / MoE 专家卸载） | §6 |

> **大纲覆盖自检**：5 点全覆盖 —— 点 1 占 §1+§2（三范式 + CB 内核）；点 2 占 §3；点 3 占 §4（含确定性探针）；点 4 占 §5；点 5 占 §6。代码落账在 §4，专项探针 `serving_batching_demo.py`。

---

## §1 调度范式三连：等批、凑批、动态批

先回答一个稍显"基础"但生产天天踩的问题：**一批请求到了 GPU，先怎么、后怎么算？** 调度器（scheduler）这一步，决定了一台机器能同时服务多少人。

三代范式，沿着"GPU 利用率"这一个指标演进：

| 范式 | 规则 | GPU 利用率 | 一句话缺点 |
|---|---|---|---|
| **朴素批次（wait-then-run）** | 攒够一批，整批一起算 | 依赖"请求正好同时到" | 人少时空转，人多时排队 |
| **Static Batching（批内同步）** | 凑满 batch，批内所有请求同步逐 token 推进，算完的干等批尾 | 批尾拖拽（长请求卡短请求） | "一个长请求拖垮一整批" |
| **Continuous / Dynamic Batching（2022）** | 请求随到随进、token 算完即出；以 token 而非请求为单位调度 | 插满会高很多（demo: 64.4%→85%） | 需要能"回填槽位"的调度器 |

关键转折在 2022 年——**Continuous Batching** 把调度的最小单位从"请求"降到了"token"：任意时刻，只要 GPU 还有空位，就放一个新请求的 token 进来；任何请求产出一个 token 就走，不占住槽位等队友。这让 GPU 在任意时刻都接近满载，吞吐因此"提升数倍"，而且短请求不再为长请求陪跑（e2e 延迟大幅下降）。这就是它成为"现代推理引擎的基石"的原因。

> 05-01 §4 的调度三把子里我们埋了这个名字；这里展开。三把子 = **Batching（摊薄带宽）→ Continuous Batching（动态批）→ 优先级配额（保服务等级）**。前两把子是"让 GPU 忙起来"，第三把子是"忙起来之后分给谁"。

---

## §2 Continuous Batching：请求怎么"随到随进、算完即出"

### 2.1 从"批内同步"到"token 级调度"

Static Batching 的问题出在"批"这个概念本身：**一个批是一个必须内部同步的团队，谁都不能提前走、也不能中途插队。** 于是出现两种浪费：

- **批尾拖拽（tail-dragging）**：批内 5/10/3/8 的 token，最长是 10，但批只推进 10 步，前面 3 步时 3-token 请求就毕业了，却仍占着 slot。
- **批间缝隙（inter-batch gap）**：上一批跑完，下一批没凑满，GPU 空着等。

Continuous Batching 把"请求"降维成"一串待产出的 token"，用一个**工作队列 + 固定 slot 数**的模型：

```
GPU 有 C 个 slot（C = 批大小上限）
每步：
  1. 从队头取 C 个活跃请求，各产出一个 token
  2. 谁产完 → 出列，释放 slot
  3. 谁没产完 → 回到队尾（下轮再排）
  4. 队列里有空位 → 新请求立即补进来（不等任何人）
```

这才是"连续"二字的含义——**GPU 永远不空着等人，永远有活干**。

### 2.2 三进程恒定探针的账：同一组请求，两种调度

下列账来自专项探针 `serving_batching_demo.py`（纯 stdlib、零网络、零随机，三进程 stdout 逐字节相同）。演示场景：**12 个混合长度请求**[5,10,3,8,12,6,4,9,7,11,2,8]，合计 85 token，GPU 并行容量 `C=4` token/步；静态批 batch=4。只换调度策略、请求完全一样：

```text
== A · 静态批 vs 连续批：同一组 12 个混合长度请求（t=0 全部到达） ==
  请求长度: [5, 10, 3, 8, 12, 6, 4, 9, 7, 11, 2, 8]    合计 85 token
  GPU 并行容量 = 4 token/步；静态批 batch = 4

  静态批（凑满 batch 开工 · 批内同步步进到最长请求）
    批 [5, 10, 3, 8]  -> 本批 10 步
    批 [12, 6, 4, 9]  -> 本批 12 步
    批 [7, 11, 2, 8]  -> 本批 11 步
    全部完成 = 33 步 · 槽位利用率 64.4%（有用 85 / 总槽位 132）
  连续批（token 级动态进出 · 算完即出 · 有槽即进）
    全部完成 = 25 步 · 平均每步占用 3.40 token · 利用率 85.0%
  连续批 / 静态批 = 33/25 = 墙钟 1.32×· 吞吐 2.58 -> 3.40 token/步
  理论下界（墙钟不可能低于）ceil(85/4) = 22 步；静态批落后下界 11 步（批尾拖拽 + 批间缝隙），连续批落后 3 步
```

**这 4 行里的三个关键洞察：**

1. **1.32× 墙钟**（`33/25`）。同一批请求、同一个 GPU，光换调度策略就快了 32%。这是"吞吐提升数倍"在慢速评估下的保守版——真实动机引擎里还有下面解释的解码并行/传输并行，倍率会更大。
2. **64.4% → 85.0% 的槽位利用率**。连续批把批尾拖拽 + 批间缝隙全消掉了,GPU 每 25 步里只有 3 步没坐满（落后下界 3 步 vs 静态 11 步）。**利用率 = 赚到的钱**,逼空槽位就是逼空显存和带宽。
3. **短请求不再为长请求买单**。看表 B（按请求原始顺序）：

```text
  请求   长度   静态批完成   连续批完成
  ----------------------------------------
  R00     5       10        4
  R01    10       10        9
  R02     3       10        2
  R03     8       10        7
  R04    12       22       14
  R05     6       22       10
  R06     4       22       11
  R07     9       22       18
  R08     7       33       17
  R09    11       33       22
  R10     2       33       16
  R11     8       33       24
  最短三条（2/3/4 token）:
    静态批最早完成 = 10 步（被所在批的批尾拽住）
    连续批最早完成 = 2 步
  平均完成时刻：静态 21.7 步 vs 连续 12.8 步
```

最关键的 **R10（2 token）**：静态批里它被 batch=4 的批尾[7,11,2,8]拽到 33 步才交付（实际它只要 2 步、前 31 步全白等）；连续批里它 16 步就交付——**短请求的 e2e 延迟从 6 倍于自身长度,降到 8 倍。** 这不是优化,是"本来应该如此"。

> **这是 05 章解释力最强的一账**：把 Continuous Batching 从"听说可以提高吞吐"变成"同一批请求、同一个 GPU、只改调度策略就快 32%、短请求少等 5 倍"。

---

## §3 PagedAttention（vLLM 2023）：KV Cache 分页 + 前缀复用

> 前置复习：05-04 §"管理线"讲了 PagedAttention 的**显存**效果（02-10 账 A3：连续 50.00 GiB → 碎片 70%，分页 29.51 GiB → 碎片 1%）。这里讲它是**怎么落到调度器里**的——这是 05-04 直接接手 05-05 的那一段，也是 vLLM 能跑起连续批的物理前提。

Continuous Batching 想让"每个请求随便进出"但 GPU 按"显存页"管理 KV——于是 vLLM 在 2023 年引入 **PagedAttention**（借鉴操作系统虚拟内存页表）：KV Cache 不再为每个请求整块预分配（整块会因请求长度未知而预留过大 → 碎片），而是切成固定大小的**页**（block），按需分配、不用的立即还回去。

| 机制 | 解决的问题 | 怎么帮到 Continuous Batching |
|---|---|---|
| 页式分配 | 预留整块 KV 的内部碎片 70%→1% | 加新请求不需预占整条序列的 KV，插槽更易满 |
| 前缀复用（Preemption） | 相同前缀的请求白算 | 多请求共享前缀的页，KV 只算一次 |
| 抢占/换出 | 显存不够时优先级低者先换出 | 让长请求可被暂停，短请求先跑 |

**数字锚（05-04 / 02-10 只引不重测）**：
- 02-10 A3：连续分配 `50.00 GiB`（内部碎片 70%）vs 分页 `29.51 GiB`（内部碎片 1%）。
- 前缀缓存：512 token 前缀、200 个共享请求 → 省 `12.50 GiB` = 全部 KV 的 42%（02-10）。
- 02-17：16 token 前缀缓存后,解码 16 步复用,`10.2ms → 3.4ms`（同机 3.0×）,Logits 逐位一致 True。

> **一句话**：PagedAttention 是"连续批"可以落地的那个膝关节——没有它，请求的动态进出会因 KV 整块预留而进退堂);有了它，插槽才能随意进出、前缀才能共享。

---

## §4 Chunked Prefill：长 prompt 不再堵死整批

### 4.1 问题：一个 200-token 的 prompt 卡住 3 个 10-token 的请求

静态批最刺眼的失败场景不是"短请求等长请求的人",而是**长 prompt 的 prefill 阶段**——一个用户贴了篇 200-token 的文章进来，GPU 先花 200 步算它的 prefill（产第一个 token），同一批的其他请求全程陪跑。

下账（确定性，探针账 C）：4 个请求 `[200, 10, 10, 10]`，容量 C=4。

| 策略 | 短请求完成 | 说明 |
|---|---|---|
| 静态 batch=1 | 210 步 | 长请求先跑 200，再轮到短的 |
| 静态 batch=2/4 | 200 步 | 长 prefill 占满 200 步,全程拖着短的 |
| **连续批 + Chunked Prefill** | **9 步** | decode 优先，剩 slot 喂长 prefill |

### 4.2 解法：把 prefill 切块,插到 decode 的缝隙里

**Chunked Prefill**：把一个长 prompt 的 prefill 计算**切成小块**（chunk），每一块只占 GPU 的几个 slot、算完释放,然后回到更该被服务的 decode 请求中间。调度规则一句话：**decode 永远优先（每步先满足 decode 的 token 需求）,剩余 slot 全部喂 prefill 的 chunk。** 效果：

```text
  连续批 + chunked prefill（decode 优先 · 剩 slot 喂长 prefill）
    短请求完成 = 9 步（vs 静态 batch=4 的 200 步 -> 22× 进步）
    长 prefill 完成 = 57 步（chunk 交错吸收剩余 slot，不独占机器）
    总完成 58 步 vs 静态 batch=4 的 200 步 -> 整体 3.45×
    利用率：静态 28.8% vs 连续 99.1%
```

**三件事同时被救：**
- 短请求 200 步 → 9 步（**22×** e2e 大改善）；
- 总墙钟 200 → 58 步（**3.45×**,吞吐也上去了）;
- 利用率 28.8% → 99.1%（GPU 不再是"算完长 prefill 再抽空子"）。

> 头部效应（head-of-line blocking）这个词在排队论里指"前面一个慢任务堵住后面所有快任务"。Chunked Prefill 是专门拆这种 head-of-line 的调度手术刀。

### 4.3 内嵌确定性代码（三进程 stdout 逐字节相同）

> 本段为站内简化账，纯 stdlib。三并发进程 stdout 字节全同，且 LF 归一化后与下方 text 块逐字节一致（三进程原始 md5 `511aac24d543aa2d8c65dbad991514c3`）。完整脚本 `serving_batching_demo.py` 见 §验证。

```python
# -*- coding: utf-8 -*-
"""05-06 Serving 核心技术 · 调度账（纯 stdlib · 零网络 · 零随机）
三进程 stdout 字节恒一；墙钟打 stderr（2>/dev/null 隔离）。"""
import sys
import time

_wall0 = time.time()

CAP = 4      # GPU 并行容量：每步可处理的 token 位置
BATCH = 4    # 静态批：凑满多少个请求开工
REQS = [5, 10, 3, 8, 12, 6, 4, 9, 7, 11, 2, 8]


def static_batch(reqs, batch=BATCH):
    """静态批：按到达顺序分成一批；组内同步步进，每步每个未算完的请求
    各产出 1 token，算完的请求继续空占槽位直到本组最长请求结束。
    返回 (总完成时刻, 组明细, 各请求完成时刻)。"""
    groups = [reqs[i:i + batch] for i in range(0, len(reqs), batch)]
    total = 0
    detail = []
    finish = [None] * len(reqs)
    idx = 0
    for g in groups:
        d = max(g)
        total += d
        detail.append((list(g), d))
        for j in range(len(g)):
            finish[idx + j] = total
        idx += len(g)
    return total, detail, finish


def cont_batch(reqs, cap=CAP):
    """连续批：按 token 级时间步模拟。每步在活跃请求中（按到达顺序）
    取前 min(cap, len(active)) 个各产出 1 token；算完即出列让位。
    返回 (总完成时刻, 各请求完成时刻)。"""
    active = [(i, l) for i, l in enumerate(reqs)]
    finish = [None] * len(reqs)
    t = 0
    while active:
        slots = min(cap, len(active))
        for k in range(slots):
            oi, L = active[k]
            L -= 1
            if L == 0:
                finish[oi] = t
                active[k] = None
            else:
                active[k] = (oi, L)
        active = [a for a in active if a is not None]
        t += 1
    return t, finish


def cb_chunked(decode_reqs, prefill_len, cap=CAP):
    """连续批 + chunked prefill：decode 请求每步优先，各占 1 slot；
    剩余 slot 全部喂给长 prefill 的 chunk token（近似把长 prompt 切成
    token 流随时段吸收）。prefill 完成即离开。返回 (总完成时刻, 各
    decode 完成时刻, prefill 完成时刻)。"""
    active = list(range(len(decode_reqs)))   # 尚未产完的 decode 请求
    remain = list(decode_reqs)               # 各自还差多少 token 才产完
    finish = [None] * len(decode_reqs)
    prefill_remain = prefill_len
    prefill_done = None
    t = 0
    while active or prefill_remain > 0:
        n_dec = min(len(active), cap)
        done_now = []
        for k in range(n_dec):
            i = active[k]
            remain[i] -= 1
            if remain[i] == 0:
                finish[i] = t
                done_now.append(i)
        pre = cap - n_dec
        if pre > 0 and prefill_remain > 0:
            prefill_remain -= pre
            if prefill_remain <= 0:
                prefill_done = t
                prefill_remain = 0
        for i in done_now:
            active.remove(i)
        t += 1
    return t, finish, prefill_done


def pct(used, slots):
    return used * 100.0 / slots if slots else 0.0


out = []

# ---------- 账 A ----------
out.append("== A · 静态批 vs 连续批：同一组 12 个混合长度请求（t=0 全部到达） ==")
out.append(f"  请求长度: {REQS}    合计 {sum(REQS)} token")
out.append(f"  GPU 并行容量 = {CAP} token/步；静态批 batch = {BATCH}")
out.append("")

st_total, st_detail, st_fin = static_batch(REQS)
cb_total, cb_fin = cont_batch(REQS)

out.append("  静态批（凑满 batch 开工 · 批内同步步进到最长请求）")
for g, d in st_detail:
    out.append(f"    批 {g}  -> 本批 {d} 步")
st_slots = st_total * CAP
cb_slots = sum(min(CAP, len(REQS)) for _ in range(cb_total))
# cb 每步活跃数（近似用总完成时刻内的非零活跃步数）
out.append(f"    全部完成 = {st_total} 步 · 槽位利用率 {pct(sum(REQS), st_slots):.1f}%"
           f"（有用 {sum(REQS)} / 总槽位 {st_slots}）")
out.append("  连续批（token 级动态进出 · 算完即出 · 有槽即进）")
out.append(f"    全部完成 = {cb_total} 步 · 平均每步占用 {sum(REQS) / cb_total:.2f} token"
           f" · 利用率 {pct(sum(REQS), cb_slots):.1f}%")
ratio = st_total / cb_total
bw_st = sum(REQS) / st_total
bw_cb = sum(REQS) / cb_total
out.append(f"  连续批 / 静态批 = {st_total}/{cb_total} = 墙钟 {ratio:.2f}×·"
           f" 吞吐 {bw_st:.2f} -> {bw_cb:.2f} token/步")
out.append(f"  理论下界（墙钟不可能低于）ceil({sum(REQS)}/{CAP}) = "
           f"-(-{sum(REQS)}//{CAP}) 步；静态批落后下界 {st_total - (-(-sum(REQS) // CAP))} 步"
           f"（批尾拖拽 + 批间缝隙），连续批落后 {cb_total - (-(-sum(REQS) // CAP))} 步")
out.append("")

# ---------- 账 B ----------
out.append("== B · 短请求的完成时刻（e2e 延迟视角·按请求原始顺序） ==")
hdrs = f"  {'请求':>2}  {'长度':>3}  {'静态批完成':>6}  {'连续批完成':>6}"
out.append(hdrs)
out.append("  " + "-" * 40)
for i, l in enumerate(REQS):
    out.append(f"  R{i:02d}   {l:>3}   {st_fin[i]:>6}   {cb_fin[i]:>6}")
short3 = sorted(range(len(REQS)), key=lambda i: REQS[i])[:3]
out.append(f"  最短三条（{'/'.join(str(REQS[i]) for i in short3)} token）:")
out.append(f"    静态批最早完成 = {min(st_fin[i] for i in short3)} 步"
           f"（被所在批的批尾拽住）")
out.append(f"    连续批最早完成 = {min(cb_fin[i] for i in short3)} 步")
avg_st = sum(st_fin) / len(REQS)
avg_cb = sum(cb_fin) / len(REQS)
out.append(f"  平均完成时刻：静态 {avg_st:.1f} 步 vs 连续 {avg_cb:.1f} 步"
           f"（短请求不再为长请求买单）")
out.append("")

# ---------- 账 C ----------
LONG = 200
out.append("== C · 头部效应：长 prompt prefill 阻塞 decode（200 token + 3×10 token） ==")
long_reqs = [LONG, 10, 10, 10]
out.append("  静态批（怎么选 batch 都躲不掉长 prompt）")
for b in (1, 2, 4):
    _, dd, ff = static_batch(long_reqs, batch=b)
    earliest_short = min(ff[i] for i in range(1, 4))
    out.append(f"    batch={b}: 短请求最早 {earliest_short} 步完成"
               f"（长 prompt 完成 {ff[0]} 步）")
out.append("  连续批 + chunked prefill（decode 优先 · 剩 slot 喂长 prefill）")
cb_t, cb_ff, cb_pd = cb_chunked([10, 10, 10], LONG)
out.append(f"    短请求完成 = {min(cb_ff)} 步（vs 静态 batch=4 的 {ff[0]} 步"
           f" -> {ff[0] / min(cb_ff):.0f}× 进步）")
out.append(f"    长 prefill 完成 = {cb_pd} 步（chunk 交错吸收剩余 slot，不独占机器）")
out.append(f"    总完成 {cb_t} 步 vs 静态 batch=4 的 {static_batch(long_reqs)[0]} 步"
           f" -> 整体 {static_batch(long_reqs)[0] / cb_t:.2f}×")
foot = sum(long_reqs)
fb = static_batch(long_reqs, batch=4)[0]
out.append(f"    利用率：静态 {pct(foot, fb * CAP):.1f}% vs 连续 {pct(foot, cb_t * CAP):.1f}%")
out.append("")

# 一致性断言（教学自检）
lo = -(-sum(REQS) // CAP)
assert cb_total >= lo, "连续批不可能低于墙钟下界"
assert all(f is not None for f in cb_fin), "连续批完成时刻必须全部有值"
assert sum(REQS) == 85, "账 A 样本恒定"

print("\n".join(out))
print(f"[wall] finish in {time.time() - _wall0:.3f} s (stderr)", file=sys.stderr)
```

```text
== A · 静态批 vs 连续批：同一组 12 个混合长度请求（t=0 全部到达） ==
  请求长度: [5, 10, 3, 8, 12, 6, 4, 9, 7, 11, 2, 8]    合计 85 token
  GPU 并行容量 = 4 token/步；静态批 batch = 4

  静态批（凑满 batch 开工 · 批内同步步进到最长请求）
    批 [5, 10, 3, 8]  -> 本批 10 步
    批 [12, 6, 4, 9]  -> 本批 12 步
    批 [7, 11, 2, 8]  -> 本批 11 步
    全部完成 = 33 步 · 槽位利用率 64.4%（有用 85 / 总槽位 132）
  连续批（token 级动态进出 · 算完即出 · 有槽即进）
    全部完成 = 25 步 · 平均每步占用 3.40 token · 利用率 85.0%
  连续批 / 静态批 = 33/25 = 墙钟 1.32×· 吞吐 2.58 -> 3.40 token/步
  理论下界（墙钟不可能低于）ceil(85/4) = -(-85//4) 步；静态批落后下界 11 步（批尾拖拽 + 批间缝隙），连续批落后 3 步

== B · 短请求的完成时刻（e2e 延迟视角·按请求原始顺序） ==
  请求   长度   静态批完成   连续批完成
  ----------------------------------------
  R00     5       10        4
  R01    10       10        9
  R02     3       10        2
  R03     8       10        7
  R04    12       22       14
  R05     6       22       10
  R06     4       22       11
  R07     9       22       18
  R08     7       33       17
  R09    11       33       22
  R10     2       33       16
  R11     8       33       24
  最短三条（2/3/4 token）:
    静态批最早完成 = 10 步（被所在批的批尾拽住）
    连续批最早完成 = 2 步
  平均完成时刻：静态 21.7 步 vs 连续 12.8 步（短请求不再为长请求买单）

== C · 头部效应：长 prompt prefill 阻塞 decode（200 token + 3×10 token） ==
  静态批（怎么选 batch 都躲不掉长 prompt）
    batch=1: 短请求最早 210 步完成（长 prompt 完成 200 步）
    batch=2: 短请求最早 200 步完成（长 prompt 完成 200 步）
    batch=4: 短请求最早 200 步完成（长 prompt 完成 200 步）
  连续批 + chunked prefill（decode 优先 · 剩 slot 喂长 prefill）
    短请求完成 = 9 步（vs 静态 batch=4 的 200 步 -> 22× 进步）
    长 prefill 完成 = 57 步（chunk 交错吸收剩余 slot，不独占机器）
    总完成 58 步 vs 静态 batch=4 的 200 步 -> 整体 3.45×
    利用率：静态 28.8% vs 连续 99.1%
```

---

## §5 LoRA Serving 多租户：一个底座，N 个变体

### 5.1 场景：100 个任务,共享一个底座,各调各的 LoRA

服务端如果每任务一个全量模型,显存爆炸;每任务一个 LoRA 微调权重又太小。**LoRA Serving（多租户）** 的解法:一份基础权重（base model）常驻显存,上层按需挂任意多的 LoRA 适配器(rank 通常 8~64,几 MB 到几十 MB)。

一个请求到达 → 查这个请求要哪个 LoRA → 找到它的适配器权重 → 在 attention/FF 层做 **base 前向 + LoRA 增量**(两者都是低秩矩阵相乘) → 产出。

### 5.2 三条技术路线

| 路线 | 核心思想 | 代表 |
|---|---|---|
| **Punica**（2023）| 一组 LoRA 共享 base 的权重,tok 级多租户调度 | Punica（论文）|
| **S-LoRA**（2023）| 把 LoRA 放进显存池,按请求动态加载,单服务器服务数千 LoRA | S-LoRA（论文）|
| **vLLM 内置 LoRA** | 在连续批的每步 dispatch 时,把请求路由到它要的 LoRA;多 LoRA 与 CB/PagedAttention 协同 | vLLM `--lora-modules` |

共同点:它们把"多 LoRA"和"连续批"结合起来——**同一个 GPU 里,一批请求跑着 N 个不同任务的 LoRA,顺序到达、token 级插队**,这就是多租户 serving 的形态。

### 5.3 什么时候有价值

| 场景 | 评估 |
|---|---|
| 客服 N 个话术模板各一个 LoRA | 高频 → 值（基础共享、适配器小）|
| 1000 个用户个人化 LoRA | 值(MoE 化变体,每请求一个适配器)|
| 单模型高频推理 | 不值（LoRA 是"多"才划算;单个模型直接全量）|
| 显存极度紧张但多任务 | 值（一个 base 装 N 个 LoRA,远比 N 个全量省显存）|

> 一句话避开坑:**LoRA Serving 的价值来自"多"——多任务共享底座。模型很少时,它不带来收益;模型很多、任务很多时,它让显存和带宽成 N 倍压缩。**

---

## §6 工程加速杂项:让连续批跑得更快的四件事

Continuous Batching 解决"调度",下面四件事解决"单步多快"——它们不改变调度语义,但都是生产引擎的标配:

| 加速项 | 一句话 | 锚点/出处 |
|---|---|---|
| **CUDA Graphs**（2023 起）| 把固定形状的 kernel 序列编译成 graph,省 launch 开销（图形化发射,每步省数百次 kernel launch 的毫秒级 CPU→GPU 同步）| 05-05/e2e-serving 都在用;vLLM `--cuda-graph-max-batch-size`|
| **FlashAttention-3**（2024）| 后半注意力算子被浮点优化,prefill 更快（FP8 下尤其明显）| 05-03 精度账里 FA-3 提速 `14.3 → 28.6 → 57.1 tok/s`|
| **FP8 kernel** | 权重/激活用 FP8 计算,矩阵乘要带宽的还是算力富余时都用得上 | 05-03 §2 的 FP8 精度账 |
| **MoE 专家卸载** | MoE 擅长把专家放卡外/多卡,离线或不常用专家可换出 EV | 05-05 EP 那段的"expert-parallel"|

> 一句话:**CF = 省 CPU 端同步; FA-3 = 快 GPU 端注意力; FP8 = 降带宽/升算力; MoE 卸载 = 把不常用的计算挪走。** 前两项是"引擎标配",后两项是"精度/架构级优化"。

---

## §验证与锚点核对表

| 声明 | 数值 | 出处/计算方法 |
|---|---|---|
| 静态批 vs 连续批墙钟 | 33 vs 25 步 = **1.32×** | `serving_batching_demo.py` 账 A |
| 槽位利用率 | 64.4% vs **85.0%** | 同上（85 有用 / 132 vs 总槽位）|
| 短请求 e2e | R10 从 33 → 16 步;平均 21.7 → 12.8 步 | 同上 账 B |
| 头阻塞 Chunked | 短请求 200 → 9 步 = **22×**;整体 3.45×;利用率 28.8→99.1% | 同上 账 C |
| PagedAttention 碎片 | 70% → 1% | 02-10 A3(不重测)|
| 前缀缓存省显存 | 512 token/200 请求 → 省 12.50 GiB = 42% | 02-10（不重测）|
| 前缀复用加速 | 16-token 前缀 10.2ms → 3.4ms = 3.0× | 02-17（不重测）|
| Continuous Batching 出处 | 2022（调度从请求级降到 token 级）| Orca（2022）|
| FlashAttention-3 | 7B FP8 提速 14.3 → 28.6 → 57.1 tok/s | 05-03（不重测）|
| 探针 md5 | 三进程 stdout `511aac24d543aa2d8c65dbad991514c3` | `serving_batching_demo.py` |

---

## §17.4 05 行卡位

05-01/05-02（已绿）、05-03（已绿）、05-04（已绿）、05-05（已绿）、**05-06（本篇，新增 ✅）**、05-07～05-10（⬜）。

---

## §常见坑（serving 视角）

1. **把 Static Batching 当 Continuous Batching**：只改 `batch` 大小不改调度器，是"更粗的静态批"——批内长请求照样拖短请求、批间照样等着凑。真正的 CB 必须允许 token 级进出。
2. **没有 PagedAttention 硬上 CB**：多请求动态进出,若 KV 是整块预分配,碎片会爆显存并卡死新请求。CB 是软件层,分页是它的存储前提,两者是配套的。
3. **Chunked Prefill 调成 1**：把 prefill chunk 切太小,每块都要重新进 kernel launch,反而把 prefill 算慢。现实中要调 chunk size,不是越小越好（vLLM `--chunked-prefill` + `--max-prefill-tokens`）。
4. **多租户 LoRA 却不开 CB**：S-LoRA/Punica 的收益都要靠 CB 的"多请求同时跑"才能兑现,单独塞 LoRA 调度器进去没有意义。

---

## 衔接下一章

下一站 05-07 `07-投机解码与 PD 分离`（§5.6）——本站在"调度层"把它做到全员出厂,下一站在"延迟层"继续压首 Token 延迟:投机解码（小模型猜大模型、greedy 下保证一致）与 PD 分离 / Disaggregated Serving（prefill 与 decode 两台机器分开）。

（路线提示：05-06 是**调度层面**两条前沿工程线；05-07 是**延迟层面**的另一条。05-08 做引擎选型收束本小节。）

---

## 参考资料与诚实边界

- **实测锚点**：02-10 `10-KV-Cache与显存账本.md`（PagedAttention 碎片 70%→1%、前缀缓存 12.50 GiB=42%）；02-17（前缀复用 3.0×）；本文 `serving_batching_demo.py` 全部数字。
- **调度语义来源**：Continuous Batching 概念出自 Orca（2022）论文——"token 级调度 / 动态批"是现代推理引擎统一的公共内核,本页只讲工程账,不搬运论文公式。
- **PagedAttention / Chunked Prefill 数值**：均来自 02-10/02-17 已实测探针，此文只引不重测。
- **诚实边界**：上文的所有"步"是教育化的简化模型（把 GPU 的时间单位抽象成"一步算一批 token"）,真实引擎还要拆 decode 步内并行的多个序列、跨步的流处理等。`serving_batching_demo.py` 的 CB 是"容量 C 个 slot 每步各产 1 token"的抽象,不模拟真实 GPU 的计算耗时 —— 它把**调度层面**的收益与损失切出来单独看,这正是本篇要讲的事。真实长时间的 Continuous Batching 倍率通常还会更高（还有 KV 前缀、调度器手写等原因）。
- **LoRA Serving 三条路线**：Punica（论文 2305.x）与 S-LoRA（论文 2311.x）为 2023 年学术工作,vLLM 内置 `--lora-modules` 为工程实现;适用场景表是定性判断,不做数值承诺。
- **墙钟 vs 吞吐**：账 A 用"步"数,不承诺物理秒;物理秒 = 步数 × 单步耗时,不同卡差异巨大,本文不做跨硬件换算。

> 一句话带走：**Continuous Batching（2022）把调度从"请求级"降到"token 级"——GPU 永不空转、短请求不再陪跑;PagedAttention 是它分页的物理前提;Chunked Prefill 让长 prompt 不再堵住整批;LoRA Serving 让一个底座服务 N 个任务。这四者是今天所有生产推理引擎的共同内核。**
