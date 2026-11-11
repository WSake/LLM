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
