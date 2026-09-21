# -*- coding: utf-8 -*-
"""上下文工程四账实测（知识地图 §6.3）：装什么 / 装多少 / 怎么排 / 要不要带。

零网络、零下载、不训练模型。只量确定性工程维度（token/成本/组合枚举/命中/纯度/路由判定）；
模型侧现象（注意力 primacy/recency、真实上下文增益）一律挂靠既有章节，不实测。
token 估算=中文 1 字计 1 + 英文/数字连续段计 1（确定性近似，非真实 tokenizer，见诚实边界）。

复现：python code/notebooks/_tools/context_engineer_demo.py
"""
import re
import sys
import datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def tok(text: str) -> int:
    """确定性 token 估算：中文 1 字=1 · 英文/数字连续段=1。"""
    return len(re.findall(r'[一-鿿]', text)) + len(re.findall(r'[A-Za-z0-9]+', text))


PIN = 2.50  # 输入 $/百万 token（要素价格事实，非本机实测；沿 01/02 台账同价目表）


def cost_usd(n: int) -> float:
    return n / 1_000_000 * PIN


W = sys.stderr.write
w0 = datetime.datetime.now()

print("=" * 72)
print("前置：Context Engineering 从「给什么」（§6.3/§6.10/§8）· 零网络零下载 · 四账实测")
print("  Prompt=怎么说（02），Context=给什么（本篇）：装什么/装多少/怎么排/要不要带")
print("  token 估算=中文字符+英文/数字词（近似子词系，非真实 tokenizer）· 价格=要素事实 PIN 2.50$/M")
print("=" * 72)

# ---------------- 实验 A：装什么 —— 上下文来源组合账 ----------------
print()
print("[实验 A] 装什么 —— 上下文来源组合账（最简集 vs 全全集）")
print("  综合任务 = 查评分[检索] + 记预算[记忆] + 加日程[工具] + 回复简洁[约束]；"
      "另 3 项可选（指令前缀 + 两段陪跑）")

SRC = {
    "RETRIEVAL": "检索：万泰餐厅·综合评分 4.7/5 · 环境 4.5/5 · 服务 4.8/5，位于中关村大街 27 号",
    "MEMORY": "记忆：用户偏好预算三千以内；常去店=万泰餐厅",
    "TOOL": "工具结果：本店可订餐位状态正常，可加入日程提醒",
    "SYS_CONSTRAINT": "约束：回复用一句话；先给结论再给依据；不要列清单",
    "SYS_PROMPT": "指令：你是一名餐饮助手，负责订位/查询/提醒",
    "ADS": "陪跑-广告：本周套餐 B 立减 20 元，满 100 可用，详情致电 400-888-0000",
    "UNREL": "陪跑-无关：系统升级公告，周五 02:00-06:00 例行维护，期间服务可能抖动",
}
keys = list(SRC)
REQ = {"RETRIEVAL", "MEMORY", "TOOL", "SYS_CONSTRAINT"}
toks = {k: tok(v) for k, v in SRC.items()}
req_tok = sum(toks[k] for k in REQ)

subsets = 0
ok_subsets = 0
min_tok = None
full_tok = sum(toks.values())
for mask in range(1 << len(keys)):
    sel = [keys[i] for i in range(len(keys)) if (mask >> i) & 1]
    subsets += 1
    if REQ.issubset(set(sel)):
        ok_subsets += 1
        st = sum(toks[k] for k in sel)
        if min_tok is None or st < min_tok:
            min_tok = st

print(f"  来源(7 项) token 明细：{ {k: toks[k] for k in keys} }")
print(f"  全子集枚举 2^{len(keys)} = {subsets} 个子集；可作答（含 4 必需来源）= {ok_subsets} 个")
print(f"  最简可作答集（只带 4 必需）：token={min_tok}  ${cost_usd(min_tok):.5f}  "
      f"纯度={req_tok}/{min_tok}={req_tok / min_tok:.3f}")
print(f"  全全集（7 项全带）：token={full_tok}  ${cost_usd(full_tok):.5f}  "
      f"纯度={req_tok}/{full_tok}={req_tok / full_tok:.3f}")
print(f"  全带比最简贵：+{full_tok - min_tok} tok（+{(full_tok - min_tok) / min_tok * 100:.1f}%）"
      f"· 纯度 {req_tok / min_tok:.3f}→{req_tok / full_tok:.3f}（陪跑把信息稀释）")
print(f"  陪跑代价行：ADS={toks['ADS']} tok · UNREL={toks['UNREL']} tok 均不增加可答性")
print("  台账：上下文每一段都有价格——「带全」是最常见的隐性浪费；最简集=只带可答必需")
print("  [CACHE] 前缀缓存（挂靠 01-20 缓存输入半价）：越靠前的段跨查询越可复用→最简集的前缀=缓存友好前缀")

# ---------------- 实验 B：装多少 —— 长度预算账 ----------------
print()
print("[实验 B] 装多少 —— 长度预算账（top-k 截断曲线）")
print("  大上下文 = 10 块按相关度降序；10 个答案词分布在前 5 块（后 5 块零命中=纯陪跑）")

BLOCKS = [
    "万泰餐厅位于中关村大街 27 号，人均 120 元，烤鸭是招牌。店内装修简约，灯光偏暖",
    "综合评分 4.7/5，环境 4.5/5，服务 4.8/5，适合商务宴请。另有露天座位与包间，整体安静",
    "营业时间 11:00-21:30，周末需提前 1 天可订，包间 6 个。非高峰时段无需排队，可现场点单",
    "用户预算三千以内，可选双人套餐 288 或四人套餐 588。以上价格不含酒水，会员另有折扣",
    "预约方式：电话 400-888-0000 或小程序，支持当日取消。到店出示订单号即可验证",
    "厨房主打京味，辣椒来自四川，池子里养着活海鲜",
    "员工福利：每周五下午茶，团建一年两次",
    "门店装修于 2024 年完成，新增包间 6 个",
    "周边地铁 4 号线可达，公交 26 路在中关村站停靠",
    "招聘：急招服务员与后厨帮工，薪资面议，包吃住",
]
AW = ["万泰餐厅", "中关村大街", "烤鸭", "评分 4.7", "商务宴请", "营业时间",
      "提前 1 天可订", "预算三千", "双人套餐", "400-888-0000"]
for i, b in enumerate(BLOCKS):
    hits = [w for w in AW if w in b]
    assert i in (0, 1, 2, 3, 4) or not hits, "陪跑块不应含答案词：" + str((i, hits))

btok = [tok(b) for b in BLOCKS]
cum = [sum(btok[: k]) for k in range(1, 11)]
hits_at = []
for k in range(1, 11):
    seen = set()
    for w in AW:
        for b in BLOCKS[:k]:
            if w in b:
                seen.add(w)
    hits_at.append(len(seen))

print("  k       token     命中(10)   成本($)     累积@k")
for k in range(1, 11):
    print("  %2d      %5d      %5d     %.5f     %.3f" % (
        k, cum[k - 1], hits_at[k - 1], cost_usd(cum[k - 1]), hits_at[k - 1] / 10))
full_b = cum[9]
top5_b = cum[4]
# 压缩=句子级：只保留含答案词的句子（沿 08-06 提取式压缩的思路，词面规则实现）
compressed = 0
for b in BLOCKS[:5]:
    kept = [s for s in re.split(r'[。；]', b) if any(w in s for w in AW)]
    compressed += tok("".join(kept))
print(f"  拐点：k=5 token={top5_b} 命中 10/10；再往后 token 涨、命中不涨（后 5 块零贡献）")
print(f"  策略账：I 全量(k=10) token={full_b}  ${cost_usd(full_b):.5f}")
print(f"          II top-5     token={top5_b} 命中 10/10 → 省 {(1 - top5_b / full_b) * 100:.1f}% 零损失")
print(f"          III 压缩(只留含答案词句子) token={compressed} → 省 {(1 - compressed / top5_b) * 100:.1f}%（相对 top-5）")
print("  挂靠：08-06 上下文压缩率 88.2%（字符口径）· 08-13 检索成本线 k=5/10/20 逐档")
print("  台账：更多块 ≠ 更多答案——先画覆盖拐点再定 k；预算到拐点为止，剩下的钱给下一轮查询")

# ---------------- 实验 C：怎么排 —— 位置效应账 ----------------
print()
print("[实验 C] 怎么排 —— 位置效应账（答案块在第几才露面）")
print("  6 块上下文；答案块依次放位置 1..6，量首屏（前 3 块）命中与噪声前置 token")

CB = [
    "系统导语：你是一名行程助手，只输出结论",
    "检索：万泰餐厅地址与电话，营业至 21:30",
    "检索：评分与服务建议",
    "记忆：用户上次备注「工作日下午方便」",
    "答案块（核心结论）：建议预订周六 18:00 四人桌，人均 120 元",
    "工具结果：日历 2026-09-26 晚间空闲",
]
cbtok = [tok(b) for b in CB]
print("  pos  噪声前置(token)  首屏(前3块)")
for p in range(1, 7):
    off = sum(cbtok[: p - 1])
    print("   %d       %4d          %s" % (p, off, "YES" if p <= 3 else "no"))

orig_off = sum(cbtok[: 4])      # 答案在 pos5（原文顺序）
orig_hit = 0                     # 前 3 块不含答案
rel_off = 0                      # 相关块置顶 → 答案在 pos1
rel_hit = 1
print(f"  原文顺序（答案 pos5）：噪声前置 {orig_off} tok · 首屏命中 {orig_hit}")
print(f"  相关块置顶（答案 pos1）：噪声前置 {rel_off} tok · 首屏命中 {rel_hit} → 首屏 0→1 · 噪声前置 {orig_off}→{rel_off} tok")
print("  挂靠：注意力 primacy/recency=模型侧现象（本文不实测，只挂靠）· 02-20 KV 预算递减")
print("  台账：先看到什么决定先答什么——把相关块置顶是零成本改造，能救回被埋在最底下的结论")

# ---------------- 实验 D：要不要带 —— 路由闸 ----------------
print()
print("[实验 D] 要不要带 —— 路由闸（8 条查询全带 vs 路由）")
print("  路由信号（词面，确定性）：含历史词（上次/之前/最近/常去）或系统单号($ID/ZX-数字) → 带上下文")

QUERIES = [
    ("q1", "根据我上次说的预算，帮我订常去那家餐厅的位子", True),
    ("q2", "查询订单 ZX-8891 的物流状态", True),
    ("q3", "帮我把张三加进明天的会议日程", True),
    ("q4", "用我上次记的偏好推荐一道菜", True),
    ("q5", "写一首关于春天的诗", False),
    ("q6", "今天天气怎么样", False),
    ("q7", "圆周率前 20 位是什么", False),
    ("q8", "把英文 greeting 翻译成日文", False),
]
HIS = ("上次", "之前", "最近", "常去")
CTX = "上下文：用户档案摘要 + 近 5 条会话要点 + 今日日程 + 检索命中 3 条"
CTOK = tok(CTX)


def route(q: str) -> bool:
    if any(h in q for h in HIS):
        return True
    if re.search(r'[A-Z]{1,3}-\d+', q):
        return True
    return False


all_tok = 0
route_tok = 0
fpr = 0
fnr = 0
hit = 0
rows = []
for qid, q, gnd in QUERIES:
    r = route(q)
    all_tok += tok(q) + CTOK
    route_tok += tok(q) + (CTOK if r else 0)
    if r == gnd:
        hit += 1
        verdict = "OK"
    elif gnd:
        fnr += 1
        verdict = "FNR"
    else:
        fpr += 1
        verdict = "FPR"
    rows.append((qid, "T" if gnd else "F", r, verdict))

print("  q     GND  路由  判定")
for qid, g, r, v in rows:
    print("  %s    %s    %s    %s" % (qid, g, "带" if r else "不带", v))
print(f"  命中 {hit}/{len(QUERIES)} · FNR={fnr}（该带没带，人名类上下文引用漏网）· FPR={fpr}")
print(f"  token 账：全带 8 条={all_tok} tok  ${cost_usd(all_tok):.5f}；路由后={route_tok} tok  ${cost_usd(route_tok):.5f}"
      f" → 省 {(1 - route_tok / all_tok) * 100:.1f}%")
print("  挂靠：08-10 Agentic RAG 决策（judge 把关省 25.4%）——「要不要去拿」是同一类闸；本篇规则是词面版")
print("  台账：路由是 Context 工程的第一个省钱闸；规则词面永远有漏网，实装要选召回率高的信号并人审关键路径")

# ---------------- 台账汇总 ----------------
print()
print("=" * 72)
print("台账汇总（A 装什么 / B 装多少 / C 怎么排 / D 要不要带）")
print("  A  最简 {min_tok} tok → 全集 {full_tok} tok（+{extra:.1f}%）· 纯度 1.000→{pure:.3f}".format(
    min_tok=min_tok, full_tok=full_tok, extra=(full_tok - min_tok) / min_tok * 100,
    pure=req_tok / full_tok))
print("  B  top-5={top5_b} tok 命中 10/10 → 省 {s1:.1f}% 零损失 · 压缩再省 {s2:.1f}%".format(
    top5_b=top5_b, s1=(1 - top5_b / full_b) * 100, s2=(1 - compressed / top5_b) * 100))
print("  C  答案块 pos5→pos1：首屏 0→1 · 噪声前置 {orig_off}→0 tok".format(orig_off=orig_off))
print("  D  路由 {hit}/8 · 省 {s:.1f}% token · FNR=1（人名引用漏网）".format(
    hit=hit, s=(1 - route_tok / all_tok) * 100))
print("一句话：Context 工程=给什么、给多少、怎么摆、何时给——四个都是可度量的工程决策，不是塞得越多越好。")
print("  落点：最简集（装什么）· 覆盖拐点（装多少）· 相关置顶（怎么排）· 路由闸（要不要带）")

e = datetime.datetime.now()
W("WALL total=%.3fs（纯计算，仅 stderr）\n" % (e - w0).total_seconds())
print("done · 一键复现：python code/notebooks/_tools/context_engineer_demo.py")
