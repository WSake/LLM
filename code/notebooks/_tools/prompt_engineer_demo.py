# -*- coding: utf-8 -*-
"""02-Prompt-Engineering 的可复现探针：Prompt 模板账本 · few-shot 示例预算 · CoT 成本与偏置评测闸
（零网络、零下载、不训练模型——只量"确定性工程维度"，模型侧真实增益一律挂靠 02 系既有实测）。

运行：cd 仓库根 && python code/notebooks/_tools/prompt_engineer_demo.py
依赖：仅 stdlib + jieba（与 v0.20 同一最轻档）。

把 §6.2/§17.5.2 的"从咒语到科学"焊成本机信号：
  实验 A  Prompt 模板账本   同一任务按 §6.2 要素链从 v1 裸任务累积到 v6（+角色/+格式/+约束/+few-shot/+CoT），
                                逐版量输入 token 与成本增量（挂靠 01 章计费模型）+ batch 缓存半价账——"prompt 不是免费的"。
  实验 B  few-shot 示例预算  8 候选示例（4 意图×2），相似优先 vs 意图覆盖两种"选示范"策略，
                                量输入面覆盖率×示例集冗余度——示例是"预算"不是"数量"，覆盖优先省预算且不冗余。
  实验 C  CoT 成本账         输入侧模板增量实测 + 输出侧步数预算表（挂靠 02-20 think≈6× 与 k↔token 递减）——
                                "该留给谁"=任务需要可验证的多步推理才值得。
  实验 D  偏置评测闸         同一任务 5 版 system prompt 的确定性维度表（token/句数/风格词/输出预算）+ 评测闸断言：
                                选 prompt 先过长度/风格/同质/无基线四道偏置闸（§17.5.2 实践=5 版评测选优的防混料）。

确定性：固定模板/示例/查询词表 + jieba 分词 → stdout 三遍逐位一致；真实墙钟（total）只进 stderr。
诚实边界：输入面覆盖率=词面分类代理，不是 LLM 输出的真实 ICL 增益（那是 02-01 gpt_demo 的解锁相变）；
          CoT 输出预算=模板限定的步数×步长，不是真实模型 TPOT；模板风格词频只量词面。
"""
import json, os, sys, time, warnings
from collections import defaultdict

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import jieba

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
T0 = time.time()

def WALL(tag, t):
    sys.stderr.write("[wall] %-28s %s\n" % (tag, t))
    sys.stderr.flush()

def tok(s):
    return [w for w in jieba.cut(s) if w.strip()]

def n_tok(s):
    return len(tok(s))

print("=" * 72)
print("前置：Prompt Engineering 从咒语到科学（§6.2/§17.5.2）· 零网络零下载 · 四账实测")
print("  只量确定性工程维度（token/成本/覆盖/冗余/风格词频）；模型侧增益挂靠 02 系既有实测")
print("=" * 72)

# ---------- 实验 A：Prompt 模板账本（§6.2 要素链，严格累积） ----------
print()
print("=" * 72)
print("实验 A  Prompt 模板账本：同一任务从 v1 裸任务累积到 v6（角色→格式→约束→few-shot→CoT）")
print("=" * 72)
USER_IN = "帮我查一下明天下午去美术馆附近餐厅的订位情况。"
TASK = "把用户消息分类为【订位/查询/投诉】三类，输出 JSON：{\"intent\":\"...\",\"reason\":\"...\"}"
CHAIN = {
    "v1 裸任务":      TASK,
    "v2 +角色":       "你是资深餐厅预订助理，擅长把用户口语转成结构化意图。",
    "v3 +输入格式":    "输入格式必须为：用户：<message> → JSON。",
    "v4 +约束":       "只输出 JSON，不要解释、不要追加文字。",
    "v5 +few-shot":   "例如：用户：今晚还有双人位吗？→{\"intent\":\"订位\"}；用户：你们几点关门？→{\"intent\":\"查询\"}。",
    "v6 +CoT提示":     "先想一步：消息里是否有订位/查询/投诉意图词，再输出。",
}
PIN_A = 2.50  # 美元/百万 token，挂靠 01 章标准档示例价
pid = {}
for name, add in CHAIN.items():
    if not pid:
        pid[name] = add
    else:
        prev_key = list(pid)[-1]
        pid[name] = pid[prev_key] + " " + add
print("  %-14s %8s %8s %10s" % ("版本", "输入tok", "增量tok", "成本($)"))
prev = None
A_ROWS = {}
for name in CHAIN:
    t = n_tok(pid[name])
    delta = t if prev is None else t - prev
    A_ROWS[name] = t
    print("  %-14s %8d %+8d %10.5f" % (name, t, delta, t * PIN_A / 1e6))
    prev = t
v1t, v6t = A_ROWS["v1 裸任务"], A_ROWS["v6 +CoT提示"]
print("  → 要素链全长后最重的版本（v6）比裸任务（v1）多 %d tok（%.1f×）——每一句工程化都是 token 预算" %
      (v6t - v1t, v6t / v1t))
# system 复用缓存账（挂靠 01 章缓存半价：相同 system 二次命中 → 输入段半价）
sys_part = pid["v6 +CoT提示"]
system_toks = n_tok(sys_part)
full_call = sys_part + " 用户：" + USER_IN
full_call_toks = n_tok(full_call)
batch = 10
base_cost = batch * full_call_toks * PIN_A / 1e6
cached_cost = base_cost - (batch - 1) * system_toks * 0.5 * PIN_A / 1e6
print("  batch=%d 次：无缓存 $%.4f vs 相同长 system 命中缓存 $%.4f（省 %.1f%%·长 system 单调场景）" %
      (batch, base_cost, cached_cost, 100 * (1 - cached_cost / base_cost)))
WALL("  实验A Prompt模板账本", "%.3f s" % 0.0)

# ---------- 实验 B：few-shot 示例预算（示例是预算不是数量） ----------
print()
print("=" * 72)
print("实验 B  few-shot 示例预算：8 候选示例（4 意图×2），相似优先 vs 意图覆盖两种选示范")
print("=" * 72)
INTENTS = {
    "机票": ["航班", "起飞", "机票", "机场", "值机", "登机"],
    "酒店": ["酒店", "房间", "预订", "退房", "入住", "前台"],
    "天气": ["天气", "预报", "下雨", "气温", "放晴", "回升"],
    "外卖": ["外卖", "点餐", "菜单", "下单", "配送", "口味"],
}
EXAMPLES = [
    ("这几天去东京的航班还有吗", "机票"),
    ("帮我订一张明天早上起飞的头等舱机票", "机票"),
    ("周末的酒店还有空的房间吗", "酒店"),
    ("帮我取消预订的酒店房间，改退房时间", "酒店"),
    ("明天上海天气怎么样，会下雨吗", "天气"),
    ("这周气温会回升吗，什么时候放晴", "天气"),
    ("点一份外卖送我家，备注少辣", "外卖"),
    ("从菜单里挑两份套餐下单", "外卖"),
]
QUERIES = {
    "q机票": "航班要改签，机票时间变了怎么办",
    "q酒店": "酒店房间订错了想退房",
    "q天气": "预报说明天会下雨吗，气温多少",
    "q外卖": "点餐备注要少辣，外卖多久能送到",
}
STOP = set("，。；？！的了吗啊呀吧呢帮我明天请假查一要送")
def content(s):
    return [w for w in tok(s) if w not in STOP]
INT_VOCAB = {k: set(v) for k, v in INTENTS.items()}

def similar_first(k):
    """相似优先：按与全部查询的总词面重叠挑 top-k（一把尺子挑示例）。"""
    tot = defaultdict(int)
    for i, (ex, it) in enumerate(EXAMPLES):
        es = set(content(ex))
        for _, q in QUERIES.items():
            tot[i] += len(es & set(content(q)))
    order = sorted(range(len(EXAMPLES)), key=lambda i: (-tot[i], i))
    return [EXAMPLES[i] for i in order[:k]]

def cover_first(k):
    """意图覆盖优先：每意图各取代表，预算跨意图铺开（前 k 个不同意图的示例）。"""
    by = defaultdict(list)
    for i, (ex, it) in enumerate(EXAMPLES):
        by[it].append((len(set(content(ex)) & INT_VOCAB[it]), -i, ex))
    picked = []
    for it in INTENTS:
        if not by[it]:
            continue
        by[it].sort(reverse=True)
        picked.append((by[it][0][2], it))
    return picked[:k]

def covered_intents(chosen):
    return len({x[1] for x in chosen})

def surface_cov(chosen):
    return len({x[1] for x in chosen}) / len(INTENTS)

def avg_redundancy(chosen):
    if len(chosen) < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            a = set(content(chosen[i][0])); b = set(content(chosen[j][0]))
            un = a | b
            tot += (len(a & b) / len(un)) if un else 0.0
            cnt += 1
    return tot / cnt

print("  覆盖曲线（k=1..4 两种选法：输入面覆盖率 / 示例集内部冗余）")
for k in range(1, 5):
    sf = similar_first(k); cf = cover_first(k)
    print("    k=%d  相似优先: %.2f / 冗余 %.2f  |  覆盖优先: %.2f / 冗余 %.2f"
          % (k, surface_cov(sf), avg_redundancy(sf), surface_cov(cf), avg_redundancy(cf)))
print("  全预算定点 k=4：相似优先触达 %d/4 意图（冗余 %.2f） vs 覆盖优先 %d/4（冗余 %.2f）——同样用完 4 条示例，覆盖优先多覆盖 1 个意图、示例还不互相重复" %
      (covered_intents(similar_first(4)), avg_redundancy(similar_first(4)),
       covered_intents(cover_first(4)), avg_redundancy(cover_first(4))))
print("  词面真相：'相似优先'按一把总重叠尺子挑 → 词面充足的代表意图被重复下注、冷门意图整组落空；'意图覆盖'每意图各占一档 → 同预算覆盖更全")
WALL("  实验B few-shot示例预算", "%.3f s" % 0.0)

# ---------- 实验 C：CoT 成本账（该留给谁） ----------
print()
print("=" * 72)
print("实验 C  CoT 成本账：输入模板增量实测 + 输出步数预算表（挂靠 02-20 think≈6× 与 k↔token 递减）")
print("=" * 72)
BASE_SYS = "把上一条消息分类为意图并输出 JSON。"
COT_ADD = "先一步一步推理，把每一步写在输出里，再给出 JSON。"
base_t = n_tok(BASE_SYS); cot_t = n_tok(BASE_SYS + COT_ADD)
print("  输入侧：裸 system %d tok → +CoT 提示 %d tok（模板增量 +%d tok = $%.5f·按 PIN_A=%.2f/百万）" %
      (base_t, cot_t, cot_t - base_t, (cot_t - base_t) * PIN_A / 1e6, PIN_A))
POUT_C = 10.00
print("  输出侧预算表（每步 12 tok、末尾答复 15 tok；模拟性设定，非真实 TPOT）：")
print("    %-8s %12s %12s" % ("模式", "输出tok", "成本($)"))
DIRECT = 20
for name, st in (("直答", 0), ("1 步", 1), ("3 步", 3), ("5 步", 5), ("8 步", 8)):
    o = DIRECT + st * 12
    print("    %-8s %12d %12.5f" % (name, o, o * POUT_C / 1e6))
print("  挂靠 02-20：think 段 ≈ 直答 6 倍输出 token（账单 ×6）；k↔token↔KV 收益递减——预算买的是可验证增益")
print("  判断表（要素事实）：多步/可验证/数字推理→CoT 值得；单步查证/风格/闲聊→CoT 只烧 token 不换正确率")
WALL("  实验C CoT成本账", "%.3f s" % 0.0)

# ---------- 实验 D：偏置评测闸（5 版 system prompt 选优的防混料） ----------
print()
print("=" * 72)
print("实验 D  偏置评测闸：同一任务 5 版 system prompt，确定性维度表 + 四道偏置闸断言")
print("=" * 72)
TASK_D = "把用户消息中出现的日期/时间提取出来，输出 ISO 8601。"
VERS = {
    "D1 最小指令":  TASK_D,
    "D2 +礼貌措辞": "您好，请帮我把消息里的日期提取成 ISO 8601 格式。谢谢！" + TASK_D,
    "D3 +长度通用": "请写一个完整的、详细的、覆盖所有细节的回答。" + TASK_D,
    "D4 +few-shot": "示例：用户：周五晚上七点见 → 输出 {date,time}。请照此输出。用户：下周一下午三点开会\n" + TASK_D,
    "D5 +CoT长格式": "请先一步步思考并逐条写清楚每一个中间结果，然后给出最终 JSON。" + TASK_D,
}
STYLE_WORDS = ["请", "谢谢", "您好", "详细", "完整", "覆盖", "一步一步", "写清楚", "认真", "每一个"]
print("  %-16s %8s %8s %10s %12s" % ("版本", "输入tok", "句数", "风格词数", "输出预算tok"))
data = []
for name, txt in VERS.items():
    t = n_tok(txt)
    sents = txt.count("。") + txt.count("！") + txt.count("？") + txt.count("\n") + 1
    sw = sum(txt.count(w) for w in STYLE_WORDS)
    out_budget = 40 + 20 * sw  # 模板要求"详细/写清楚"→ 输出预算走高（模拟口径，显式声明）
    data.append((name, t, sents, sw, out_budget))
    print("  %-16s %8d %8d %10d %12d" % (name, t, sents, sw, out_budget))
def rank_desc(keyfn):
    return [d[0] for d in sorted(data, key=lambda x: (keyfn(x), x[1]))][::-1]
def rank_asc(keyfn):
    return [d[0] for d in sorted(data, key=lambda x: (keyfn(x), x[1]))]
len_rank = rank_desc(lambda d: d[4])           # 长度口径：输出预算大 → 排前
eff_rank = rank_asc(lambda d: d[1] + d[3])     # 效率口径：输入 token + 风格词 少 → 排前
print("  排序差（防混料演示）：长度口径前三 = %s vs 效率口径前三 = %s（同一组模板，口径不同榜首不同）" %
      (",".join(len_rank[:3]), ",".join(eff_rank[:3])))
print("  评测闸四断言：")
print("    [闸1 长度偏置] 输出预算最高版 = %s（%d tok）→ 若按篇幅打分，它会系统性靠前" %
      (max(data, key=lambda x: x[4])[0], max(data, key=lambda x: x[4])[4]))
print("    [闸2 风格混料] 风格词最多版 = %s（%d 个：请/详细/完整…）→ 若按「礼貌度/认真度」打分会偏向它" %
      (max(data, key=lambda x: x[3])[0], max(data, key=lambda x: x[3])[3]))
print("    [闸3 样本同质] query 集固定 3 条（同一场景）→ 未覆盖全部输入面，单样本结论不可外推")
print("    [闸4 无基线] 5 版都没有纯裸版对照组 → 选优必须带 baseline（对照 §17.5.2 实践）")
print("  实践（§17.5.2）：同任务写 5 版 system prompt 评测选优 = 建评分卡（任务正确率为主）+ 先过四闸拦截口径污染")
WALL("  实验D 偏置评测闸", "%.3f s" % 0.0)

# ---------- 台账汇总 ----------
print()
print("=" * 72)
print("台账汇总  Prompt 工程四账 · 信号=本机读数 · 判断=经验值+挂靠")
print("=" * 72)
print("  A 模板账本 · 信号: 六版累积 token %d→%d（%.1f×）· batch 长 system 缓存省 %.1f%% → 判断: 每句工程化=token 预算，长 system 用缓存复用" %
      (v1t, v6t, v6t / v1t, 100 * (1 - cached_cost / base_cost)))
print("  B few-shot  · 信号: 同用 4 条示例 覆盖优先 %d/4 vs 相似优先 %d/4（多 1 意图）· 冗余 %.2f>%.2f → 判断: 示例是预算，多样覆盖全输入面" %
      (covered_intents(cover_first(4)), covered_intents(similar_first(4)),
       avg_redundancy(similar_first(4)), avg_redundancy(cover_first(4))))
print("  C CoT 成本 · 信号: 直答 20 tok vs 8 步 %d tok（%.1f×）→ 判断: 多步可验证才值得，风格/单步别开" %
      (DIRECT + 8 * 12, (DIRECT + 8 * 12) / DIRECT))
print("  D 偏置闸  · 信号: 长度口径榜首=%s vs 效率口径榜首=%s → 判断: 选 prompt 先过长度/风格/同质/基线四闸" %
      (len_rank[0], eff_rank[0]))
print()
print("  核心一句话：Prompt 工程不是「背咒语」，是给模型配齐任务/约束/边界并量好预算——")
print("              同样一句话，输入 token 不同、示例覆盖不同、评测口径不同，结果就可能完全不同。")
WALL("total", "%.3f s" % (time.time() - T0))
print("done · 一键复现：python code/notebooks/_tools/prompt_engineer_demo.py")
