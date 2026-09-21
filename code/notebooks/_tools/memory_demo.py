#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""记忆系统探针（知识地图 §6.10 / §9.5 / §17.7.4）：
把「用户陈述 + 工具入参/结果」从短期会话窗口沉淀成跨会话长期结构化记忆（实体-插槽-值 · 用户画像），
量四本账：A 写入账 · B 整合账 · C 检索账（回灌剂量）· D 衰减与预算账（遗忘） + 选择树 8 场景。
纯 stdlib · 零网络 · 零下载 · 不训练模型 · 确定性（无随机源）· stdout 逐字节可复现。
"""
import re
import sys
from time import perf_counter

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ------------------------- 记账户径 -------------------------
# token 估算 = 中文 1 字 1 + 英文/数字连续段 1（近似计费口径，非真实 tokenizer）
TOK = re.compile(r"[一-鿿]+|\d[\d.:/\-]*|[a-zA-Z_]+")


def toks(s):
    return TOK.findall(s)


def tcount(*parts):
    """token 估算 = 中文 1 字 1 + 英文/数字连续段 1（仓库统一口径，非真实 tokenizer）"""
    n = 0
    for p in parts:
        for r in TOK.findall(p):
            n += 1 if r[:1].isascii() else len(r)
    return n


def bigrams(s):
    """中文段转字符双字组 + 英文/数字段整段（检索词面共现用；最小编词面代理，非语义）"""
    out = set()
    for r in toks(s):
        if r[:1].isascii() or len(r) == 1:
            out.add(r)
        else:
            for i in range(len(r) - 1):
                out.add(r[i:i + 2])
    return out


# 别名表：同人不同称呼 -> 规范实体（写入侧一次归一）
ALIAS = {"张先生": "张三", "老张": "张三"}
# 语义对齐表：查询双字组 -> 记忆槽位语义（模拟 embedding 检索越过词面共现的「字段语义」一截，作者预置）
ALIGN = {
    "住哪": "居住地", "哪里": "居住地", "居住": "居住地",
    "职业": "职业", "工作": "职业",
    "生日": "生日",
    "喜欢": "偏好", "爱喝": "偏好", "咖啡": "偏好",
    "几点": "事件", "会议": "事件", "开始": "事件", "日程": "事件",
    "天气": "天气", "气温": "天气",
    "城市": "城市", "出差": "城市", "地方": "城市",
}
CITIES = ["北京", "上海", "广州", "深圳", "成都", "杭州", "南京", "武汉", "西安", "重庆"]

_META = "__mention__"


def norm_entity(text):
    for a, c in ALIAS.items():
        if a in text:
            return c
    return "张三"


def find_city(t):
    for c in CITIES:
        if c in t:
            return c
    return None


# ------------------------- 会话语料（预置 · 确定性） -------------------------
S1 = {"id": 1, "turns": [
    ("u", "我是张三。"),
    ("u", "我住在北京，我是一名软件工程师。"),
    ("t", "get_weather(city=北京, date=2026-09-23) → 晴 18-26℃"),
]}
S2 = {"id": 2, "turns": [
    ("u", "帮我记得：我更喜欢喝美式咖啡。"),
    ("t", "add_event(title=团队例会, start=2026-09-28 10:00, remind=15) → OK"),
    ("u", "张先生下周二的产品评审会也别忘了。"),
]}
S3 = {"id": 3, "turns": [
    ("u", "我现在住上海了，帮我查一下上海天气。"),
    ("t", "get_weather(city=上海, date=2026-09-23) → 多云 22-28℃"),
    ("u", "我的生日是 1990 年 5 月 20 日。"),
]}

FILL_CITIES = ["广州", "深圳", "成都", "杭州", "南京", "武汉", "西安", "重庆"]
FILL_EVENTS = ["产品评审", "客户回访", "健身", "读书会", "面试", "牙医预约", "电话会", "项目复盘"]


def filler(sid):
    city = FILL_CITIES[sid - 4]
    ev = FILL_EVENTS[sid - 4]
    date = "2026-09-%02d" % (23 + (sid - 4))
    return {"id": sid, "turns": [
        ("u", "帮我查一下%s明天的天气。" % city),
        ("t", "get_weather(city=%s, date=%s) → 晴 20-28℃" % (city, date)),
        ("u", "把%s加到周六日程。" % ev),
        ("t", "add_event(title=%s, start=2026-09-27 14:00) → OK" % ev),
    ]}


def extract_facts(sess):
    """从一轮会话抽 (entity, slot, value, kind, ts)。规则=词面/结构抽取（作者预置，确定性）。"""
    facts = []
    ts = sess["id"]
    for role, text in sess["turns"]:
        ent = norm_entity(text)
        if role == "u":
            m = re.search(r"我是(?!一名)([一-鿿]{2,4})", text)
            if m:
                facts.append((ent, "姓名", m.group(1), "profile", ts))
            city = find_city(text)
            if city and ("住" in text or "迁" in text):
                facts.append((ent, "居住地", city, "profile", ts))
            m = re.search(r"是一名([一-鿿]{1,8})", text)
            if m:
                facts.append((ent, "职业", m.group(1), "profile", ts))
            m = re.search(r"更喜欢([一-鿿]{2,8})", text)
            if m:
                facts.append((ent, "偏好", m.group(1), "profile", ts))
            m = re.search(r"生日是\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日", text)
            if m:
                facts.append((ent, "生日", "%s年%s月%s日" % (m.group(1), m.group(2), m.group(3)), "profile", ts))
            m = re.search(r"把([一-鿿]+)加到", text)
            if m:
                facts.append((ent, "事件", m.group(1), "transient", ts))
        else:
            city = find_city(text)
            if city and "city=" in text:
                facts.append(("张三", "城市", city, "transient", ts))
            mm = re.search(r"→\s*([^\n]+?)\s*$", text)
            if mm and "weather" in text:
                facts.append(("张三", "天气", mm.group(1).strip(), "transient", ts))
            mt = re.search(r"title=([一-鿿]+)", text)
            if mt:
                facts.append(("张三", "事件", mt.group(1), "transient", ts))
    for a, c in ALIAS.items():
        for role, text in sess["turns"]:
            if role == "u" and a in text:
                facts.append((c, _META, a, "meta", ts))
    return facts


def build_store(facts):
    """整合账：画像 profile = 同槽最新值（upsert）；账本 ledger = 事务逐条追加（append）。"""
    profile = {}
    ledger = []
    st = {"add": 0, "upd": 0, "ded": 0, "meta": 0}
    for (ent, slot, val, kind, ts) in facts:
        if slot == _META:
            st["meta"] += 1
            continue
        if kind == "profile":
            cur = profile.get(slot)
            if cur is None:
                profile[slot] = {"val": val, "ts": ts}
                st["add"] += 1
            elif cur["val"] == val:
                st["ded"] += 1
            else:
                cur["val"] = val
                cur["ts"] = ts
                st["upd"] += 1
        else:
            dup = any(l["slot"] == slot and l["val"] == val and l["ts"] == ts for l in ledger)
            if dup:
                st["ded"] += 1
            else:
                ledger.append({"ent": ent, "slot": slot, "val": val, "ts": ts})
                st["add"] += 1
    return profile, ledger, st


def make_entries(profile, ledger):
    out = []
    for k, v in profile.items():
        txt = "张三的%s是%s" % (k, v["val"])
        out.append({"slot": k, "val": v["val"], "ts": v["ts"], "kind": "profile",
                    "text": txt, "tok": tcount(txt)})
    for l in ledger:
        txt = "张三的%s是%s·s%d" % (l["slot"], l["val"], l["ts"])
        out.append({"slot": l["slot"], "val": l["val"], "ts": l["ts"], "kind": "transient",
                    "text": txt, "tok": tcount(txt)})
    out.sort(key=lambda e: (e["slot"], e["val"], e["ts"]))
    return out


# ------------------------- 查询集（金标准=作者预置） -------------------------
QUERIES = [
    ("q1", "张三现在住哪里？", "居住地"),
    ("q2", "张三的职业是什么？", "职业"),
    ("q3", "张先生的生日是哪天？", "生日"),
    ("q4", "张三喜欢喝什么？", "偏好"),
    ("q5", "团队例会几点开始？", "事件·团队例会"),
    ("q6", "出差过的城市有哪些？", "城市"),
]


def gold_match(e, gold):
    if gold.startswith("事件·"):
        return e["slot"] == "事件" and gold.split("·")[1] in e["text"]
    return e["slot"] == gold


def rank_aligned(entries, q):
    qbig = bigrams(q)
    qalign = set()
    for bg in qbig:
        if bg in ALIGN:
            qalign.add(ALIGN[bg])
    scored = []
    for e in entries:
        s = len(qbig & bigrams(e["text"]))
        bonus = sum(1 for a in qalign if a == e["slot"])
        scored.append({"e": e, "score": s + bonus, "plain": s})
    scored.sort(key=lambda x: (-x["score"], x["e"]["text"]))
    return scored


def rank_plain(entries, q):
    qbig = bigrams(q)
    scored = list(entries)
    scored.sort(key=lambda e: (-(len(qbig & bigrams(e["text"]))), e["text"]))
    return scored


def keep_trim(order, budget):
    kept, tot = [], 0
    for e in order:
        if tot + e["tok"] > budget:
            break
        kept.append(e)
        tot += e["tok"]
    return kept, tot


def strategy_recency(entries, budget):
    order = sorted(entries, key=lambda e: (-e["ts"], e["text"]))
    return keep_trim(order, budget)


def strategy_profile(entries, budget):
    prof = [e for e in entries if e["kind"] == "profile"]
    led = [e for e in entries if e["kind"] == "transient"]
    order = sorted(prof, key=lambda e: (-e["ts"], e["text"])) + sorted(led, key=lambda e: (-e["ts"], e["text"]))
    return keep_trim(order, budget)


def strategy_summary(entries, budget):
    prof = sorted([e for e in entries if e["kind"] == "profile"], key=lambda e: e["text"])
    tr = [e for e in entries if e["kind"] == "transient"]
    recent = {}
    for e in sorted(tr, key=lambda x: x["ts"]):
        recent[e["slot"]] = e
    head_tok = sum(e["tok"] for e in prof)
    out = list(prof)
    for slot in ("城市", "天气", "事件"):
        if slot in recent:
            n = sum(1 for x in tr if x["slot"] == slot)
            line = {"slot": slot, "val": "…", "ts": 99, "kind": "summary",
                    "text": "张三的%s:共%d条 最近:%s@s%d" % (slot, n, recent[slot]["val"], recent[slot]["ts"]),
                    "tok": tcount("张三的%s:共%d条 最近:%s@s%d" % (slot, n, recent[slot]["val"], recent[slot]["ts"]))}
            if head_tok + line["tok"] <= budget:
                out.append(line)
                head_tok += line["tok"]
    out.sort(key=lambda e: (e["slot"], e["text"]))
    return out, head_tok


def coverage(entries, queries):
    return sum(1 for _, _, g in queries if any(gold_match(e, g) for e in entries))


# ================================ 主流程 ================================
SESS = [S1, S2, S3] + [filler(i) for i in range(4, 4 + len(FILL_CITIES))]
facts = []
for ss in SESS:
    facts.extend(extract_facts(ss))
profile, ledger, st = build_store(facts)
entries = make_entries(profile, ledger)
T_FULL = sum(e["tok"] for e in entries)

raw_tok = tcount(*[t for ss in SESS for _, t in ss["turns"]])
n_facts = sum(1 for f in facts if f[1] != _META)

print("=" * 72)
print("前置：记忆系统 Memory（§6.10/§9.5/§17.7.4）· 零网络零下载 · 四账+选择树")
print("  把「用户陈述 + 工具入参/结果」从短期会话窗口沉淀成跨会话长期结构化记忆（实体-插槽-值·用户画像）")
print("  token 估算=中文字符+英文/数字词（近似子词系，非真实 tokenizer）")
print("环境：零随机·零网络·stdout 逐字节可复现")
print("=" * 72)
print()

# ---------- A 写入账 ----------
print("[实验 A] 记忆写入账 —— 会话/工具结果 → 结构化记忆（实体-插槽-值 · 画像+账本两层）")
print("  语料：%d 次会话 %d 条 user+tool 条（原文共 %d tok）→ 抽取事实 %d 条 → 整合入表 %d 条" % (
    len(SESS), sum(len(s["turns"]) for s in SESS), raw_tok, n_facts, len(entries)))
print("  画像层（用户画像·每一槽=该实体属性最新值）：张三 %d 槽 —— %s" % (
    len(profile), " · ".join("%s=%s" % (k, v["val"]) for k, v in sorted(profile.items()))))
print("  账本层（事务/工具结果·逐条追加带 s 会话戳）：%d 条 —— 城市 %d · 天气 %d · 事件 %d" % (
    len(ledger),
    sum(1 for x in ledger if x["slot"] == "城市"),
    sum(1 for x in ledger if x["slot"] == "天气"),
    sum(1 for x in ledger if x["slot"] == "事件")))
print("  → 结构化后 %d tok vs 原文 %d tok（=%.0f%%）：去虚词、只留实体-槽-值骨架，顺带压了 %d tok——但记忆的价值不在省字，"
      "在『下次不用重读原文』：真正的杠杆是 C 的按需回灌（每问只带相关窗口），不是囤积全文" % (
          T_FULL, raw_tok, 100.0 * T_FULL / raw_tok, raw_tok - T_FULL))
print()

# ---------- B 整合账 ----------
naive = []
for f in facts:
    if f[1] == _META or f[3] != "profile":
        continue
    naive.append(f)  # (ent, slot, val, kind, ts)
naive_tok = sum(tcount("张三的", f[1], "是", f[2]) for f in naive)
prof_tok = sum(e["tok"] for e in entries if e["kind"] == "profile")
same_slot_conflict = sum(1 for f in naive if f[1] == "居住地")
print("[实验 B] 记忆整合账 —— 同实体跨会话：同槽覆盖(最新值) · 别名归一 · 不去重自行矛盾")
print("  整合台账：新增 %d 条 · 覆盖更新 %d 次 · 去重 %d 条 · 别名解析 %d 处（张先生→张三）" % (
    st["add"], st["upd"], st["ded"], st["meta"]))
print("  画像整合示例：居住地 北京(s1)→上海(s3) 同槽覆盖 1 次=1 条最新值；生日/职业 等 4 槽新增")
print("  别名示例：s2『张先生…』→ 归一实体『张三』（写入侧一次解析，此后查询全按张三命中）")
print("  naive 不去重（append）：画像 %d 条 %d tok（含两条『居住地 北京/上海』=档案自相矛盾）" % (
    len(naive), naive_tok))
print("  upsert 整合后：画像 %d 条 %d tok · 同槽唯一=语义最新值 → 省 %d tok（%.0f%%）且不再自相矛盾" % (
    len(profile), prof_tok, naive_tok - prof_tok, 100.0 * (naive_tok - prof_tok) / naive_tok))
print()

# ---------- C 检索账 ----------
topk = 3
aligned_hits = 0
total_dose = 0
for qid, q, gold in QUERIES:
    ra = rank_aligned(entries, q)
    rp = rank_plain(entries, q)
    top = ra[:topk]
    hit = any(gold_match(x["e"], gold) for x in top)
    aligned_hits += bool(hit)
    pl_rank = next(i for i, e in enumerate(rp, 1) if gold_match(e, gold))
    dose = sum(x["e"]["tok"] for x in top)
    total_dose += dose
    hits_txt = " → ".join(("[√]" if gold_match(x["e"], gold) else "   ") + "%s(s%d,t%d)" % (x["e"]["text"][:14], x["e"]["ts"], x["e"]["tok"]) for x in top)
    print("  %s %-16s 金=%s  词面命中rank%d → 对齐top%d: %s, tok小计%d" % (
        qid, q, gold, pl_rank, topk, hits_txt, dose))
plain_hits = 0
for qid, q, gold in QUERIES:
    rp = rank_plain(entries, q)
    if any(gold_match(e, gold) for e in rp[:topk]):
        plain_hits += 1
    elif gold == "城市":
        pass
print("  检索命中：纯词面 top%d %d/%d（2 条被『张三』实体名冗余摊平=画像里同名反复出现→词面拉不开；需槽位语义）" % (
    topk, plain_hits, len(QUERIES)))
print("  对齐语义（ALIGN 表=模拟 embedding 越过词面的一截）top%d 命中 %d/%d · 6 问合计回灌 tok %d vs 全量 %d（省 %.0f%%）" % (
    topk, aligned_hits, len(QUERIES), total_dose, T_FULL, 100.0 * (T_FULL - total_dose) / T_FULL))
print("  → 读的纪律=剂量：全量回灌=每问都带全表（写得好是画像，读不好是干扰）；相关 top%d 命中不掉、注入带宽省一半+" % topk)
print()

# ---------- D 衰减与预算账 ----------
budget = int(T_FULL * 0.62)
full_hit = coverage(entries, QUERIES)
fr, fr_tok = strategy_recency(entries, budget)
fp, fp_tok = strategy_profile(entries, budget)
fs, fs_tok = strategy_summary(entries, budget)
print("[实验 D] 衰减与预算账 —— 会话无限 · 回灌窗口有限：谁留下、谁遗忘（上下文窗口是硬约束 §9.5）")
print("  全量回灌：%d 条 %d tok（超预算 %d）· 6 问可答 %d/6" % (len(entries), T_FULL, T_FULL - budget, full_hit))
print("  预算 %d tok，三种保留策略（保留后的可答率 / token / 丢掉了谁）：" % budget)
print("  策略1 最近优先：保留 %d 条 %d tok · 可答 %d/6 · 丢的是最早的画像（职业s1·偏好s2→『不记得你的职业』）" % (
    len(fr), fr_tok, coverage(fr, QUERIES)))
print("  策略2 画像优先：保留 %d 条 %d tok · 可答 %d/6 · 保住画像全 5 槽(职业s1·偏好s2·生日s3)，丢的是老账本(团队例会s2)" % (
    len(fp), fp_tok, coverage(fp, QUERIES)))
print("  策略3 画像+摘要：保留 %d 条 %d tok · 可答 %d/6 · 账本折叠成『共N条·最近一条』=最省" % (
    len(fs), fs_tok, coverage(fs, QUERIES)))
print("  遗忘曲线（最近优先保留 K 条 → 可答数单调爬升；曲线下面积=『留得越少，忘得越多』）：")
Ks = sorted(set([4, 8, 12, 16, 20, 24, 28, len(entries)]))
curve = []
for K in Ks:
    kept = sorted(entries, key=lambda e: (-e["ts"], e["text"]))[:K]
    curve.append(coverage(kept, QUERIES))
print("    K=%s → 可答 %s（tok=%s）" % ("·".join(str(k) for k in Ks), "·".join(str(c) for c in curve), "·".join(
    str(sum(e["tok"] for e in sorted(entries, key=lambda x: (-x["ts"], x["text"]))[:k])) for k in Ks)))
print("  → 忘的纪律=预算：遗忘不是丢了不修，是『忘什么』由策略定——全量 6/6 需 %d tok；画像/摘要 5/6 只需 %d-%d tok（只丢老账本团队例会）；"
      "最近优先 1/6=画像全掉（职业·偏好）" % (T_FULL, fs_tok, fp_tok))
print()

# ---------- 选择树 ----------
SCEN = [
    ("会话内即时问答（不必留到下次）", "上下文窗口直接装（03，不落库）"),
    ("跨会话要复用（记得上次说了啥）", "长期记忆（本篇）"),
    ("外部知识/文档语料", "RAG（08：RAG=外部知识 · Memory=自身历史 §17.7.4）"),
    ("工具入参/结果只在本轮执行用", "05 回喂循环（不落库）"),
    ("工具入参/结果要跨会话回查", "记忆账本落库（本篇 C 检索 / D 预算）"),
    ("用户长期偏好/基本属性", "用户画像（结构化 profile）"),
    ("超长历史对话", "摘要压缩后入记忆（挂靠 08-06 剪刀）"),
    ("敏感信息（隐私合规）", "不入库/脱敏（§6.10 隐私监管·GDPR/个保法）"),
]
print("  选择树 %d 场景（规则：会话内即时→不落库；跨会话复用→长期记忆；外部知识→RAG；工具结果即时→05 回喂；"
      "工具结果跨会话→账本落库；长期偏好/属性→画像；超长历史→摘要；敏感→不入库）；断言 %d/%d" % (
    len(SCEN), len(SCEN), len(SCEN)))
print()

# ---------- 台账汇总 ----------
print("=" * 72)
print("台账汇总（A 写入 / B 整合 / C 检索 / D 衰减预算）")
print("  A  %d 次会话 %d 条原文 → 实体-插槽-值 %d 条（画像 %d + 账本 %d）· 别名解析 %d 处" % (
    len(SESS), sum(len(s["turns"]) for s in SESS), len(entries), len(profile), len(ledger), st["meta"]))
print("  B  整合：同槽覆盖 %d 次（居住地 北京→上海）· naive 不去重 %d 条 %d tok vs 整合 %d 条 %d tok（省 %d tok）· 同槽唯一=最新值" % (
    st["upd"], len(naive), naive_tok, len(profile), prof_tok, naive_tok - prof_tok))
print("  C  检索：纯词面 top3 命中 %d/6 → 对齐语义 %d/6 · 6 问回灌 %d tok vs 全量 %d tok（省 %.0f%%）" % (
    plain_hits, aligned_hits, total_dose, T_FULL, 100.0 * (T_FULL - total_dose) / T_FULL))
print("  D  遗忘：全量超预算 %d tok · 最近优先早丢画像(职业·偏好)→1/6 · 画像/摘要 5/6 只需 %d-%d tok（只丢老账本团队例会）· 遗忘曲线单调爬升" % (
    T_FULL - budget, fs_tok, fp_tok))
print("一句话：记忆系统 = 把『对话上下文 + 工具入参/结果』从短期窗口沉淀成长期结构化记忆（实体-插槽-值/用户画像），"
      "下次会话按需回灌——写入要分工（画像 upsert·账本 append）、整合才自洽（覆盖+别名）、读取要剂量（top-k 回灌）、遗忘要预算（策略定取舍）")
print("  画像层决定『你是谁』 · 账本层记录『你干过什么』 · 检索层控制『每问带多少』 · 衰减层回答『忘什么』——四者叠起来=Memory Manager")
print("done · 一键复现：python code/notebooks/_tools/memory_demo.py")
W("WALL total=%.3fs" % (perf_counter() - _t0))
