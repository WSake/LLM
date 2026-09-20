# -*- coding: utf-8 -*-
"""
eval_gate_demo.py —— 04-训练体系 10-训练闭环中的评测门禁（知识地图 §4.10 / §17.3 收尾）的确定性实证
=====================================================================================================
§4.10 说：训练循环里要有"评测门禁"——预训练看 PPL + 少量下游抽样、SFT/RL 后跑全量精选集；
三条训练失败模式（loss 降但能力跌 / 遗忘 / 奖励黑客）必须在训练中段就被拦住。
本篇把"门禁"拍成三段本机真跑 + 一套规则表：

  [A] 认证门禁（过拟合闸）：train 上 CE 单调降 = "看起来成功"；hold-out 评测先降后见顶回升 =
      过拟合拐点。门禁规则：评测连续 3 次未创新低 → 早停 + 回滚到 best。
  [B] 回归门禁（遗忘闸）：先把旧能力句训到背熟，再换新语料续训——新语料 loss 降的同时、
      旧能力回归集逐点跌 = 遗忘曲线。门禁规则：回归分跌破阈值 → 回滚到"旧能力可接受"的
      checkpoint，量化"再想多训几步 = 丢多少旧能力"。
  [C] 泄漏门禁（假健康闸）：把评测句塞进训练集，两臂对照——被污染的评测分虚低（假健康），
      全新 hold-back 句上两臂几乎无差 = 泄漏没带来真本事，评测锁必须 hold-out。
  [D] 门禁规范表（规则，账算非实测）：评测集四划分 / 频率纪律 / 触发动作 / 三条失败模式的门禁
      （含 RL 后格式门禁——奖励黑客）清单。

[A][B][C] 均为本机真跑：torch CPU 单线程、固定 seed、GRU-48 小字符语言模型，科学数字逐位复现，
只有墙钟在跑次间浮动。[A] 的语料是同一主题拆成的 train 8 句 + dev 6 句（模拟真实预训练
train/dev 同源抽样）；[B][C] 用 24 句互不相同的中文事实句（旧能 8 / 新能 8 / 评测 4 / 回退 4）。
整脚本纯 CPU，墙钟 1-2 分钟级（训练 + 逐点评测主导）。
"""
import sys, time, math

if sys.stdout.encoding and sys.stdout.encoding.lower() in ("gbk", "gb2312"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

os_environ = __import__("os").environ
os_environ.setdefault("OMP_NUM_THREADS", "1")
os_environ.setdefault("MKL_NUM_THREADS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F

t0 = time.perf_counter()
torch.manual_seed(0)
torch.set_num_threads(1)

print("=" * 66)
print("eval_gate_demo：训练闭环中的评测门禁（04 训练体系 §4.10/§17.3 收尾）——过拟合·遗忘·假健康")
print("=" * 66)
print("[0] 口径：§4.10 说训练循环里要有评测门禁。本篇 [A][B][C] 在本机 CPU 上真训练一个")
print("    GRU-48 小字符语言模型：[A] 用同源预训练语料（train 8 句 / dev 6 句），[B][C] 用 24 句")
print("    互不相同的中文事实句（旧能 8 / 新能 8 / 评测 4 / 回退 4）。三条门禁各抓一种训练失败模式：")
print("    [A] 过拟合（train CE 降、hold-out CE 先降后升）、[B] 遗忘（新语料 loss 降、旧能力回归集跌）、")
print("    [C] 评测泄漏（被污染的评测分虚低、hold-back 揭穿）。[D] 为规则表（账算非实测）。")

# ---------------------------------------------------------------------------
# 语料：[A] 用自带的"预训练语料"——同一主题的两笔短句（train 8 句 / dev 6 句未见于训练），
#       模拟真实预训练里 train/dev 同源抽样；[B][C] 用 24 句互不相同的中文事实句——旧能 8 / 新能 8 / 评测 4 / 回退 4
# ---------------------------------------------------------------------------
A_TRAIN = ["南极洲是地球上最寒冷的大陆",
           "厚重的冰盖覆盖着大部分陆地",
           "企鹅是南极海岸的常客",
           "它们靠海洋里的磷虾维持生命",
           "南极海域盛产数量惊人的磷虾",
           "冬季南极会出现极夜现象",
           "极夜期间的天空总是灰暗",
           "极昼来临时阳光可以持续数月"]
A_EVAL = ["地球上最寒冷的大陆是南极洲",
          "南极洲的大部分陆地被冰盖覆盖",
          "南极的冬季会出现漫长的极夜",
          "磷虾是维持企鹅生命的海洋生物",
          "南极的日照在冬季变得短暂",
          "灰暗的天空在极夜持续很久"]
OLD8 = ["大熊猫以竹子为主要食物",
        "地球绕太阳公转一圈大约一年",
        "水受热后会变成水蒸气",
        "光在真空中的速度约为每秒三十万千米",
        "柠檬汁属于酸性液体",
        "蝙蝠靠回声定位寻找飞虫",
        "唐朝的首都设在长安城",
        "铁在潮湿空气中容易生锈"]
NEW8 = ["海龟的性别由孵化温度决定",
        "铅笔芯的主要材料是石墨",
        "蜜蜂用跳舞向同伴传递花源位置",
        "量子比特可以同时处于叠加状态",
        "纸币的原料来自棉花纤维",
        "北极熊的皮肤其实是黑色",
        "仙人掌的刺是变形的叶子",
        "火星的一天约比地球长四十分钟"]
EVAL4 = ["长颈鹿靠心脏泵血供脑部",
         "月亮绕地球运行一周大约一个月",
         "玻璃的主要成分是二氧化硅",
         "鲸鱼是通过肺部呼吸的哺乳动物"]
HOLD4 = ["盐湖里的盐分来自上游岩石",
         "风车的转速受风速控制",
         "木材在真空中不会燃烧",
         "火山口周围常有温泉"]

ALL = A_TRAIN + A_EVAL + OLD8 + NEW8 + EVAL4 + HOLD4
chars = sorted(set("".join(ALL)))
ch2i = {c: i for i, c in enumerate(chars)}
V = len(chars)


class CharLM(nn.Module):
    def __init__(self, voc, d=48, p=0.25):
        super().__init__()
        self.emb = nn.Embedding(voc, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.proj = nn.Linear(d, voc)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h, _ = self.gru(self.drop(self.emb(x)))
        return self.proj(self.drop(h))


def ids_list(sents):
    return [[ch2i[c] for c in s] for s in sents]


def train_step(net, opt, s_ids, n_round=1):
    """训 n_round 步；每步把当前句子集整批过一遍（小语料下 1 步 = 全集合 1 遍）。
    y[t] = x[t+1]，末位无目标，pad 位置 ignore。"""
    net.train()
    tot = 0.0
    for _ in range(n_round):
        maxlen = max(len(t) for t in s_ids)
        xbt = torch.zeros(len(s_ids), maxlen, dtype=torch.long)
        ybt = torch.full((len(s_ids), maxlen), -100, dtype=torch.long)
        for i, t in enumerate(s_ids):
            L = len(t)
            xbt[i, :L] = torch.tensor(t)
            ybt[i, :L - 1] = torch.tensor(t[1:])
        opt.zero_grad()
        loss = F.cross_entropy(net(xbt).reshape(-1, V), ybt.reshape(-1))
        loss.backward()
        opt.step()
        tot += float(loss.detach())
    return tot / n_round


def eval_sent(net, s):
    """左对齐逐前缀算 next-char：句平均 CE 与 argmax 命中率。dropout 关闭 → 评估确定。
    返回的命中率是 next-char argmax 判对比例，不是整句整对。"""
    ids = [ch2i[c] for c in s]
    net.eval()
    ces, hits = [], 0
    with torch.no_grad():
        for t in range(1, len(ids)):
            x = torch.tensor(ids[:t]).unsqueeze(0)
            logits = net(x)[0, -1]
            ce = -F.log_softmax(logits, dim=-1)[ids[t]]
            ces.append(float(ce))
            if int(logits.argmax()) == ids[t]:
                hits += 1
    return sum(ces) / len(ces), hits / len(ces)


def eval_mean_ce(net, sents):
    """整批 1 次 forward 拿句平均 next-char CE（pad ignore），dropout 关闭。"""
    ids = ids_list(sents)
    maxlen = max(len(t) for t in ids)
    xbt = torch.zeros(len(ids), maxlen, dtype=torch.long)
    ybt = torch.full((len(ids), maxlen), -100, dtype=torch.long)
    for i, t in enumerate(ids):
        L = len(t)
        xbt[i, :L] = torch.tensor(t)
        ybt[i, :L - 1] = torch.tensor(t[1:])
    net.eval()
    with torch.no_grad():
        logits = net(xbt)
        loss = F.cross_entropy(logits.reshape(-1, V), ybt.reshape(-1))
    return float(loss)


def regression_score(net, sents, thr=0.7):
    """回归分 = sents 里 next-char 命中率 ≥ thr 的句子数（能力"背对"句数）。"""
    ok = 0
    for s in sents:
        _, hit = eval_sent(net, s)
        if hit >= thr:
            ok += 1
    return ok


def make_net():
    torch.manual_seed(0)
    return CharLM(V)


def make_opt(net):
    return torch.optim.Adam(net.parameters(), lr=3e-3)


# ---------------------------------------------------------------------------
# [A] 认证门禁（预训练 dev-PPL 闸）：过拟合拐点 + 早停回滚
# ---------------------------------------------------------------------------
print("\n[A] 认证门禁（预训练 dev-PPL 闸，本机真跑）：train 8 句 / dev 6 句（同源同主题两笔）")
A_SENTS = ids_list(A_TRAIN)
A_EVAL_S = ids_list(A_EVAL)
dev_chars = set("".join(A_EVAL))
train_chars = set("".join(A_TRAIN))
print(f"    语料：同一条'南极洲+极地动物'主题拆两份——train 8 句 / dev 6 句未见于训练；"
      f"dev 共 {len(dev_chars)} 个不同字、与 train 共享 {len(dev_chars & train_chars)} 个"
      f"（{100.0*len(dev_chars & train_chars)/len(dev_chars):.0f}%），刻意模拟真实预训练的'train/dev 同源抽样'")
A_STEPS, EV = 216, 8          # 逐 8 步评一次；门禁 = dev-PPL 连续 3 次未创新低 → 早停回滚
netA = make_net(); optA = make_opt(netA)
best_e, best_step, stall, curve, stop_step = 1e9, 0, 0, [], None
for st in range(0, A_STEPS + 1, EV):
    if st > 0:
        train_step(netA, optA, A_SENTS, n_round=EV)
    tr = eval_mean_ce(netA, A_TRAIN)
    ev = eval_mean_ce(netA, A_EVAL)
    curve.append((st, tr, ev))
    if ev < best_e - 1e-9:
        best_e, best_step, stall = ev, st, 0
    else:
        stall += 1
        if stall >= 3:            # 门禁现在就执行：连续 3 次未创新低 → 早停回滚
            stop_step = st
            break
if stop_step is None:
    stop_step = A_STEPS
print(f"    训练口径：V={V} · GRU-48 · lr=3e-3 · 每步过全集合 · 逐 {EV} 步评测")
print("    dev 门禁用句（未见于训练）：" + "／".join(A_EVAL))
print(f"    step  训练CE  dev-PPL")
for st, tr, ev in curve:
    mark = "   ← 过拟合拐点(best)" if st == best_step else ""
    mark = f"   ← 门禁执行:早停+回滚(best@step{best_step})" if st == stop_step else mark
    print(f"    {st:>4}  {tr:.4f}  {math.exp(ev):7.2f}{mark}")
last_ev = curve[-1][2]
print(f"    → dev-PPL 先降后升：best 在 step {best_step}（dev CE {best_e:.4f} ≈ PPL {math.exp(best_e):.1f}），"
      f"训练 CE 却一路降着——train 在'变好'，dev 在'变差' = 过拟合已开始")
print(f"    → 门禁动作：连续 3 次未创新低 → 在 step {stop_step} 早停并回滚到 best 步的参数"
      f"（这 {stop_step - best_step} 步训下来 dev 不仅没创新低、还亏了 {last_ev - best_e:+.4f}；"
      f"照默认计划继续训到 step {A_STEPS}，dev-PPL 只会顺着拐头更差）")
assert min(ev for _, _, ev in curve) == best_e
assert stop_step >= best_step

# ---------------------------------------------------------------------------
# [B] 回归门禁：遗忘曲线 + 回滚
# ---------------------------------------------------------------------------
print("\n[B] 回归门禁（遗忘闸，本机真跑）：旧能 8 句先训到背熟 → 换新能 8 句续训，旧能回归集逐点跌")
S1, S2 = 96, 288
netB = make_net(); optB = make_opt(netB)
for st in range(S1):
    train_step(netB, optB, ids_list(OLD8), n_round=1)
reg0 = regression_score(netB, OLD8)
print(f"    阶段一：训旧能 8 句 {S1} 步 → 旧能回归分 {reg0}/8（回归分 = next-char 命中率≥0.7 的句子数）")
print("    阶段二：换新能 8 句续训，每 8 步记录 新句CE 与 旧能回归分（回归门禁逐点读数）")
rowsB, bestB = [], None
for st in range(0, S2 + 1, EV):
    if st > 0:
        train_step(netB, optB, ids_list(NEW8), n_round=EV)
    new_ce = eval_mean_ce(netB, NEW8)
    reg = regression_score(netB, OLD8)
    rowsB.append((st, new_ce, reg))
    if bestB is None or (reg >= 6 and new_ce < bestB[1]):
        bestB = (st, new_ce, reg)
trigger, trigger_row = None, None
for i, (st, new_ce, reg) in enumerate(rowsB):
    if reg <= 4 and len([1 for _, _, r in rowsB[max(0, i - 1):i + 1] if r <= 4]) >= 2:
        trigger, trigger_row = st, (new_ce, reg)
        break
print("    step  新句CE  旧能回归")
for st, new_ce, reg in rowsB:
    mark = "   ← 门禁触发" if st == trigger else ""
    print(f"    {st:>4}  {new_ce:.4f}   {reg}/8{mark}")
if trigger is None:
    trigger, trigger_row = rowsB[-1][0], (rowsB[-1][1], rowsB[-1][2])
bs, bn, br = bestB
fs, fn, fr = rowsB[-1]
jt, tn, tr_ = trigger, *trigger_row
print(f"    → 门禁规则：旧能回归分连续 2 次 ≤4/8 → 触发。实测触发步 step {jt}（旧能 {tr_}/8 · 新句 {tn:.4f}）")
print(f"    → 回滚点 = 旧能仍≥6 且新句 CE 最低的 step {bs}（旧能 {br}/8 · 新句 {bn:.4f}）——停回这类点，"
      f"'旧能用'和'新句已学'两头都保住")
print(f"    → 放任从 step {bs} 硬训到 step {fs}：旧能 {br}/8 → {fr}/8（丢 {br - fr}/8），"
      f"新句 CE {bn:.4f} → {fn:.4f}——为了深训这最后 ~{bn - fn:.3f} 的 CE 进度，旧能掉了 {br - fr}/8。")
print(f"      回归门禁拦的就是这笔账：触发即回滚，再用重放/重新配比在保住旧能的前提下把新句重新练上。")
assert reg0 == 8
assert bestB is not None and br >= 6

# ---------------------------------------------------------------------------
# [C] 泄漏门禁：评测句进训练 = 假健康，hold-back 揭穿
# ---------------------------------------------------------------------------
print("\n[C] 泄漏门禁（假健康闸，本机真跑）：评测 4 句里的 2 句塞进训练集（泄漏臂）vs 干净臂，同训练步数")
C_STEPS = 120
def run_arm(sents):
    net = make_net(); opt = make_opt(net)
    ids = ids_list(sents)
    for _ in range(C_STEPS):
        train_step(net, opt, ids, n_round=1)
    return net
clean = run_arm(OLD8 + NEW8)
leak = run_arm(OLD8 + NEW8 + EVAL4[:2])
ce_clean_e = eval_mean_ce(clean, EVAL4)
ce_leak_e = eval_mean_ce(leak, EVAL4)
ce_clean_leaked = eval_mean_ce(clean, EVAL4[:2])
ce_leak_leaked = eval_mean_ce(leak, EVAL4[:2])
ce_clean_unk = eval_mean_ce(clean, EVAL4[2:])
ce_leak_unk = eval_mean_ce(leak, EVAL4[2:])
ce_clean_h = eval_mean_ce(clean, HOLD4)
ce_leak_h = eval_mean_ce(leak, HOLD4)
print(f"    训练：干净臂 16 句 / 泄漏臂 16+2 句（评测集里塞进「{EVAL4[0]}」「{EVAL4[1]}」）× {C_STEPS} 步")
print("    评测对象 →       干净臂    泄漏臂    Δ（泄漏−干净）")
print(f"    评测集 4 句      {ce_clean_e:.4f}  {ce_leak_e:.4f}   {ce_leak_e - ce_clean_e:+.4f}")
print(f"      ↳ 泄漏的 2 句  {ce_clean_leaked:.4f}  {ce_leak_leaked:.4f}   {ce_leak_leaked - ce_clean_leaked:+.4f}")
print(f"      ↳ 没泄漏的 2 句 {ce_clean_unk:.4f}  {ce_leak_unk:.4f}   {ce_leak_unk - ce_clean_unk:+.4f}")
print(f"    hold-back 4 句   {ce_clean_h:.4f}  {ce_leak_h:.4f}   {ce_leak_h - ce_clean_h:+.4f}")
print(f"    → 泄漏臂的『评测分数』看着好了 {ce_clean_e - ce_leak_e:.4f}——"
      f"正是塞进去那 2 句被背出来的假健康；但 hold-back 全新句上两臂只差 {abs(ce_leak_h - ce_clean_h):.4f}，"
      f"泄漏没带来真本事，只撬开了评测锁。")
assert ce_leak_e < ce_clean_e, "泄漏臂评测CE应更低（假健康物证）"
assert abs(ce_leak_h - ce_clean_h) < 0.5, "hold-back 上两臂应接近"

# ---------------------------------------------------------------------------
# [D] 门禁规范表（规则，账算非实测）
# ---------------------------------------------------------------------------
print("\n[D] 门禁规范表（规则，账算非实测）：评测门禁的四件套")
print("    ① 评测集四划分：train（训练）/ dev-评测（早停·调超参·看PPL）/ regression（能力回归，SFT·RL后全量跑）/ hold-back（污染审计，永不进训练）")
print("    ② 频率纪律：warmup 内豁免评测；其后每 k 步固定同种子子样本，跨运行可比")
print("    ③ 触发动作：评测连续 N 次未创新低 → 早停+回滚 best；回归分跌破阈值 → 停/回滚/换锚；泄漏审计 → 重出评测集")
print("    ④ 三条失败模式各自的门禁：过拟合 = dev 评测见顶（本段 [A]）；遗忘 = 回归集能力跌（[B]）；假健康 = 评测泄漏（[C]）；")
print("       RL 后格式崩 = 奖励黑客 → 跑格式+非格式混合回归集：格式合规率涨但能力分崩就触发")
print("    行业实践（非本机实测）：预训练每万步打 PPL + 少量下游抽样；SFT/RL 后跑全量精选回归集；W&B/MLflow 记录评测与触发")

print(f"\n墙钟 {1000*(time.perf_counter()-t0):.0f} ms（[A][B][C] 本机真跑，科学数字逐位一致，只有墙钟浮动）")
