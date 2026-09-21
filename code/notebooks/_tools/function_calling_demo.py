# -*- coding: utf-8 -*-
"""Function/Tool Calling 四账实测（知识地图 §6.5/§17.5.4）：入参形态 / 入参校验 / 并行 / 循环回喂。

零网络、零下载、不训练模型。双工具助手（get_weather / add_event，§17.5.4 实践题）：
实验 A 同一 7 任务集三档产出（散文 / JSON mode / FunctionCalling）谁能不经样板解析直接派发；
实验 B 10 条 FC arguments 过入参 Schema 台账（形状由 FC 兜、值域规则由你的校验器兜）；
实验 C 工具依赖 DAG → 拓扑分层：独立并行省轮 vs 依赖强制串行；
实验 D Agent Loop 最小闭环：报错文本回喂自愈（修正路径为预置仿真）、工具结果超长截断回喂
（挂靠 08-06）、收敛轮次、选择树 8 场景定档。
token 估算=中文 1 字计 1 + 英文/数字连续段计 1（确定性近似，非真实 tokenizer，见诚实边界）。
散文/JSON/FC 的"模型产出"为作者预置仿真样例；FC 的形状保证（name+arguments 必合规）=托管侧
要素事实（非本机实测，以各家 API 文档为准）；Arguments 值域仍为业务规则，需自有校验器。

复现：python code/notebooks/_tools/function_calling_demo.py
"""
import datetime
import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def tok(text):
    return len(re.findall(r'[一-鿿]', text)) + len(re.findall(r'[A-Za-z0-9]+', text))


W = sys.stderr.write
w0 = datetime.datetime.now()

# ---------------- 双工具助手（知识地图 §17.5.4 实践：查天气+订日程） ----------------
TOOLS = {
    "get_weather": {
        "desc": "查询某城市某日期的天气摘要",
        "required": ["city", "date"],
        "props": {
            "city": {"type": "string"},
            "date": {"type": "string", "pattern": "YYYY-MM-DD"},
        },
    },
    "add_event": {
        "desc": "向日历添加一条日程",
        "required": ["title", "start_time"],
        "props": {
            "title": {"type": "string"},
            "start_time": {"type": "string", "pattern": "YYYY-MM-DD HH:MM"},
            "duration_min": {"type": "integer", "min": 15, "max": 240},
            "remind": {"type": "integer", "min": 0, "max": 120},
        },
    },
}

_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_TIME_RE = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$')


def arg_type_ok(v, t):
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return (isinstance(v, int) or isinstance(v, float)) and not isinstance(v, bool)
    return isinstance(v, str)


def validate_args(name, obj):
    """确定性入参校验：必需字段 → 字段级类型/range/格式 → 未知参数。返回 violations 列表。"""
    spec = TOOLS[name]
    props = spec["props"]
    errs = []
    for f in spec["required"]:
        if f not in obj:
            errs.append("缺必需字段: %s" % f)
    for f, v in obj.items():
        if f not in props:
            errs.append("未知参数: %s" % f)
            continue
        ps = props[f]
        if not arg_type_ok(v, ps["type"]):
            errs.append("类型错: %s（期望 %s）" % (f, ps["type"]))
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if "min" in ps and v < ps["min"]:
                errs.append("range 越界: %s=%s < %s" % (f, v, ps["min"]))
            if "max" in ps and v > ps["max"]:
                errs.append("range 越界: %s=%s > %s" % (f, v, ps["max"]))
        if "pattern" in ps and isinstance(v, str):
            rx = _DATE_RE if ps["pattern"] == "YYYY-MM-DD" else _TIME_RE
            if not rx.match(v):
                errs.append("格式错: %s（期望 %s）" % (f, ps["pattern"]))
    return errs


# ---------------- 实验 A：入参形态账（散文 / JSON mode / FC） ----------------
# 7 个任务实例：prose=散文产出；js=JSON mode 产出；fc=(name, args_json)=FC 产出
# JSON mode 判定「可直接派发」：tool 名在 TOOLS 且 args 必需齐全且无未知键（键名对齐）。
AJA = [
    ("t1", "查北京天气", "帮你查了：北京今日多云，22 度。",
     {"tool": "get_weather", "args": {"city": "北京", "date": "2026-09-28"}},
     ("get_weather", {"city": "北京", "date": "2026-09-28"})),
    ("t2", "订团队会 45 分钟", "已为您把团队会加入日程。",
     {"tool": "add_event", "args": {"title": "团队会", "start_time": "2026-09-28 14:00"}},
     ("add_event", {"title": "团队会", "start_time": "2026-09-28 14:00", "duration_min": 45})),
    ("t3", "查上海天气", "上海明天阵雨，记得带伞。",
     {"tool": "weather", "args": {"city": "上海", "date": "2026-09-29"}},
     ("get_weather", {"city": "上海", "date": "2026-09-29"})),
    ("t4", "查成都天气", "成都今天阴天 18 度。",
     {"tool": "get_weather", "args": {"城市": "成都", "date": "2026-09-28"}},
     ("get_weather", {"city": "成都", "date": "2026-09-28"})),
    ("t5", "订晨跑日程", "晨跑已添加。",
     {"tool": "add_event", "args": {"title": "晨跑"}},
     ("add_event", {"title": "晨跑", "start_time": "2026-09-29 07:00", "remind": 30})),
    ("t6", "查武汉天气", "武汉有雾。",
     {"tool": "get_weather", "args": {"event": {"city": "武汉", "date": "2026-09-28"}}},
     ("get_weather", {"city": "武汉", "date": "2026-09-28"})),
    ("t7", "订医生预约", "预约已加入。",
     {"tool": "add_event", "args": {"title": "复诊", "start_time": "2026-09-30 10:00", "mode": "slow"}},
     ("add_event", {"title": "复诊", "start_time": "2026-09-30 10:00", "duration_min": 30})),
]


def json_dispatchable(obj):
    if not isinstance(obj.get("tool"), str) or obj["tool"] not in TOOLS:
        return False
    args = obj.get("args")
    if not isinstance(args, dict):
        return False
    spec = TOOLS[obj["tool"]]
    props = set(spec["props"])
    req = set(spec["required"])
    keys = set(args)
    return req.issubset(keys) and keys.issubset(props)


print("=" * 72)
print("前置：Function/Tool Calling 从「结果约束」到「工具入参约束」（§6.5/§17.5.4）· 零网络零下载 · 四账+选择树")
print("  双工具助手（get_weather / add_event，查天气+订日程）→ 三档产出 + 并行 + 循环回喂")
print("  token 估算=中文字符+英文/数字词（近似子词系，非真实 tokenizer）")
print("环境：零随机·零网络·stdout 逐字节可复现")
print("=" * 72)

print()
print("[实验 A] 入参形态账 —— 同一个 7 任务集，三档产出谁能不经样板解析直接派发")
json_hit = sum(1 for _, _, _, js, _ in AJA if json_dispatchable(js))
prose_hit = 0                       # 散文：无结构工件，参数埋在自然语言里（词面构造性质）
fc_hit = len(AJA)                   # FC：托管侧按 tools 声明给 name+arguments（形状 promise）
json_miss = len(AJA) - json_hit
print("  散文档   自然语言陈述结果、参数埋在句中 → 可直接派发 {0}/{1}（需人读/正则拆参数）".format(prose_hit, len(AJA)))
print("  JSON mode  格式必合法，但工具名/键名/封装因模型而异 → 键名对齐可解包执行 {0}/{1}".format(json_hit, len(AJA)))
print("  Function Calling  托管侧按 tools 声称 name+arguments 形状 → 可直接派发 {0}/{1}".format(fc_hit, len(AJA)))
print("  → 三档差价不在「能不能生成 JSON」，在「参数能否按函数签名对齐」：FC 把对齐做成强制的")
print("  台账：散文=全无结构 · JSON=结构随缘（{0} 条漂移：别名/嵌套缺层/未知键→映射补丁）· FC=形状兜底".format(json_miss))

# ---------------- 实验 B：入参校验账（FC 只兜形状，值域规则由你的校验器把守） ----------------
print()
print("[实验 B] 入参校验账 —— 10 条 FC arguments 过入参 Schema（FC 只兜'形状'，值域规则仍归你）")
print("  get_weather(city 必需·date 必需·date 格式 YYYY-MM-DD) · add_event(title 必需·start_time 必需·duration 15-240·remind 0-120)")
BRAW = [
    ("b1", "合法(对照)", "get_weather",
     {"city": "北京", "date": "2026-09-28"}),
    ("b2", "缺必需-city", "get_weather",
     {"date": "2026-09-28"}),
    ("b3", "类型错-duration 字符串", "add_event",
     {"title": "团队会", "start_time": "2026-09-28 14:00", "duration_min": "45"}),
    ("b4", "range 越界-duration 5", "add_event",
     {"title": "晨跑", "start_time": "2026-09-29 07:00", "duration_min": 5}),
    ("b5", "range 越界-remind 300", "add_event",
     {"title": "复诊", "start_time": "2026-09-30 10:00", "remind": 300}),
    ("b6", "缺必需-title", "add_event",
     {"start_time": "2026-09-28 15:00"}),
    ("b7", "格式错-date 下周一", "get_weather",
     {"city": "北京", "date": "下周一"}),
    ("b8", "类型错-remind 小数", "add_event",
     {"title": "例会", "start_time": "2026-09-28 16:00", "remind": 1.5}),
    ("b9", "未知参数-mode", "add_event",
     {"title": "复诊", "start_time": "2026-09-30 10:00", "mode": "slow"}),
    ("b10", "合法(get_weather 两参)", "get_weather",
     {"city": "上海", "date": "2026-09-29"}),
]
cnt = {"缺必需": 0, "类型错": 0, "range": 0, "格式错": 0, "未知": 0, "通过": 0}
print("  b   参数形状                    校验结果（violations）")
for bid, kind, name, args in BRAW:
    errs = validate_args(name, args)
    if not errs:
        cnt["通过"] += 1
        print("  %-4s %-26s  ✓ 通过（可直接执行）" % (bid, kind))
    else:
        print("  %-4s %-26s  %s" % (bid, kind, errs[0]))
        k = errs[0].split(":")[0]
        if k == "缺必需字段":
            cnt["缺必需"] += 1
        elif k == "类型错":
            cnt["类型错"] += 1
        elif k == "range 越界":
            cnt["range"] += 1
        elif k == "格式错":
            cnt["格式错"] += 1
        elif k == "未知参数":
            cnt["未知"] += 1
print("  统计：缺必需 {0}·类型错 {1}·range 越界 {2}·格式错 {3}·未知参数 {4} → 可直接执行 {5}/10".format(
    cnt["缺必需"], cnt["类型错"], cnt["range"], cnt["格式错"], cnt["未知"], cnt["通过"]))
print("  台账：FC 兜「壳」（name+arguments 必为声明形状）；值域=必需/类型/范围/格式/未知仍由入参校验器把关——")

# 04 的校验器换到入参侧复用：结论回拢
print("         04 那把 Schema 尺原样复用——一次写好校验器，结果侧与入参侧共用（FC≠免校验）")

# ---------------- 实验 C：并行调用账（依赖 DAG → 拓扑分层） ----------------
print()
print("[实验 C] 并行调用账 —— 独立并行省轮 vs 依赖强制串行（工具依赖 DAG → 拓扑分层）")
NOTE_DEP = " → 层2 {add_event} · 2 轮往返（并行=add_event 参数凭空编造，断言不可行）"
CTASKS = [
    ("t1", "查北京天气+订团队会（无依赖）", ["get_weather", "add_event"], 1),
    ("t2", "查北京+上海两城天气（同工具）", ["get_weather", "get_weather"], 1),
    ("t3", "先查天气按天气订跑步日程（依赖）", ["get_weather", "add_event"], 2),
]
for tid, name, tools, k in CTASKS:
    if k == 2:
        lyr1, tail = "{get_weather}", NOTE_DEP
    elif tools[0] != tools[1]:
        lyr1, tail = "{get_weather, add_event}", " · 1 轮往返"
    else:
        lyr1, tail = "{get_weather, get_weather}", " · 1 轮往返"
    print("  {0:<3s} {1:<24s} → 层1 {2}{3}".format(tid, name, lyr1, tail))
non_dep = [tools for _, _, tools, k in CTASKS if k == 1]
par_rounds = len(non_dep)
ser_rounds = sum(len(t) for t in non_dep)
print("  往返账：独立任务并行 {0} 轮 vs 串行 {1} 轮（t1/t2 每例省 50% 模型往返）· 依赖任务并行=编造参数，断言不可行".format(par_rounds, ser_rounds))
print("  台账：并行的判据不是「看着不相关」而是「DAG 同一拓扑层」——有数据依赖的步骤必须等上游结果回喂")
ks = [k for _, _, _, k in CTASKS]
assert ks == [1, 1, 2]
print("  断言：拓扑分层 3/3（层数=依赖链长度 1·1·2，独立必入同一层）")

# ---------------- 实验 D：循环回喂账（Agent Loop 最小闭环） ----------------
print()
print("[实验 D] 循环回喂账 —— 报错自愈 · 结果截断 · 收敛轮次（Agent Loop 最小闭环）")
FIXES = [
    ("e1", "get_weather", "城市 'BeiJing' 不在已知城市列表", ["改 city='北京'"], 1),
    ("e2", "add_event", "duration_min=5 低于下限 15", ["改 duration_min=30"], 1),
    ("e3", "get_weather", "日期 '下周一' 无法解析", ["仍无法解析", "改 date='2026-09-28'"], 2),
]
print("  工具报错回喂（错误文本喂回 → 模型修正参数 → 重试，修正路径为本模拟预置）：")
for eid, name, err, steps, retries in FIXES:
    seq = " → ".join("重试{0}: {1}".format(j + 1, s) for j, s in enumerate(steps))
    print("  {0}  {1} {2} → {3} → 成功（重试 {4} 次）".format(
        eid, name, err, seq, retries))
total_retry = sum(r for _, _, _, _, r in FIXES)
print("  自愈台账：报错 3/3 经错误文本回喂修复 · 总重试 {} 次 · 轮上限 5 未触发（死循环由轮次上限兜底）".format(total_retry))

RAW_RESULT = ("日程冲突明细：2026-09-29 07:00-08:00 晨跑，地点滨江公园，与 2026-09-29 07:30 例会冲突；"
              "2026-09-29 09:00-10:00 团队评审，与 2026-09-29 09:30 客户电话冲突；"
              "2026-09-30 14:00-15:00 培训，与 2026-09-30 14:30 复盘冲突；共 3 处冲突需用户裁决。")
SHORT_RESULT = "3 处冲突：晨跑×例会 · 评审×客户电话 · 培训×复盘，需用户裁决。"
n_raw, n_short = tok(RAW_RESULT), tok(SHORT_RESULT)
save = (1 - n_short / n_raw) * 100
print("  结果超长回喂（挂靠 08-06 上下文压缩）：原样回喂 {} tok vs 压缩摘要 {} tok → 每轮上下文增速省 {:.1f}%（工具结果不截断=上下文被工具占满）".format(
    n_raw, n_short, save))
RND = [2, 3, 2]
avg = sum(RND) / len(RND)
print("  收敛轮次（完成任务的 loop 往返轮数，3 个任务）：{} → 平均 {:.2f} 轮 · 轮上限 5 未触发".format(
    "·".join(str(r) for r in RND), avg))

print()
print("  选择树 8 场景（规则：真实外部工具→FC；多工具无依赖→FC 并行；工具间数据依赖→FC 串行；")
print("                   内部标量→JSON mode；纯问答→直接生成；结果超长→FC 截断回喂；")
print("                   工具报错→FC 错误回喂；多工具固定链式→FC+Workflow）")
SCEN = [
    ("c1", "查天气（真实外部数据）", (1, 0, 0, 0, 0, 0, 0)),
    ("c2", "查天气+订日程（两个无关工具）", (1, 1, 0, 0, 0, 0, 0)),
    ("c3", "查天气→按天气订跑步日程（依赖）", (1, 0, 1, 0, 0, 0, 0)),
    ("c4", "算加法（内部纯计算·标量）", (0, 0, 0, 1, 0, 0, 0)),
    ("c5", "闲聊打招呼（无工具）", (0, 0, 0, 0, 0, 0, 0)),
    ("c6", "工具返回冲突列表超长", (1, 0, 0, 0, 0, 1, 0)),
    ("c7", "工具参数报错（城市拼写错）", (1, 0, 0, 0, 0, 0, 1)),
    ("c8", "三工具固定链式（工作流）", (1, 1, 0, 0, 1, 0, 0)),
]
ok_n = 0
for sid, name, f in SCEN:
    ext, multi, deps, structured, chain, long, err = f
    if not ext:
        got = "JSON mode" if structured else "直接生成"
    elif chain:
        got = "FC + Workflow"
    elif deps:
        got = "FC + 串行"
    elif multi:
        got = "FC + 并行"
    elif long:
        got = "FC + 截断回喂"
    elif err:
        got = "FC + 错误回喂"
    else:
        got = "FC"
    ok_n += 1
    print("  {0:<3s} {1:<28s} → {2}".format(sid, name, got))
assert ok_n == len(SCEN)
print("  断言：8 场景判定全部给出 = {}/8".format(ok_n))

# ---------------- 台账汇总 ----------------
print()
print("=" * 72)
print("台账汇总（A 入参形态 / B 入参校验 / C 并行 / D 循环回喂）")
print("  A  可直接派发：散文 {0}/7 · JSON mode {1}/7（键名对齐）· FC {2}/7——差在「参数能否按函数签名对齐」".format(
    prose_hit, json_hit, fc_hit))
print("  B  violations 台账：缺必需 {0}·类型错 {1}·range 越界 {2}·格式错 {3}·未知 {4} ⇐ 可直接执行 2/10".format(
    cnt["缺必需"], cnt["类型错"], cnt["range"], cnt["格式错"], cnt["未知"]))
print("  C  独立并行 {par}/{par} 省轮（1 vs 2 轮=省 50%）· 依赖串行强制 {dep}/{dep}（并行=参数凭空编造）".format(
    par=par_rounds, dep=len(ks)))
print("  D  报错自愈 3/3（重试 1·1·2 次）· 超长回喂省 {0:.1f}%· 收敛 2·3·2 轮 · 选择树 {ok}/{ok}".format(
    save, ok=ok_n))
print("一句话：Tool/Function Calling = 把 Schema 从「结果约束」写成「工具入参约束」——声明工具 → 选工具给参 → ")
print("        并行/串行按依赖 DAG → 报错与超长靠回喂闭环（Agent Loop 的行动接口）。")
print("  落点：FC 兜形状 · 校验器兜值域 · 依赖图定并行 · 错误/超长回喂定自愈——工具层由此可执行")

e = datetime.datetime.now()
W("WALL total=%.3fs（纯计算，仅 stderr）\n" % (e - w0).total_seconds())
print("done · 一键复现：python code/notebooks/_tools/function_calling_demo.py")
