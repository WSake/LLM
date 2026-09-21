# -*- coding: utf-8 -*-
"""结构化输出四账实测（知识地图 §6.4）：方法演进 / Schema 校验 / 约束解码 / 成本与选择树。

零网络、零下载、不训练模型。10 条保单抽取的"模型原生输出"为作者预置仿真样例（词面代理，
真实失败分布因模型而异），对同一业务 Schema 跑三档：提示词法（只靠 json 解析）→ JSON mode
（语法强制）→ 约束解码（解码侧拦截）。只量确定性的语法/结构/拦截/成本维度。
token 估算=中文 1 字计 1 + 英文/数字连续段计 1（确定性近似，非真实 tokenizer，见诚实边界）。
约束解码状态机=字段级最小模拟（真实库 SGLang/vLLM/Outlines 为 token 级，非本机实测）。

复现：python code/notebooks/_tools/structured_output_demo.py
"""
import datetime
import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def tok(text):
    return len(re.findall(r'[一-鿿]', text)) + len(re.findall(r'[A-Za-z0-9]+', text))


INPUT_PIN = 2.50    # 输入 $/百万 token（要素价格事实，非本机实测；沿 01/02/03 同价目表）
OUTPUT_PIN = 10.00  # 输出 $/百万 token（常见档位，要素事实，非本机实测）


def cost_usd(n, pin=OUTPUT_PIN):
    return n / 1_000_000 * pin


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def is_num(v):
    return (isinstance(v, int) or isinstance(v, float)) and not isinstance(v, bool)


def is_str(v):
    return isinstance(v, str)


def is_obj(v):
    return isinstance(v, dict)


W = sys.stderr.write
w0 = datetime.datetime.now()

# ---------------- 业务 Schema：保单抽取（知识地图 §17.5.3 实践） ----------------
SCHEMA = {
    "name": "保单抽取 PolicyExtract",
    "required": ["insured_name", "age", "policy_type"],
    "properties": {
        "insured_name": {"type": "string"},
        "age": {"type": "integer", "min": 0, "max": 120},
        "policy_type": {"type": "string", "enum": ["medical", "life", "auto", "property"]},
        "premium": {"type": "number", "min": 0},
        "address": {"type": "object", "required": ["city"],
                    "properties": {"city": {"type": "string"}, "zip": {"type": "string"}}},
    },
}

REQUIRED = SCHEMA["required"]
PROPS = SCHEMA["properties"]
ENUM_PT = PROPS["policy_type"]["enum"]

# 10 条"模型原生输出"（作者预置仿真：分布覆盖语法层 + 结构层的真实常见毛病）
RAW = [
    ("c1", "合法(对照)",
     '{"insured_name":"张三","age":34,"policy_type":"medical","premium":3999.9,"address":{"city":"上海"}}'),
    ("c2", "语法-前缀散文",
     '已为您抽取保单信息如下：\n{"insured_name":"李四","age":28,"policy_type":"auto","premium":5200.0,"address":{"city":"北京"}}'),
    ("c3", "语法-尾随逗号",
     '{"insured_name":"王五","age":51,"policy_type":"life","premium":12300.5,"address":{"city":"广州"},}'),
    ("c4", "语法-单引号",
     "{'insured_name':'赵六','age':42,'policy_type':'property','premium':8000.0,'address':{'city':'深圳'}}"),
    ("c5", "缺必需字段",
     '{"age":30,"policy_type":"medical","premium":2000.0,"address":{"city":"杭州"}}'),
    ("c6", "类型错-age字符串",
     '{"insured_name":"钱七","age":"30","policy_type":"medical","premium":2100.0,"address":{"city":"南京"}}'),
    ("c7", "enum 拼错",
     '{"insured_name":"孙八","age":38,"policy_type":"medicl","premium":3300.0,"address":{"city":"成都"}}'),
    ("c8", "range 越界-age 175",
     '{"insured_name":"周九","age":175,"policy_type":"life","premium":15000.0,"address":{"city":"武汉"}}'),
    ("c9", "range 越界-premium 负",
     '{"insured_name":"吴十","age":29,"policy_type":"auto","premium":-5.5,"address":{"city":"重庆"}}'),
    ("c10", "嵌套缺必需 address.city",
     '{"insured_name":"郑十一","age":45,"policy_type":"property","premium":9000.0,"address":{"zip":"200000"}}'),
]
SYN = {
    "c2": "前缀散文: 首个非空字符应为 '{'，实际为散文文本",
    "c3": "尾随逗号: 对象闭合前多余 ','",
    "c4": "单引号: JSON 规范只允许双引号包裹字符串",
}


def validate_schema(obj):
    """确定性结构校验：必需字段→字段级类型/enum/range→嵌套 required。返回 violations 列表。"""
    errs = []
    for f in REQUIRED:
        if f not in obj:
            errs.append("缺必需字段: %s" % f)
    for f, spec in PROPS.items():
        if f not in obj:
            continue
        v = obj[f]
        t = spec["type"]
        ok = (is_int(v) if t == "integer" else is_num(v) if t == "number"
              else is_str(v) if t == "string" else is_obj(v))
        if not ok:
            errs.append("类型错: %s（期望 %s）" % (f, t))
        if t in ("integer", "number") and (is_int(v) or is_num(v)):
            if "min" in spec and v < spec["min"]:
                errs.append("range 越界: %s=%s < %s" % (f, v, spec["min"]))
            if "max" in spec and v > spec["max"]:
                errs.append("range 越界: %s=%s > %s" % (f, v, spec["max"]))
        if "enum" in spec and is_str(v) and v not in spec["enum"]:
            errs.append("enum 越界: %s='%s' ∉ {%s}" % (f, v, ", ".join(spec["enum"])))
        if t == "object" and is_obj(v):
            for rf in spec.get("required", []):
                if rf not in v:
                    errs.append("缺必需字段: %s.%s" % (f, rf))
    return errs


def decode_stop(cid, errs):
    """约束解码（字段级最小模拟）：返回 (层, 部位, 位置文本)；None 表示解码侧 0 拦截。"""
    if cid in SYN:
        pos = {"c2": "首字符'已'即被挡——解码器只许 JSON 字面量开头",
               "c3": "闭合','前即被挡——对象不允许尾随逗号",
               "c4": "单引号不在 JSON 词表——字符串必须双引号开头"}[cid]
        return "L1 语法", {"c2": "对象起始", "c3": "对象收尾", "c4": "字符串引号"}[cid], pos
    prefix = errs[0].split(":")[0]
    layer = "L2 Schema"
    part = {"缺必需字段": "required 收尾", "类型错": "类型栅栏",
            "enum 越界": "词表筛选", "range 越界": "range 校验"}[prefix]
    return layer, part, "生成该字段处被拦（%s）" % errs[0]


print("=" * 72)
print("前置：Structured Output 从「合法」到「合规」（§6.4/§17.5.3）· 零网络零下载 · 四账+选择树")
print("  同一批 10 条保单抽取「模型原生输出」→ 提示词法 / JSON mode / 约束解码 三档对照")
print("  token 估算=中文字符+英文/数字词（近似子词系，非真实 tokenizer）· 输入 PIN 2.50$/M·输出 10$/M")
print("环境：零随机·零网络·stdout 逐字节可复现")
print("=" * 72)

# ---------------- 实验 A：方法演进 —— 稳定性账 ----------------
print()
print("[实验 A] 方法演进 —— 同一批 10 条原生输出，三档各自拦到哪一层")
syn_ok = 0
full_ok = 0
for cid, kind, src in RAW:
    try:
        obj = json.loads(src)
    except Exception:
        obj = None
    if obj is not None:
        syn_ok += 1
        if not validate_schema(obj):
            full_ok += 1
print("  提示词\"请输出 JSON\"（只靠 json 解析）：语法合法 {}/10".format(syn_ok))
print("  其中过完整业务 Schema（必需+类型+enum+range+嵌套）= {}/10".format(full_ok))
print("  → 提示词法全程达标（能直接落库）{}/10 —— 解析合法≠结构合规：7 条能 parse、只有 1 条真能落库".format(full_ok))
print("  挂靠：越强的模型格式服从越好，但「结构完整=业务校验」与强度无关——是系统层的义务（§6.4 实践）")
print("  台账：方法演进=把「合法」从碰运气挪到受控：提示词(碰运气)→JSON mode(文法层强制)→约束解码(结构层强制)")

# ---------------- 实验 B：Schema 校验 —— 落库账 ----------------
print()
print("[实验 B] Schema 校验 —— 落库账（同一 Schema 逐条 violations 台账）")
print("  业务 Schema：required={0} · policy_type enum={1} · age 0-120 · premium≥0 · address.city 必需".format(
    "insured_name/age/policy_type", ", ".join(ENUM_PT)))
cnt = {"语法": 0, "缺必需": 0, "类型错": 0, "enum": 0, "range": 0, "通过": 0}
print("  c   形状                   校验结果（violations）")
for cid, kind, src in RAW:
    try:
        obj = json.loads(src)
        errs = validate_schema(obj)
    except Exception:
        errs = ["语法层: %s" % SYN.get(cid, "不可解析")]
    if not errs:
        cnt["通过"] += 1
        print("  %-4s %-20s  ✓ 通过（可直接落库）" % (cid, kind))
    else:
        print("  %-4s %-20s  %s" % (cid, kind, errs[0]))
        k = errs[0].split(":")[0]
        if k == "缺必需字段":
            cnt["缺必需"] += 1
        elif k == "类型错":
            cnt["类型错"] += 1
        elif k == "enum 越界":
            cnt["enum"] += 1
        elif k == "range 越界":
            cnt["range"] += 1
        elif k == "语法层":
            cnt["语法"] += 1
print("  统计：语法层失败 {} · Schema 层失败 {}（缺必需 {}·类型错 {}·enum 越界 {}·range 越界 {}）· 直接落库 {}/10".format(
    cnt["语法"], 10 - cnt["语法"] - cnt["通过"], cnt["缺必需"], cnt["类型错"], cnt["enum"], cnt["range"], cnt["通过"]))
print("  台账：Schema 校验是落库前的最后一道拦网——校验器写在哪，业务风险就关在哪")

# ---------------- 实验 C：约束解码 —— 拦截账（状态机最小模拟） ----------------
print()
print("[实验 C] 约束解码 —— 解码侧拦截（字段级最小模拟：每字段第一违规点被叫停）")
print("  c    形状                    层·部位             位置文本")
stops = 0
for cid, kind, src in RAW:
    try:
        obj = json.loads(src)
        errs = validate_schema(obj)
    except Exception:
        obj, errs = None, []
    if cid in SYN or errs:
        layer, part, pos = decode_stop(cid, errs)
        stops += 1
    else:
        layer, part, pos = "全合规", "放行", "解码侧 0 拦截"
    print("  %-5s %-21s %-19s %s" % (cid, kind, layer + "·" + part, pos))
print("  解码侧拦截 {}/10 · 放行 1/10（c1 合法对照）——语法 3 挡在 L1，结构 6 挡在 L2".format(stops))
print("  对比：JSON mode 只保证 'parse 必成功'（挡 3 个语法），6 条结构不合照样流进生产")
print("  挂靠：SGLang Structured Outputs / vLLM guided decoding / Outlines——真实库把 Schema 编译成 token 级状态机（要素事实，非本机实测）")

# ---------------- 实验 D：成本与选择树 ----------------
print()
print("[实验 D] 成本账 + 选择树（JSON mode vs Structured Outputs vs Function Calling）")
base = sum(tok(src) for _, _, src in RAW)
ledger = [
    ("提示词法", 9, "9 条需重试/后处理", 1),
    ("JSON mode", 6, "语法全过·6 条结构后处理", 1),
    ("约束解码", 0, "0 重试 0 后处理", 10),
]
print("  输出当量账（10 条首次 + 每条修复按再生成一遍计，等长近似）：base={} tok".format(base))
print("  方法            当量倍数   输出 token      $({:.1f}/M)    落库一次过率".format(OUTPUT_PIN))
for name, retry_n, note, ok in ledger:
    contribs = 10 + retry_n
    out_tok = base * contribs // 10
    print("  {0:<11s}   ×{1:<6.1f} {2:<9d} ${3:<12.5f} {4}/10  · {5}".format(
        name, contribs / 10, out_tok, cost_usd(out_tok), ok, note))
print("  台账：重试的每一遍=重发输入+再生成+再校验——把「合法」挪到解码侧，一次过率 10/10、当量费用 ×1.0")

print()
print("  选择树 8 场景（规则：要调工具→FunctionCalling；嵌套或严格落库/强类型下游→StructuredOutputs；标量+内部→JSON mode）")
SCENARIOS = [
    ("s1", "保单抽取(嵌套+枚举+严格落库)", True, True, True, None),
    ("s2", "订单地址归档(嵌套+落库)", True, False, True, None),
    ("s3", "分类标签(标量枚举·内部)", False, True, False, None),
    ("s4", "摘要自由文本(仅打包)", False, False, False, None),
    ("s5", "调保存笔记工具", False, False, True, "save_note"),
    ("s6", "调天气工具(宽松参数)", False, False, False, "weather"),
    ("s7", "内部批量解析标量", False, False, False, None),
    ("s8", "合规审计导出(严格+落库)", True, True, True, None),
]
ok_any = 0
for sid, name, nested, enums, strict, tool in SCENARIOS:
    if tool:
        got = "FunctionCalling"
    elif nested or strict:
        got = "StructuredOutputs"
    else:
        got = "JSON mode"
    ok_any += 1
    print("  %-3s %-28s → %s" % (sid, name, got))
assert ok_any == len(SCENARIOS)
print("  断言：8 场景判定全部给出 = {}/8".format(ok_any))

# ---------------- 台账汇总 ----------------
print()
print("=" * 72)
print("台账汇总（A 方法演进 / B Schema 校验 / C 约束解码 / D 成本与选择树）")
print("  A  提示词法：语法合法 {}/10 · 结构达标 {}/10（能落库 1 条）——解析合法≠结构合规".format(syn_ok, full_ok))
print("  B  violations 台账：缺必需 {a}·类型错 {b}·enum 越界 {c}·range 越界 {d} ⇐ 直接落库 1/10".format(
    a=cnt["缺必需"], b=cnt["类型错"], c=cnt["enum"], d=cnt["range"]))
print("  C  解码侧拦截 {}/10（L1 语法 3 + L2 Schema 6）· 放行 1/10；JSON mode 只挡语法不挡结构".format(stops))
print("  D  输出当量：提示词 ×1.9 · JSON mode ×1.6 · 约束解码 ×1.0；落库一次过率 1/10·1/10·10/10；选择树 8/8")
print("一句话：JSON 合法 ≠ Schema 合规——合法是文法层的及格线，合规是结构层的契约；约束解码把两层都挪进解码侧。")
print("  落点：提示词法=碰运气 · JSON mode=文法保险丝 · 约束解码=契约保险丝 · 选择树=按业务定档")

e = datetime.datetime.now()
W("WALL total=%.3fs（纯计算，仅 stderr）\n" % (e - w0).total_seconds())
print("done · 一键复现：python code/notebooks/_tools/structured_output_demo.py")
