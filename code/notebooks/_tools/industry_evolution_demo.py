# -*- coding: utf-8 -*-
"""
14-垂直领域 00 篇探针：通用到行业演化链（§14.0/§14.1 · 🔴 会画架构）
垂直模型 = 通用模型 + 领域资产（数据/知识/流程/合规）
纯 stdlib · 零网络 · 零随机 · 零第三方：全部判定=确定性函数，断言不过立即抛错。
用法: python code/notebooks/_tools/industry_evolution_demo.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

PASS = 0
FAIL = 0


def ok():
    global PASS
    PASS += 1


bad = lambda: globals().__setitem__("FAIL", FAIL + 1)

CHAIN = ["通用底座", "领域CPT", "领域SFT", "领域RL", "RAG实时知识", "Agent工作流", "产品化"]
PAYLOAD = {
    "领域CPT": "补知识(领域语料:医学教材/财报/代码库/法条)",
    "领域SFT": "补行为(问答对/报告模板/专家改写)",
    "领域RL": "补偏好(专家打分/可验证标准)",
    "RAG实时知识": "实时知识(指南/台账/私有库)",
    "Agent工作流": "检索+调用内部系统+流程编排",
    "产品化": "权限/审计/人审/评测/合规",
}
PRODUCT_KIT = ["权限", "审计", "人审", "评测", "合规"]


def hop_for(need):
    # 需求关键词 -> 演化链主流段（判型：先看症状落到哪一段）
    m = {
        "语料": "CPT", "术语": "CPT", "教材": "CPT", "法条": "CPT",
        "问答对": "SFT", "报告模板": "SFT", "改写": "SFT",
        "专家打分": "RL", "可验证": "RL",
        "更新": "RAG", "实时": "RAG", "私有库": "RAG", "台账": "RAG",
        "流程": "Agent", "内部系统": "Agent", "编排": "Agent",
    }
    for k, v in m.items():
        if k in need:
            return v
    return "工程"


print("== A 造账：七段链怎么画（§14.1 会画架构）==")
try:
    assert len(CHAIN) == 7 and CHAIN[0] == "通用底座" and CHAIN[5] == "Agent工作流" and CHAIN[6] == "产品化"
    ok()
    print("A1 七段链 7/7 排序正确（通用底座→产品化）：", " → ".join(CHAIN))
except AssertionError:
    bad(); print("A1 FAIL")

try:
    assert PAYLOAD["领域CPT"].startswith("补知识")
    assert PAYLOAD["领域SFT"].startswith("补行为")
    assert PAYLOAD["领域RL"].startswith("补偏好")
    ok()
    print("A2 三补映射 3/3（CPT补知识·SFT补行为·RL补偏好）：", PAYLOAD["领域CPT"], "|", PAYLOAD["领域SFT"], "|", PAYLOAD["领域RL"])
except AssertionError:
    bad(); print("A2 FAIL")

try:
    assert PAYLOAD["RAG实时知识"].startswith("实时知识")
    assert PAYLOAD["Agent工作流"].startswith("检索")
    assert len(PRODUCT_KIT) == 5
    ok()
    print("A3 工程双段+产品化五件套 5/5：RAG→", PAYLOAD["RAG实时知识"], "· Agent→", PAYLOAD["Agent工作流"], "· 产品化=", "/".join(PRODUCT_KIT))
except AssertionError:
    bad(); print("A3 FAIL")

NEED_HOP = {
    "病历术语密度高·医学语料齐": "CPT",
    "标注问答对/报告模板齐": "SFT",
    "专家打分/可验证标准": "RL",
    "指南/私有库持续更新": "RAG",
    "要调内部系统/流程编排": "Agent",
}
try:
    assert all(hop_for(k) == v for k, v in NEED_HOP.items())
    ok()
    print("A4 五需求判型 5/5（先判型再动手）：", ["病历语料→CPT", "问答对→SFT", "专家打分→RL", "私有库更新→RAG", "流程→Agent"])
except AssertionError:
    bad(); print("A4 FAIL")

print("== B 资产账：四栏模板 × 四行业（§14.0 稀缺资产）==")
ASSETS = ["领域数据", "领域知识库", "业务流程建模", "合规与责任边界"]
SECTOR = {
    "医疗": {"领域数据": 3, "领域知识库": 2, "业务流程建模": 2, "合规与责任边界": 3, "难": "领域数据"},
    "金融": {"领域数据": 3, "领域知识库": 2, "业务流程建模": 2, "合规与责任边界": 2, "难": "领域数据"},
    "法律": {"领域数据": 2, "领域知识库": 3, "业务流程建模": 1, "合规与责任边界": 2, "难": "领域知识库"},
    "代码": {"领域数据": 1, "领域知识库": 1, "业务流程建模": 3, "合规与责任边界": 1, "难": "业务流程建模"},
}
try:
    assert ASSETS == ["领域数据", "领域知识库", "业务流程建模", "合规与责任边界"]
    ok()
    print("B1 四类领域资产 4/4（§14.0 真正稀缺的四样）：", " / ".join(ASSETS))
except AssertionError:
    bad(); print("B1 FAIL")

total = {s: sum(v for k, v in d.items() if k in ASSETS) for s, d in SECTOR.items()}
try:
    assert total == {"医疗": 10, "金融": 9, "法律": 8, "代码": 6}
    assert total["医疗"] >= total["金融"] >= total["法律"] >= total["代码"]
    ok()
    print("B2 四行业难度总分（每栏1-3，越大越稀缺）：医疗10 · 金融9 · 法律8 · 代码6（模型能力可买，资产余额难造）")
except AssertionError:
    bad(); print("B2 FAIL")

MAP = {
    "领域数据": ["04-01 预训练数据工程", "治理管线"],
    "领域知识库": ["08-01 检索", "08-03 重排", "11-05 RAG评测"],
    "业务流程建模": ["09-02 主循环", "09-04 行动推理", "09-08 多Agent"],
    "合规与责任边界": ["12-安全", "11-07 安全评测"],
    "领域SFT行为": ["04-06 全参/PEFT"],
    "领域RL偏好": ["04-05 RL数据·可验证奖励"],
}
try:
    assert len(MAP) == 6 and all(k in MAP and len(MAP[k]) >= 1 for k in MAP)
    ok()
    print("B3 资产→挂靠章 6/6（四大资产+两训练行为对应 04/08/09/11/12 已测锚点）：")
    for k, v in MAP.items():
        print("      ", k, "→", " / ".join(v))
except AssertionError:
    bad(); print("B3 FAIL")

try:
    hard = {s: max((v for k, v in d.items() if k in ASSETS)) for s, d in SECTOR.items()}
    assert all(SECTOR[s]["难"] in ASSETS and SECTOR[s][SECTOR[s]["难"]] == hard[s] for s in SECTOR)
    ok()
    print("B4 各行业'最难资产' 4/4（难的不是模型是资产）：",
          {s: SECTOR[s]["难"] for s in SECTOR})
except AssertionError:
    bad(); print("B4 FAIL")

print("== C 选择账：该不该为行业重训（跳步决策 + 成本档）==")
NEED_COMBO = {
    "医院助手(病历术语·报告模板·禁止替代诊断)": ["CPT", "SFT", "合规"],
    "客服答疑(产品手册实时更新·FAQ)": ["RAG"],
    "信贷审核(报表结构化·反洗钱留痕)": ["SFT"],
    "法律检索(法条更新·判例引用可溯源)": ["RAG"],
    "代码助手(模型已强·仓库索引+CI)": ["工程", "合规"],
}


def combo_for(need):
    c = []
    if "病历" in need or "医学语料" in need:
        c.append("CPT")
    if "报告" in need or "模板" in need or "结构化" in need:
        c.append("SFT")
    if "实时" in need or "更新" in need or "手册" in need:
        c.append("RAG")
    if "索引" in need or "CI" in need:
        c.append("工程")
    if "助手" in need or "人工" in need or "禁止替代" in need:
        c.append("合规")
    return c or ["RAG"]


try:
    assert all(combo_for(k) == v for k, v in NEED_COMBO.items())
    ok()
    print("C1 五行业需求→方案组合 5/5：", {k.split("(")[0]: "/".join(v) for k, v in NEED_COMBO.items()})
except AssertionError:
    bad(); print("C1 FAIL")

ENG, SFT, CPT = 1.0, 10.0, 100.0
try:
    assert SFT / ENG == 10.0 and CPT / SFT == 10.0 and CPT / ENG == 100.0
    ok()
    print("C2 三档成本级（教学层数量级·真实取决于算力与语料量）：纯工程×1 · 领域SFT×10 · 领域CPT×100 —— 能工程先工程，能SFT不CPT")
except AssertionError:
    bad(); print("C2 FAIL")


def cost_level(combo):
    if "CPT" in combo:
        return "CPT级×100"
    if "SFT" in combo:
        return "SFT级×10"
    return "工程级×1"


try:
    lv = {k.split("(")[0]: cost_level(v) for k, v in NEED_COMBO.items()}
    assert lv == {"医院助手": "CPT级×100", "客服答疑": "工程级×1", "信贷审核": "SFT级×10",
                  "法律检索": "工程级×1", "代码助手": "工程级×1"}
    ok()
    print("C3 五需求成本档 5/5：", lv, "（法律检索只上RAG不训模型=引用校验是工程不是重训）")
except AssertionError:
    bad(); print("C3 FAIL")

print("== D 合规与评测账：行业红线 + 评测四件套（§14.8 点到）==")
try:
    AVAL = ["通用基准", "领域真题", "专家盲评", "安全用例"]
    assert len(AVAL) == 4
    ok()
    print("D1 领域评测集四件套 4/4：", "/".join(AVAL), "（评测集本身也是商业资产）")
except AssertionError:
    bad(); print("D1 FAIL")

try:
    REG = ["数据来源合法", "脱敏", "模型备案/审计", "输出留痕", "人审+风险评估"]
    assert len(REG) == 5
    ok()
    print("D2 合规五动作 5/5：", "/".join(REG))
except AssertionError:
    bad(); print("D2 FAIL")

REDLINE = {
    "医疗": "不能替代执业判断→辅助+人工确认",
    "金融": "不得主动荐股·数值不许编→RAG+校验+落库留痕",
    "法律": "引用必须可溯源→GraphRAG+引用校验",
    "代码": "变更过测试+安全审查→CI+审查门禁",
}


def blocked(s, action):
    if s == "医疗" and "诊断" in action and "人工" not in action:
        return REDLINE["医疗"]
    if s == "金融" and "荐股" in action:
        return REDLINE["金融"]
    if s == "法律" and "引用" in action and "出处" not in action:
        return REDLINE["法律"]
    if s == "代码" and "测试" not in action and ("提交" in action or "合并" in action):
        return REDLINE["代码"]
    return None


DANGER = {
    "医疗": "模型输出确定性诊断结论",
    "金融": "给用户荐股说这只会涨",
    "法律": "引用判例但追溯不到原文依据",
    "代码": "跳过CI直接合并代码",
}
try:
    res = {s: blocked(s, a) for s, a in DANGER.items()}
    assert all(res.values()) and all(r.startswith(rl.split("→")[0]) for r, rl in zip(res.values(), (REDLINE[s] for s in REDLINE)))
    ok()
    print("D3 四行业危险动作→红线拦截 4/4：", {s: v.split("→")[0] for s, v in res.items()}, "（拦截≠拒绝人审，是强制升人工）")
except AssertionError:
    bad(); print("D3 FAIL")

print()
print(f"SUMMARY  PASS {PASS} · FAIL {FAIL}" if FAIL == 0 else f"SUMMARY  PASS {PASS} · FAIL {FAIL}  [需修复]")
