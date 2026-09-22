#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""09-Agent体系 · 09-Computer-Use（知识地图 §9.9 Computer Use）探针：
§9.9 原话："是什么：Agent 直接操作操作系统 GUI（点击/输入/截图），无需 API。
代表：Anthropic Computer Use（2024-10）、OpenAI Operator（2025-01，云端浏览器）、
UI-TARS（字节开源端到端 UI Agent）、OS-Copilot（研究基线）。
技术点：截图→视觉理解→坐标/动作输出→执行；难点在 GUI 像素级理解与误操作控制
（安全护栏）。评测：OSWorld、V* Bench、Mind2Web。"
四账 A-D 实测（全部=作者预置仿真 + 本机确定性计算）：
  A 为什么要 GUI（无需 API）——3 个任务（旧版计费系统桌面端/运营后台内网网页/有开放
    API 的内部服务）× 三种接法前置（Function Calling 要接口文档 · MCP 要 Server ·
    Computer Use 只要屏幕看得见）——FC 可用 1/3、MCP 1/3、Computer Use 3/3：没有
    接口的世界（老桌面系统/第三方 SaaS）只有 GUI 能进；有 API 时 API 1 次调用 vs GUI
    平均 4 轮小循环（账 B）
  B 截图→视觉理解→动作→执行管线——虚拟"偏好设置"窗口（7 元素元素表驱动），任务
    「把亮度调到 80%」：每轮 截图(采样视图)→视觉理解(OCR 定位亮度行的 [ + ] 控件)
    →动作输出(click(+))→执行(亮度 +10)→新截图，4 步收敛（3 次 +1 次保存）＝一次
    正确操作=多次小循环，每步一个失败点
  C 像素级理解与坐标精度（难点一）——同一「立即购买」按钮 × 4 屏幕（默认/缩放125%/
    窗口平移/纯图标版）× 3 定位法——固定坐标 1/4（只默认屏中，缩放/平移/图标全漂=
    坐标是窗口与分辨率的函数，记忆基准点=猜运气）· OCR 文本定位 3/4（文字按钮缩放
    平移都能读文本重定位，纯图标无文本=死）· 视觉语义定位 4/4（认形状+标签，图标也
    认）——GUI 的坐标不是稳定契约，像素级理解才是（16-Qwen2.5-VL/UI-TARS 路线要
    『看图找元素』的原因）
  D 误操作控制与安全护栏（难点二+应用落点）——GUI 无 schema 可校验，护栏从 API 的
    『参数校验』退化成『确认框文本匹配+高危动作人工审核+限额+快照回滚』：6 个触点
    （删报表/付款/出厂设置/永久操作/外链/诱导『跳过确认』钮）三档护栏——无护栏 6/6
    直出（误删报表落盘）→ 审核+确认框匹配 2/6（拦危险确认『确定』+永久操作转人工，
    剩 2 条低危诱导漏网）→ 全配(+限额+快照回滚) 0/6（付款限额拦、误触可快照恢复）=
    GUI 误操作必须『可逆』（OSWorld 评测『动作可逆/影响可控』是独立维度）
口径：屏幕状态/元素表/坐标/护栏判定 = 作者预置仿真（非真实 Computer Use 系统运行·
     不 import 任何框架）；A-D 统计 = 本机确定性计算（纯 stdlib · 零随机 · 零网络 ·
     stdout 逐字节可复现；wall-clock 只进 stderr）。代表产品（Claude Computer Use/
     Operator/UI-TARS/OS-Copilot）与评测（OSWorld/V* Bench/Mind2Web）= 要素事实
     （写作环境无外网未复核，非本机实测）；browser-use/Playwright MCP = 下一站
     10-浏览器Agent 的框架点。
关联：08-多智能体（把 loop 拆成 N 个 vs 本篇把 loop 接到 GUI=手脚延伸）· 07-Agent安全
      （六护栏→本篇账 D GUI 版退化矩阵）· 05-Coding-Agent（Action=写代码 vs 本篇
      Action=点屏幕）· 02-Agent-Loop（六状态机=本管线骨架）· 06 章 03-Context /
      10-多模态 §10.8（看懂→动手）· 知识地图 §9.9/§17.7.6 · LINKS。
"""
import sys
from time import perf_counter

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ================= 账 A：为什么需要 Computer Use（无需 API）=================
# 任务元组：(任务名, 场景描述, FC 判定(有接口文档?), MCP 判定(有 Server?), 手段结论)
TASKS_A = [
    ("旧计费系统·月度报表", "纯桌面 GUI 老系统", "无",      "无",        "唯一入口"),
    ("运营后台·改缓存配置", "网页后台(内网)",     "无",      "无",        "唯一入口"),
    ("内部服务·生成汇总邮件", "内部 API 服务",    "有文档",  "有 Server", "也可(绕远)"),
]
METHODS_A = ["Function Calling(要接口)", "MCP(要 Server)", "Computer Use(要 GUI)"]


def _usable_A(task):
    apif, mcpf = task[2], task[3]
    return [
        "✅" if apif == "有文档" else "❌",
        "✅" if mcpf == "有 Server" else "❌",
        "✅",  # GUI 永远在——只要这系统长得出来（老系统桌面端/网页都能截图定位）
    ]


def expA():
    print("=" * 72)
    print("[账 A] 为什么需要 Computer Use——没有 API 的世界，GUI 是唯一入口")
    print("=" * 72)
    print("  同一个任务，三种接法前置条件不一样：FC 要接口文档、MCP 要 Server、")
    print("  Computer Use 只要『屏幕看得见』（截图→理解→点）。逐任务看可用性：")
    print(f'    {"":<6}{METHODS_A[0]:<22}{METHODS_A[1]:<18}{METHODS_A[2]}')
    avail = []
    for t in TASKS_A:
        row = _usable_A(t)
        avail.append(row)
        print(f'    {t[0]:<18} {" ".join(row)}    {t[1]}\n')
    n_fc = sum(1 for r in avail if r[0] == "✅")
    n_mcp = sum(1 for r in avail if r[1] == "✅")
    n_cu = sum(1 for r in avail if r[2] == "✅")
    assert n_fc == 1 and n_mcp == 1 and n_cu == 3
    print(f"  → 可用性：FC {n_fc}/3 · MCP {n_mcp}/3 · Computer Use {n_cu}/3")
    print("    老桌面系统/第三方 SaaS/内网后台往往没有接口文档与 Server——这是常态，")
    print("    所以 Computer Use 的存在理由就是『没有 API 的世界，屏幕点得通』；")
    print("    反过来说有 API 时 API 1 次调用就够，GUI 反而要 4 轮小循环（账 B）。")
    print("  → 断言：Computer Use 3/3 唯一全覆盖 · FC/MCP 都被『无接口』卡在 1/3")


# ================= 账 B：截图→视觉理解→动作→执行 管线 =================
# 虚拟"偏好设置"窗口（元素表驱动；坐标即"截图视图"里控件的位置）
SET_ELEMS = [
    ("title", "偏好设置", 40, 1, "text"),
    ("label", "亮度", 4, 3, "text"),
    ("bar",   "██████░░░░", 10, 3, "bar"),     # 亮度条：brightness/10 格
    ("minus", "[-]", 32, 3, "button"),
    ("plus",  "[+]", 37, 3, "button"),
    ("save",  "[保存]", 8, 5, "button"),
    ("cancel", "[取消]", 16, 5, "button"),
]
GOAL_B = 80
STEP_B = 10


def _snap_bar(brightness):
    filled = brightness // STEP_B
    return "█" * filled + "░" * (100 // STEP_B - filled)


def _snapshot_state(brightness):
    # 截图=对屏幕状态采样出一份视图：亮度 bar 是当前状态的可视化
    return {eid: (name, x, y, kind) for eid, name, x, y, kind in SET_ELEMS}, _snap_bar(brightness)


def _perceive_find_plus(snap_elems):
    # 视觉理解：OCR 读文本 → 找到亮度 label 所在行的 [ + ] 控件，输出目标+物理坐标
    assert snap_elems["label"][0] == "亮度"
    plus = snap_elems["plus"]
    return "plus", plus[1], plus[2]


def _execute_click(tgt, brightness, saved):
    if tgt == "plus":
        brightness = min(100, brightness + STEP_B)
    elif tgt == "save":
        saved = True
    return brightness, saved


def expB():
    print()
    print("=" * 72)
    print("[账 B] 管线闭环：截图→视觉理解→动作→执行→新截图（一次正确操作=多次小循环）")
    print("=" * 72)
    print("  虚拟『偏好设置』窗口（元素表驱动）——任务：把亮度从 50% 调到 80%")
    print(f'    {"步":>2}  {"动作输出":<14}{"坐标":<10}亮度条        状态')
    brightness, saved = 50, False
    steps = 0
    while brightness < GOAL_B or not saved:
        steps += 1
        snap_elems, _ = _snapshot_state(brightness)
        if brightness < GOAL_B:
            tgt, x, y = _perceive_find_plus(snap_elems)      # 视觉理解 → 目标控件
        else:
            tgt, x, y = "save", SET_ELEMS[-2][2], SET_ELEMS[-2][3]
        brightness, saved = _execute_click(tgt, brightness, saved)
        state = ("完成:保存已点" if (brightness >= GOAL_B and saved)
                 else ("未达标·继续调亮" if brightness < GOAL_B else "亮度达标·准备保存"))
        print(f'    {steps:>2}  click({tgt:<9})   ({x},{y})   {_snap_bar(brightness)}   {state}')
    assert steps == 4 and brightness == GOAL_B and saved
    print(f"\n  → 收敛 {steps} 步：3 次 [ + ]（50→60→70→80）+ 1 次[保存]。每个动作前都有一轮")
    print("    『截图-理解』，所以一次正确 GUI 操作=多次小循环（vs API 1 次调用）。")
    print("    每步都是一个失败点：截图糊/理解错/坐标偏/点错钮——计算机使用又贵又难的原因。")
    print("  → 断言：4 步收敛 · 亮度 50→80 精确停止 · 保存被点击=管线全闭环")


# ================= 账 C：像素级理解与坐标精度（难点一）=====================
# 同一「立即购买」按钮 × 4 屏幕（默认 / 缩放125% / 窗口平移 / 纯图标版）
SCREENS_C = [
    ("默认 100%",      ("buy", "立即购买", 520, 380, 120, 40, "text")),
    ("缩放 125%",      ("buy", "立即购买", 650, 475, 150, 50, "text")),   # 物理坐标随缩放漂
    ("窗口平移(+120,+40)", ("buy", "立即购买", 640, 420, 120, 40, "text")),  # 窗口挪按钮跟着墨
    ("纯图标版(🛒)",   ("buy", "🛒", 520, 380, 48, 48, "icon")),           # 无文本，OCR 死
]
FIXED_CX, FIXED_CY = 580, 400   # 固定坐标法记住的"默认屏按钮中心"


def _bbox_hit(screen, cx, cy):
    _, name, x, y, w, h, kind = screen
    return x <= cx <= x + w and y <= cy <= y + h


def _method_fixed(screen):
    return _bbox_hit(screen, FIXED_CX, FIXED_CY)   # 永远点记忆里的 (580,400)


def _method_ocr(screen):
    _, name, x, y, w, h, kind = screen
    return kind != "icon"        # 有文本就能读出来重定位；图标无文本=认不出


def _method_vision(screen):
    _, name, x, y, w, h, kind = screen
    return True                  # 视觉语义：文字按钮读文本、图标按形状+标签认（购物车）


def expC():
    print()
    print("=" * 72)
    print("[账 C] 像素级理解与坐标精度——GUI 的坐标不是稳定契约，像素理解才是")
    print("=" * 72)
    print("  任务：四屏都点击『立即购买』。三种定位法（记住基准坐标 / OCR 读文本定位 / ")
    print("  视觉语义定位）逐屏命中：")
    print(f'    {"屏幕背景":<18}{"固定坐标(基准)":>14}{"OCR文本定位":>12}{"视觉语义定位":>12}')
    hits = {"fixed": 0, "ocr": 0, "vis": 0}
    for desc, screen in SCREENS_C:
        fx = _method_fixed(screen)
        oc = _method_ocr(screen)
        vi = _method_vision(screen)
        hits["fixed"] += int(fx)
        hits["ocr"] += int(oc)
        hits["vis"] += int(vi)
        print(f'    {desc:<18}{("✅ 中点" if fx else "❌ 点空"):>14}'
              f'{("✅ 重定位" if oc else "❌ 无文本"):>12}'
              f'{("✅ 认出" if vi else "❌ 漏认"):>12}')
    assert hits == {"fixed": 1, "ocr": 3, "vis": 4}, hits
    print(f"\n  → 固定坐标 {hits['fixed']}/4：只在默认屏中——缩放 125%、窗口平移后按钮物理坐标"
          f"全漂（点空），连图标版也因新中心落在原 bbox 外而 miss；记住坐标=猜运气。")
    print(f"    OCR 文本定位 {hits['ocr']}/4：『立即购买』这种文字按钮，缩放/平移都能"
          f"『读文本→重定位』，但纯图标按钮没有文字=认不出（死）。")
    print(f"    视觉语义定位 {hits['vis']}/4：认形状+标签，图标也认（🛒→购物车）——四屏全过。")
    print("  → 所以计算机使用必须『截图→视觉理解→定位』而不是『记坐标』：")
    print("    坐标是窗口/分辨率的函数，元素身份才是屏幕里稳定的锚点（Qwen2.5-VL/UI-TARS 路线）")


# ================= 账 D：误操作控制与安全护栏（难点二+应用落点）=============
# GUI 无 schema 可校验 → 护栏从 API 的"参数校验"退化成"确认框文本匹配+高危人工审核
# +限额+快照回滚"。事件元组：(id, 动作, 危险类型 or None, 带确认框?, 确认框是"确定"?)
DANGER_D = [
    (1, "删除月度报表", "删除", True, True),       # 危险确认"确定"→确认框文本匹配拦
    (2, "支付 ¥520",   "付款", True, True),        # 付款确认→匹配+限额拦
    (3, "恢复出厂设置", "永久", True, True),        # 不可逆→高危人工审核拦
    (4, "跳过二次确认(永久操作)", "永久", False, False),  # 高风险但无确认框→审核拦(永久)
    (5, "点击外部链接", None, True, False),          # 低危外链→两边都不拦=漏
    (6, "误触『跳过确认』诱导钮", None, False, False),  # 危险流程旁诱导按钮→不拦=漏
]


def _guard_d(evt, subset):
    eid, act, dtype, has_confirm, confirm_danger = evt
    g = set(subset)
    if "快照回滚" in g:
        return False                                  # 任何误操都能快照恢复→不产生残留
    if "确认框文本匹配" in g and has_confirm and confirm_danger:
        return False                                  # 确认框里是危险"确定"→拦截
    if "高危人工审核" in g and (dtype in ("删除", "付款", "永久")):
        return False                                  # 高危动作转人工，不放行自动执行
    if "动作限额" in g and dtype in ("付款",):
        return False
    return True                                       # 仍放行=漏网


def expD():
    print()
    print("=" * 72)
    print("[账 D] 误操作控制与安全护栏——GUI 无 schema，护栏=确认框匹配+审核+限额+回滚")
    print("=" * 72)
    print("  6 个触点（删报表/付款/出厂设置/永久操作/外链/诱导『跳过确认』钮）。GUI 的")
    print("  护栏没有『参数 schema』可校验，只能退化成『确认框文本匹配 + 高危动作人工审核")
    print("  + 动作限额 + 快照回滚』：")
    cfg_d = {
        "无护栏(直出)":           [],
        "审核+确认框匹配":          ["确认框文本匹配", "高危人工审核"],
        "全配(审核+匹配+限额+回滚)": ["确认框文本匹配", "高危人工审核", "动作限额", "快照回滚"],
    }
    print(f'    {"护栏配置":<28}{"漏网":>4}  仍触发的动作（残留影响）')
    residual = {}
    for nm, subset in cfg_d.items():
        leaked = [e for e in DANGER_D if _guard_d(e, subset)]
        residual[nm] = len(leaked)
        names = " · ".join(a for _, a, *_ in leaked) if leaked else "无（全部拦截）"
        print(f'    {nm:<28}{str(len(leaked)) + "/6":>6}  {names}')
    assert residual["无护栏(直出)"] == 6
    assert residual["审核+确认框匹配"] == 2
    assert residual["全配(审核+匹配+限额+回滚)"] == 0
    print(f"\n  → 无护栏 {residual['无护栏(直出)']}/6 直出：Agent 误删报表直接落盘、误付 ¥520 已扣款；")
    print(f"    +确认框匹配+高危审核 {residual['审核+确认框匹配']}/6：危险确认『确定』被匹配、永久操作转人工，"
          f"剩 2 条低危诱导（外链/『跳过确认』钮）自动放行；")
    print(f"    全配(+限额+快照回滚) {residual['全配(审核+匹配+限额+回滚)']}/6：付款限额拦、任何误触可快照恢复"
          f"→ 误操作『可逆』=OSWorld 评测里『动作可逆/影响可控』独立维度的数字版。")
    print("  → 与 07 对比：API 护栏管『参数对不对』，GUI 护栏只能管『确认对不对』——")
    print("    没有 schema，就得靠确认框匹配 + 人工审核 + 快照回滚层层兜住。")


def main():
    print("=" * 72)
    print("computer_use_demo：09-Agent体系 · 09-Computer-Use（§9.9 GUI 直接操作·无需 API）")
    print("    关联：地图 §9.9 ｜ 技术点=截图→视觉理解→坐标/动作→执行 ｜ 难点=像素+误操作")
    print("=" * 72)
    print("[0] 口径：屏幕状态/元素表/坐标/护栏判定 = 作者预置仿真（非真实 Computer Use 系统"
          "运行·不 import 任何框架）；A-D 统计 = 本机确定性计算；代表产品与评测 = "
          "要素事实（Claude Computer Use·Operator·UI-TARS·OS-Copilot·OSWorld·V*·Mind2Web）")
    expA()
    expB()
    expC()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 为什么 GUI / B 管线 / C 像素与坐标 / D 护栏）")
    print("  A 无 API 的世界只有 GUI 能进：FC 1/3·MCP 1/3·Computer Use 3/3；有 API 时 API 1 次 vs GUI 4 轮")
    print("  B 一次正确操作=多次小循环：截图→理解→动作→执行→新截图 4 步收敛（3 次 +1 保存），每步一个失败点")
    print("  C GUI 坐标不是契约：固定坐标 1/4（缩放/平移/图标全漂）·OCR 3/4（图标无文本死）·视觉语义 4/4")
    print("  D GUI 护栏=确认框匹配+人工审核+限额+快照回滚：6→2→0，误触可回滚；OSWorld『动作可逆』独立维度")
    print("一句话：Computer Use 把 Agent 的『手脚』接到 GUI——没有 API 的世界只有屏点得通；"
          "但坐标会漂、像素会糊、点错难撤销，所以要看图找元素（视觉语义），"
          "并用确认框匹配+人工审核+快照回滚兜住误操作（§9.9 + 09-07 六护栏 GUI 版）。")
    print("done · 一键复现：python code/notebooks/_tools/computer_use_demo.py")


if __name__ == "__main__":
    main()
    W(f"[computer_use_demo] wall-clock {perf_counter() - _t0:.3f} s")
