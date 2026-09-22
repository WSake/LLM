#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""09-Agent体系 · 10-浏览器Agent（知识地图 §9.10 Browser Agent）探针：
§9.10 原话："代表：Gemini in Chrome、OpenAI AgentKit/Operator、Anthropic 浏览器工具、
开源 browser-use / Skyvern / Playwright MCP。形态：云端浏览器（Operator 类）、浏览器
插件（本地）、协议化（Playwright MCP server 暴露给任何 Agent）。技术点：DOM 可访问性
树比纯视觉更可靠；登录/验证码是工程难点；anti-bot 对抗。应用：订票、比价、数据分析
抓取、Deep Research。"
四账 A-D 实测（全部=作者预置仿真 + 本机确定性计算）：
  A 三种形态怎么选——同一批浏览器任务 × 三种部署形态（云端浏览器 / 浏览器插件 /
    Playwright MCP 协议化）：云端 1/3（公开网开箱即用，但内网进不来·数据出域风险·
    批任务成本高，Operator 类）、插件 2/3（本地执行走内网/数据不外出·但绑定浏览器
    实例难调度）、协议化 3/3（MCP server 常驻，一次部署任意 Agent/调度器可驱动，
    三场景全覆盖）＝没有一个形态包打天下，协议化最通用（§9.10 三形态）
  B DOM 可访问性树 vs 纯视觉——同一"确认下单"按钮 × 4 个页面状态（静态/促销浮层遮
    挡/JS 动态后加载/无文本图标钮×aria-label）× 2 种"读屏"法：DOM 可访问性树
    （拿 role/text/aria 属性/rect，元素在树里就在）4/4 vs 纯视觉（截一张图认元素，
    被遮挡/未加载/无文本=盲）1/4——§9.10『DOM 可访问性树比纯视觉更可靠』的数字版：
    结构是"先见"，像素是"单帧快照"（09 账 C 视觉语义 4/4 在"没结构可用"的通用 GUI
    成立；浏览器里有 DOM=更稳的锚）
  C 登录/验证码为什么是工程难点——同一"拉下载个人报表"任务 × 4 个认证档位：公开页
    自动 100%·表单登录（凭证+session 保持）自动 100%·图形验证码自动 30%/人工接管
    70%·SSO+MFA 自动 10%/人工 90%——登录态管理（session 复用）是基本功，验证码/
    MFA 是反自动化的护城河→工程现实=凭证仓库+会话复用+失败转人工兜底
  D anti-bot 对抗与护栏——同一"10 分钟抓 100 个商品页"任务 × 2 配置：无护栏（120ms
    连发·同 UA·无重试）被 anti-bot 拦 38/成功 62 → 配护栏（限速≥2s·指纹稳定·退避
    重试·验证码人工兜底）拦 3/成功 97——反爬侮辱的正是"机器特征"：高频+同指纹+
    无节奏；护栏四件套=07 六护栏的浏览器版（限速=动作限额·指纹匹配=最小权限·退避
    重试=错误重试·验证码转人工=人工确认）
口径：任务/页面状态/认证档位/请求节奏与拦截判定 = 作者预置仿真（非真实浏览器 Agent
     运行·不 import Playwright/浏览器·零网络）；A-D 统计 = 本机确定性计算（纯 stdlib
     · 零随机 · 零网络 · stdout 逐字节可复现；wall-clock 只进 stderr）。代表产品
     （Gemini in Chrome / AgentKit / Operator / Anthropic 浏览器工具 / browser-use /
     Skyvern / Playwright MCP）= 要素事实（写作环境无外网未复核，非本机实测）。
关联：09-Computer-Use（通用 GUI 无结构→视觉语义 4/4 vs 本篇浏览器里有 DOM→结构更稳）
      · 07-Agent安全（六护栏→账 D 浏览器版）· 06-05 Function-Calling（账 B：DOM 定位
      媲美 API 的"字段寻址"，纯视觉=无 schema 的像素寻址）· 07-11-MCP（协议化=
      账 A 的 Playwright MCP）· 10-多模态 §10.8（Qwen2.5-VL 能操作 UI=视觉路线与
      DOM 路线在浏览器里互补）· 知识地图 §9.10/§17.7.6 · LINKS。
"""
import sys
from time import perf_counter

sys.stdout.reconfigure(encoding="utf-8")
_t0 = perf_counter()


def W(t=""):
    sys.stderr.write(t + "\n")


# ================= 账 A：三种形态（云端 / 插件 / 协议化）怎么选 =================
# 任务元组：(任务, 网络边界, 数据敏感, 调度特征)
TASKS_A = [
    ("第三方网站订票",    "公开网可达",    "个人出行数据",   "即点即用"),
    ("内网 BI 报表拉取",  "仅内网可达",    "敏感·不可出域",  "按需/定时"),
    ("每天批量抓价落库",  "公开网可达",    "公开行情",       "脚本化·全天候"),
]
FORMS_A = ["云端浏览器(Operator类)", "浏览器插件(本地)", "Playwright MCP(协议化)"]


def _usable_A(task):
    net, sens, sched = task[1], task[2], task[3]
    cloud = "✅" if net == "公开网可达" and sens == "个人出行数据" else "❌"
    ext = "✅" if net == "仅内网可达" or sched == "即点即用" else "❌"
    mcp = "✅"  # MCP server 部署在目标网络侧：内网可部署、公开网可部署、常驻可调度
    return [cloud, ext, mcp]


def expA():
    print("=" * 72)
    print("[账 A] 三种形态怎么选——云端 / 插件 / 协议化，没有一个包打天下")
    print("=" * 72)
    print("  §9.10 形态：云端浏览器（Operator 类）· 浏览器插件（本地）· 协议化")
    print("  （Playwright MCP server 暴露给任何 Agent）。同一批任务换三种形态看可用性：")
    print(f'    {"":<6}{FORMS_A[0]:<26}{FORMS_A[1]:<22}{FORMS_A[2]}')
    avail = []
    for t in TASKS_A:
        row = _usable_A(t)
        avail.append(row)
        print(f'    {t[0]:<14} {" ".join(row)}    {t[1]}·{t[3]}\n')
    n_c = sum(1 for r in avail if r[0] == "✅")
    n_e = sum(1 for r in avail if r[1] == "✅")
    n_m = sum(1 for r in avail if r[2] == "✅")
    assert n_c == 1 and n_e == 2 and n_m == 3
    print(f"  → 云端 {n_c}/3 · 插件 {n_e}/3 · 协议化 {n_m}/3")
    print("    云端=开箱即用但受网络边界约束（内网进不来·数据出域风险·批任务成本高）；")
    print("    插件=本地执行数据不外走、但绑一个浏览器实例难独立调度；")
    print("    协议化=MCP server 常驻目标网络侧，一次部署任意 Agent/调度器驱动=最通用。")
    print("  → 断言：协议化 3/3 全覆盖 · 云端卡『内网/数据敏感』 · 插件卡『脚本化调度』")


# ================= 账 B：DOM 可访问性树 vs 纯视觉 =================
# 同一"确认下单"按钮的 4 个页面状态：(状态, 可从 a11y 树定位吗?, 当前帧截图看得见吗?)
#   一页：静态 / 浮层遮挡 / JS 动态后加载 / 无文本图标钮(仅 aria-label)
PAGES_B = [
    ("静态正常",                True, True),   # 都在——DOM 读到文本+rect，截图也看得清
    ("促销浮层遮挡按钮",          True, False),  # DOM/树里有文本与 rect，截图上被浮层盖住
    ("JS 动态后加载",            True, False),  # 首帧不在 DOM，但轮询 a11y 树 1s 后进树可读；截图拍到 loading
    ("无文本图标钮(aria-label)",  True, False),  # 只有 aria-label，树里可读；像素无文本=盲
]


def _method_dom(locatable):
    # DOM 可访问性树：元素最终从树里可定位（静态=立即可读，动态=轮询等进树，aria=兜底）
    return locatable


def _method_vis(can_see):
    # 纯视觉：对当前这一帧截图认元素——被遮挡/没加载/无文本 都看不到
    return can_see


def expB():
    print()
    print("=" * 72)
    print("[账 B] DOM 可访问性树 vs 纯视觉——结构是『先见』，像素是『单帧快照』")
    print("=" * 72)
    print("  同一『确认下单』按钮 × 4 个页面状态 × 2 种读屏法（断言论）：")
    print(f'    {"页面状态":<22}{"DOM可访问性树":>14}{"纯视觉(截图)":>14}')
    hits = {"dom": 0, "vis": 0}
    for desc, locatable, can_see in PAGES_B:
        d = _method_dom(locatable)
        v = _method_vis(can_see)
        hits["dom"] += int(d)
        hits["vis"] += int(v)
        print(f'    {desc:<22}{("✅ 读role/text/aria" if d else "❌ 盲目"):>14}'
              f'{("✅ 看清" if v else "❌ 盲"):>14}')
    assert hits == {"dom": 4, "vis": 1}, hits
    print(f"\n  → DOM 可访问性树 {hits['dom']}/4：元素从树里可定位就有 role/text/aria/rect——")
    print("    浮层遮挡没关系（能先关浮层再点）、动态加载可以轮询等它进树、无文本按钮有")
    print(f"    aria-label 兜底。纯视觉 {hits['vis']}/4：每一帧截图都是『快照』——被遮挡看到")
    print("    的是浮层文字、没加载拍到的是 loading、无文本直接认不出（09 账 C 同款死法）。")
    print("  → 所以 §9.10 说『DOM 可访问性树比纯视觉更可靠』：浏览器里有结构可用，就不再")
    print("    需要靠像素猜（09 账 C 视觉语义 4/4 是『没有结构』的通用 GUI 兜底；有 DOM")
    print("    又更稳的锚时就该用 DOM）。")


# ================= 账 C：登录 / 验证码——为什么是工程难点 =================
# 认证档位：(档位, 是否可自动, 自动化成功率%, 人工接管率%)
AUTH_C = [
    ("公开页(无需登录)",        True, 100, 0),
    ("表单登录 + session 保持",  True, 100, 0),
    ("图形验证码",             False, 30, 70),
    ("SSO + 二次认证(MFA)",    False, 10, 90),
]


def expC():
    print()
    print("=" * 72)
    print("[账 C] 登录 / 验证码——同一任务，认证门槛越高自动化越难")
    print("=" * 72)
    print("  任务：让浏览器 Agent 拉取『我的个人报表』。四个认证档位的自动化难度：")
    print(f'    {"认证档位":<26}{"自动化成功率":>14}{"人工接管率":>14}')
    auto = {}
    for nm, can, a, m in AUTH_C:
        auto[nm] = (can, a, m)
        print(f'    {nm:<26}{str(a) + "%":>14}{str(m) + "%":>14}')
    assert [v[1] for v in auto.values()] == [100, 100, 30, 10]
    print(f"\n  → 公开页 100%：无需登录态。表单登录 100%：一次登录成功后 session 复用（cookie/")
    print("    token 带进后续请求），不是每次都重新登——这是浏览器 Agent 的基本功（09 账 C：")
    print("    '会话'身份不是页面上的坐标，是 HTTP 层的东西）。")
    print(f"    图形验证码 30%：只有规则恰好能过的简单图形能自动，其余转人工 70%；")
    print(f"    SSO+MFA 10%：OAuth 前段可走，TOTP/设备确认把自动化钉死，人工接管 90%。")
    print("    → 登录态管理是基本功，验证码/MFA 是反自动化的护城河——工程现实=凭证仓库 +")
    print("      session 复用 + 失败转人工兜底（登录页是 Agent 最常卡壳的墙）。")


# ================= 账 D：anti-bot 对抗与护栏 =================
# "10 分钟抓 100 个商品页"任务：无护栏 vs 配护栏，被 anti-bot 拦下多少
REQS_D = 100
CFG_D = {
    "无护栏(120ms连发·同UA·无重试)":   {"blocked": 38},   # 高频+同指纹+无节奏 → 风控判机器
    "配护栏(限速2s·指纹稳·退避重试·验证码转人工)": {"blocked": 3},  # 机器特征消除 → 仅真验证码拦
}
GUARD_MAP = [
    ("限速 ≥2s/请求",        "07 六护栏·动作限额",     "把高频机器节奏降回人类节奏"),
    ("UA/指纹稳定",          "07 六护栏·最小权限",     "伪装成正常用户=最小暴露面"),
    ("429 退避重试(上限2次)", "07 六护栏·错误重试",     "被限速不被封号"),
    ("验证码 → 转人工接管",   "07 六护栏·人工确认",     "自动化做不到的确定性兜底"),
]


def expD():
    print()
    print("=" * 72)
    print("[账 D] anti-bot 对抗——反爬侮辱的正是『机器特征』，护栏把它抹掉")
    print("=" * 72)
    print(f"  同一任务：10 分钟内抓取 {REQS_D} 个商品页。两种配置的拦截结果：")
    print(f'    {"配置":<36}{"被拦(429/403/验证码)":>22}{"成功":>8}')
    res = {}
    for nm, c in CFG_D.items():
        ok = REQS_D - c["blocked"]
        res[nm] = (c["blocked"], ok)
        print(f'    {nm:<36}{str(c["blocked"]) + "/100":>20}{str(ok) + "/100":>9}')
    assert res["无护栏(120ms连发·同UA·无重试)"] == (38, 62)
    assert res["配护栏(限速2s·指纹稳·退避重试·验证码转人工)"] == (3, 97)
    print(f"\n  → 无护栏被拦 38：120ms 连发=流量模式被判机器，同 UA=指纹一致暴露自动化，"
          f"429 不重试=直接断；")
    print(f"    配护栏只拦 3：限速把节奏降回人类、指纹稳定不暴露、429 退避重试续上，")
    print(f"    剩 3 个是真验证码页→转人工接管（账 C 的墙在这里再次出现）。")
    print("  → 护栏四件套是 07 六护栏的浏览器版：")
    for item, src, why in GUARD_MAP:
        print(f"    · {item:22} = {src:16}（{why}）")
    print("  → 结论：anti-bot 不是要绕开，是要『别长着一张机器的脸』——节奏、指纹、重试、" +
          "人工兜底，四件套齐了对站点的冲击也小、任务的成功率也高。")


def main():
    print("=" * 72)
    print("browser_agent_demo：09-Agent体系 · 10-浏览器Agent（§9.10 Browser Agent）")
    print("    关联：地图 §9.10 ｜ 三形态 云端/插件/协议化 ｜ DOM 比纯视觉可靠 ｜ 登录验证码难点")
    print("=" * 72)
    print("[0] 口径：任务/页面状态/认证档位/请求节奏与拦截判定 = 作者预置仿真（非真实浏览器"
          "Agent 运行·不 import Playwright/浏览器·零网络）；A-D 统计 = 本机确定性计算；"
          "代表产品与框架 = 要素事实（Gemini in Chrome·AgentKit·Operator·browser-use·"
          "Skyvern·Playwright MCP）")
    expA()
    expB()
    expC()
    expD()
    print()
    print("=" * 72)
    print("台账汇总（A 形态 / B 读屏 / C 登录验证码 / D anti-bot）")
    print("  A 云端 1/3（内网/数据敏感/批任务卡）·插件 2/3（本地但绑浏览器）·协议化 3/3 最通用")
    print("  B DOM 可访问性树 4/4 vs 纯视觉 1/4（遮挡/动态加载/无文本=盲）——结构先见，像素单帧")
    print("  C 公开 100%·表单登录+session 100%·验证码 30%（人工 70%）·SSO+MFA 10%（人工 90%）")
    print("  D 无护栏 38 拦 → 护栏四件套 3 拦：限速（节奏）+指纹稳定（身份）+退避重试+验证码转人工")
    print("一句话：浏览器 Agent = 把 Computer Use 收进 DOM 有结构的浏览器里——形态先选对"
          "（协议化最通用）、读屏用 DOM 而不是猜像素、登录态管好，剩下的交给护栏四件套："
          "做自己的事、别长一张机器的脸（§9.10）。")
    print("done · 一键复现：python code/notebooks/_tools/browser_agent_demo.py")


if __name__ == "__main__":
    main()
    W(f"[browser_agent_demo] wall-clock {perf_counter() - _t0:.3f} s")
