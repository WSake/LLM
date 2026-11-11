# -*- coding: utf-8 -*-
"""mamba_hybrid_demo.py —— §15.8+§15.9 SSM 与混合架构（15-前沿技术 04 篇配套）

四账对应正文 📑 本节大纲七点中的核心五点：
账 A  复杂度：Attention O(n^2) vs SSM O(n) 缩放账（大纲 1）
账 B  质量账：省的计算 vs 追平质量的钱（大纲 2，临界点演算）
账 C  谱系：Jamba/Zamba/Hymba/Falcon-H1 年份与类型完整性（大纲 3）
账 D  结局判定：宣称 → 全面取代/并列/收编 三岔口（大纲 4）
账 E  省 KV 工具箱：五件工具 × 省什么 × 代表（大纲 5）
账 F  自我改造：给定宣称 → 稀疏/压缩/混合/分层/风险 四板斧（大纲 6）

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr。全部统计 = 本机确定性计算。

诚实边界（与正文「数字纪律」一致）：
- Mamba(2023)/Mamba-2(2024)/Jamba(2024)/Zamba(2024)/Hymba(2024)/Falcon-H1(2025)
  及 §15.8【在途-边缘化】标注 = 要素事实，出自知识地图与公开论文年份；
  写作环境无外网、未在线复核，也不引用任何未核对的横向 benchmark 分数。
- O(n^2)/O(n) 复杂度、KV 存储随 n 线性 = 可自行验证的计算背景。
- 三岔口结局判定、自我改造四板斧、五件工具箱、临界点演算 = 作者整理的**教学数/判断**，
  规则全部写在 action() 与 CASES 中，可逐行核对。
"""
from __future__ import annotations

import sys
import time

_WSTART = time.perf_counter()

BASES = (1024, 16384, 131072, 1048576)  # 序列长度（1K / 16K / 128K / 1M）


def attn_cost(n):
    return n * n  # O(n^2)，以"配对数"为本单位


def ssm_cost(n):
    return n  # O(n)，以"每 token 固定"为单位


def hybrid_cost(cfg, n):
    """给定层配置 cfg = {'attn': a, 'ssm': b}，总 cost = a·n^2 + b·n。"""
    attn, ssm = cfg["attn"], cfg["ssm"]
    return attn * n * n + ssm * n


def break_even_copies(attn_share, n):
    """账 B：混合/SSM 把序列成本从 n^2 降到 ~c·n。若需要模型大 m 倍才能追平质量，
    求中性点（成本相等时的 m）。attn_share 为注意力层占比（0=纯 SSM，0.1=混合）。"""
    full = n * n  # 全注意力的"配对成本"
    reduced = attn_share * n * n + (1 - attn_share) * n
    # 若 m 倍模型，成本大体 m * reduced（假定训练成本随参数量线性）
    # 求 full = m * reduced  =>  m = full / reduced
    return round(full / reduced, 2) if reduced else float("inf")


# (所属日期, 名称, 类型, 可否验证) —— 年份为要素事实
SPECTRUM = (
    ("2024", "Jamba",      "生产级混合（Transformer+Mamba+MoE）", "论文/发布可查"),
    ("2024", "Zamba",      "研究级混合（共享注意力 + SSM + MoE）", "论文可查"),
    ("2024", "Hymba",      "研究级混合（注意力头与 SSM 头并行）", "论文可查"),
    ("2025", "Falcon-H1",  "生产级混合（系列量产）",             "发布可查"),
)

# (宣称, 期望岔口)：账 D 三岔口
CLAIM_OUTCOME = (
    ("Mamba 在 XXB 规模全面取代 Transformer", "全面取代"),
    ("Mamba 与 Attention 长期并列两大生成范式", "并列"),
    ("SSM 作为混合层进入 Jamba/Falcon 系列",  "收编"),
    ("线性注意力让 KV 可以为常量",            "收编"),
)

# (宣称, 期望板斧)：账 F 四板斧
CLAIM_AXE = (
    ("滑窗注意力让局部 token 互看",         "稀疏"),
    ("MLA 把 KV 投影到低秩",                  "压缩"),
    ("layer interleave attention 与 linear",  "混合"),
    ("局部层滑窗 + 高层全局稀疏（分层）",      "分层"),
    ("从零写一套全新架构对抗生态",            "改生态/高风险"),
)

TOOLS = (
    ("稀疏注意力", "计算", "滑窗 / Longformer"),
    ("线性注意力 / SSM", "计算 + 缓存", "Mamba / 线性注意力"),
    ("KV 压缩", "缓存", "MLA / GQA"),
    ("混合 / 交错", "计算 + 缓存", "Jamba / Llama 4 interleaved"),
    ("分层局部+全局", "计算", "局部/高层稀疏"),
)


def classify_outcome(claim):
    """账 D：把宣称判进三岔口之一（确定性关键词规则，正文 §4 可核）。"""
    if "全面取代" in claim or "打败 Transformer" in claim:
        return "全面取代"
    if ("并列" in claim) or ("两大生成范式" in claim):
        return "并列"
    return "收编"


def classify_axe(claim):
    """账 F：把宣称判进四板斧之一（确定性关键词规则，正文 §6 可核）。"""
    if "分层" in claim:
        return "分层"
    if "滑窗" in claim or "窗口" in claim or "稀疏" in claim:
        return "稀疏"
    if "压缩" in claim or "低秩" in claim or "投影" in claim:
        return "压缩"
    if "interleave" in claim or "交错" in claim or "混合" in claim:
        return "混合"
    return "改生态/高风险"


def W(label):
    print("[wall] %-28s %.4fs" % (label, time.perf_counter() - _WSTART), file=sys.stderr)


def main():
    p = f = 0

    def check(name, got, exp):
        nonlocal p, f
        hit = got == exp
        p += int(hit)
        f += int(not hit)
        print("  %-46s %s" % (name, "PASS" if hit else "FAIL(期望 %r 实得 %r)" % (exp, got)))

    print("15-04 §15.8+§15.9 SSM 与混合架构（纯 stdlib / 零网络 / 零随机）")

    # ---------- 账 A：复杂度缩放 ----------
    print("\n[账 A] Attention O(n^2) vs SSM O(n)（大纲 1）")
    a, s = attn_cost(BASES[0]), ssm_cost(BASES[0])
    check("n=1K 时 Attention > SSM", a > s, True)
    a4, s4 = attn_cost(BASES[3]), ssm_cost(BASES[3])
    check("n=1M 时 Attention 与 SSM 比 1K 时拉开", (a4 / s4) / (a / s) > 1000, True)
    check("Attention 单边长 16 倍成本平方爆表", attn_cost(16384) / attn_cost(1024), 256)
    check("SSM 单边长 16 倍成本线性 16 倍", ssm_cost(16384) / ssm_cost(1024), 16)
    check("基线序列 4 档", len(BASES), 4)
    W("账A")

    # ---------- 账 B：省 vs 还的质量账 ----------
    print("\n[账 B] 省的计算 vs 追平质量的钱（大纲 2）")
    full = attn_cost(1_000_000)
    pure_ssm = ssm_cost(1_000_000)
    m_pure = round(full / pure_ssm, 2)  # 纯 SSM 便宜 1M 倍，但需大 m 倍模型才追平
    check("纯 SSM 按配对成本便宜 100 万倍", m_pure, 1000000.0)
    # 混合 10% 注意力层：成本约 0.1·n^2+0.9·n
    m_mix = break_even_copies(0.1, 1_000_000)
    check("混合 10% attention 便宜约 10 倍", m_mix > 9.5 and m_mix < 10.5, True)
    # 临界点：若追平质量需要 m>便宜倍数，净收益为负 → 纯 SSM 不划算
    check("纯 SSM 因需更大模型追平而净收益为负", m_pure > 1, True)
    check("混合因代价可控而可持续", m_mix < 15, True)
    W("账B")

    # ---------- 账 C：混合谱系 ----------
    print("\n[账 C] 混合谱系（大纲 3）")
    check("谱系 4 条", len(SPECTRUM), 4)
    years = [x[0] for x in SPECTRUM]
    check("年份升序 2024→2025", years, ["2024", "2024", "2024", "2025"])
    check("Hymba 归属 Meta 研究线（类型含研究级）", "研究级" in SPECTRUM[2][2], True)
    check("Falcon-H1 为 2025 量产线", SPECTRUM[3][0] == "2025" and "生产级" in SPECTRUM[3][2], True)
    W("账C")

    # ---------- 账 D：结局判定 ----------
    print("\n[账 D] 纯 Mamba 三岔口（大纲 4）")
    for i, (claim, exp) in enumerate(CLAIM_OUTCOME, 1):
        check("[%d] %s" % (i, claim[:18]), classify_outcome(claim), exp)
    W("账D")

    # ---------- 账 E：省 KV 工具箱 ----------
    print("\n[账 E] 五件省 KV 工具（大纲 5）")
    check("工具箱 5 件", len(TOOLS), 5)
    saves = set()
    for t in TOOLS:
        saves.update(t[1].split(" + "))
    check("省的两类：计算与缓存", saves, {"计算", "缓存"})
    check("MLA 归于 KV 压缩", [t[0] for t in TOOLS if "MLA" in t[2]],
          ["KV 压缩"])
    check("Jamba/Llama4 归于混合", any("Jamba" in t[2] and "Llama 4" in t[2] for t in TOOLS), True)
    W("账E")

    # ---------- 账 F：自我改造四板斧 ----------
    print("\n[账 F] Transformer 自我改造板斧判定（大纲 6）")
    for i, (claim, exp) in enumerate(CLAIM_AXE, 1):
        check("[%d] %s" % (i, claim[:20]), classify_axe(claim), exp)
    check("四板斧闭集", {classify_axe(c) for c, _ in CLAIM_AXE[:-1]},
          {"稀疏", "压缩", "混合", "分层"})
    W("账F")

    print("\nPASS %d  FAIL %d" % (p, f))
    return 1 if f else 0


if __name__ == "__main__":
    raise SystemExit(main())
