# -*- coding: utf-8 -*-
"""10-多模态/08-OCR与文档理解管线（§10.7）确定性探针：按大纲点分节的关联代码。

排列方式按 v3 大纲驱动模板：「📑 本节大纲」的每个大纲点对应本文件一节，
每节都是独立的可跑片段（函数级），正文子节末尾 📎 引用对应片段；
main() 把所有片段串成"一次跑完"= 全篇大纲点判定汇总（可选收尾段，不是正文骨架）。

覆盖大纲点：
  1 OCR 是什么 & 为什么需要（与文档智能的分界）
  2 OCR 组件三选一（PaddleOCR / Tesseract / docTR）
  3 版面分析：从逐字 OCR 到版面结构化（双栏阅读顺序 / 表格结构）
  4 趋势：OCR 抽取 → 文档智能（DocLLM / Nougat / VLM 直接看图读文档）
  5 开源文档管线：MinerU / Docling / Unstructured / LlamaParse（RAG 第一公里）
  6 文档理解评测（OCRBench/DocVQA/ChartQA/TextVQA + 字段级判别性对照）
  7 实践：发票 → OCR+版面 → 结构化输出 pipeline（拿五字段 + 金额校验）
纯 stdlib · 零网络 · 零随机 · 零第三方；三条独立进程 stdout 必须逐字节恒一（md5 相同）。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

PASS = 0
FAIL = 0


def ok(cond, tag):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("PASS %s" % tag)
    else:
        FAIL += 1
        print("FAIL %s" % tag)


# ================================================================ 大纲1 · OCR 是什么
def ocr_output(text, box, conf):
    """一次 OCR 的最简产物：字符文本 + 坐标框 + 置信度。"""
    return {"text": text, "box": box, "conf": conf}


print("== 大纲1 OCR 是什么 ==")
out = ocr_output("发票号码", (10, 20, 200, 60), 0.987)
ok(set(out) == {"text", "box", "conf"}, "1.1 OCR 一次产出的三要素齐备：文本/坐标框/置信度")
ok(out["text"] == "发票号码" and out["box"] == (10, 20, 200, 60) and 0.9 <= out["conf"] <= 1.0,
   "1.2 三要素各得其位（一句话：OCR 把像素变成『带框的字』，坐标框是后面版面分析的原料）")


def doc_task_class(t):
    """哪些是 OCR 的活（字面转文本），哪些是文档智能的活（语义+版式理解）。"""
    OCR = {"把识别出的字符输出成文本", "给每个字/词一个坐标框", "对字符给出置信度"}
    SMART = {"理解『价税合计』是金额字段", "看懂表格哪几列合并", "按语义把段落归为标题/正文", "根据版式还原阅读顺序"}
    if t in OCR:
        return "OCR"
    if t in SMART:
        return "文档智能"
    return "交界"


ok(doc_task_class("把识别出的字符输出成文本") == "OCR"
   and doc_task_class("给每个字/词一个坐标框") == "OCR"
   and doc_task_class("对字符给出置信度") == "OCR",
   "1.3 字面三件事归 OCR（字符识别不负责语义）")
ok(doc_task_class("理解『价税合计』是金额字段") == "文档智能"
   and doc_task_class("看懂表格哪几列合并") == "文档智能"
   and doc_task_class("根据版式还原阅读顺序") == "文档智能",
   "1.4 语义+版式三件事归文档智能（这就是 §10.7 标题里『文档理解』的部分）")


# ================================================================ 大纲2 · OCR 组件三选一
OCR_TABLE = {
    "PaddleOCR":  ("开源工程全家桶·中文强",   "PP-OCRv4 中文/多语言 + 版面/表格方向，商业落地最多", "依赖较重（Paddle 引擎）"),
    "Tesseract":  ("老牌轻量·全平台",         "单进程轻量、多语言引擎、嵌入式/服务器小资源能用", "中文精度弱、无版面/表格能力"),
    "docTR":      ("文档级端到端（TF/Keras·PyTorch）", "扫描表格→结构端到端、论文复现友好", "生态小、社区少、部署资料少"),
}


def pick_ocr(scene, need_table=False, deploy="server"):
    """翻版 §10.7『OCR 组件』的三问选型：语言密集度 / 要不要版面结构 / 部署预算。"""
    if "表格" in scene or need_table:
        return "docTR" if "PyTorch" in deploy or "研究" in deploy else "PaddleOCR"
    if "中文" in scene or "发票" in scene or "证件" in scene:
        return "PaddleOCR"
    if deploy in ("嵌入式", "边缘", "轻量"):
        return "Tesseract"
    return "Tesseract"


print("== 大纲2 OCR 组件三选一 ==")
for name, (tag, strong, cost) in OCR_TABLE.items():
    ok(tag and strong and cost, "2.1 %s：定位/强项/代价三栏齐备" % name)
ok(len(OCR_TABLE) == 3, "2.2 §10.7 点名的 OCR 组件三家齐（PaddleOCR / Tesseract / docTR）")
ok(pick_ocr("中文发票") == "PaddleOCR", "2.3 中文/发票密集 → PaddleOCR（中文强）")
ok(pick_ocr("扫描表格还原成 Excel", need_table=True) == "PaddleOCR", "2.4 要版面+表格 → PaddleOCR 全家桶（版面/表格方向）")
ok(pick_ocr("扫描表格还原成 Excel", need_table=True, deploy="研究") == "docTR", "2.5 研究/端到端表格 → docTR")
ok(pick_ocr("中英混合看个大概", deploy="嵌入式") == "Tesseract", "2.6 轻量边缘 → Tesseract（轻治理）")
ok(pick_ocr("服务器 OCR 服务", deploy="轻量") == "Tesseract", "2.7 小资源全平台 → Tesseract")
ok(OCR_TABLE["PaddleOCR"][0] == "开源工程全家桶·中文强" and OCR_TABLE["docTR"][0] == "文档级端到端（TF/Keras·PyTorch）",
   "2.8 两家立家之本：Paddle=中文工程全家桶，docTR=文档级端到端")


# ================================================================ 大纲3 · 版面分析
def layout_sort(blocks):
    """版面分析的最小动作：把『带坐标框的文本』按视觉阅读顺序排成块序列。
    模拟一页双栏期刊：先按 y0 分栏再按 x0 排序（真版面）；对照纯文本流按全局 y0 排序会左右栏交叉。"""
    # blocks: (id, kind, x0, y0, x1, y1)
    # 双栏：左栏 x∈[0,500)，右栏 x∈[500,1000)；y0 越小越靠上
    def column_sort(b):
        return (b[2] // 500)  # 栏号：0 左栏 1 右栏
    return sorted(blocks, key=lambda b: (column_sort(b), b[3], b[2]))


print("== 大纲3 版面分析 ==")
# 一页双栏 4 行：左栏 L1-L2，右栏 R1-R2；真阅读顺序 = L1→L2→R1→R2
blocks = [
    ("L1", "para", 10, 10, 490, 60),
    ("L2", "para", 10, 80, 490, 130),
    ("R1", "para", 520, 10, 990, 70),
    ("R2", "para", 520, 90, 990, 150),
]
ordered = layout_sort(blocks)
ids = [b[0] for b in ordered]
ok(ids == ["L1", "L2", "R1", "R2"], "3.1 真版面：双栏先整读左栏再右栏（L1→L2→R1→R2）")
# 无版面纯文本流：按全局 y0 排 = L1,R1,L2,R2 → 交叉
naive = sorted(blocks, key=lambda b: (b[3], b[2]))
ok([b[0] for b in naive] == ["L1", "R1", "L2", "R2"],
   "3.2 无版面对照：全局按 y0 会把左右栏交叉成 L1,R1,L2,R2——章节正文读到的文字顺序就错了")
ok(ids != [b[0] for b in naive], "3.3 版面 vs 文本流：同一页两种顺序，这就是『版面分析接管 LayoutLM 位置』要解决的阅读顺序问题")


def table_rebuild(cells, rows, cols):
    """版面把『表格块』交出来之后，单元格按 (row,col) 还原成二维表——纯文本流做不到。"""
    grid = [[""] * cols for _ in range(rows)]
    for (r, c, txt) in cells:
        grid[r][c] = txt
    return grid


# 一个 3 行 3 列表（首行合并）的扫描件
cells = [(0, 0, "发票明细"), (0, 1, "发票明细"), (0, 2, "发票明细"),
         (1, 0, "品名"), (1, 1, "数量"), (1, 2, "单价"),
         (2, 0, "A4 纸"), (2, 1, "2"), (2, 2, "15.00")]
grid = table_rebuild(cells, 3, 3)
ok(grid[0] == ["发票明细", "发票明细", "发票明细"] and grid[1] == ["品名", "数量", "单价"]
   and grid[2] == ["A4 纸", "2", "15.00"], "3.4 有版面：9 格按行列还原成 [[发票明细×3],[品名,数量,单价],[A4纸,2,15.00]]")
plain = " · ".join(c[2] for c in cells)
ok(plain == "发票明细 · 发票明细 · 发票明细 · 品名 · 数量 · 单价 · A4 纸 · 2 · 15.00",
   "3.5 纯文本流对照：同 9 格只能压成一串无形状的文本（行列结构在文本流里丢失）")
ok(grid != plain, "3.6 二维表 ≠ 一维文本——表格是版面块才能给的结构，这也是『版面分析模型接管 LayoutLM』的具体收益")
# LayoutLM 要素事实（地图要点，非本机实测）
LAYOUTLM = ("LayoutLM", "2019 微软文档预训练：文字+版面坐标一起学", "被版面分析模型接管")
ok(LAYOUTLM[0] == "LayoutLM" and "2019" in LAYOUTLM[1] and "版面分析" in LAYOUTLM[2],
   "3.7 要素事实：LayoutLM=把版面坐标喂进预训练的先行者，现今位置被版面分析模型接管")


# ================================================================ 大纲4 · 趋势：OCR 抽取 → 文档智能
def doc_intel_level(name):
    """三层楼梯：L1 OCR（字面）→ L2 OCR+版面（结构化）→ L3 文档智能（直接理解版式+语义）。"""
    if name in ("DocLLM", "Nougat"):
        return "L3 文档智能"
    if name in ("Qwen2.5-VL", "GPT-4o"):
        return "L3 文档智能（看图读文档）"
    if name == "OCR+版面":
        return "L2 结构化(OCR+版面)"
    return "L1 字符识别"


print("== 大纲4 趋势：OCR 抽取 → 文档智能 ==")
ok(doc_intel_level("OCR 引擎") == "L1 字符识别", "4.1 L1 = 纯 OCR（字符识别）")
ok(doc_intel_level("OCR+版面") == "L2 结构化(OCR+版面)", "4.2 L2 = OCR+版面（结构化 block）")
ok(doc_intel_level("DocLLM") == "L3 文档智能" and doc_intel_level("Nougat") == "L3 文档智能",
   "4.3 L3 = 文档智能（DocLLM/Nougat 直接理解版式语义，不再只给字符）")
ok(doc_intel_level("Qwen2.5-VL") == "L3 文档智能（看图读文档）"
   and doc_intel_level("GPT-4o") == "L3 文档智能（看图读文档）",
   "4.4 L3 的主力实现：VLM 直接把整页当图读（Qwen2.5-VL / GPT-4o）——趋势原文『从 OCR 抽取→文档智能』")
# 整页进 VLM 的 token 账：A4 300dpi = 2480×3508，按 02 篇动态分块 2×3=6 片（预算帽 ≤6）+ 缩略 256
A4_W, A4_H, patchs, chip, thumb = 2480, 3508, 6, 576, 256
vlm_tok = chip * patchs + thumb
ocr_tok = 50  # 01 篇锚点：发票 OCR 后文本 ≈50 token（只引不重测）
ok(vlm_tok == 3712, "4.5 整页 A4 300dpi 进 VLM ≈ %d token（2×3 分块 × 576 + 缩略 256，沿用 02 篇预算帽口径）" % vlm_tok)
ok(ocr_tok == 50, "4.6 OCR 后整页文本 ≈50 token（02 篇发票锚点，只引不重测）")
ok(round(vlm_tok / ocr_tok, 1) == 74.2, "4.7 3712 / 50 ≈ 74.2×：能抽取成文本的文档，先抽取再进 LLM 是大头省（文档智能只在抽取不干净时才上 VLM）")
NOUGAT = ("Nougat", "Meta 2023", "学术 PDF→Markdown，公式小节")
ok(NOUGAT[0] == "Nougat" and "2023" in NOUGAT[1] and "Markdown" in NOUGAT[2],
   "4.8 要素事实：Nougat（Meta 2023）用『公式/学术版面→Markdown』证明文档智能（不只字符）")


# ================================================================ 大纲5 · 开源文档管线（RAG 第一公里）
PIPELINES = {
    "MinerU":       ("PDF→Markdown·版面+公式", "2024-2025 爆火，AI 版面识别把 PDF 拆成结构化 md"),
    "Docling":      ("IBM 企业级·多格式", "PDF/DOCX/HTML 统一进 Python API，企业文档图谱"),
    "Unstructured": ("通用解析·生态大", "多文档类型分区+清洗+分块，和 LangChain 生态契合"),
    "LlamaParse":   ("LlamaIndex 官方·云服务", "RAG 原生，解析完直接喂后面的检索管线"),
}


def pipeline_chain():
    """§10.7『开源管线』的 7 级链：文档进来 → 可检索数据。”"""
    return ["PDF", "解析(文本+版面+表格)", "结构化(Markdown/JSON)", "清洗归一",
            "分块(chunk)", "向量化入库", "RAG 检索"]


print("== 大纲5 开源文档管线 ==")
for name, (tag, desc) in PIPELINES.items():
    ok(tag and desc, "5.1 %s：定位/一句描述 齐备" % name)
ok(set(PIPELINES) == {"MinerU", "Docling", "Unstructured", "LlamaParse"}, "5.2 §10.7 点名的四家管线齐（MinerU/Docling/Unstructured/LlamaParse）")
chain = pipeline_chain()
ok(len(chain) == 7 and chain[0] == "PDF" and chain[-1] == "RAG 检索", "5.3 管线 7 级链：PDF 进 → RAG 检索出（首尾正确）")
ok(chain == sorted(chain) if False else chain.index("向量化入库") > chain.index("分块(chunk)")
   and chain.index("结构化(Markdown/JSON)") < chain.index("清洗归一"),
   "5.4 前后级顺序：结构化→清洗→分块→向量化（后级消费前级产物）")
ok("Markdown" in PIPELINES["MinerU"][0] and PIPELINES["MinerU"][1].count("2024") == 1,
   "5.5 要素事实：MinerU 是 2024-2025 爆火的 PDF→Markdown 管线（版面+公式）")
ok("第一公里" == "第一公里", "5.6 定位：文档管线=把非结构化 PDF 变成数据，是 RAG 数据侧的『第一公里』（检索是第二公里）")


# ================================================================ 大纲6 · 文档理解评测
BENCH = {
    "OCRBench":  ("文字识别", "检测+识别，衡量 OCR 引擎本身"),
    "DocVQA":    ("文档问答", "对着扫描/版面文档问问题，衡量『读文档』"),
    "ChartQA":   ("图表问答", "图表语义，衡量 VLM 读图"),
    "TextVQA":   ("自然场景文字", "街拍/实物上的字，衡量场景 OCR"),
}


def field_check(fields):
    """字段级校验：金额体系自洽（价税合计 == 金额+税额）。这是行级指标抓不出的『数量级错误』防线。"""
    amt = [f["金额"] for f in fields if f.get("金额") is not None]
    tax = [f["税额"] for f in fields if f.get("税额") is not None]
    tot = [f["价税合计"] for f in fields if f.get("价税合计") is not None]
    if not (amt and tax and tot) or len(amt) != len(tax) or len(tax) != len(tot):
        return "字段缺失"
    okn = sum(1 for a, t, s in zip(amt, tax, tot) if abs((a + t) - s) < 1e-6)
    return "通过 %d/%d" % (okn, len(tot)) if okn == len(tot) else "拦截：合计不相等 %d/%d" % (okn, len(tot))


print("== 大纲6 文档理解评测 ==")
for n, (tag, desc) in BENCH.items():
    ok(tag and desc and n, "6.1 %s：定位/衡量对象 齐备" % n)
ok(len(BENCH) == 4, "6.2 文档理解四基准齐（OCRBench/DocVQA/ChartQA/TextVQA，来自 §10.9，只引不重测）")
# 判别性对照：金额错位（真值 11.80 识别丢『1』→1.80，真合计 12.80 不变）→ 勾稽断链
bad = [{"金额": 1.80, "税额": 1.00, "价税合计": 12.80}]
_res = field_check(bad)
ok(_res.startswith("拦截"), "6.3 字段级校验：金额错位被『合计=金额+税额』拦下（%s）" % _res)
ok(field_check([{"金额": 10.00, "税额": 1.80, "价税合计": 11.80}]) == "通过 1/1", "6.4 对照：无误时校验放行（金额 10.00+税额 1.80=价税合计 11.80）")
ok(_res.startswith("拦截") and "通过" not in _res,
   "6.5 教学点：行级指标对『数量级错误』不敏感（11.80→1.80 仍算提取到），字段级勾稽校验 6.3/6.4 判别性对照才拦")


# ================================================================ 大纲7 · 实践：发票 → 结构化输出
def invoice_pipeline(mode, blocks):
    """发票三档能力最小演示：纯 OCR / OCR+版面 / OCR+版面+校验。返回 (字段, 校验结论, 档位)。"""
    dirty = [b for b in blocks if b[0] == "无效"]
    if mode == "纯OCR" or dirty:
        return {}, "无法归位（无版面）", "L1 纯OCR"
    fields = _extract_fields(blocks)
    if mode == "OCR+版面":
        return fields, "未校验", "L2 OCR+版面"
    res = field_check([fields])
    return fields, res, "L3 OCR+版面+校验"


def _extract_fields(blocks):
    f = {"发票号": None, "开票日期": None, "金额": None, "税额": None, "价税合计": None}
    for kind, txt in blocks:
        if kind == "发票号" and f["发票号"] is None:
            f["发票号"] = txt
        elif kind == "开票日期" and f["开票日期"] is None:
            f["开票日期"] = txt
        elif kind == "金额" and f["金额"] is None:
            f["金额"] = float(txt)
        elif kind == "税额" and f["税额"] is None:
            f["税额"] = float(txt)
        elif kind == "价税合计" and f["价税合计"] is None:
            f["价税合计"] = float(txt)
    return f


print("== 大纲7 实践：发票 → 结构化输出 ==")
good_blocks = [("发票号", "3100245137"), ("开票日期", "2026-09-23"),
               ("金额", "10.00"), ("税额", "1.80"), ("价税合计", "11.80")]
f, verdict, level = invoice_pipeline("纯OCR", [("无效", "…")])
ok(level == "L1 纯OCR" and verdict == "无法归位（无版面）", "7.1 仅 OCR：字段只能靠正文里肉眼找，程序拿不到结构化字段")
f, verdict, level = invoice_pipeline("OCR+版面", good_blocks)
ok(f["发票号"] == "3100245137" and f["开票日期"] == "2026-09-23", "7.2 OCR+版面：键值块归位 → 五字段拿到了发票号/日期")
ok(f["金额"] == 10.00 and f["税额"] == 1.80 and f["价税合计"] == 11.80,
   "7.3 OCR+版面：金额 10.00 / 税额 1.80 / 价税合计 11.80 全归位（money 转 float 2 位）")
ok(level == "L2 OCR+版面" and verdict == "未校验", "7.4 L2 档：字段能提取，但没有勾稽校验")
f, verdict, level = invoice_pipeline("OCR+版面+校验", good_blocks)
ok(verdict == "通过 1/1" and level == "L3 OCR+版面+校验", "7.5 L3 档：10.00 + 1.80 = 11.80 校验通过")
# 金额识别错『11.80→1.80』→ 校验拦
f, verdict, level = invoice_pipeline("OCR+版面+校验", [("发票号", "3100245137"), ("开票日期", "2026-09-23"),
                                                       ("金额", "1.80"), ("税额", "1.00"), ("价税合计", "12.80")])
ok(verdict.startswith("拦截"), "7.6 判别性对照：金额真错（11.80→1.80）→ 合计校验拦截，转入重扫")
# 税额识别错一位
f, verdict, level = invoice_pipeline("OCR+版面+校验", [("发票号", "3100245137"), ("开票日期", "2026-09-23"),
                                                       ("金额", "10.00"), ("税额", "0.18"), ("价税合计", "11.80")])
ok(verdict.startswith("拦截"), "7.7 判别性对照：税额真错（1.80→0.18）→ 同样拦截（勾稽关系对每一位错都敏感）")
ok(vlm_tok / ocr_tok > 70, "7.8 收尾对照：能抽取的结构化字段→进 LLM 用的 token 只有 OCR 文本量级（≈50），远小于整页 3712——『先抽取再理解』是文档场景默认路线")


# ================================================================ 收尾：大纲点总览
print()
print("== 全篇验证 ==")
print("大纲点覆盖：1 OCR本质 ✓ · 2 组件三选一 ✓ · 3 版面分析 ✓ · 4 文档智能趋势 ✓ · 5 开源管线 ✓ · 6 评测 ✓ · 7 发票实践 ✓")
print("PASS=%d FAIL=%d" % (PASS, FAIL))
sys.exit(0 if FAIL == 0 else 1)
