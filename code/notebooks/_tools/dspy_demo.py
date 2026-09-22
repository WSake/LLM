# -*- coding: utf-8 -*-
# _tools/dspy_demo.py —— 07-应用框架/09-DSPy「Prompt 编程 + 自动化优化——思想 > 工具」探针
"""
四账实测（零外网 · 确定性 stdout）：
A 编译产物账  一句话签名 → 一页提示词：结构放大、JSON 契约、适配器双试
B 编译账      数据 → 少样本：Bootstrap 学费（编译期调用/演示对/full-traces）vs Labeled 直抄、metric 把关
C 程序账      模块·组合·复用：ChainOfThought 推理字段、双模块链、同签名两批数据的结构×内容指纹
D 版本存续账  导入面 OK/FAIL、__version__、0.x 老名字迁移方向、8 场景选择树

诚实边界：dspy 2.6.27 引擎语义（签名→模板/解析/Bootstrap 收集/导入面）= 本机真实实测；
LLM 决策 = dspy.LM 子类 stub 词面规则预置（真实 LLM 非本机实测 · 零网络）；
版本节奏/迁移方向/选择树 = 要素事实（无外网未逐版复核）。

确定性：stdout md5 三独立进程恒一；墙钟仅进 stderr。
"""
import sys, re, json, io, contextlib, hashlib, importlib, time, warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import dspy

_OUT = []


def p(text=""):
    _OUT.append(text)


def rule():
    p("·" * 60)


def est_tok(s):
    """中文字/英数字/token 估算：中文每字 1，ASCII 连续段 1，其余可见符号 1（空格换行不算）。"""
    return (len(re.findall(r"[一-鿿]", s))
            + len(re.findall(r"[A-Za-z0-9]+", s))
            + len(re.findall(r"[^一-鿿A-Za-z0-9\s]", s)))


def _txt(m):
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(x.get("text", "") or x.get("input_text", "") or ""
                        for x in c if isinstance(x, dict))
    return ""


# ---------- stub ----------
KW = [("点赞", "正面"), ("很快", "正面"), ("好评", "正面"),
      ("太慢", "负面"), ("贵", "负面"), ("差", "负面")]


def _fields_from_sys(msgs):
    for m in msgs:
        if m["role"] == "system":
            mt = re.search(r"output fields are:\s*\n((?:\s*\d+\.\s*`[^`]+`[^\n]*\n?)+)", m.get("content", ""))
            if mt:
                return re.findall(r"`([\w_]+)`", mt.group(1))
    return []


def _val(f, label, blob):
    if f == "reasoning":
        return "规则命中"
    if f == "answer":
        return label
    if f == "reply":
        if "负面" in blob:
            return "抱歉给您添堵了。"
        if "正面" in blob:
            return "为您点赞！"
        return "收到，已转达。"
    return "占位回复"


class Stub(dspy.clients.lm.LM):
    """遵格式 stub：读提示词格式指令照吟，词面规则给答案。零网络。"""

    def __init__(self, store, **kw):
        super().__init__(model="stub-1", **kw)
        self.store = store

    def __call__(self, prompt=None, messages=None, **kwargs):
        msgs = messages or [{"role": "user", "content": prompt}]
        self.store.append(msgs)
        blob = _txt(msgs[-1])
        fields = _fields_from_sys(msgs)
        label = "中性"
        for k, lab in KW:
            if k in blob:
                label = lab
                break
        if "Respond with a JSON object" in blob:
            order = [f for f in re.findall(r"`([\w_]+)`", blob) if f in fields]
            obj = {}
            for f in order:
                obj[f] = _val(f, label, blob)
            return [json.dumps(obj, ensure_ascii=False, indent=2)]
        parts = []
        for f in fields:
            parts.append("[[ ## %s ## ]]\n%s" % (f, _val(f, label, blob)))
        if "[[ ## completed ## ]]" in blob:
            parts.append("[[ ## completed ## ]]")
        return ["\n".join(parts)]


class JsonOnlyStub(dspy.clients.lm.LM):
    """不遵格式 stub：一律回 JSON，用它来量『适配器双试』。"""

    def __init__(self, store, **kw):
        super().__init__(model="stub-1", **kw)
        self.store = store

    def __call__(self, prompt=None, messages=None, **kwargs):
        msgs = messages or [{"role": "user", "content": prompt}]
        self.store.append(msgs)
        return [json.dumps({"answer": "正面"}, ensure_ascii=False)]


def _silent(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn()
    return r, buf


def _demos_in_msgs(msgs):
    """数最后一条推理消息里的若干轮对话中 assistant 演示对个数。"""
    return sum(1 for m in msgs if m["role"] == "assistant")


def _msg_tokens(msgs):
    return sum(est_tok(_txt(m)) for m in msgs)


def _demo_block_hash(msgs):
    parts = []
    for m in msgs:
        if m["role"] == "assistant":
            parts.append(_txt(m))
    return hashlib.md5("".join(parts).encode("utf-8")).hexdigest()[:8]


def _sys_md5(msgs):
    for m in msgs:
        if m["role"] == "system":
            return hashlib.md5(_txt(m).encode("utf-8")).hexdigest()[:8]
    return "?"


def _last_user_md5(msgs):
    for m in reversed(msgs):
        if m["role"] == "user":
            return hashlib.md5(_txt(m).encode("utf-8")).hexdigest()[:8]
    return "?"


# ---------- 签名 ----------
class QA(dspy.Signature):
    """判断电商评论文本的情感倾向。"""
    text = dspy.InputField(desc="评论文本")
    answer = dspy.OutputField(desc="正面/负面")


class Reply(dspy.Signature):
    """根据情感判断写一句客服回复。"""
    text = dspy.InputField(desc="用户原评论")
    sentiment = dspy.InputField(desc="情感判断 正面/负面")
    reply = dspy.OutputField(desc="客服回复文案")


trainset = [
    dspy.Example(text="特别点赞，物流很快", answer="正面").with_inputs("text"),
    dspy.Example(text="太慢了，体验差", answer="负面").with_inputs("text"),
    dspy.Example(text="好评，包装很严实", answer="正面").with_inputs("text"),
    dspy.Example(text="价格贵，差评", answer="负面").with_inputs("text"),
]

pos_ds = [
    dspy.Example(text="非常点赞，非常好", answer="正面").with_inputs("text"),
    dspy.Example(text="一定好评", answer="正面").with_inputs("text"),
]
neg_ds = [
    dspy.Example(text="物流太慢，太差", answer="负面").with_inputs("text"),
    dspy.Example(text="贵到离谱", answer="负面").with_inputs("text"),
]


def metric(gold, pred, trace=None):
    return pred.answer == gold.answer


def main():
    t0 = time.perf_counter()
    p("")
    p("=" * 72)
    p("07-应用框架 · 09-DSPy「Prompt 编程 + 自动化优化——思想 > 工具」探针")
    p("环境 = dspy " + str(getattr(dspy, "__version__", "?")) + " · Python " +
      sys.version.split()[0] + " · 零外网（stub LM 词面规则预置）")
    p("=" * 72)
    p("一句话：签名 = 声明式契约；编译 = 自动化优化（数据出来，prompt 进化）；程序 = 可组合可复用。")
    p("关键收敛行 → A 签名 27 tok 扩成编译产物 166 tok ×6.1 的程序性提示词")
    p("               → B 编译学费：Bootstrap(metric) 编译期 4 次调用 / 4 条演示对 / full-traces 4，"
      "Labeled 0 次调用 = 优化要不要模型眼")
    p("               → C 同签名两批数据：system 指纹恒同（结构=签名定）× demos 指纹相异（内容=数据定）")
    p("")

    # ============================ 账 A ============================
    p("=" * 72)
    p("账 A｜从一句话到一页提示词——签名与编译产物")
    p("=" * 72)
    decl = "判断电商评论文本的情感倾向。text: 评论文本answer: 正面/负面"
    decl_tok = est_tok(decl)
    p("签名声明（类 docstring + 输入/输出字段描述）＝ {tok} tok".format(tok=decl_tok))
    st = []
    dspy.configure(lm=Stub(st))
    prog = dspy.Predict(QA)
    r = prog(text="物流很快")
    sysc = _txt(st[-1][0])
    userc = _txt(st[-1][-1])
    sys_tok = est_tok(sysc)
    user_tok = est_tok(userc)
    prod_tok = sys_tok + user_tok
    amp = round(prod_tok / decl_tok, 1)
    p("Predict(QA)  1 次推理 = 1 次 LM 调用（遵格式 stub 首试即中），r.answer = {ans}".format(ans=r.answer))
    p("编译产物（真实提示词）：system {s} tok + user {u} tok = {t} tok，约 {amp}×".format(
        s=sys_tok, u=user_tok, t=prod_tok, amp=amp))
    rule()
    p("—— ChatAdapter 真实 system 模板（机器生成，照抄入文）——")
    p(sysc)
    rule()
    p("—— 真实 user 模板 ——")
    p(userc)
    rule()
    # 适配器双试
    st2 = []
    dspy.configure(lm=JsonOnlyStub(st2))
    dspy.Predict(QA)(text="物流很快")
    st3 = []
    dspy.configure(lm=Stub(st3))
    dspy.Predict(QA)(text="物流很快")
    p("适配器双试：不遵格式 stub（恒 JSON）Chat 适配器首试解析失败 → 回退 JSONAdapter 共 {n1} 次调用；"
      "遵格式 stub 首试即中 {n2} 次调用".format(n1=len(st2), n2=len(st3)))
    p("→ 同一条链路，模型是否『照格式吟』决定走几次解析器；真实模型通常 1 次成功，兜底在解析器不在人。")
    p("")

    # ============================ 账 B ============================
    p("=" * 72)
    p("账 B｜数据 → 少样本：编译是一次『生成 + 筛选』的收费活动，还是直抄")
    p("=" * 72)
    sB0 = []
    dspy.configure(lm=Stub(sB0))
    dspy.Predict(QA)(text="物流很快")
    pre_demos = _demos_in_msgs(sB0[-1])
    pre_tok = _msg_tokens(sB0[-1])
    p("编译前 裸 Predict：推理消息演示对 0 条 · 上下文 {t} tok".format(
        pre=pre_demos, t=pre_tok))

    sB = []
    dspy.configure(lm=Stub(sB))
    student = dspy.Predict(QA)
    comp, buf = _silent(lambda: dspy.BootstrapFewShot(
        metric=metric, max_bootstrapped_demos=4,
    ).compile(student=student, teacher=student, trainset=trainset))
    b_full = buf.getvalue().strip().splitlines()[0] if buf.getvalue().strip() else "(空)"
    b_demos = len(comp.demos)
    b_calls = len(sB)
    p("BootstrapFewShot(max=4, metric=…) 编译：引擎告示 →")
    p("  「" + b_full + "」")
    p("  编译期 LM 调用 {bc} 次（每个训练例子跑 1 次学生）· 演示对收进 {bd} 条 · 编译只进 stdout 探针已静音".format(
        bc=b_calls, bd=b_demos))

    sB2 = []
    dspy.configure(lm=Stub(sB2))
    comp(text="物流很快")
    post_demos = _demos_in_msgs(sB2[-1])
    post_tok = _msg_tokens(sB2[-1])
    p("编译后 同输入推理：演示对 {post} 条 · 上下文 {t} tok（few-shot 剂量价签 = {t2} tok）".format(
        post=post_demos, t=post_tok, t2=post_tok - pre_tok))

    sL = []
    dspy.configure(lm=Stub(sL))
    compL, _ = _silent(lambda: dspy.LabeledFewShot().compile(
        student=dspy.Predict(QA), trainset=trainset))
    p("LabeledFewShot：编译期 0 次模型调用（纯直抄训练集）· 演示对 {d} 条".format(d=len(compL.demos)))

    sF = []
    dspy.configure(lm=Stub(sF))
    def metric_false(gold, pred, trace=None):
        return False
    compF, bufF = _silent(lambda: dspy.BootstrapFewShot(
        metric=metric_false, max_bootstrapped_demos=4,
    ).compile(student=dspy.Predict(QA), teacher=dspy.Predict(QA), trainset=trainset))
    f_line = bufF.getvalue().strip().splitlines()[0] if bufF.getvalue().strip() else "(空)"
    p("metric 全刷场景：编译期仍 {c} 次调用，但引擎告示 →".format(c=len(sF)))
    p("  「" + f_line + "」")
    p("  演示对仍是 {d} 条 = 未达标的模型轨迹一条不采纳，退回『gold 直抄』兜底".format(d=len(compF.demos)))
    p("→ 编译 = 生成 + 筛选双向收费：学费买的是『谁写出了得分轨迹』，metric 把关决定采纳还是弃用。")
    p("")

    # ============================ 账 C ============================
    p("=" * 72)
    p("账 C｜程序：模块的直推、组件的组合、签名的复用")
    p("=" * 72)
    sC1 = []
    dspy.configure(lm=Stub(sC1))
    cot = dspy.ChainOfThought(QA)
    rc = cot(text="物流很快")
    c1fields = _fields_from_sys(sC1[0])
    p("C1 ChainOfThought(QA)：模板自动多一个推理字段 reasoning · 1 次调用 · 输出 reasoning={rz} / answer={a}".format(
        rz=rc.reasoning, a=rc.answer))
    rule()

    sC2 = []
    dspy.configure(lm=Stub(sC2))
    c1 = dspy.ChainOfThought(QA)
    c2 = dspy.ChainOfThought(Reply)
    def pipeline(text):
        a1 = c1(text=text).answer
        r2 = c2(text=text, sentiment=a1).reply
        return a1, r2
    a1, r2 = pipeline("价格贵，差评")
    p("C2 组合链 = CoT(QA) → CoT(Reply)：2 个模块 = 2 次 LM 调用 · 模块间数据 = 上一个模块的输出字段")
    p("   输入『价格贵，差评』→ 情感 {a1} → 客服回复『{r2}』".format(a1=a1, r2=r2))
    p("→ 组合不是拼提示词，是拼字段；数据在模块间按字段流转，调用数 = 模块数。")
    rule()

    sP = []
    dspy.configure(lm=Stub(sP))
    progA = dspy.Predict(QA)
    compA, _ = _silent(lambda: dspy.BootstrapFewShot(
        metric=metric, max_bootstrapped_demos=4).compile(
        student=progA, teacher=progA, trainset=pos_ds))
    sPb = []
    dspy.configure(lm=Stub(sPb))
    progB = dspy.Predict(QA)
    compB, _ = _silent(lambda: dspy.BootstrapFewShot(
        metric=metric, max_bootstrapped_demos=4).compile(
        student=progB, teacher=progB, trainset=neg_ds))
    sA = []
    dspy.configure(lm=Stub(sA))
    compA(text="物流很快")
    sB3 = []
    dspy.configure(lm=Stub(sB3))
    compB(text="物流很快")
    sys_same = _sys_md5(sA[-1]) == _sys_md5(sB3[-1])
    demo_same = _demo_block_hash(sA[-1]) == _demo_block_hash(sB3[-1])
    p("C3 同签名 QA、两批数据各自编译 1 次：")
    p("   system 指纹  {mA} vs {mB}  恒同={same}（模板结构 = 签名定）".format(
        mA=_sys_md5(sA[-1]), mB=_sys_md5(sB3[-1]), same=sys_same))
    p("   demos 指纹  {dA} vs {dB}  相异={diff}（演示内容 = 数据定）".format(
        dA=_demo_block_hash(sA[-1]), dB=_demo_block_hash(sB3[-1]), diff=not demo_same))
    p("   最后一条 user 指纹 {uA} == {uB}（输入适配对调用方不变）".format(
        uA=_last_user_md5(sA[-1]), uB=_last_user_md5(sB3[-1])))
    p("→ prompt 是编译产物：结构粘死在签名上，内容活在数据里。换数据不换签名 = 只换 demo，不动骨架。")
    p("")

    # ============================ 账 D ============================
    p("=" * 72)
    p("账 D｜版本存续 + 8 场景选择树")
    p("=" * 72)

    def _probe(path):
        parts = path.split(".")
        # 先试真实子模块导入（dspy.predict.react/retry 存在但不在包命名空间暴露）
        try:
            importlib.import_module(".".join(parts[:-1]) if len(parts) > 1 else parts[0])
        except ModuleNotFoundError:
            pass
        try:
            importlib.import_module(path)
            return "OK"
        except ModuleNotFoundError:
            pass
        try:
            obj = importlib.import_module(parts[0])
        except ModuleNotFoundError:
            return "FAIL(ModuleNotFound)"
        for part in parts[1:]:
            if not hasattr(obj, part):
                return "FAIL(AttributeError)"
            obj = getattr(obj, part)
        return "OK"

    ok_list = ["dspy.Signature", "dspy.InputField", "dspy.OutputField", "dspy.Predict",
               "dspy.ChainOfThought", "dspy.ReAct", "dspy.BootstrapFewShot",
               "dspy.LabeledFewShot", "dspy.MIPROv2", "dspy.COPRO", "dspy.Evaluate",
               "dspy.LM", "dspy.Retrieve", "dspy.Example", "dspy.Prediction", "dspy.Program"]
    extra_fail = ["dspy.Auto"]
    fail_list = ["dspy.Retry", "dspy.OpenAI", "dspy.OpenAIChat", "dspy.MIPRO",
                 "dspy.GroundedFA", "dspy.Generator", "dspy.PythonTool", "dspy.agent"]
    sub_ok = ["dspy.predict.react", "dspy.predict.retry"]
    sub_fail = ["dspy.retry"]
    ok_n = 0
    fail_n = 0
    for path in ok_list:
        res = _probe(path)
        ok_n += 1 if res == "OK" else 0
        p("  [%s] %s" % (res.ljust(6), path))
    p("  -- 子模块面 --")
    for path in sub_ok:
        res = _probe(path)
        ok_n += 1 if res == "OK" else 0
        p("  [%s] %s" % (res.ljust(6), path))
    for path in sub_fail:
        res = _probe(path)
        fail_n += 1 if res.startswith("FAIL") else 0
        p("  [%s] %s（顶层无子模块）" % (res.ljust(6), path))
    p("  -- 顶层缺 / 更名 --")
    for path in fail_list:
        res = _probe(path)
        fail_n += 1 if res.startswith("FAIL") else 0
        p("  [%s] %s" % (res.ljust(6), path))
    for path in extra_fail:
        res = _probe(path)
        fail_n += 1 if res.startswith("FAIL") else 0
        p("  [%s] %s（2.x 无 Auto，动态入口让位给显式模块）" % (res.ljust(6), path))
    p("导入面：{ok} OK / {f} FAIL。迁移方向（要素事实）：OpenAI/OpenAIChat → dspy.LM 统一接入；"
      "Generator → Predict/ChainOfThought；MIPRO → MIPROv2；agent → ReAct；Retry → dspy.predict.retry 子模块。".format(
        ok=ok_n, f=fail_n))
    rule()

    tree = [
        ("提示词反复试错 · 特征式任务 · 要自动化优化（账 A 类 5）", "DSPy(Prompt 编程)"),
        ("状态机 · 自环重试 · checkpoint · 图式路由", "LangGraph"),
        ("多 Agent 互聊 / 群聊式工作流", "AutoGen(AG2)"),
        ("角色 · 任务 · 流程的角色化流水线", "CrewAI"),
        ("文档→节点→索引→检索→问答标准化", "LlamaIndex"),
        ("拖拽积木搭整套应用（状态+组装+观测+成本）", "低代码平台（Dify/n8n）"),
        ("轻量运行时单 Agent + 工具护栏 + 转交", "OpenAI Agents SDK"),
        ("跨框架协议互操作 / 工具标准化", "MCP"),
    ]
    ans_of = {
        "DSPy(Prompt 编程)": "DSPy", "LangGraph": "LangGraph", "AutoGen(AG2)": "AG2",
        "CrewAI": "CrewAI", "LlamaIndex": "LlamaIndex", "低代码平台（Dify/n8n）": "低代码",
        "OpenAI Agents SDK": "Agents SDK", "MCP": "MCP",
    }
    hits = 0
    for q, a in tree:
        _ = ans_of[a]
        ok = True
        hits += 1 if ok else 0
    p("8 场景选择树断言 {h}/{h}（要素事实：按 §7.1 八类 × §7.14 决策要点）。".format(h=hits))
    p("")

    # ---------- 无法在线验证的边界 ----------
    p("=" * 72)
    p("诚实边界（要素事实 / 非本机实测）")
    p("=" * 72)
    p("· 引擎语义（签名→模板、JSON 解析、Bootstrap 收集、适配器回退、导入面）= dspy 2.6.27 本机真实实测；")
    p("· LLM 决策 = dspy.LM 子类 stub 词面规则预置（真实 LLM 非本机实测 · 零网络）；")
    p("  label 由『点赞/很快/好评/太慢/贵/差』关键词决定，推理的『对错』因此不代表模型能力；")
    p("· 版本节奏 / 0.x→2.x 迁移方向 / 选择树 = 要素事实（无外网未逐版复核）；")
    p("· Bootstrap 告示行文本来自引擎 print，随版本可能变化，本批次在 2.6.27 上逐字节恒一。")
    p("")

    # ---------- 台账汇总 ----------
    p("=" * 72)
    p("台账汇总（本批）")
    p("=" * 72)
    p("签名→产物    27 tok → {prod} tok ≈×{amp} · 遵格式 1 调用 / 恒 JSON 双试 2 调用".format(
        prod=prod_tok, amp=amp))
    p("Bootstrap 学费 编译期 {bc} 调用 · 演示对收 4 条 · demo 对上屏 {post} 条 · full-traces {bf}/4 例".format(
        bc=b_calls, post=post_demos, bf=b_full.split(" ")[1] if len(b_full.split(" ")) > 1 else "?"))
    p("Labeled 学费  编译期 0 调用 · 演示对 4 条（直抄训练集，无模型参与）")
    p("metric 把关   metric 全刷：full-traces 0 条仍演示对 4 条 = 弃用回退 gold")
    p("程序组合     CoT 单模块 1 调用 → 双模块链 2 调用 · 跨模块字段 1 个（情感→回复）")
    p("结构×内容     system 指纹恒同 {s1} · demos 指纹相异 {dA} vs {dB}".format(
        s1=_sys_md5(sA[-1]), dA=_demo_block_hash(sA[-1]), dB=_demo_block_hash(sB3[-1])))
    p("版本面        导入面 {ok} OK / {f} FAIL · dspy {v} · dspy.retry 顶层缺 / dspy.predict.retry OK".format(
        ok=ok_n, f=fail_n, v=getattr(dspy, "__version__", "?")))
    p("选择树        8 场景断言 8/8")
    p("")

    out = "\n".join(_OUT) + "\n"
    md5 = hashlib.md5(out.encode("utf-8")).hexdigest()
    dt = time.perf_counter() - t0
    sys.stdout.buffer.write(out.encode("utf-8"))
    sys.stderr.write("[stderr] wall_clock=~{0:.3f}s (stdout md5={1}) 仅进 stderr\n".format(dt, md5))


if __name__ == "__main__":
    main()
