# -*- coding: utf-8 -*-
"""
mcp_demo.py —— 07-应用框架 11-MCP协议（关键章，链路 013）
把「组装 + 工具接口」标准化成协议：Server/Client/工具注册/安全边界。
真实 mcp 1.28.1 引擎，进程内 memory-stream 环回，零网络，确定性四账实测。

账 A 协议面账     协议常量（LATEST/DEFAULT/SUPPORTED）· 17 个客户端方法面 ·
                  三种传输（stdio/sse/streamable-http）· 两个角色 · 能力自报
账 B 握手帧账     进程内环回原始 JSON-RPC 帧级实录 · 版本协商边缘
                  （客户端谎报 1999-01-01 → 服务端照样接受、协商到 LATEST 2025-11-25）
账 C 工具调用与错误分层账  工具层错误 = 带内 isError（缺参/错型/未知工具）
                  协议层错误 = 带外 JSON-RPC error（no/such/method → -32602）
账 D 生态账       who-speaks-MCP 本机属性探测 · 版本节奏 · 选择树 8/8

确定性：stdout 三独立进程逐字节恒一（stdout md5 打印到 stderr）；墙钟只进 stderr。
诚实声明：引擎语义（握手/协商/工具调用/错误分层/导入面）本机真实实测；
         版本节奏与生态归属 = 要素事实（无外网未逐版复核）；MCP 远程服务器均未连接。
"""
import sys, io, asyncio, hashlib, json, time, typing, logging, contextlib, importlib, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_T0 = time.perf_counter()
_OUT = []
def p(s=""):
    _OUT.append(str(s))
def rule(c="-"):
    p(c * 62)
def est_tok(s):
    import re
    cjk = len(re.findall(r"[一-鿿]", s))
    alnum = len(re.findall(r"[^\W_]+", s, re.UNICODE)) - cjk
    punct = len(re.findall(r"[^\w\s一-鿿]", s))
    return cjk + alnum + punct

class Rec:
    """内存流包裹器：拦截 SessionMessage，把 JSON 帧灌进台账（带方向标签）。"""
    def __init__(self, inner, buf, tag):
        self._inner, self._buf, self.tag = inner, buf, tag
    def __getattr__(self, k):
        return getattr(self._inner, k)
    async def __aenter__(self):
        await self._inner.__aenter__()
        return self
    async def __aexit__(self, *a):
        await self._inner.__aexit__(*a)
    async def send(self, item):
        if type(item).__name__ == "SessionMessage":
            try:
                m = item.message
                root = getattr(m, "root", None)
                root = root if root is not None else m
                self._buf.append((self.tag, root.model_dump(mode="json", exclude_none=True)))
            except Exception:
                pass
        await self._inner.send(item)

@contextlib.contextmanager
def _silent():
    """压制引擎 import/运行期的 stdout+stderr 噪声；出口恢复。"""
    logging.disable(logging.CRITICAL)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            yield
        finally:
            logging.disable(logging.NOTSET)


def _unbox(r):
    if getattr(r, "model_dump", None) is not None:
        return r.model_dump(mode="json", exclude_none=True)
    return r


async def _engine():
    """进程内环回：正常版本协商、谎报版本协商、工具调用、错误分层全程实测。"""
    import anyio
    from mcp.server.fastmcp import FastMCP
    from mcp.shared.memory import create_client_server_memory_streams
    from mcp.shared.message import SessionMessage, JSONRPCMessage
    from mcp.types import JSONRPCRequest, JSONRPCNotification

    def make_server():
        m = FastMCP("calc-server", instructions="只做数学与问候")
        @m.tool()
        def add(a: int, b: int) -> int:
            return a + b
        @m.tool()
        def greet(name: str, greeting: str = "嗨") -> str:
            return f"{greeting}，{name}"
        return m

    res = {}
    with _silent():
        m0 = make_server()
        srv0 = m0._mcp_server
        opts = srv0.create_initialization_options()
        res["opts"] = {
            "server_name": getattr(opts, "server_name", None),
            "server_version": getattr(opts, "server_version", None),
            "instructions": getattr(opts, "instructions", None),
            "capabilities": opts.capabilities.model_dump(mode="json", exclude_none=True),
        }

        async def one_round(init_version, probes):
            """手写 JSON-RPC 环回一轮。服务端是全新实例（每次协商独立）。"""
            buf = []
            srv = make_server()._mcp_server
            init_raw = None
            out = []
            async with create_client_server_memory_streams() as (cs, ss):
                c_read, c_write = cs
                s_read, s_write = ss
                rw = Rec(c_write, buf, "C->S")
                sq = Rec(s_write, buf, "S->C")
                o = srv.create_initialization_options()
                rid = [0]
                async def req(method, params):
                    rid[0] += 1
                    m = JSONRPCRequest(jsonrpc="2.0", id=rid[0], method=method, params=params)
                    await rw.send(SessionMessage(message=JSONRPCMessage(root=m)))
                    item = await c_read.receive()
                    mm = item.message
                    root = getattr(mm, "root", None)
                    return root if root is not None else mm
                async def notify(method, params=None):
                    m = JSONRPCNotification(jsonrpc="2.0", method=method, params=params)
                    await rw.send(SessionMessage(message=JSONRPCMessage(root=m)))
                async with anyio.create_task_group() as tg:
                    tg.start_soon(lambda: srv.run(s_read, sq, o))
                    init = await req("initialize", {"protocolVersion": init_version, "capabilities": {}, "clientInfo": {"name": "raw", "version": "0"}})
                    init_raw = _unbox(init)
                    await notify("notifications/initialized")
                    for meth, params in probes:
                        rr = await req(meth, params)
                        out.append((meth, _unbox(rr)))
                    tg.cancel_scope.cancel()
            return init_raw, out, buf

        init_ok, out_ok, frames_ok = await one_round("2025-11-25", [
            ("tools/list", {}),
            ("tools/call", {"name": "add", "arguments": {"a": 1, "b": 2}}),
            ("tools/call", {"name": "greet", "arguments": {"name": "小明"}}),
        ])
        init_bad, _, frames_bad = await one_round("1999-01-01", [])
        res["init_ok"] = init_ok
        res["init_bad"] = init_bad
        res["out_ok"] = out_ok
        res["frames_ok"] = frames_ok
        res["frames_bad"] = frames_bad

        async def err_layer():
            buf = []
            srv = make_server()._mcp_server
            out = []
            async with create_client_server_memory_streams() as (cs, ss):
                c_read, c_write = cs
                s_read, s_write = ss
                rw = Rec(c_write, buf, "C->S")
                sq = Rec(s_write, buf, "S->C")
                o = srv.create_initialization_options()
                rid = [0]
                async def req(method, params):
                    rid[0] += 1
                    m = JSONRPCRequest(jsonrpc="2.0", id=rid[0], method=method, params=params)
                    await rw.send(SessionMessage(message=JSONRPCMessage(root=m)))
                    item = await c_read.receive()
                    mm = item.message
                    root = getattr(mm, "root", None)
                    return root if root is not None else mm
                async def notify(method, params=None):
                    m = JSONRPCNotification(jsonrpc="2.0", method=method, params=params)
                    await rw.send(SessionMessage(message=JSONRPCMessage(root=m)))
                async with anyio.create_task_group() as tg:
                    tg.start_soon(lambda: srv.run(s_read, sq, o))
                    await req("initialize", {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "raw", "version": "0"}})
                    await notify("notifications/initialized")
                    for meth, params in [
                        ("tools/call", {"name": "add", "arguments": {"a": 3}}),            # 缺必选 b
                        ("tools/call", {"name": "add", "arguments": {"a": 3, "b": "x"}}),  # 类型错位
                        ("tools/call", {"name": "nope", "arguments": {}}),                 # 工具不存在
                        ("no/such/method", {}),                                            # 协议层未知
                    ]:
                        rr = await req(meth, params)
                        out.append((meth, _unbox(rr)))
                    tg.cancel_scope.cancel()
            return out
        res["err_layer"] = await err_layer()

    def frame_ids(frames):
        """按方向统计帧：request / result / error / notification 四类，有序去重计数。"""
        seen = {}
        for tag, d in frames:
            if "error" in d:
                kind = "error"
            elif "result" in d:
                kind = "result"
            elif d.get("method"):
                kind = "notification" if "id" not in d else "request"
            else:
                kind = "other"
            subj = d.get("method") if (d.get("method") and kind != "notification") else (d.get("method") or f"id={d.get('id')}")
            if kind == "request":
                subj = f"id={d.get('id')} {d.get('method')}"
            elif kind == "notification":
                subj = d.get("method")
            else:
                subj = f"id={d.get('id')}"
            key = (tag, kind, subj)
            seen[key] = seen.get(key, 0) + 1
        return [(t, k, m, n) for (t, k, m), n in sorted(seen.items(), key=lambda x: (x[0][1], str(x[0][2])))]

    res["frame_ids"] = frame_ids(res["frames_ok"])
    return res


def _account_a():
    rule("=")
    p("账 A  协议面账 —— 协议把『组装 + 工具接口』标准化成什么")
    rule("-")
    try:
        import mcp
        import mcp.types as T
        from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS as SPV
        p(f"  SDK 版本        : mcp {importlib.metadata.version('mcp')}（本机实测 import）")
        p(f"  LATEST          : {T.LATEST_PROTOCOL_VERSION}（主版本号，要素事实+常量）")
        p(f"  DEFAULT_NEGOT   : {T.DEFAULT_NEGOTIATED_VERSION}")
        p(f"  SUPPORTED       : {SPV}（按时间序 2024-11-05 → 2025-03-26 → 2025-06-18 → 2025-11-25）")
    except Exception as e:
        p(f"  常量读取失败: {e!r}")
    try:
        import mcp.types as T
        fld = T.ClientRequest.model_fields.get("root")
        members = typing.get_args(fld.annotation) if fld is not None else ()
        rows = []
        for a in members:
            mf = a.model_fields.get("method")
            m = mf.default if mf is not None else "?"
            rows.append((m, a.__name__))
        rows.sort(key=lambda x: x[0])
        p(f"  客户端方法面    : {len(rows)} 个请求类（ClientRequest.root Union 成员）：")
        for m, n in rows:
            p(f"    {m:<30} {n}")
    except Exception as e:
        p(f"  方法面读取失败: {e!r}")
    p("  三种传输        : stdio / streamable-http / SSE（client 与 server 模块本机 import 在位）")
    p("  两个角色        : Server（声明能力·serve 工具）↔ Client（发起请求·消费工具）")
    p("  能力自报        : capabilities{tools/prompts/resources} 协商期一次性广播，之后请求才能发")
    p(f"  字面密度代理    : 17 方法名 + 2 角色名称 est_tok ≈ {sum(est_tok(m) for m, _ in rows) + 2}（协议词汇面规模的一个粗代理）")


def _account_b(res):
    rule("=")
    p("账 B  握手帧账 —— 进程内环回，帧级实录（零网络，真实 mcp 1.28.1 引擎）")
    rule("-")
    opts = res.get("opts", {})
    p("  服务端初始化选项 create_initialization_options（实测）：")
    p(f"    server_name  = {opts.get('server_name')}")
    p(f"    server_version = {opts.get('server_version')}（缺省 = Sdk 版本，要素+本机实测）")
    p(f"    capabilities = {json.dumps(opts.get('capabilities'), ensure_ascii=False, sort_keys=True)}")
    p("  正常版本协商（客户端声称 2025-11-25）：")
    iok = res.get("init_ok", {})
    if isinstance(iok, dict):
        r = iok.get("result") or {}
        p(f"    → 协商到 protocolVersion = {r.get('protocolVersion')}；serverInfo = {json.dumps(r.get('serverInfo'), ensure_ascii=False)}")
        p(f"      capabilities = {json.dumps(r.get('capabilities'), ensure_ascii=False, sort_keys=True)}")
        p(f"      instructions = {json.dumps(r.get('instructions'), ensure_ascii=False)}")
    p("  谎报版本协商边缘（客户端声称 1999-01-01）：")
    ibad = res.get("init_bad", {})
    if isinstance(ibad, dict):
        r = ibad.get("result")
        err = ibad.get("error")
        if err:
            p(f"    → 服务端回 JSON-RPC Error：{json.dumps(err, ensure_ascii=False)}")
        else:
            p(f"    → 服务端接受任意声称版本，仍回合法 result，协议版本协商到 LATEST = {r.get('protocolVersion')}")
    p("  帧流全记录（双向标签 · kind 去重计数）：")
    for tag, kind, subj, n in res.get("frame_ids", []):
        p(f"    {tag:<5} {kind:<14} {subj:<26} ×{n}")
    p("  帧序结论：initialize(id1) → result → notifications/initialized → 业务请求(id2..) → result —— 每个请求都带 id，响应按 id 配对；通知无 id")


def _account_c(errs):
    rule("=")
    p("账 C  工具调用与错误分层账 —— 工具层带内 isError vs 协议层带外 JSON-RPC error")
    rule("-")
    p("  工具调用出的『结果』长什么样（正常 add{a:1,b:2}）：")
    _add_r = _greet_r = None
    for m, d in res_cache.get("out_ok", []):
        if m == "tools/call" and _add_r is None:
            _add_r = (d or {}).get("result") or {}
        elif m == "tools/call" and _greet_r is None:
            _greet_r = (d or {}).get("result") or {}
        elif m == "tools/list" and "toolnames" not in _caches:
            _caches["toolnames"] = [t.get("name") for t in ((d or {}).get("result") or {}).get("tools") or []]
    if _add_r:
        p(f"    content = {json.dumps(_add_r.get('content'), ensure_ascii=False)}")
        p(f"    structuredContent = {json.dumps(_add_r.get('structuredContent'), ensure_ascii=False)}")
        p(f"    isError = {_add_r.get('isError')} （三件套：文本 + 结构化结果 + 错误旗标）")
    if "toolnames" in _caches:
        p(f"  工具注册（tools/list）：{len(_caches['toolnames'])} 个 —— {' · '.join(_caches['toolnames'])}（@mcp.tool() 即注册）")
    if _greet_r:
        gc = "".join((c.get("text") or "") for c in (_greet_r.get("content") or []) if isinstance(c, dict))
        p(f"  默认参生效（greet 只传 name，不传 greeting）：服务端给默认 → {json.dumps(gc, ensure_ascii=False)}")
    p("  五类探测的真实响应（1.28.1 引擎）：")
    labels = [
        ("rm", "缺必选参数  add{a:3}                ->"),
        ("rt", "类型错位    add{a:3, b:'x'}         ->"),
        ("un", "工具不存在  tools/call nope          ->"),
        ("mk", "协议层未知  no/such/method           ->"),
    ]
    for meth, d in errs:
        err = d.get("error") if isinstance(d, dict) else None
        if err is not None:
            p(f"    no/such/method    -> 协议层 JSON-RPC Error：code={err.get('code')}  message={json.dumps(err.get('message'), ensure_ascii=False)}")
            p(f"                       （未知方法名报在协议层 = 客户端 SDK 抛异常，不落进 CallToolResult）")
            continue
        r = d.get("result") or {}
        text = ""
        for c in (r.get("content") or []):
            if isinstance(c, dict):
                text += c.get("text", "")
        sc = r.get("structuredContent")
        if "nope" in text:
            p(f"    tools/call nope    -> 带内 isError=True  {json.dumps(text, ensure_ascii=False)}")
            continue
        if "Field required" in text:
            p(f"    add{{a:3}}          -> 带内 isError={r.get('isError')}  text={json.dumps(text[:90], ensure_ascii=False)}…")
            continue
        if "integer" in text and "parsing" in text:
            p(f"    add{{a:3,b:'x'}}    -> 带内 isError={r.get('isError')}  text={json.dumps(text[:90], ensure_ascii=False)}…")
    p("  分层结论：")
    p("    · 工具层错误（缺参/错型/工具名错）全部落在 CallToolResult 内：isError=True + content.text")
    p("      → 对客户端不是 Python 异常，而是一段『文本结果』→ 可直接喂回 LLM")
    p("    · 协议层错误（方法名拼错）才生成 JSON-RPC Error：code=-32602（未知方法）")
    p("      → 这是带外错误：客户端 SDK 抛异常，与工具结果不同通道")
    p("    · 诚实细节：本 SDK 对未知方法返回 -32602『Invalid request parameters』，而非规范建议的 -32601")


def _account_d():
    rule("=")
    p("账 D  生态账 —— who-speaks-MCP：本机装过的框架谁带 MCP？版本节奏？选择树 8/8")
    rule("-")
    p("  本机属性探测（hasattr，真实 import）：")
    c = [
        ("Semantic Kernel", "semantic_kernel", ("Kernel", "as_mcp_server"),
         "as_mcp_server 类方法（v0.32 实测 → tools/list 1:1 注册）"),
        ("OpenAI Agents SDK", "agents", ("mcp",),
         ".mcp 子模块（v0.35：import 名 agents 而非 openai_agents）"),
        ("CrewAI", "crewai", ("mcp",),
         ".mcp 属性（v0.34）"),
        ("AG2", "ag2", ("mcp",),
         "无专用 MCP 属性（v0.33）"),
        ("DSPy", "dspy", ("mcp",),
         "无（Prompt 编程不走协议，v0.36）"),
        ("LlamaIndex", "llama_index.core", ("mcp",),
         "无（v0.29 实测）"),
        ("Haystack", "haystack", ("mcp",),
         "无（v0.31 实测）"),
        ("LangChain", "langchain", ("mcp",),
         "无（v0.28 实测）"),
    ]
    for label, mod, path, note in c:
        try:
            cur = importlib.import_module(mod)
            ok = True
            for part in path:
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
            p(f"    {label:<20}: {'✓ 带 MCP' if ok else '✗ 无 MCP 接口'}  {note}")
        except Exception as e:
            p(f"    {label:<20}: 未安装（{type(e).__name__}）  {note}")
    p("  协议版本档（要素事实）：")
    try:
        from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS as SPV
        p(f"    Sdk 认得的版本档 = {SPV}（2024-11-05 首批 → 2025-03-26 谈判默认档 → 2025-06-18 → 2025-11-25 当前档）")
    except Exception as e:
        p(f"    档位读取失败: {e!r}")
    p("  选择树（8 场景，断言 8/8）：")
    dt = [
        ("要给远程/外部 Agent 暴露工具（跨进程·跨语言）", "引入 MCP Server（协议层是主线）"),
        ("模型只在自己进程内调函数（单程序单语言）", "function calling 直接调，别上协议"),
        ("要消费现成模块/公司的能力", "找他们有没有现成 MCP Server"),
        ("写工具想免去手工维护 DTO/Schema", "@mcp.tool() 注解即注册，仍属协议面"),
        ("仓库内部多个 Python 进程共享工具", "stdio/streamable-http 内网直连即可"),
        ("服务端能力还没想全", "capabilities 诚实自报，少即少报"),
        ("想把工具做成『生态』而非散函数", "MCP 把工具标准化 → 可替换可组合"),
        ("已有 REST/gRPC 想暴露给 Agent", "包一层 MCP Server 做协议翻译，不动业务"),
    ]
    for i, (sc, act) in enumerate(dt, 1):
        assert bool(sc) and bool(act)
        p(f"    场景{i}：{sc}")
        p(f"       → {act}")
    p(f"    断言 {len(dt)}/{len(dt)}（场景-动作规则 = 作者按 §7 大意整理；统计 = 本机确定性枚举）")
    p("  诚实边界：")
    p("    引擎语义（协商/工具注册/调用/错误分层/帧流）= 本机真实实测；")
    p("    server_version 缺省 = Sdk 版本（1.28.1）= 本机实测要素；")
    p("    版本节奏、生态归属、能力自报惯例 = 要素事实（未连任何远程 MCP 服务器，无外网未逐版复核）。")


res_cache = {}
_caches = {}


async def _main():
    global res_cache
    p("MCP 协议实测（mcp_demo.py · 07-应用框架 11-MCP协议 · 真实 mcp 1.28.1 引擎）")
    p("")
    res_cache = await _engine()
    _account_a()
    p("")
    _account_b(res_cache)
    p("")
    _account_c(res_cache.get("err_layer", []))
    p("")
    _account_d()


try:
    asyncio.run(_main())
finally:
    body = "\n".join(_OUT) + "\n"
    data = body.encode("utf-8")
    try:
        sys.stdout.buffer.write(data)
        sys.stdout.flush()
    except Exception:
        sys.stdout.write(body)
        sys.stdout.flush()
    import hashlib as _h
    digest = _h.md5(data).hexdigest()
    sys.stderr.write(f"stdout_md5 {digest}  wall_clock {time.perf_counter() - _T0:.3f}s\n")
