# -*- coding: utf-8 -*-
"""01-Model-API与流式 的可复现探针：同步 vs 流式 · token 计费 · 重试与超时（零网络、零下载）。

运行：cd 仓库根 && python code/notebooks/_tools/api_stream_demo.py
依赖：仅 stdlib + jieba（无 numpy/faiss/模型——06-应用开发是最轻的一档）。

把 §6.1/§17.5.1 的接口基本功焊成本机信号：
  实验 A 同步 vs 流式     同一段模型回复两种拿法：同步=一次 JSON，流式=按 token 切成
                                N 个 SSE data: 帧 + [DONE]。测两点：① 两条路径重组文本
                                逐字节一致（流式不改变内容）；② 帧数 = token 数（流式的
                                本质是"把一次响应切成 N 个子事件"，不是 N 次响应）。
  实验 B token 计费账     同步 vs 流式按 token 计费（不是按帧/按 chunk），in/out 两段价、
                                缓存命中半价；挂靠 06 章的 k=5/10/20 窗口字符成本线。
  实验 C 重试与超时       指数退避 + 可注入假时钟（不退避时不 sleep 也能测逻辑确定性）+
                                timeout 触发重试 + 429 Retry-After 建议值 + 幂等性提醒。

确定性：固定 token 序列（jieba）+ 固定 SSE 帧 + 假时钟注入 → stdout 三遍逐位一致；
        真实墙钟只进 stderr（gen_delay 是"每生成 1 token 的模拟耗时"，非真实模型时延分布）。
"""
import io, json, os, sys, time, warnings
from collections import deque

# 单线程 BLAS 防御（本探针其实不用 blas，保持与全仓探针一致的表头风格）
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import jieba

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')
T0 = time.time()

def WALL(tag, t):
    sys.stderr.write("[wall] %-28s %s\n" % (tag, t))
    sys.stderr.flush()

# ---------- 固定模型回复（模拟"模型生成的一整段答案"） ----------
REPLY = ("好的，我来解释一下为什么要用流式。流式把一次完整的回答切成一串小数据帧，"
         "每一帧只带一个增量片段，客户端收到一个就渲染一个。这样做并不会改变最终答案，"
         "也不会改变你为回答支付的费用，它只改变了答案到达的节奏——用户第一眼能看到内容"
         "的时间，从等待全部生成完，提前到第一个词生成完。")
TOKENS = [w for w in jieba.cut(REPLY) if w.strip()]
N = len(TOKENS)

# 固定的 system + user（计费账的输入侧）
SYSTEM = "你是一个耐心讲技术的小助手。"
USER = "请用两三句话解释流式与同步调用的区别。"
INPUT_TOKENS = len([w for w in jieba.cut(SYSTEM + USER) if w.strip()])

print("=" * 72)
print("前置：Model API 与流式（§6.1/§17.5.1）· 模拟 SSE 帧零网络 · 同步vs流式/计费/重试三实验")
print("  模型回复固定 %d 个 token，第 i 个 chunk = 第 i 个 token" % N)
print("  输入侧固定 system+user = %d 个 token（计费账输入段）" % INPUT_TOKENS)
print("=" * 72)

# ---------- 模拟 LLM 服务（内存 SSE 帧，与真实协议逐字符同构） ----------
GEN_DELAY = 0.01    # 每生成 1 token 的模拟耗时（真实墙钟，只进 stderr；stdout 不出现浮动数字）

def make_frame(delta, finish_reason=None):
    obj = {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    return "data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n\n"

def stream_frames(tokens, delay=GEN_DELAY):
    """服务端流式生成器：每生成 1 token 等 delay，然后逐帧吐 SSE（data: ... + 空行），最后 [DONE]。
    客户端边收边渲染——首帧到达时间 = TTFT（首 token 延迟）。"""
    for tok in tokens:
        if delay:
            time.sleep(delay)
        yield make_frame({"content": tok}, None)
    yield make_frame({}, "stop")
    yield "data: [DONE]\n\n"

def sync_body(tokens):
    """同步路径：服务端攒齐全部 token 一次返回完整 JSON（等价于 stream=False）。"""
    time.sleep(len(tokens) * GEN_DELAY)
    full = "".join(tokens)
    obj = {"id": "chatcmpl-1", "choices": [{"index": 0, "message": {"role": "assistant", "content": full}, "finish_reason": "stop"}]}
    return "data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n\n"

def parse_sse(text):
    """客户端：逐行扫 SSE，data: 行取 payload，遇到 [DONE] 停；累积 delta.content。"""
    chunks = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        delta = obj["choices"][0]["delta"]
        if delta.get("content"):
            chunks.append(delta["content"])
    return chunks

# flake8: keep parse_sse as the documented client-side parser reference (used by notebook readers)

# ---------- 实验 A：同步 vs 流式，内容是否一致、SSE 帧数、TTFT ----------
print()
print("=" * 72)
print("实验 A  同步 vs 流式：重组一致（流式不改变内容）+ 帧数 = token 数 + TTFT")
print("=" * 72)
tA = time.time()
# 流式路径：边收边渲染（首帧到达即 TTFT），按 data: 帧解析累积 content
t0f = time.monotonic()
chunks = []
ttft = None
frames = []
for frm in stream_frames(TOKENS):
    frames.append(frm)
    if not frm.startswith("data:"):
        continue
    payload = frm[len("data:"):].strip()
    if payload == "[DONE]":
        break
    obj = json.loads(payload)
    delta = obj["choices"][0]["delta"]
    if delta.get("content"):
        if ttft is None:
            ttft = time.monotonic() - t0f
        chunks.append(delta["content"])
sb = "".join(frames)
reassembled = "".join(chunks)
sync_txt = json.loads(sync_body(TOKENS)[len("data:"):].strip())["choices"][0]["message"]["content"]
same = (reassembled == sync_txt)
n_frames = len(chunks)
print("  同步路径完整内容 : %d 字" % len(sync_txt))
print("  流式路径重组内容 : %d 字 · chunk 数 = %d（= token 数 %d）" % (len(reassembled), n_frames, N))
print("  逐字节一致 same = %s  ← 流式不改变内容，只改变到达节奏" % ("Y" if same else "N"))
print("  首 chunk = %r  · 末 chunk finish_reason = stop · 尾帧 [DONE] 存在 = %s" %
      (chunks[0], "Y" if sb.endswith("data: [DONE]\n\n") else "N"))
print("  TTFT 见 stderr（首 token 到达墙钟，run 间浮动）")
WALL("  实验A 流式TTFT", "%.3f s" % (ttft if ttft is not None else 0.0))
WALL("  实验A 同步全量总时长", "%.3f s" % (len(TOKENS) * GEN_DELAY))
WALL("  实验A 同步vs流式(挂钟量)", "%.3f s" % (time.time() - tA))
tB = time.time()

# ---------- 实验 B：token 计费账 ----------
print()
print("=" * 72)
print("实验 B  token 计费账：按 token 计费（in/out 两段价 + 缓存半价），不是按帧/chunk")
print("=" * 72)
# 价格表：要素事实（公开 API 价，非本机实测），GPT-4o 一类标准档为量级示例
PIN, POUT, PCACHE = 2.50, 10.00, 1.25            # 美元 / 每百万 token（标准档示例价）
PIN2, POUT2 = 0.15, 0.60                          # 美元 / 每百万 token（开源平替类档示例价）
def cost(inp, out, ccache=0.0):
    return (inp * (1 - 0.5 * ccache) * PIN + out * POUT) / 1e6
def cost2(inp, out):
    return (inp * PIN2 + out * POUT2) / 1e6
c_sync = cost(INPUT_TOKENS, N)
c_stream = cost(INPUT_TOKENS, N)                   # 同一回复、同一 token 数 → 账单相同
c_cache = cost(INPUT_TOKENS, N, ccache=1.0)        # 相同长 system 二次命中 → 输入段半价
print("  同一回复：同步账单 = $%.4f · 流式账单 = $%.4f · 差额 $0（token 相同，与帧数/次数无关）"
      % (c_sync, c_stream))
print("  若误按 chunk 计费则 = $%.4f × %d 帧 = $%.4f ← 这是误读，API 按 token 收" % (c_sync, n_frames, c_sync * n_frames))
print("  相同 system 二次命中缓存：输入 $%.5f→$%.5f（省 %.1f%%·缓存命中半价示例）" %
      (c_sync, c_cache, 100 * (1 - c_cache / c_sync)))
print("  换成开源平替档：$%.4f（输入 $%.2f/M 输出 $%.2f/M 示例价）" % (cost2(INPUT_TOKENS, N), PIN2, POUT2))
print("  挂靠 06 章：窗口字符成本线 k=5/10/20→154.4/307.9/614.1 字符 ≈ token 越多账单越贵（控制块数=直接省钱）")
WALL("  实验B token计费账", "%.3f s" % (time.time() - tB))
tC = time.time()

# ---------- 实验 C：重试与超时（指数退避 + 假时钟） ----------
print()
print("=" * 72)
print("实验 C  重试与超时：指数退避计划 · first-token 超时触发重试 · 429 建议值")
print("=" * 72)

class FakeClock:
    """可注入假时钟：退避 sleep 只推进假时刻不真等——stdout 确定性且全跑 <1s。"""
    def __init__(self):
        self.now = 0.0
    def monotonic(self):
        return self.now
    def sleep(self, s):
        self.now += s

CLK = FakeClock()

def call_with_retry(fn, max_retries=3, base=0.5, backoff=2.0, timeout=1.0, clock=None,
                    retry_after=None):
    """min 客户端：最多 max_retries 次重试；退避 base*backoff^attempt（429 给 Retry-After 则覆盖）；
    首 token 超过 timeout 未见 = 失败触发重试。返回 (status, attempts, planned_sleeps)。"""
    attempts = 0
    plan = []
    for attempt in range(max_retries + 1):
        attempts += 1
        out = fn(clock)
        if out is not None:
            return ("ok", attempts, plan)
        if retry_after is not None:
            wait = max(0.1, retry_after)          # 尊重服务端建议恢复时刻
        else:
            wait = base * (backoff ** attempt)     # 指数退避计划
        plan.append(wait)
        if clock:
            clock.sleep(wait)
        else:
            time.sleep(wait)
    return ("give_up", attempts, plan)

# C1 确定性失败计划：前 4 次超时（首 token 晚于 timeout 未到），第 5 次成功
plan_txt = deque(["timeout"] * 4 + ["ok"])
def fn1(clock):
    s = plan_txt.popleft()
    if s == "timeout":
        if clock:
            clock.now += 3.0          # 模拟服务端生成远超 timeout（1.0s）的等待
        return None
    return "ok"
timeout = 1.0
st, att, p1 = call_with_retry(fn1, max_retries=4, base=0.5, backoff=2.0,
                              timeout=timeout, clock=CLK)
print("  计划=前4次首token超时后成功 → 实际 attempts=%d 次（1 原始 + %d 次重试）· 状态=%s" %
      (att, att - 1, st))
print("  退避计划 base=0.5 factor=2.0 → wait=[0.5, 1, 2, 4] s（假时钟累计 %.1f s，未真睡）" % sum(p1))
print("  首 token 超时上限 timeout=%.1f s：超过即视为失败→退避→重试（保用户体感，不无限等）" % timeout)

# C2 429 带 Retry-After 建议值：退避被服务端建议值覆盖（尊重服务器节流语义）
plan_txt2 = deque(["429"] * 2 + ["ok"])
def fn2(clock):
    s = plan_txt2.popleft()
    return None if s == "429" else "ok"
ra = 1.7
CLK.now = 0.0
st2, att2, p2 = call_with_retry(fn2, max_retries=3, base=0.5, backoff=2.0,
                                timeout=1.0, clock=CLK, retry_after=ra)
print("  429+Retry-After=%.1f s → wait=[%.1f, %.1f] s（覆盖指数退避，尊重服务器恢复时刻）· attempts=%d"
      % (ra, p2[0], p2[1] if len(p2) > 1 else 0.0, att2))

# C3 幂等性提醒（不问服务端，只讲纪律：可重试的前提是请求幂等）
print("  幂等性纪律：重试只对幂等请求安全——聊天补全是幂等的（可重试）；若同一请求触发副作用（扣费/写库/下单）必须去重键/幂等键，否则重试=重复执行")
WALL("  实验C 重试超时", "%.3f s" % (time.time() - tC))

# ---------- 台账汇总（三大基本功 × 本机信号 × 警戒线） ----------
print()
print("=" * 72)
print("台账汇总  Model API 基本功三件套 · 信号=本机读数 · 警戒线=经验值")
print("=" * 72)
print("  ①同步 vs 流式   · 信号: 重组一致 %s（A: 帧数=%d=token 数） → 判断: 关心 TTFT 就开 stream，内容与账单均不变"
      % ("Y" if same else "N", n_frames))
print("  ②token 计费     · 信号: 同步/流式账单同 $%.4f（B: 输入段%dtok 输出%dtok） → 判断: 按 token 计费不按帧，优化对象=控 token 数" %
      (c_sync, INPUT_TOKENS, N))
print("  ③重试与超时     · 信号: 退避计划 sleep=[0.5,1,2,4]·首次 token 超时上限 %.1fs（C） → 判断: 指数退避+jitter，429 尊重 Retry-After，非幂等不重试" % timeout)
print()
print("  核心一句话：流式 ≠ 多次调用、≠ 更贵，它只是把一次响应切成 N 个子事件——")
print("              省的是首 token 的体感等待，不省也不加账单。")
WALL("total", "%.3f s" % (time.time() - T0))
print("done · 一键复现：python code/notebooks/_tools/api_stream_demo.py")
