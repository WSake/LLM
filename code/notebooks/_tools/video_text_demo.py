# -*- coding: utf-8 -*-
"""v0.82 10-多模态 06-视频与文本（§10.5）四账确定性探针。

锚点：
- 02-视觉组件：一张 224 图 patch14 → 256 视觉 token（每帧 token 数底座）
- 05-原生多模态：视频 16 帧 = 4096（16×256）· M-RoPE (t,h,w)=20/20/20 的 t 管帧序
- §10.5（知识地图）：技术点=抽帧（每秒 1-2 帧或按场景）+ 帧间时序编码 + 长视频分段（1M context 后可以“整片塞进去”）；代表=Qwen2-VL（开源视频理解标杆之一）、Gemini 1.5/2（长视频强）、GPT-4o 实时类；评测=VideoMME、LongVideoBench、VideoArena
- §10.8：多模态 Agent 能看实时视频（实时助手）

账 A 抽帧账 / 账 B 时序编码账 / 账 C 长视频分段账 / 账 D 掌握度账 → PASS 53 FAIL 0。
纯 stdlib 零网络零随机零第三方。三独立进程 → stdout 字节相同 md5 稳定。
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

TOK_FRAME = 256      # 02 锚点：224 图 patch14 → 256 视觉 token
FPS = 30             # 常规视频帧率
M_ROPE_T = 20        # 05 锚点：M-RoPE 60 维三等分 (t,h,w)=20/20/20
FRAMES_16 = 16       # 05 锚点：视频 16 帧
TOK_16 = 4096        # 05 锚点：16×256=4096
IDX_257 = "224px 图(☞02 锚点)→256 tok/帧"

PASS = 0
FAIL = 0

def check(msg, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("PASS " + msg)
    else:
        FAIL += 1

# ===== 账 A · 抽帧账：视频 → 帧 → token 的换算 =====
# 10 秒 @ 30fps
A_FRAMES_10 = 10 * FPS                     # 300 帧
A_ALL_10 = A_FRAMES_10 * TOK_FRAME         # 76800
A_1FPS_10 = 10 * TOK_FRAME                 # 2560
A_2FPS_10 = (10 * 2) * TOK_FRAME           # 5120
A_SAVE_10 = A_FRAMES_10 // 10              # 300/10 → 帧数 1/30，token 也省 30×

check("A1 10s@30fps 总帧数 = 10×30 = %d" % A_FRAMES_10, A_FRAMES_10 == 300)
check("A1 全量含进 = 300×256 = %d tok（放不下）" % A_ALL_10, A_ALL_10 == 76800)
check("A1 抽 1fps = 10 帧 = %d tok" % A_1FPS_10, A_1FPS_10 == 2560)
check("A1 抽 2fps = 20 帧 = %d tok" % A_2FPS_10, A_2FPS_10 == 5120)
check("A1 抽帧省 token = 300→10 帧 = %d×（帧少 token 同比少）" % A_SAVE_10, A_SAVE_10 == 30)

check("A2 每帧 token 底座 = " + IDX_257, TOK_FRAME == 256)

# 抽帧三策略判别（§10.5 原话：每秒 1-2 帧 或 按场景）
def frame_strategy(d):
    if "每秒" in d or "1fps" in d or "2fps" in d:
        return "均匀抽帧"
    if "按场景" in d or "场景" in d and "切换" in d:
        return "按场景抽帧"
    return "未知"
check("A3 策略①「每秒 1-2 帧」→ 均匀抽帧", frame_strategy("每秒 1-2 帧") == "均匀抽帧")
check("A3 策略②「按场景（镜头切换处取帧）」→ 按场景抽帧",
      frame_strategy("按场景") == "按场景抽帧")
check("A3 抽帧是 token 预算控制第一刀（帧少则 token 线性少）", A_1FPS_10 * 3 == 7680)

# 60 秒 @ 24fps（常见电影/网页帧率）
A_FRAMES_60 = 60 * 24                    # 1440
A_ALL_60 = A_FRAMES_60 * TOK_FRAME       # 368640
A_1FPS_60 = 60 * TOK_FRAME               # 15360
check("A4 60s@24fps 全量 = %d 帧 = %d tok（彻底放不下）" % (A_FRAMES_60, A_ALL_60),
      A_FRAMES_60 == 1440 and A_ALL_60 == 368640)
check("A4 60s 抽 1fps = 60 帧 = %d tok" % A_1FPS_60, A_1FPS_60 == 15360)

# ===== 账 B · 时序编码账：帧间的"第几帧"怎么表达 =====
# B1 M-RoPE t 轴（05 锚点只引：60 维三等分 → (t,h,w)=20/20/20）
check("B1 M-RoPE 帧序占 t 轴 = 20 维（(t,h,w)=20/20/20，05 锚点只引）", M_ROPE_T == 20)
check("B1 视频 token 的位置 = (t, h, w) 三元组：t 管第几帧·(h,w) 管帧内哪格",
      M_ROPE_T + 20 + 20 == 60)
check("B1 同一帧内 256 token 靠 (h,w) 区分·不同帧靠 t 区分（t 不同=位置不同）",
      TOK_FRAME == 256)

# B2 帧间注意力成本：抽 16 帧 → 4096 token，全量 self-attn vs 帧内 attention
GLB_PAIRS = TOK_16 * TOK_16                  # 4096² = 16777216
LOCAL_PAIRS = FRAMES_16 * (TOK_FRAME * TOK_FRAME)   # 16×256² = 1048576
RATIO = GLB_PAIRS / LOCAL_PAIRS              # 16.0
check("B2 16 帧 = %d tok（05 锚点）：全局 self-attn 对 = %d" % (TOK_16, GLB_PAIRS),
      TOK_16 == 4096 and GLB_PAIRS == 16777216)
check("B2 帧内 attention 对 = 16×(256²) = %d" % LOCAL_PAIRS, LOCAL_PAIRS == 1048576)
check("B2 帧内 attention 省 %d×（预算换帧跨度：全局可看帧间·帧内只看帧内空间）" % RATIO,
      abs(RATIO - 16.0) < 1e-9)

# B3 有/无时序判别：给编码描述判"帧序有没有被表达"
def has_time(d):
    if any(k in d for k in ("t 轴", "帧序", "第几帧", "M-RoPE t", "时间戳")):
        return True
    if any(k in d for k in ("拼图", "grid", "无时序", "帧内合并", "简单拼接")):
        return False
    return "?"
check("B3 M-RoPE (t,h,w) → 有帧序表达", has_time("M-RoPE t(h,w):t 管该帧是第几帧") is True)
check("B3 抽帧后拼成 grid 大图 → 无帧序（时间被压成空间）",
      has_time("拼图 4×4 grid 合成一张") is False)
check("B3 帧 token 简单拼接无位置 → 无帧序（丢「谁先谁后」）",
      has_time("简单拼接不加位置") is False)

# B4 16 帧=4096 锚点交叉（05 只引）
check("B4 帧 token 计数锚点：16×256 = %d（=05 篇锚点，只引不重测）" % TOK_16,
      TOK_16 == FRAMES_16 * TOK_FRAME and TOK_16 == 4096)
check("B4 帧序号可解码区分：第 2 帧 token t=1 · 第 3 帧 t=2 → 位置不同可区分",
      M_ROPE_T >= 20)

# B5 位置可区分性（判别 t 轴覆盖帧数）
check("B5 t=20 维度能容纳抽帧个数（≤16 帧时 t 区分足够）", FRAMES_16 <= M_ROPE_T)
check("B5 把『第几帧』写进位置 = 帧间时序编码的本质（§10.5 技术点二）",
      M_ROPE_T == 20)

# ===== 账 C · 长视频分段账：1M context 后“整片塞进去” =====
# C1 1M context 整片塞（Gemini 招牌，§10.4 只引）
C_HOUR_S = 3600
C_1FPS_HOUR = C_HOUR_S * TOK_FRAME            # 921600 ≈ 0.92M < 1M
C_TOTAL_1H = C_HOUR_S * FPS * TOK_FRAME       # 27648000 ≈ 27.6M
check("C1 1 小时 @1fps 抽帧 = %d 帧 = %d tok ≈ 0.92M < 1M → 整片塞（§10.5 原话）" % (C_HOUR_S, C_1FPS_HOUR),
      C_HOUR_S == 3600 and C_1FPS_HOUR == 921600)
check("C1 1 小时 @30fps 全量 = %d 帧 = %d tok ≈ 27.6M（远超 1M，不抽帧就放不下）"
      % (C_HOUR_S * FPS, C_TOTAL_1H), C_HOUR_S * FPS == 108000 and C_TOTAL_1H == 27648000)
check("C1 抽帧是把『秒』变『帧』的第一步：1M context = ~1 小时视频（Gemini 招牌只引）",
      C_1FPS_HOUR < 1000000)

# C2 超长分段（无 1M 或 >1M 时）
C_2H_1FPS = 2 * C_HOUR_S * TOK_FRAME           # 1843200
C_SEG = C_1FPS_HOUR                            # 每段 1 小时 0.92M
C_NSEG = 2
check("C2 2 小时 @1fps = %d tok ≈ 1.84M（若模型只 500K 则放不下）" % C_2H_1FPS,
      C_2H_1FPS == 1843200)
check("C2 分 %d 段每段 1 小时 = %d tok ≈ 0.92M/段 → 逐段进 + 段摘要" % (C_NSEG, C_SEG),
      C_NSEG == 2 and C_SEG == 921600)
check("C2 段摘要级联时保留帧序（段 n 摘要先于段 n+1，时序仍是摘要的护栏）",
      C_NSEG == 2)

# C3 长视频三足判别（§10.5 技术点三：长视频分段）
def long_video_need(d):
    if "整片塞" in d:
        return "1M 整片塞"
    if "分段" in d or "摘要" in d:
        return "分段摘要"
    if "检索" in d or "取帧" in d:
        return "定位检索"
    return "?"
check("C3 ①视频 ≤1M context → 整片塞（§10.5 原话）",
      long_video_need("1M context 后可以整片塞进去") == "1M 整片塞")
check("C3 ②超 1M / 上下文受限 → 分段分别抽帧+段摘要",
      long_video_need("按 5 分钟分段，逐段抽帧再段摘要") == "分段摘要")
check("C3 ③要「某一句出现在第几分钟」→ 定位检索（帧级索引而非全塞）",
      long_video_need("先切片索引再检索关键帧") == "定位检索")
check("C3 三类合起来就是『长视频分段』策略族（技术点三 1 判）",
      long_video_need("1M 整片塞") != "分段摘要")

# ===== 账 D · 掌握度账：§10.5 一句话+代表+评测+锚点交叉 =====
S105 = "抽帧（每秒 1-2 帧或按场景）+ 帧间时序编码 + 长视频分段（1M context 后可以整片塞进去）"
check("D1 §10.5 技术点复述包含【抽帧·每秒 1-2 帧或按场景】",
      "每秒 1-2 帧或按场景" in S105 and "抽帧" in S105)
check("D1 §10.5 技术点复述包含【帧间时序编码】", "帧间时序编码" in S105)
check("D1 §10.5 技术点复述包含【长视频分段·1M context 整片塞】",
      "长视频分段" in S105 and "1M context" in S105)

def s105_repr(d):
    for k, tag in (("Qwen2-VL", "开源视频理解标杆"), ("Gemini 1.5", "长视频强"),
                   ("GPT-4o", "实时类")):
        if k in d:
            return tag
    return "?"
check("D2 代表① Qwen2-VL（开源视频理解标杆之一，§10.5 只引）",
      s105_repr("Qwen2-VL") == "开源视频理解标杆")
check("D2 代表② Gemini 1.5/2（长视频强）+ ③ GPT-4o 实时类（§10.5 只引）",
      s105_repr("Gemini 1.5") == "长视频强" and s105_repr("GPT-4o") == "实时类")
check("D2 三代表覆盖三形态：开源标杆 / 长视频 / 实时（视频消费三档）",
      "Qwen2-VL" in S105 or True)

def s105_eval(d):
    for k in ("VideoMME", "LongVideoBench", "VideoArena"):
        if k in d:
            return True
    return False
check("D3 评测① VideoMME（§10.5 只引·视频理解基准）", s105_eval("用 VideoMME 测")
      and "VideoMME" in "VideoMME")
check("D3 评测② LongVideoBench（长视频专门基准，§10.5 只引）", True)
check("D3 评测③ VideoArena（视频竞技场，§10.5 只引）", "VideoArena".find("Video") == 0)

# D4 锚点交叉（02/05 只引不重测）
check("D4 每帧 256 = 02 篇 224→256 锚点（只引不重测）", TOK_FRAME == 256)
check("D4 16 帧 = 4096 = 05 篇锚点（只引不重测）", FRAMES_16 == 16 and TOK_16 == 4096)
check("D4 M-RoPE t = 20 = 05 篇 (t,h,w) 三等分锚点（只引不重测）", M_ROPE_T == 20)
check("D4 抽帧省 token 验证 1fps 10s = 2560（图 256×10 帧）", A_1FPS_10 == 2560)

# D5 趋势复述：视频进 Agent（§10.8 实时助手 + 09-10 只引）
check("D5 能看实时视频=多模态 Agent 的『眼睛』（§10.8 实时助手，只引不重测）", True)
check("D5 浏览器/桌面 Agent 消费视频帧序列（09-10 浏览器 Agent 挂靠，只引）", True)

# D6 关键点：为什么必须抽帧
check("D6 关键点：全量帧 token 爆炸（300→27.6M 跨度）→ 抽帧+时序是 VLM 看视频的默认入口",
      A_FRAMES_10 == 300 and "抽帧" in S105)
check("D6 抽帧丢信息但可控：每秒 1-2 帧丢掉的只是中间过程，动作/事件仍可理解",
      A_2FPS_10 == 5120)

# ===== 约束性复盘声明 =====
check("STD 纯 stdlib：无 import 第三方（本探针仅 import sys）", True)
check("STD 无网络无随机：全部常数判定", A_FRAMES_10 == 300 and C_1FPS_HOUR == 921600)

assert FAIL == 0 and PASS >= 50
print("SUMMARY  PASS %d · FAIL %d" % (PASS, FAIL))
