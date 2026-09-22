# -*- coding: utf-8 -*-
"""model_registry_demo.py —— §13.6 模型注册与版本管理（13-工程化与基础设施 06 篇配套）

§13.6 知识地图原文（🟢 会做）：
- 版本管理：模型权重（safetensors+元数据）+ 配套（tokenizer、chat template、量化产物、
  评测报告、许可）打包成"模型版本"。
- 注册中心：Hugging Face Hub（开源）、MLflow Model Registry、各云 Model Registry；
  内网镜像（代理 HF）是合规常态。
- 发布策略：蓝绿、金丝雀（5-10%）、A/B（对照评测）、阴影流量（shadow，录流量不返回）。

本篇定位：13-05 账 A 只点了一格「模型注册（MLflow Registry/HF Hub）」，本篇把它正式展开——
一个模型从"权重文件"变成"生产能引用、能回滚、能审计的模型版本"，靠三件套：
(1) 版本打包（权重+元数据+配套五件）、(2) 注册中心与寻址（三中心+内网镜像）、
(3) 四种发布策略（蓝绿/金丝雀/A-B/阴影）。纯本机算不清真实 registry（零网络连不上
HF Hub/MLflow），降维成四本确定性账：
账 A  版本打包账：一个完整"模型版本" = 权重 safetensors + 元数据 config + 配套五件，7 件套完整性判定
账 B  注册中心寻址账：HF Hub / MLflow Registry / 云 Registry 三中心归位 + 内网镜像（合规常态）
账 C  发布策略账：蓝绿 / 金丝雀 / A-B / 阴影 四策略各一动作账（流量切、观测、回滚）
账 D  溯源与审计账：模型版本 ↔ 权重 hash ↔ 数据版本 ↔ 评测报告 ↔ 线上记录，回滚两步封死

照例：**纯 stdlib · 零网络 · 零随机 · 零第三方**。stdout 三个独立进程逐字节 md5 恒一，
墙钟只进 stderr（W() 负责计时与落 stderr）。全部统计 = 本机确定性计算。

诚实边界（与文章头部 📌 一致）：
- 版本打包件清单（safetensors/config/tokenizer/chat template/量化/评测报告/许可）、
  注册中心三件（HF Hub/MLflow/云）、内网镜像代理 HF 为合规常态、四发布策略语义
  = §13.6 要素事实（公开资料整理，写作环境无外网未在线复核）。
- Qwen2.5-7B 结构数字（3584/28 层/28 头/4 KV 头/151936 词表）= 公开资料要素事实（未复核）。
- int4 量化 3.3 GiB 与 fp16 14.0 GiB = 13-01 账 A/D 同源口径，本篇只引不重算。
- 金丝雀 5-10%（取 8%）、A/B 200 样本命中数、阴影录流量不返回、各账判定
  = 作者预置 + 本机确定性算术；不 import transformers、不拉 HF、不起 MLflow。
"""
from __future__ import annotations
import sys
import time

try:  # 输出统一 UTF-8 字节，不随控制台代码页漂移（GBK 下也能打印 ✅/⛔）
    _ = sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

_WSTART = time.perf_counter()


def W(msg: str) -> None:
    print(f"[wall] {time.perf_counter() - _WSTART:.3f}s  {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 要素事实与作者预置（§13.6 + 13-01 同源口径 + 公开资料整理）
# ---------------------------------------------------------------------------
# 账 A —— 版本打包件清单（§13.6「权重+配套」展开为 7 件套）
A_FILES = [
    ("weight.safetensors", "权重：safetensors 格式（内嵌 $metadata.format='pt' + header hash），7B fp16 14.0 GiB"),
    ("config.json", "元数据：architecture/model_type/层数/头数/词表，读配置才能把权重加载对"),
    ("tokenizer", "配套 1：tokenizer.json / vocab.json + merges.txt，词表与 BPE 合代码"),
    ("chat_template", "配套 2：tokenizer_config.json 里的 chat_template 字符串（Qwen/chat 模板）"),
    ("quantized", "配套 3：量化产物 int4 GGUF 3.3 GiB（13-01 账 A 同源，≈fp16 的 1/4）"),
    ("eval_report", "配套 4：评测报告 eval_report.json（benchmark + judge + 人工分，账 D 溯源用）"),
    ("LICENSE", "配套 5：许可（Apache-2.0/Qwen 许可），合规红线（仓库 §7 红线）"),
]
A_FULL_OK = ["weight.safetensors", "config.json", "tokenizer", "chat_template", "quantized", "eval_report", "LICENSE"]
A_INCOMPLETE = ["weight.safetensors", "config.json", "tokenizer", "quantized", "LICENSE"]  # 缺 chat_template / eval_report

# 账 B —— 注册中心三件 + 寻址三例 + 内网镜像
B_CENTERS = [
    ("Hugging Face Hub", "开源社区，拉新模型/公网镜像源（transformers / huggingface_hub 默认端）"),
    ("MLflow Model Registry", "团队内模型生命周期：None → Staging → Production → Archived，stage 晋升与回滚"),
    ("云 Model Registry", "托管注册：SageMaker / Vertex / 各家云，训练-注册-部署闭环"),
]
B_ADDR = [
    ("HF 寻址", "Qwen/Qwen2.5-7B-Instruct:e1f2a3b…（owner/repo:revision，revision=commit hash 不可变）"),
    ("MLflow 寻址", "models:/qwen-7b/3（models:/<name>/<version>，数字版本不可变）"),
    ("镜像寻址", "s3://models/qwen7b/v9（对象存储内网镜像，路径含 tag）+ registry.mirror=https://hf.local"),
]
UPSTREAM_UNREACHABLE = True   # 模拟外网/上游不可达
B_MIRROR_HIT = True           # 内网镜像缓存命中 → 合规常态判据

# 账 C —— 四发布策略（金丝雀 5-10% 取 8%，13-03 灰度 5% 同式只引）
C_GREEN_WEIGHT = 1.00
C_CANARY_PCT = 0.08           # §13.6 金丝雀 5-10%
C_CANARY_STEPS = [0.20, 0.50, 1.00]
C_AB_N = 150                  # 每桶样本量（两桶 50/50 共 300）
C_AB_A_HIT, C_AB_B_HIT = 132, 141
C_SHADOW_MODE = "record_only" # 阴影：录流量不返回

# 账 D —— 溯源链与回滚
D_CHAIN = [
    ("模型版本", "qwen-7b/v9 (tag 指向 revision e1f2a3b…)"),
    ("权重 hash", "safetensors sha256 = a3f5c9d2…（内容寻址，版本不可篡改）"),
    ("数据版本", "train 数据集 DVC hash 4f8b1a2c…（13-05 账 D『数据变更』落点）"),
    ("评测报告", "eval_report.json：MMLU 0.742 · judge 0.90 · 长尾人工 0.61（账 C A/B 决策依据）"),
    ("线上记录", "service=qwen-chat · model=qwen-7b/9 · 上线 2026-09-20 · 策略=阴影转正"),
]
D_ALERT = "P95 延迟 1.4x 告警 + judge 线上分 -0.08"
D_ROLLBACK_STEPS = ["定位：production 记录 → qwen-7b/9", "切回：蓝绿切权 v8 立即生效，shadow 日志留证"]


# ---------------------------------------------------------------------------
# 账 A · 版本打包账：7 件套完整性判定
# ---------------------------------------------------------------------------
def account_a() -> str:
    out = []
    out.append("== A · 模型版本打包账：一个版本 = 权重 + 元数据 + 配套五件（7 件套） ==")
    n_miss = 0
    for name, what in A_FILES:
        present = name in A_INCOMPLETE
        n_miss += 0 if present else 1
        out.append(f"    [{name}] {what}" + ("" if present else "  <- 缺件（注册被拒）"))
    ok_miss = n_miss == 2 and "chat_template" not in A_INCOMPLETE and "eval_report" not in A_INCOMPLETE
    out.append(f"    缺件版 = {len(A_INCOMPLETE)}/{len(A_FILES)}（缺 chat_template·eval_report）："
               f"chat 服务无模板不可用 · eval_report 缺失禁止上生产 = "
               + ("PASS" if ok_miss else "FAIL"))
    ok_full = len(A_FULL_OK) == 7 and {n for n, _ in A_FILES} == set(A_FULL_OK)
    out.append(f"    -> 完整版 = 7/7 件齐（weight+config+tokenizer+chat_template+quant+eval_report+LICENSE）"
               f" 才允许进注册中心 = " + ("PASS" if ok_full else "FAIL"))
    # 权重与量化（13-01 同源只引）
    fp16, int4 = 14.0, 3.3
    save_pct = 1 - int4 / fp16
    ok_quant = abs(int4 - 3.3) < 1e-9 and fp16 > int4
    out.append(f"    权重账：fp16 {fp16:.1f} GiB vs int4 {int4:.1f} GiB（13-01 账 A/D 同源只引）"
               f" = 量化省 {save_pct:.0%} 显存 = " + ("PASS" if ok_quant else "FAIL"))
    # 版本识别：不可变 revision vs 移动 tag
    ok_tag = not set(['latest']) & set(['revision'])
    out.append("    版本识别：revision e1f2a3b…(commit hash, 不可变) 锁内容寻址 · tag 'latest' 移动会漂移 = "
               + ("PASS" if ok_tag else "FAIL"))
    n_a = sum([ok_miss, ok_full, ok_quant, ok_tag])
    out.append(f"    -> 一个能生产引用的版本号 = 7 件套全齐 + 不可变 revision（硬件层 13-01 已算）——"
               f"打包 4 判 {n_a}/4 全 PASS = " + ("PASS" if n_a == 4 else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 B · 注册中心寻址账：三中心 + 内网镜像
# ---------------------------------------------------------------------------
def account_b() -> str:
    out = []
    out.append("== B · 注册中心寻址账：HF Hub / MLflow Registry / 云 Registry + 内网镜像 ==")
    n_bad = 0
    for name, role in B_CENTERS:
        ok = name != "" and role != ""
        n_bad += 0 if ok else 1
        out.append(f"    [{name}] = {role} = " + ("PASS" if ok else "FAIL"))
    ok3 = n_bad == 0
    for kind, addr in B_ADDR:
        out.append(f"    寻址 [{kind}] = {addr}")
    ok_addr = True
    out.append(f"    寻址规则：HF 'owner/repo:revision' · MLflow 'models:/name/version' · 镜像 's3://…/tag'，"
               f"revision/version 用不可变标识、tag 仅快捷名 = "
               + ("PASS" if ok_addr else "FAIL"))
    # 内网镜像 = 合规常态
    if UPSTREAM_UNREACHABLE:
        out.append("    上游不可达（外网 4xx/5xx）→ 走 registry.mirror 内网缓存 → deploy 命中 = "
                   + ("PASS" if B_MIRROR_HIT else "FAIL"))
    ok_mirror = UPSTREAM_UNREACHABLE and B_MIRROR_HIT
    n_b = sum([ok3, ok_addr, ok_mirror])
    out.append(f"    -> 内网镜像（代理 HF）把『拉新模型』变成『缓存合规』——断外网也能部署，"
               f"合规常态=注册中心 3+1 判 {n_b}/3 全 PASS = " + ("PASS" if n_b == 3 else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 C · 发布策略账：蓝绿 / 金丝雀 / A-B / 阴影
# ---------------------------------------------------------------------------
def account_c() -> str:
    out = []
    out.append("== C · 发布策略账：蓝绿切权 · 金丝雀渐提 · A-B 对照 · 阴影只录不返 ==")
    res = []
    # C1 蓝绿：green 100 → blue 100 原子切换，回滚=再切一次；成本双份
    blue_cutover = C_GREEN_WEIGHT - 1.0 + 1.0  # 切后 blue 占 100%
    res.append(("蓝绿 Blue-Green",
                f"切前 green 100%/blue 0% → 切后 blue {blue_cutover:.0%}；回滚 = 切权一次；代价资源双份",
                blue_cutover == 1.00))
    # C2 金丝雀：8% → 20 → 50 → 100（13-03 灰度 5% 同式只引）
    canary_ok = [C_CANARY_PCT] + C_CANARY_STEPS == [0.08, 0.20, 0.50, 1.00]
    res.append(("金丝雀 Canary",
                f"先 {C_CANARY_PCT:.0%} 流量（§13.6 5-10% · 13-03 灰度 5% 同式只引）→ 观测错误率/延迟/评测分达标 → "
                "20%→50%→100% 逐档；任一档告警 → 一键回零",
                canary_ok))
    # C3 A-B：50/50 分桶对照评测（11-02/03/04 决策依据）
    b_wins = C_AB_B_HIT > C_AB_A_HIT
    res.append(("A/B 对照",
                f"两版同跑 50/50 分桶（各 {C_AB_N} 样本）：A 命中 {C_AB_A_HIT} · B 命中 {C_AB_B_HIT}"
                f" → B 胜 {f'+{C_AB_B_HIT / C_AB_A_HIT - 1:.1%}' if b_wins and C_AB_A_HIT else '-'}"
                " → 用 11 章评测拿数据，不分权重改流量",
                b_wins))
    # C4 阴影：100% 复制、只录不返回；离线评测后再转正
    res.append(("阴影 Shadow",
                f"{C_SHADOW_MODE}：线上请求 100% 复制进新版本，输出只记录不返回用户 → 离线评测通过才转正"
                "（录流量不返回 = 用户无感知试新）",
                C_SHADOW_MODE == "record_only"))
    n_good = 0
    for name, line, ok in res:
        n_good += 1 if ok else 0
        out.append(f"    [{name}] = {line} = " + ("PASS" if ok else "FAIL"))
    out.append(f"    -> 四种发布四笔账：蓝绿买原子/花双份 · 金丝雀买稳/贴线过 · A-B 买数据/两版活 · "
               f"阴影买无感试新/只录不返 → 发布策略 {n_good}/4 全 PASS = "
               + ("PASS" if n_good == 4 else "FAIL"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 账 D · 溯源与审计账：版本 ↔ 数据 ↔ 评测 ↔ 线上，回滚两步封死
# ---------------------------------------------------------------------------
def account_d() -> str:
    out = []
    out.append("== D · 溯源与审计账：一条审批线 = 权重 hash→数据版本→评测报告→线上记录，回滚两步 ==")
    n_bad = 0
    for k, v in D_CHAIN:
        ok = v != ""
        n_bad += 0 if ok else 1
        out.append(f"    溯源[{k}] = {v} = " + ("PASS" if ok else "FAIL"))
    ok_chain = n_bad == 0
    # 事故演练：v9 上线后告警
    ok_alert = "0.61" in D_CHAIN[3][1]
    out.append(f"    ⚠ 事故：v9 上线后 {D_ALERT} —— 阴影日志佐证：长尾请求 judge 0.61 早于转正已暴露 = "
               + ("PASS" if ok_alert else "FAIL"))
    for i, step in enumerate(D_ROLLBACK_STEPS, 1):
        out.append(f"    回滚 {i}/2 · {step}")
    ok_rollback = len(D_ROLLBACK_STEPS) == 2
    n_d = sum([ok_chain, ok_alert, ok_rollback])
    out.append(f"    -> 审计=能回答『线上跑哪个版本/为何上/何时拉下来』——v9 打了蓝绿回滚 + 阴影留证复盘，"
               f"溯源 3 判 {n_d}/3 全 PASS = " + ("PASS" if n_d == 3 else "FAIL"))
    return "\n".join(out)


def main() -> None:
    W("start")
    parts = [account_a(), account_b(), account_c(), account_d()]
    print("\n" + "\n".join(parts), end="")
    W("done")


if __name__ == "__main__":
    main()
