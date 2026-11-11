# -*- coding: utf-8 -*-
"""对照索引一致性检查（PLAN §1.5 / CONTRIBUTING 承诺的 CI 第二道闸）。

规则：仓库里"已存在的"每一样东西，都必须在 `00-知识地图/对照索引.md` 里有登记——
  1. 每个编号顶层目录（00-16 及其它 NN- 目录）须出现在索引 §1 总览表；
  2. 每个目录下的正文 md（非地图/索引自身）须可按路径在索引 §2 命中；
  3. 每个 `code/notebooks/*.ipynb` 须在索引 §3 命中；
  4. `code/notebooks/_tools/` 下的篇章探针（排除 build_/exec_ 辅助脚本）须全部出现在
     索引 §5 统计快照的探针名单中，且声明数 == 名单数 == 名单文件全部落盘；
  5. 索引里出现的相对链接是否存在（兜底，与 check_links.py 互补）。

用于：新增探针 / 正文 / notebook 而不更新对照索引 → CI 失败。

用于：新增任何内容文件而不更新对照索引 → CI 失败，强制"写一篇挪一篇"。
用法：python tools/check_index_consistency.py   （仓库根运行）
"""
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.getcwd()
INDEX = os.path.join(ROOT, '00-知识地图', '对照索引.md')


def read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


def main():
    index_text = read(INDEX)
    bad = []

    # 1) 编号顶层目录必须在 §1 出现（索引里以 `NN-名称` 反问号表格的方式出现）
    num_dirs = sorted(
        d for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d)) and re.match(r'^\d{2}-', d)
    )
    for d in num_dirs:
        if ('`%s`' % d) not in index_text:
            bad.append('顶层目录未在对照索引 §1 登记: %s' % d)

    # 2) 目录内正文 md（顶层平铺）须路径命中索引 §2
    #    路径在索引里以 ../NN-xxx/file.md 形式出现，故用"NN-xxx/file.md"子串匹配
    for d in num_dirs:
        for fn in sorted(os.listdir(os.path.join(ROOT, d))):
            if not fn.lower().endswith('.md'):
                continue
            rel = '%s/%s' % (d, fn)
            if rel in INDEX or fn == '对照索引.md' or fn == '大模型知识地图_完整版.md':
                continue
            if rel not in index_text:
                bad.append('正文未在对照索引 §2 登记: %s' % rel)

    # 3) code/notebooks 的 ipynb 须在索引 §3 命中
    nb_dir = os.path.join(ROOT, 'code', 'notebooks')
    if os.path.isdir(nb_dir):
        for fn in sorted(os.listdir(nb_dir)):
            if not fn.lower().endswith('.ipynb'):
                continue
            if 'code/notebooks/%s' % fn not in index_text:
                bad.append('notebook 未在对照索引 §3 登记: code/notebooks/%s' % fn)

    # 4) 篇章探针登记：磁盘上的探针须全部进索引名单，名单也要与声明数对得上
    #    （堵住「写了 _tools/xxx_demo.py 却漏登记 §5 统计快照」的漏洞）
    tools_dir = os.path.join(ROOT, 'code', 'notebooks', '_tools')
    AUX_PREFIX = ('build_', 'exec_')      # notebook 构建/执行辅助脚本，非篇章探针
    probe_files = sorted(
        f for f in os.listdir(tools_dir)
        if f.endswith('.py') and not f.startswith(AUX_PREFIX)
    ) if os.path.isdir(tools_dir) else []
    m = re.search(r'篇章探针另置于 `code/notebooks/_tools/`（(\d+) 个：(.*?)）',
                  index_text, re.S)
    if not m:
        bad.append('对照索引 §5 缺少篇章探针统计名单')
    else:
        declared = int(m.group(1))
        listed = re.findall(r'`([a-z0-9_]+\.py)`', m.group(2))
        if declared != len(listed):
            bad.append('探针声明数 %d 与名单数 %d 不一致' % (declared, len(listed)))
        if len(set(listed)) != len(listed):
            bad.append('探针名单存在重复登记')
        for n in listed:
            if not os.path.isfile(os.path.join(tools_dir, n)):
                bad.append('探针名单中的文件不存在: %s' % n)
        for n in probe_files:
            if n not in listed:
                bad.append('篇章探针未在对照索引登记: code/notebooks/_tools/%s' % n)

    # 5) 索引内相对链接兜底（复用 check_links 同款逻辑的轻量版）
    LINK = re.compile(r'(?<!!)\[[^\]]*\]\(([^)]+)\)')
    for m in LINK.finditer(index_text):
        tgt = m.group(1)
        if tgt.startswith(('http', 'mailto', 'www', '#', '/', '<', '{')):
            continue
        full = os.path.normpath(os.path.join(os.path.dirname(INDEX), tgt))
        if not os.path.exists(full):
            bad.append('对照索引内死链: %s' % tgt)

    if bad:
        print('data-notice: 对照索引一致性发现 %d 处违规' % len(bad))
        for x in bad:
            print('::error::%s' % x)
        sys.exit(1)
    print('index-consistency OK: 目录 %d · notebook %d · 探针 %d' % (
        len(num_dirs),
        len([f for f in os.listdir(nb_dir) if f.endswith('.ipynb')]),
        len(probe_files)))


if __name__ == '__main__':
    main()
