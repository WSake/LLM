# -*- coding: utf-8 -*-
"""对照索引一致性检查（PLAN §1.5 / CONTRIBUTING 承诺的 CI 第二道闸）。

规则：仓库里"已存在的"每一样东西，都必须在 `00-知识地图/对照索引.md` 里有登记——
  1. 每个编号顶层目录（00-16 及其它 NN- 目录）须出现在索引 §1 总览表；
  2. 每个目录下的正文 md（非"旧提纲"、非地图/索引自身）须可按路径在索引 §2 命中；
  3. 每个 `code/notebooks/*.ipynb` 须在索引 §3 命中；
  4. 索引里出现的相对链接是否存在（兜底，与 check_links.py 互补）。

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

    # 2) 目录内正文 md（顶层平铺，排除"旧提纲"与规划性文件）须路径命中索引 §2
    #    路径在索引里以 ../NN-xxx/file.md 形式出现，故用"NN-xxx/file.md"子串匹配
    for d in num_dirs:
        for fn in sorted(os.listdir(os.path.join(ROOT, d))):
            if not fn.lower().endswith('.md'):
                continue
            rel = '%s/%s' % (d, fn)
            if rel in INDEX or fn == '对照索引.md' or fn == '大模型知识地图_完整版.md':
                continue
            if '旧提纲' in fn:      # 旧提纲集合登记即可，不逐篇强制
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

    # 4) 索引内相对链接兜底（复用 check_links 同款逻辑的轻量版）
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
    print('index-consistency OK: 目录 %d · notebook %d' % (
        len(num_dirs), len([f for f in os.listdir(nb_dir) if f.endswith('.ipynb')])))


if __name__ == '__main__':
    main()
