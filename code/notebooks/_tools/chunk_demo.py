# -*- coding: utf-8 -*-
"""04-分块与元数据.md 的可复现实验：固定切 vs 换行递归切（拿知识地图开刀）。

用法：仓库根 python code/notebooks/_tools/chunk_demo.py
依赖：无。输出即为 04 章 3 节引用的实数字。"""
import io, re, statistics, sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DOC = r'00-知识地图/大模型知识地图_完整版.md'


def split_fixed(text, size=512):
    return [text[i:i + size] for i in range(0, len(text), size)]


def split_recursive(text, limit=512):
    lines, chunks, cur = text.split('\n'), [], ''
    for line in lines:
        if len(cur) + len(line) + 1 > limit and cur.strip():
            chunks.append(cur)
            cur = line
        elif not line.strip() and cur.strip():
            chunks.append(cur)
            cur = ''
        else:
            cur = (cur + '\n' if cur else '') + line
        if len(cur) >= limit * 3:                      # 兜底：超长段硬切
            for i in range(0, len(cur), limit):
                chunks.append(cur[i:i + limit])
            cur = ''
    if cur.strip():
        chunks.append(cur)
    return chunks


def mid_line_cuts(text, chunks):
    cuts, p = 0, 0
    for c in chunks[:-1]:
        p += len(c)
        if p < len(text) and text[p] != '\n' and text[p - 1] != '\n':
            cuts += 1
    return cuts


def main():
    text = io.open(DOC, encoding='utf-8').read()
    n_head = sum(1 for l in text.split('\n') if re.match(r'^\s*#{1,4}\s', l))
    print('源文档: %d 字符 / %d 行 / 标题 %d 条' % (len(text), len(text.split('\n')), n_head))

    fixed = split_fixed(text)
    print('固定 %d 硬切    : %d 块；切线落在行中间 %d 处'
          % (512, len(fixed), mid_line_cuts(text, fixed)))

    rec = split_recursive(text)
    med = statistics.median(map(len, rec))
    print('换行递归分块     : %d 块；切线落在行中间 %d 处；中位块长 %d 字符'
          % (len(rec), mid_line_cuts(text, rec), med))


if __name__ == '__main__':
    main()
