# -*- coding: utf-8 -*-
"""Markdown 内部链接死链检查（GitHub 语义：相对链接一律按所在文件目录解析）。
   用法: python tools/check_links.py [仓库根]
   外链、锚点、常见写法忽略；裸路径（无 ./ ../ 前缀）同样校验。"""
import os, re, sys, pathlib

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def rel_md_files(root):
    for p in pathlib.Path(root).rglob('*.md'):
        if '.git' in p.parts:
            continue
        yield p

LINK = re.compile(r'(?<!!)\[[^\]]*\]\(([^)]+)\)')
SKIP_PREFIX = ('http://', 'https://', 'mailto:', 'www.', '#', '/', '<', '{')
SKIP_SUBSTR = ('github.com', 'arxiv.org', 'opensource.org')

def main():
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    root = os.path.abspath(root)
    bad = []
    checked = 0
    skipped_py = 0
    for p in rel_md_files(root):
        rel = str(p.relative_to(root)).replace('\\', '/')
        text = p.read_text(encoding='utf-8')
        for m in LINK.finditer(text):
            target = m.group(1)
            # 跳过外链/锚点/占位
            if target.startswith(SKIP_PREFIX):
                continue
            if any(s in target for s in SKIP_SUBSTR):
                continue
            # 路径含空格 → 跳过（避免误报）
            if ' ' in target:
                continue
            # 去掉锚点
            tgt = target.split('#')[0]
            if not tgt:
                continue
            full = os.path.normpath(os.path.join(os.path.dirname(str(p)), tgt))
            if not os.path.exists(full):
                bad.append((rel, target))
            else:
                checked += 1
    # 报告
    if bad:
        print(f'❌ 发现 {len(bad)} 个死链:')
        for f, t in bad:
            print(f'   {f}  →  {t}')
        sys.exit(1)
    print(f'✅ 相对链接全部有效（共校验 {checked} 条）')

if __name__ == '__main__':
    main()
