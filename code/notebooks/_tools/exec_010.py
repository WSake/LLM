# -*- coding: utf-8 -*-
"""exec-harness：顺序执行 notebook 010 的代码单元，把真实 stdout 以 text/plain 输出注入回 ipynb。"""
import contextlib, io as _io, json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

NB = r'F:\00AI创业\01LLM\LLM\code\notebooks\010-RAGAS评测一个RAG系统.ipynb'
nb = json.load(_io.open(NB, encoding='utf-8'))

ns = {'__name__': '__main__'}
order = []
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        order.append(i)

print('>>> 顺序执行 %d 个 code 单元 ...' % len(order))
exec_count = 0
for i in order:
    c = nb['cells'][i]
    src = ''.join(c['source'])
    exec_count += 1
    c['execution_count'] = exec_count
    buf = _io.StringIO()
    try:
        ns['__display_output__'] = []
        with contextlib.redirect_stdout(buf):
            exec(compile(src, '<cell %d>' % exec_count, 'exec'), ns)
    except Exception as e:
        import traceback
        buf.write('\n[CELL %d 异常] %r\n' % (exec_count, e))
        traceback.print_exc(file=buf)
        c['outputs'] = [{'output_type': 'error', 'ename': type(e).__name__,
                         'evalue': str(e), 'traceback': buf.getvalue().splitlines()}]
        c['metadata'].setdefault('tags', []).append('exec-error')
        # 继续执行，保留中间结果便于定位
        nxt = buf.getvalue() + '\n'
        print('  !! cell %d ERROR' % exec_count)
        print(nxt[:4000])
        continue
    out = buf.getvalue()
    txt = out if out else '<无输出>'
    c['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': txt.splitlines(keepends=True)}]
    if out.strip():
        print('  [cell %d] ok · %d 行输出' % (exec_count, out.count('\n') + 1))
    else:
        print('  [cell %d] ok · 无输出' % exec_count)

with _io.open(NB, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write('\n')
print('>>> 完成，输出已注入 back; code 错误单元:', sum(1 for c in nb['cells'] if c['cell_type']=='code' and 'exec-error' in c['metadata'].get('tags', [])))
