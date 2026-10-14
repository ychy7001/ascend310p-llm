"""以关闭思考的真实 Chat 模板验证共享文档 A/B/A 和独立冷参考。"""
import json
import time
import urllib.request
from pathlib import Path

BASE = 'http://127.0.0.1:18024'
MODEL = 'qwen36-w4a8'
common = ('Read the document and answer the final question.\n' +
          'A routine note about memory testing.\n' * 100 +
          'The secret code is ASCEND732941. The project owner is LinMei.\n' +
          'A routine note about memory testing.\n' * 100)
questions = {'A': ('What is the secret code? Reply with only the code.', 'ASCEND732941'),
             'B': ('Who is the project owner? Reply with only the name.', 'LinMei')}
rows = []
for label, key, salt in [('A-cold','A','opi-prefix-v2'), ('A-hot','A','opi-prefix-v2'),
                         ('B-shared','B','opi-prefix-v2'), ('A-after-B','A','opi-prefix-v2'),
                         ('B-cold','B','opi-prefix-cold-v2')]:
    question, expected = questions[key]
    body = {'model':MODEL, 'messages':[{'role':'user','content':common+question}],
            'temperature':0, 'max_tokens':64, 'cache_salt':salt,
            'chat_template_kwargs':{'enable_thinking':False}}
    started = time.perf_counter()
    req = urllib.request.Request(BASE+'/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=180) as response:
        result = json.load(response)
    text = result['choices'][0]['message']['content'].strip()
    row = {'label':label, 'expected':expected, 'correct':text==expected,
           'elapsed_s':time.perf_counter()-started, 'response':result}
    rows.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)
Path('/data/qwen36-deploy/prefix-chat-c16-mtp0.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
assert all(r['correct'] for r in rows), 'Prefix-state answer mismatch'
