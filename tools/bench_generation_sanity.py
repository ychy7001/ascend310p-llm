# SPDX-License-Identifier: Apache-2.0
"""Small deterministic service-generation sanity gate for W4A8/MTP."""
import argparse, json, re, time, urllib.request

PROMPTS = [
    ('english', 'Explain in three short sentences why INT4 weights reduce memory traffic.'),
    ('中文', '用三句话说明 INT4 权重为什么能减少内存流量。'),
    ('code', 'Write a Python function that returns the sum of a list of integers.'),
    ('math', 'Compute 37 multiplied by 19 and show the arithmetic.'),
    ('repeat', 'quantization bandwidth precision ' * 32),
]

def call(base, model, prompt, max_tokens):
    body={'model':model,'messages':[{'role':'user','content':prompt}],
          'temperature':0,'max_tokens':max_tokens,'stream':False,
          'chat_template_kwargs':{'enable_thinking':False}}
    req=urllib.request.Request(base.rstrip('/')+'/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    start=time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r: payload=json.load(r)
    elapsed=time.perf_counter()-start
    choice=payload.get('choices',[{}])[0]; text=((choice.get('message') or {}).get('content') or '')
    usage=payload.get('usage') or {}
    issues=[]
    if not text.strip(): issues.append('empty_text')
    if re.search(r'(?<![A-Za-z])(?:nan|inf|-inf)(?![A-Za-z])', text.lower()): issues.append('nonfinite_literal')
    if len(text) > 2 and text.count(text[:min(20,len(text))]) > 8: issues.append('repeated_prefix')
    return {'label':None,'elapsed_s':elapsed,'text':text,'usage':usage,'finish_reason':choice.get('finish_reason'),'issues':issues}

ap=argparse.ArgumentParser(); ap.add_argument('--base-url',required=True); ap.add_argument('--model',required=True); ap.add_argument('--out',required=True); ap.add_argument('--max-tokens',type=int,default=128); a=ap.parse_args()
rows=[]
for label,prompt in PROMPTS:
    row=call(a.base_url,a.model,prompt,a.max_tokens); row['label']=label; rows.append(row); print(json.dumps({'label':label,'usage':row['usage'],'issues':row['issues']}),flush=True)
result={'status':'PASS' if all(not r['issues'] for r in rows) else 'FAIL','model':a.model,'cases':len(rows),'issues':sum((r['issues'] for r in rows),[]),'rows':rows}
with open(a.out,'w') as f: json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':result['status'],'cases':len(rows),'issues':result['issues']}))
if result['status']!='PASS': raise SystemExit(1)
