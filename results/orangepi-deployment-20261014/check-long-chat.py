"""真实 Chat 模板、关闭思考的长文检索 canary；不是正式长文质量集。"""
import argparse
import hashlib
import json
import time
import urllib.request
import sys
from pathlib import Path
sys.path.insert(0, '/data/qwen36-deploy/repo/tools')
from bench_server_phases import snapshot, prefix_snapshot, METRICS
p=argparse.ArgumentParser()
p.add_argument('--tokens',type=int,required=True)
p.add_argument('--label',required=True)
p.add_argument('--spec',action='store_true')
a=p.parse_args()
base='http://127.0.0.1:18024'; model='qwen36-w4a8'
def post(path,body):
    req=urllib.request.Request(base+path,data=json.dumps(body).encode(),headers={'Content-Type':'application/json','Connection':'close'})
    with urllib.request.urlopen(req,timeout=1800) as r:return json.load(r)
def tokenize(text):
    return post('/tokenize',{'model':model,'prompt':text,'add_special_tokens':False})['tokens']
# Explicit template matches the checkpoint's non-thinking Chat generation path.
prefix=tokenize('<|im_start|>user\nRead the following notes and find the secret code.\nDocument:\n')
needle=tokenize('\nThe secret code is ASCEND732941.\n')
suffix=tokenize('\nEnd of notes. Return only the secret code.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
budget=a.tokens-len(prefix)-len(needle)-len(suffix)
filler=tokenize('Ordinary note: calibration and memory measurements are recorded carefully. '*(budget//10+100))[:budget]
assert len(filler)==budget
ids=prefix+filler[:budget//2]+needle+filler[budget//2:]+suffix
assert len(ids)==a.tokens
rows=[]
for kind in ('cold','hot'):
    before=snapshot(base,a.spec); cache_before=prefix_snapshot(base)
    started=time.perf_counter()
    result=post('/v1/completions',{'model':model,'prompt':ids,'max_tokens':64,'temperature':0,'cache_salt':f'opi-needle-{a.label}-{a.tokens}'})
    elapsed=time.perf_counter()-started
    after=snapshot(base,a.spec); cache_after=prefix_snapshot(base)
    delta={k:after[k]-before[k] for k in before}
    row={'kind':kind,'tokens':a.tokens,'input_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
         'elapsed_s':elapsed,'phase_delta':delta,'cache_delta':{k:cache_after[k]-cache_before[k] for k in cache_before},
         'correct':result['choices'][0]['text'].strip()=='ASCEND732941','response':result}
    rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    Path(f'/data/qwen36-deploy/needle-{a.label}-{a.tokens}.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
assert all(row['correct'] for row in rows), 'Needle answer mismatch'
