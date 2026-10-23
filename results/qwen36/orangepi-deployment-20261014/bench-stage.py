"""冷请求与热前缀分开测量，输出 phase / TTFT / TPOT 与内存快照。"""
import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path
import sys
sys.path.insert(0, '/data/qwen36-deploy/repo/tools')
from bench_http_stream import request, post
from bench_server_phases import snapshot, prefix_snapshot, METRICS

p=argparse.ArgumentParser()
p.add_argument('--label',required=True)
p.add_argument('--tokens',type=int,required=True)
p.add_argument('--repeats',type=int,default=3)
p.add_argument('--output',type=int,default=128)
p.add_argument('--spec',action='store_true')
a=p.parse_args()
base='http://127.0.0.1:18024'; model='qwen36-w4a8'
with post(base,'/tokenize',{'model':model,'prompt':'Memory calibration and bandwidth records. '*(a.tokens+1),'add_special_tokens':False}) as resp:
    tokens=json.load(resp)['tokens'][:a.tokens]
assert len(tokens)==a.tokens
salt=uuid.uuid4().hex
out=Path(f'/data/qwen36-deploy/bench-{a.label}-{a.tokens}.jsonl')
with out.open('x') as stream:
    for repeat in range(-1,a.repeats+1):
        # The last request repeats the immediately preceding salt (hot).
        hot=repeat==a.repeats
        request_salt=f'{salt}-{repeat-1 if hot else repeat}'
        before=snapshot(base,a.spec); cache_before=prefix_snapshot(base)
        result=request(base,model,tokens, min(32,a.output) if repeat==-1 else a.output,cache_salt=request_salt)
        after=snapshot(base,a.spec); cache_after=prefix_snapshot(base)
        delta={k:after[k]-before[k] for k in before}
        pre,dec=delta[METRICS[1]],delta[METRICS[2]]
        hits=cache_after['vllm:prefix_cache_hits_total']-cache_before['vllm:prefix_cache_hits_total']
        mem={}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key,val=line.split(':',1)
            if key in ('MemTotal','MemAvailable','MemFree','AnonPages','Cached','Hugetlb'):
                mem[key]=int(val.split()[0])
        row={'kind':'warmup' if repeat==-1 else 'hot' if hot else 'cold','repeat':repeat,
             'tokens':a.tokens,'cache_hits':hits,'phase_delta':delta,'memory_kB':mem,
             'prefill_tok_s':len(tokens)/pre if pre>0 else None,
             'decode_tok_s':(result['usage']['completion_tokens']-1)/dec if dec>0 else None,
             'response':result}
        stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
        print(json.dumps({k:v for k,v in row.items() if k!='response'}),flush=True)
        if not hot: assert hits==0,'Cold request had a cache hit'
