# SPDX-License-Identifier: Apache-2.0
"""Exact-token long request with a single retrieval canary; not a long-context suite."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def post(base,path,body):
    request=urllib.request.Request(base.rstrip('/')+path,data=json.dumps(body).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=3600) as response:return json.load(response)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-url',required=True)
    p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prompt-tokens',type=int,default=32768)
    p.add_argument('--needle-fraction',type=float,default=.5)
    p.add_argument('--max-tokens',type=int,default=64);a=p.parse_args()
    if not 0<=a.needle_fraction<=1:raise ValueError('Needle fraction outside 0..1')
    def tokenize(text):return post(a.base_url,'/tokenize',{'model':a.model,'prompt':text,
        'add_special_tokens':False})['tokens']
    prefix=tokenize('Read this document and locate the secret code. Ordinary notes contain no secret code.\nDocument:\n')
    needle=tokenize('\nThe secret code is ASCEND732941.\n')
    suffix=tokenize('\nEnd of document. What is the secret code? Reply with only the code.\nAnswer:')
    budget=a.prompt_tokens-len(prefix)-len(needle)-len(suffix)
    if budget<1:raise ValueError('Prompt budget too small')
    repeats=max(1,budget//12+1)
    while True:
        filler=tokenize('Ordinary note: calibration and memory measurements are recorded carefully. '*repeats)
        if len(filler)>=budget:
            filler=filler[:budget]
            break
        repeats*=2
    split=int(budget*a.needle_fraction)
    tokens=prefix+filler[:split]+needle+filler[split:]+suffix
    assert len(tokens)==a.prompt_tokens
    config={'kind':'config','model':a.model,'prompt_tokens':len(tokens),
        'prompt_ids_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        'needle_fraction':a.needle_fraction,'needle_start_token':len(prefix)+split,
        'max_tokens':a.max_tokens,'temperature':0,'protocol':'raw completion, one retrieval canary'}
    with a.out.open('x') as f:
        f.write(json.dumps(config)+'\n');f.flush()
        start=time.perf_counter()
        result=post(a.base_url,'/v1/completions',{'model':a.model,'prompt':tokens,
            'temperature':0,'max_tokens':a.max_tokens})
        choice=result['choices'][0];text=choice.get('text') or ''
        row={'kind':'result','elapsed_s':time.perf_counter()-start,'text':text,
            'expected':'ASCEND732941','retrieved':'ASCEND732941' in text,
            'usage':result.get('usage'),'finish_reason':choice.get('finish_reason')}
        if row['usage']['prompt_tokens']!=len(tokens):raise ValueError('Server prompt usage differs')
        f.write(json.dumps(row,ensure_ascii=False)+'\n');print(json.dumps(row,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
