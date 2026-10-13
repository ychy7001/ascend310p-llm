# SPDX-License-Identifier: Apache-2.0
"""Fixed-seed GSM8K subset via chat HTTP; retains every response and truncation.

Zero-shot prompt protocol, not a claim of standard lm-eval or full GSM8K score.
Download official test.jsonl separately; datasets and raw outputs stay outside Git.
"""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import random
import re
import time
import urllib.request


def extract(text):
    matches=re.findall(r'####\s*(-?\d[\d,]*(?:\.\d+)?)',text)
    return str(Decimal(matches[-1].replace(',','')).normalize()) if matches else None


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base-url',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=32)
    p.add_argument('--seed',type=int,default=310)
    p.add_argument('--indices',help='仅用于诊断指定dataset行，逗号分隔；覆盖随机抽样')
    p.add_argument('--max-tokens',type=int,default=512)
    a=p.parse_args()
    raw=a.dataset.read_bytes()
    data=[json.loads(line) for line in raw.decode().splitlines()]
    selected=sorted(int(i) for i in a.indices.split(',')) if a.indices else sorted(random.Random(a.seed).sample(range(len(data)),a.limit))
    if not selected or len(set(selected))!=len(selected) or min(selected)<0 or max(selected)>=len(data):
        p.error('Invalid dataset indices')
    config={'kind':'config','label':a.label,'model':a.model,'base_url':a.base_url,
        'dataset_sha256':hashlib.sha256(raw).hexdigest(),'dataset_rows':len(data),
        'indices':selected,'max_tokens':a.max_tokens,'temperature':0,'enable_thinking':False,
        'prompt_suffix':'Solve this problem briefly. End with #### followed by the final numeric answer.',
        'protocol':'zero-shot fixed subset, strict #### numeric extraction; not standard full GSM8K'}
    rows=[]
    with a.out.open('x') as f:
        f.write(json.dumps(config)+'\n');f.flush()
        for index in selected:
            item=data[index]
            body={'model':a.model,'messages':[{'role':'user',
                'content':item['question']+'\n\n'+config['prompt_suffix']}],
                'temperature':0,'max_tokens':a.max_tokens,
                'chat_template_kwargs':{'enable_thinking':False}}
            request=urllib.request.Request(a.base_url.rstrip('/')+'/v1/chat/completions',
                data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            start=time.perf_counter()
            with urllib.request.urlopen(request,timeout=1800) as response:
                payload=json.load(response)
            choice=payload['choices'][0]
            text=choice['message'].get('content') or ''
            predicted=extract(text);gold=extract(item['answer'])
            if gold is None:
                raise ValueError(f'Gold numeric answer missing at dataset row {index}')
            row={'kind':'answer','index':index,'gold':gold,'predicted':predicted,
                'correct':predicted==gold,'text':text,'usage':payload.get('usage'),
                'reasoning':choice['message'].get('reasoning_content'),
                'finish_reason':choice['finish_reason'],'elapsed_s':time.perf_counter()-start}
            rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps({k:row[k] for k in ['index','correct','finish_reason','elapsed_s']}),flush=True)
        summary={'kind':'summary','label':a.label,'samples':len(rows),
            'correct':sum(r['correct'] for r in rows),
            'parse_missing':sum(r['predicted'] is None for r in rows),
            'truncated':sum(r['finish_reason']=='length' for r in rows)}
        f.write(json.dumps(summary)+'\n')
        print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
