# SPDX-License-Identifier: Apache-2.0
"""复测固定会议4的负责人映射；保留原请求/全文/表格解析，供语义复核。"""
import argparse
import copy
import json
from pathlib import Path
import uuid
from bench_http_stream import post
from bench_server_phases import prefix_snapshot

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base-url',required=True)
p.add_argument('--reference-jsonl',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
source=next(r for r in map(json.loads,a.reference_jsonl.read_text().splitlines()) if r['kind']=='meeting' and r['id']==4)
expected={'周明':['回复'],'刘洋':['修复'],'何丽':['用例']}
def extract(text):
    rows={}
    for line in text.splitlines():
        if not line.strip().startswith('|'):continue
        cells=[c.strip() for c in line.strip().strip('|').split('|')]
        for owner in expected:
            if any(owner in c for c in cells):rows[owner]=cells
    return rows
with a.out.open('x') as f:
    for mode in ('original_payload','unique_salt'):
        for repeat in range(3):
            req=copy.deepcopy(source['request'])
            if mode=='unique_salt':req['cache_salt']='owners-'+uuid.uuid4().hex
            before=prefix_snapshot(a.base_url)
            with post(a.base_url,'/v1/chat/completions',req) as response:result=json.load(response)
            after=prefix_snapshot(a.base_url)
            text=result['choices'][0]['message']['content']
            rows=extract(text)
            mapping={owner:all(word in ' '.join(rows.get(owner,[])) for word in words) for owner,words in expected.items()}
            hit_delta=after['vllm:prefix_cache_hits_total']-before['vllm:prefix_cache_hits_total']
            if mode=='unique_salt' and hit_delta!=0:raise RuntimeError('Unique-salt request had prefix hits')
            row=dict(kind='meeting_owner_recheck',fixture_id=4,mode=mode,repeat=repeat,request=req,response=result,
                     extracted_owner_rows=rows,expected_owner_tasks=expected,owner_task_checks=mapping,
                     prefix_hits_delta=hit_delta,requires_semantic_review=True)
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps({k:v for k,v in row.items() if k not in ('request','response')},ensure_ascii=False),flush=True)
