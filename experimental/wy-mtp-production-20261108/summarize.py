#!/usr/bin/env python3
"""Summarize archived windows without treating capacity tests as quality gates."""
import argparse,json,statistics,re
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('windows',nargs='+',type=Path); p.add_argument('--out',required=True,type=Path); a=p.parse_args()
records=[]
for directory in a.windows:
    record={'window':directory.name,'status':json.loads((directory/'status.json').read_text()),'benchmarks':{}}
    for path in sorted(directory.glob('p*-o*.jsonl')):
        rows=[r for line in path.read_text().splitlines() if (r:=json.loads(line)).get('kind')=='request']
        valid=[r for r in rows if r.get('valid')]
        record['benchmarks'][path.stem]={'count':len(rows),'valid':len(valid),'prefill_tok_s_median':statistics.median(r['prefill_tok_s'] for r in valid) if valid else None,'decode_tok_s_median':statistics.median(r['decode_tok_s'] for r in valid if r.get('decode_tok_s')) if any(r.get('decode_tok_s') for r in valid) else None,'ttft_s_median':statistics.median(r['response']['ttft_s'] for r in valid) if valid else None,'prefix_hits':sum(r.get('prefix_cache_delta',{}).get('vllm:prefix_cache_hits_total',0) for r in rows),'draft_acceptance':[(r.get('spec_decode') or {}).get('acceptance_rate') for r in valid]}
    for path in directory.glob('full-*.json'):
        r=json.loads(path.read_text()); result=r['result']; before=r['metrics_before']; after=r['metrics_after']; pt=after['vllm:request_prefill_time_seconds_sum']-before['vllm:request_prefill_time_seconds_sum']
        record[path.stem]={'usage':result['usage'],'ttft_s':result['ttft_s'],'elapsed_s':result['elapsed_s'],'prefill_tok_s':result['usage']['prompt_tokens']/pt,'prefix_hits':r['cache_after']['vllm:prefix_cache_hits_total']-r['cache_before']['vllm:prefix_cache_hits_total']}
    path=directory/'c4-p4096-o128.jsonl'
    if path.exists():
        rows=[json.loads(x) for x in path.read_text().splitlines()]; requests=[r for r in rows if r.get('kind')=='request']; rounds=[r for r in rows if r.get('kind')=='round']; ttfts=sorted(r['ttft_s'] for r in requests if r.get('success'))
        record['concurrency4']={'requests':len(requests),'success':sum(r['success'] for r in requests),'aggregate_output_tok_s_median':statistics.median(r['aggregate_output_tok_s'] for r in rounds),'ttft_s_p50':statistics.median(ttfts),'ttft_s_empirical_p95':ttfts[min(len(ttfts)-1,int(len(ttfts)*.95))]}
    path=directory/'mixed-concurrency.json'
    if path.exists():
        rows=json.loads(path.read_text()); record['mixed_concurrency']={'requests':len(rows),'success':sum(r['success'] for r in rows),'usage':[r.get('usage') for r in rows]}
    path=directory/'resource-samples.jsonl'
    if path.exists():
        samples=[json.loads(line) for line in path.read_text().splitlines()]
        values=[int(used) for row in samples for used,total in re.findall(r'(\d+)\s*/\s*(\d+)',row['npu']) if int(total)>40000]
        record['observed_memory']={'sampling_interval_s':10,'samples':len(samples),'max_used_mb':max(values) if values else None,'scope':'npu-smi observed use, not allocator peak'}
    records.append(record)
a.out.write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(records,ensure_ascii=False,indent=2))
