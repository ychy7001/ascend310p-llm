#!/usr/bin/env python3
"""Summarize completed request records; never treat partial arms as accepted."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def median(rows, key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return statistics.median(values) if values else None


def summarize(root):
    arms={}
    for directory in sorted(root.glob('w[48]-*')):
        if not directory.is_dir(): continue
        data={}
        for path in sorted(directory.glob('*.jsonl')):
            records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            rows=[r for r in records if r.get('kind')=='request']
            config=next(r for r in records if r.get('kind')=='config')
            data[path.stem]={'request_count':len(rows),'expected_count':config['repeats'],'complete':len(rows)==config['repeats'],'all_valid':bool(rows) and all(r.get('valid') for r in rows),'prompt_tokens':config['prompt_tokens'],'output_tokens':config['output_tokens'],'prompt_hashes':sorted({r['prompt_ids_sha256'] for r in rows}),'prefill_s_median':median(rows,'prefill_seconds'),'prefill_tok_s_median':median(rows,'prefill_tok_s'),'decode_tok_s_median':median(rows,'decode_tok_s'),'ttft_s_median':statistics.median(r['response']['ttft_s'] for r in rows) if rows else None,'prefix_cache_hits':sum(r.get('prefix_cache_delta',{}).get('vllm:prefix_cache_hits_total',0) for r in rows),'response_sha256':[hashlib.sha256(r['response']['text'].encode()).hexdigest() for r in rows]}
        pdf=[]
        for path in sorted(directory.glob('pdf-*.json')):
            row=json.loads(path.read_text())
            pdf.append({k:row.get(k) for k in ('id','repeat','valid','text_sha256','prefill_seconds','first_content_s','usage')})
        arms[directory.name]={'text':data,'pdf':pdf,'has_container_inspect':(directory/'container-inspect.json').exists()}
    paired=[]
    for mtp in (0,2):
        native=arms.get(f'w4-native-mtp{mtp}',{}).get('text',{})
        torch=arms.get(f'w4-torch-mtp{mtp}',{}).get('text',{})
        for key in sorted(native.keys() & torch.keys()):
            n,t=native[key],torch[key]
            if not n['complete'] or not t['complete']:continue
            assert n['prompt_hashes']==t['prompt_hashes'], 'prompt mismatch'
            paired.append({'mtp':mtp,'case':key,'native_prefill_s':n['prefill_s_median'],'torch_prefill_s':t['prefill_s_median'],'prefill_time_reduction':1-n['prefill_s_median']/t['prefill_s_median'],'native_decode_tok_s':n['decode_tok_s_median'],'torch_decode_tok_s':t['decode_tok_s_median'],'all_valid':n['all_valid'] and t['all_valid']})
    status=json.loads((root/'status.json').read_text())
    return {'scope':'single-stream integration diagnostic; synthetic text and PDF proxy; no formal business acceptance','status':status,'arms':arms,'paired':paired}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    print(json.dumps(summarize(args.root),ensure_ascii=False,indent=2))
