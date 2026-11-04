#!/usr/bin/env python3
import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'tools'))
from inference_numeric_metadata import interval_histogram,timestamp_seconds,quantiles
out=pathlib.Path('results/decode-attribution-20261104/v2');summary=json.loads((out/'site-summary.json').read_text())
lengths=[];all_bins={};proxy_bins={}
for site in summary['sites']:
 rows=[json.loads(l) for l in (out/'raw'/(site['site']+'-metadata.jsonl')).read_text().splitlines()]
 requests=[r for r in rows if r['kind']=='request_completed' and r.get('time') and 'elapsed_s' in r]
 intervals=[(timestamp_seconds(r['time'])-r['elapsed_s'],timestamp_seconds(r['time'])) for r in requests]
 proxy=[r for r in requests if r['prompt_tokens']>=4096 and r.get('generated_tokens',r.get('completion_tokens',0))>=128]
 pi=[(timestamp_seconds(r['time'])-r['elapsed_s'],timestamp_seconds(r['time'])) for r in proxy]
 site['request_interval_histogram']=interval_histogram(intervals)
 site['meeting_proxy_interval_histogram']=interval_histogram(pi)
 site['arrival_method']='completion timestamp minus server elapsed_s (rounded to 0.01s); visible completed requests only'
 site['request_elapsed_seconds']=quantiles([r['elapsed_s'] for r in requests])
 for k,v in site['request_interval_histogram']['seconds'].items():all_bins[k]=all_bins.get(k,0.)+v
 for k,v in site['meeting_proxy_interval_histogram']['seconds'].items():proxy_bins[k]=proxy_bins.get(k,0.)+v
 metrics=site['counters'];q=metrics.get('vllm:prefix_cache_queries_total',0);h=metrics.get('vllm:prefix_cache_hits_total',0)
 site['prefix_token_hit_rate']=h/q if q else None
 lengths.extend({'site':site['site'],'input':int(r['prompt_tokens']),'output':int(r.get('generated_tokens',r.get('completion_tokens',0))),'arrival_epoch':timestamp_seconds(r['time'])-r['elapsed_s'],'duration':r['elapsed_s']} for r in proxy)
summary['pooled_request_interval_seconds']=all_bins;summary['pooled_meeting_proxy_interval_seconds']=proxy_bins
summary['interval_histogram_limitation']='Visible completed requests only; not scheduler decode batch; possible left truncation and logging gaps. Concurrency >4 belongs to the observed serving stacks and cannot be copied to seqs4.'
(out/'site-summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'raw'/'paired-lengths-arrivals.json').write_text(json.dumps(lengths))
print('proxy',summary['meeting_proxy_n']);print('busy bins',proxy_bins)
