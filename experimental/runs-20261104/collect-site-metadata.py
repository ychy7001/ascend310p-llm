#!/usr/bin/env python3
"""Read-only SSH collection; all site identities and raw metadata stay in ignored results."""
import concurrent.futures,json,pathlib,subprocess,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'tools'))
from inference_numeric_metadata import quantiles,sampled_histogram
OUT=pathlib.Path('results/decode-attribution-20261104/v2')
PARSER=pathlib.Path('tools/inference_numeric_metadata.py').read_text().split("if __name__=='__main__':")[0]
REMOTE=r'''
import subprocess
pfx=['sudo','-n'] if ROOT else []
for name in ['llm-backend-primary','llm-gateway']:
 process=subprocess.run(pfx+['podman','logs','--timestamps',name],capture_output=True,text=True)
 for line in (process.stdout+process.stderr).splitlines():
  for row in parse_line(line):print(json.dumps(row,separators=(',',':')))
# Current counters are read without running a request.
if ROOT:
 process=subprocess.run(['curl','-sS','--max-time','5','http://127.0.0.1:18081/metrics'],capture_output=True,text=True)
else:
 code="import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:18020/metrics',timeout=5).read().decode())"
 process=subprocess.run(['podman','exec','llm-backend-primary','/app/venv/bin/python','-c',code],capture_output=True,text=True)
allowed=('prefix_cache_queries_total','prefix_cache_hits_total','request_success_total','request_prompt_tokens_count','request_prompt_tokens_sum','request_generation_tokens_count','request_generation_tokens_sum','num_requests_running','num_requests_waiting','request_queue_time_seconds_sum')
for line in process.stdout.splitlines():
 if line.startswith('#'):continue
 metric=line.split('{')[0].split(' ')[0]
 if any(metric.endswith(':'+key) for key in allowed):
  try:value=float(line.split()[-1])
  except ValueError:continue
  print(json.dumps({'kind':'metric','metric':metric,'value':value}))
'''
def collect(item):
 site,host,root=item
 process=subprocess.run(['ssh',host,'python3 -'],input=PARSER+'\nROOT='+repr(root)+'\n'+REMOTE,capture_output=True,text=True,timeout=120)
 if process.returncode:raise RuntimeError(site+' metadata extraction failed')
 path=OUT/'raw'/(site+'-metadata.jsonl');path.write_text(process.stdout)
 rows=[json.loads(line) for line in process.stdout.splitlines()]
 requests=[r for r in rows if r['kind']=='request_completed']
 # A transparent length screen is a proxy, since task/caller classification is absent.
 meeting=[r for r in requests if r['prompt_tokens']>=4096 and r.get('generated_tokens',r.get('completion_tokens',0))>=128]
 concurrency=[r for r in rows if r['kind']=='concurrency_sample']
 metrics={}
 for r in rows:
  if r['kind']=='metric':metrics[r['metric']]=metrics.get(r['metric'],0)+r['value']
 def dist(rs):return {'n':len(rs),'input_tokens':quantiles([r['prompt_tokens'] for r in rs]),'output_tokens':quantiles([r.get('generated_tokens',r.get('completion_tokens',0)) for r in rs])}
 times=sorted(r['time'] for r in rows if r.get('time'))
 summary={'site':site,'first':times[0] if times else None,'last':times[-1] if times else None,
          'requests':dist(requests),'meeting_length_proxy':dist(meeting),'concurrency':sampled_histogram(concurrency),
          'concurrency_samples':len(concurrency),'concurrency_sample_quantiles':quantiles([r['running'] for r in concurrency]),
          'counters':metrics,'arrival_timestamps_available':bool(requests and all('elapsed_s' in r for r in requests)),'per_request_prefix_available':False}
 return summary,requests
with concurrent.futures.ThreadPoolExecutor() as pool:results=list(pool.map(collect,[('site01','ychy-a1',True),('site02','ails-n2',False),('site03','ails-n3',False)]))
summaries=[x[0] for x in results];requests=[r for _,rs in results for r in rs]
meeting=[r for r in requests if r['prompt_tokens']>=4096 and r.get('generated_tokens',r.get('completion_tokens',0))>=128]
summary={'sites':summaries,'pooled_requests':len(requests),'pooled_input':quantiles([r['prompt_tokens'] for r in requests]),'pooled_output':quantiles([r.get('generated_tokens',r.get('completion_tokens',0)) for r in requests]),'meeting_proxy_definition':'input >=4096 and output >=128; task type not available; not a verified meeting-only sample','meeting_proxy_n':len(meeting),'meeting_proxy_input':quantiles([r['prompt_tokens'] for r in meeting]),'meeting_proxy_output':quantiles([r.get('generated_tokens',r.get('completion_tokens',0)) for r in meeting])}
(OUT/'site-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'raw'/'paired-lengths.json').write_text(json.dumps([{'input':int(r['prompt_tokens']),'output':int(r.get('generated_tokens',r.get('completion_tokens',0)))} for r in meeting]))
print(json.dumps(summary,indent=2))
