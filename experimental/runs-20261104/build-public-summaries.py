#!/usr/bin/env python3
"""Build anonymous aggregate tables. Site request records remain in ignored raw/."""
import csv,json,pathlib
out=pathlib.Path('results/decode-attribution-20261104/v2')
s=json.loads((out/'site-summary.json').read_text())
def write(name,rows):
 with (out/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
rows=[]
for pop,keys in [('all',('pooled_requests','pooled_input','pooled_output')),('meeting_length_proxy',('meeting_proxy_n','meeting_proxy_input','meeting_proxy_output'))]:
 for typ,key in [('input',keys[1]),('output',keys[2])]:rows.append({'population':pop,'n':s[keys[0]],'tokens':typ,**s[key]})
write('length-quantiles.csv',rows)
rows=[]
for population,key in [('all_visible_request_intervals','pooled_request_interval_seconds'),('meeting_proxy_visible_request_intervals','pooled_meeting_proxy_interval_seconds')]:
 hist=s[key];total=sum(hist.values());busy=sum(v for k,v in hist.items() if k!='0')
 for k,v in sorted(hist.items(),key=lambda p:int(p[0])):rows.append({'population':population,'concurrent':int(k),'seconds':v,'fraction_of_observed_span':v/total,'fraction_of_busy_time':v/busy if k!='0' else None})
hist={str(k):sum(d['concurrency']['seconds'].get(str(k),0) for d in s['sites']) for k in range(7)};total=sum(hist.values());busy=total-hist['0']
for k,v in hist.items():rows.append({'population':'backend_running_samples_gaps_le_20s','concurrent':int(k),'seconds':v,'fraction_of_observed_span':v/total,'fraction_of_busy_time':v/busy if k!='0' else None})
write('site-concurrency-histograms.csv',rows)
r=json.loads((out/'replay-summary.json').read_text());total=r['phase_steps']['decode'];duration=r['phase_seconds']['decode']
write('replay-batch-weights.csv',[{'batch':b,'steps':r['decode_batch_steps'][str(b)],'step_fraction':r['decode_batch_steps'][str(b)]/total,'seconds':r['decode_batch_seconds'][str(b)],'decode_time_fraction':r['decode_batch_seconds'][str(b)]/duration,'wall_fraction':r['decode_batch_seconds'][str(b)]/r['wall_seconds']} for b in (1,2,3,4)])
config={'scope':'archived production replica, observation only','tp':2,'weight_quantization':'W8A8 routed experts and full-attention QKV; FP16 shared experts, linear-attention projections and lm_head','graph_sizes':[1,2],'mtp':0,'max_model_len':262144,'max_num_batched_tokens':4096,'max_num_seqs':4,'prefix_cache':True,'cold_replay_requests':11,'cold_replay_admissions_stop_seconds':480,'cold_replay_wall_seconds':r['wall_seconds'],'seed':20261104,'generation_seed':17,'independent_cache_salt_per_request':True,'arrival_model':'busy-time conditioned closed-loop occupancy targets, not exact real arrival replay','profile_windows':['b1-long','b2-long','b3-short','mixed-b4'],'profile_target_stable_forwards_per_rank':112,'mapping_forwards_per_rank':3,'mapping_real_batch':3,'inference_implementation_changed':False,'optimizations_implemented':False}
config.update(production_image_sha256='1a193ded0d75832321019542326740c7fc4509853556d032e773631466453b38',profile_capture_and_export_status='four lean windows completed remotely; exports not retrieved; final health after connectivity loss unknown')
(out/'configuration-summary.json').write_text(json.dumps(config,indent=2)+'\n')
