#!/usr/bin/env python3
"""Read the event-complete CPU prefix of interrupted B3 traces; no event recovery guessing."""
import json,resource,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path('tools').resolve()))
from decode_trace_utils import trace_events,relative_timestamp,subtract,intersection_duration,union_duration
from analyze_decode_profile import load_kernels
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
ROOT=Path('results/decode-attribution-20261104/v3')
for rank in (0,1):
 path=ROOT/'profiles'/f'b3-short-rank{rank}.json';result=json.loads(path.read_text());source=next((ROOT/'raw/profiles').glob(f'*rank{rank}_*093327*'))
 _,rows=load_kernels(next(iter(list(source.glob('PROF*/mindstudio_profiler_output/op_summary*.csv'))+list(source.glob('PROF*/device_*/summary/op_summary*.csv')))))
 lo=result['window_start_us'];hi=result['window_end_us'];origin=result['origin_epoch_us'];steps=result['steps']
 known=[(max(lo,r['start']),min(hi,r['end'])) for r in rows if r['start']<hi and r['end']>lo]
 gap=subtract([(lo,hi)],known);ops=[];sync=[];names=Counter();count=0
 with (source/'ASCEND_PROFILER_OUTPUT/trace_view.json').open() as stream:
  for e in trace_events(stream,allow_missing_final_bracket=True):
   count+=1
   if e.get('ph')!='X':continue
   start=relative_timestamp(e['ts'],origin);end=start+float(e.get('dur',0))
   if start>=hi or end<=lo:continue
   iv=(max(lo,start),min(hi,end));name=e.get('name','')
   if e.get('cat')=='cpu_op':ops.append(iv);names[name]+=1
   if 'synchronize' in name.lower():sync.append(iv)
 result['worker_cpu_op_union_ms_per_step']=union_duration(ops)/steps/1000
 result['worker_cpu_ops_in_named_kernel_gap_ms_per_step']=intersection_duration(ops,gap)/steps/1000
 result['host_sync_span_ms_per_step_overlaps_device']=union_duration(sync)/steps/1000
 result['export_metadata']['host_event_prefix_recovery']={'only_final_array_bracket_missing':True,'complete_events_read':count,'selected_cpu_op_events':sum(names.values()),'top_names':names.most_common(12),'limitation':'CPU events preserved before device append was interrupted; missing events are not synthesized; full final PyTorch report remains incomplete'}
 path.write_text(json.dumps(result,indent=2)+'\n');print(rank,result['worker_cpu_ops_in_named_kernel_gap_ms_per_step'],count,flush=True)
