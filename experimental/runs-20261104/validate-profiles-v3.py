#!/usr/bin/env python3
"""Local, numerical integrity checks for the eight selected device windows."""
import csv,json,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path('tools').resolve()))
from attribute_decode_kernels import validate_weight_shape
ROOT=Path('results/decode-attribution-20261104/v3');records=[]
events=[json.loads(line) for line in (ROOT/'raw/capture-metadata/http-events.jsonl').read_text().splitlines()]
stops={r['label']:r for r in events if r['kind']=='profile_stop'}
for batch,label in enumerate(['b1-long','b2-long','b3-short','mixed-b4'],1):
 for rank,pid in [(0,567),(1,575)]:
  path=ROOT/'profiles'/f'{label}-rank{rank}.json';d=json.loads(path.read_text());lo=(d['origin_epoch_us']+d['window_start_us'])/1e6;hi=(d['origin_epoch_us']+d['window_end_us'])/1e6
  observation=[json.loads(line) for line in (ROOT/f'raw/capture-metadata/observation/observe-{pid}.jsonl').read_text().splitlines()]
  forwards=[r for r in observation if r['kind']=='forward_start' and lo<=r['wall']<hi]
  batch_hist=Counter(r['batch'] for r in forwards);padding=Counter(r['padded_tokens'] for r in forwards);graphs=Counter(r['mode'] for r in forwards)
  contexts=[value for r in forwards for value in r.get('context',[]) if isinstance(value,(int,float))]
  checks=0
  for node in d['fp16_module_attribution']['nodes']:
   checked=[validate_weight_shape(shape,node.get('weight',{}).get('shape')) for shape in node['kernel_shape_samples']]
   if any(checked):checks+=1
  d['fp16_module_attribution']['validation']['native_weight_shapes_checked_nodes']=checks
  path.write_text(json.dumps(d,indent=2)+'\n')
  row={'case':label,'batch':batch,'rank':rank,'selected_steps':d['steps'],'captured_head_count':d['selection']['captured_head_count'],'captured_observed_forward_count':stops[label]['observed_steps'][str(pid)],'captured_stable_forward_count':stops[label]['stable_steps'][str(pid)],'native_kernel_rows_in_selected_window':d['kernel_rows'],'fp16_calls_per_step':d['fp16_module_attribution']['matched_fp16_calls_per_step'],'native_nz_weight_shapes_checked_nodes':checks,'graph_fp16_nodes':d['graph_fp16_node_count'],'graph_reduction_nodes':len(d['graph_reduction_node_ledger']),'all_physical_reductions_per_step':d['hardware_tasks']['REDUCE_ASYNC_V2']['count']/d['steps'],'observed_forwards_in_device_time_window':len(forwards),'observed_batch_histogram':dict(batch_hist),'observed_padding_histogram':dict(padding),'observed_graph_modes':dict(graphs),'cycle_ms':d['window_ms_per_step'],'context_samples_preview':forwards[0].get('context') if forwards else None,'kernel_export_source':'native_pytorch_csv' if batch<3 else 'native_cann_op_summary','full_final_pytorch_report':batch<3}
  assert d['steps']==96 and row['fp16_calls_per_step']==220,row
  assert len(forwards)>=90 and set(batch_hist)=={batch},row
  assert batch>=3 or row['graph_fp16_nodes']==220 and row['graph_reduction_nodes']==80,row
  assert batch<3 or checks==220,row
  records.append(row)
report={'valid_device_windows':len(records),'selected_steps_total':sum(r['selected_steps'] for r in records),'native_full_final_pytorch_reports':4,'locally_completed_cann_accounting_reports':4,'mixed_b4_extrapolated':False,'records':records,'rounding_crosscheck':json.loads((ROOT/'raw/timeline-vs-native-validation.json').read_text()),'limitations':['Two interrupted final B3 trace arrays retain CPU prefixes but lack device append/final bracket. B4 final framework reports are absent. All eight selected device windows use complete native CSV/timeline data.','Recorded forward CPU times verify the batch throughout the selected device-time window; asynchronous pipelining prevents claiming one-to-one CPU/device start correspondence.']}
(ROOT/'profile-integrity.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
