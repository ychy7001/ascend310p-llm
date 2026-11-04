#!/usr/bin/env python3
"""Complete CPU accounting exports from pre-existing device timelines, sequentially."""
import json,resource,sys,time
from pathlib import Path
sys.path.insert(0,str(Path('tools').resolve()))
from decode_trace_utils import trace_events
from export_decode_timeline import export
from analyze_decode_profile import analyze
from attribute_decode_kernels import attribute
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
resource.setrlimit(resource.RLIMIT_CPU,(3600,3600))
ROOT=Path('results/decode-attribution-20261104/v3');ledgers=json.loads(Path('results/decode-attribution-20261104/v2/module-shape-ledger.json').read_text())
records=[]
for case,stamp in [('b3-short','093327'),('mixed-b4','093352')]:
 for rank in (0,1):
  source=next((ROOT/'raw/profiles').glob(f'*rank{rank}_*20261104{stamp}*'))
  retrieved={r['directory'] for r in json.loads((ROOT/'raw/retrieval.json').read_text()) if r['returncode']==0}
  if source.name not in retrieved:raise RuntimeError('Retrieval incomplete: '+source.name)
  trace=source/'ASCEND_PROFILER_OUTPUT/trace_view.json';full_trace=False  # Interrupted final trace; use complete CANN timelines.
  if not full_trace:
   sources=list(source.glob('PROF*/device_*/timeline/task_time*.json'))+list(source.glob('PROF*/device_*/timeline/communication*.json'))
   assert len(sources)==2,sources
   trace=ROOT/'raw'/f'{case}-rank{rank}-device-trace.json'
   with trace.open('w') as out:
    out.write('[');first=True
    for part in sources:
     with part.open() as stream:
      for event in trace_events(stream):
       if not first:out.write(',\n')
       out.write(json.dumps(event,separators=(',',':')));first=False
    out.write(']\n')
  csv_path=next(iter(list(source.glob('PROF*/mindstudio_profiler_output/op_summary*.csv'))+list(source.glob('PROF*/device_*/summary/op_summary*.csv'))))
  start=time.monotonic();metadata={'source':'Native CANN op_summary CSV; matched 9.1.0-beta.1 offline export where missing','kernel_rows':None}
  result,rows=analyze(csv_path,trace,96,case=='mixed-b4')
  ledger=next(r for r in ledgers if r['rank']==rank)['fp16_ledger']
  result['fp16_module_attribution']=attribute(rows,ledger,result['window_start_us'],result['window_end_us'],result['steps'])
  result['export_metadata']={**metadata,'full_framework_trace_available':full_trace,'standard_pytorch_kernel_csv_available':False,'native_cann_csv_available':True,'case':case,'rank':rank}
  if not full_trace:
   for field in ['worker_cpu_op_union_ms_per_step','worker_cpu_ops_in_named_kernel_gap_ms_per_step','host_sync_span_ms_per_step_overlaps_device']:
    result[field]=None
  (ROOT/'profiles'/f'{case}-rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n')
  records.append({'case':case,'rank':rank,'steps':result['steps'],'cycle_ms':result['window_ms_per_step'],'seconds':time.monotonic()-start,'kernel_rows':metadata['kernel_rows'],'full_framework_trace':full_trace})
  (ROOT/'raw/local-completion-runs.json').write_text(json.dumps(records,indent=2)+'\n');print(records[-1],flush=True)
  del rows,result
