#!/usr/bin/env python3
"""Final CPU validation, source hashes and explicit sanitized delivery inventory."""
from datetime import datetime,timezone
import hashlib,json,re,subprocess,sys,unittest
from pathlib import Path
ROOT=Path('results/decode-attribution-20261104');OUT=ROOT/'v3'
sys.path.insert(0,str(Path('tools').resolve()))
names=['test_observe_decode','test_inference_numeric_metadata','test_analyze_decode_observation','test_decode_trace_utils','test_analyze_decode_profile','test_attribute_decode_kernels','test_map_decode_modules','test_export_decode_timeline','test_weight_decode_profiles']
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
with (OUT/'raw/unit-tests.txt').open('w') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
assert result.wasSuccessful()
profile=json.loads((OUT/'profile-integrity.json').read_text());weighted=json.loads((OUT/'weighted-composition.json').read_text());retrieval=json.loads((OUT/'raw/retrieval.json').read_text());local=json.loads((OUT/'local-parsing.json').read_text())
checks={'eight_profile_directories_rsync_returncode_zero':len(retrieval)==8 and all(r['returncode']==0 for r in retrieval),'eight_valid_device_windows':profile['valid_device_windows']==8,'96_steps_each':all(r['selected_steps']==96 for r in profile['records']),'768_selected_steps':profile['selected_steps_total']==768,'220_projection_calls_each':all(r['fp16_calls_per_step']==220 for r in profile['records']),'eager_220_native_nz_shapes_checked':all(r['native_nz_weight_shapes_checked_nodes']==220 for r in profile['records'] if r['batch']>=3),'graph_220_fp16_80_reduce_nodes':all(r['graph_fp16_nodes']==220 and r['graph_reduction_nodes']==80 for r in profile['records'] if r['batch']<=2),'selected_batches_match':all(set(r['observed_batch_histogram'])=={str(r['batch'])} for r in profile['records']),'81_physical_reductions_each_step':all(r['all_physical_reductions_per_step']==81 for r in profile['records']),'b4_directly_measured':not profile['mixed_b4_extrapolated'],'composition_closes':abs(weighted['fraction_sum']-1)<1e-12,'time_weights_close':abs(sum(weighted['time_weights'].values())-1)<1e-12,'native_vs_timeline_340480_kernels':profile['rounding_crosscheck']['native_rows']==340480 and profile['rounding_crosscheck']['all_events_agree_within_0_002_us'],'no_remote_parsing_or_npu_or_service_operations':local['remote_parsing_processes_this_round']==local['npu_workloads_this_round']==local['service_operations_this_round']==0,'initial_vendor_pool_limit_deviation_disclosed':bool(local['initial_local_vendor_pool_deviation']),'recorded_time_swaps_field_zero':local['all_recorded_time_reports_swaps']==0,'old_incomplete_conclusion_removed':all(t not in (ROOT/'README.md').read_text() for t in ['尚未取回','受阻','未能取回'])}
public=[ROOT/'README.md',*[p for p in OUT.rglob('*') if p.is_file() and 'raw' not in p.relative_to(OUT).parts and p.name not in ['artifact-sha256.json','validation.json']]]
privacy=[]
for p in public:
 text=p.read_text()
 for pattern in ['/home/','/usr/','ychy-a','ails-n',r'\b(?:\d{1,3}\.){3}\d{1,3}\b']:
  if re.search(pattern,text):privacy.append({'artifact':str(p.relative_to(ROOT)),'pattern':pattern})
checks['anonymous_public_results']=not privacy
assert all(checks.values()),{k:v for k,v in checks.items() if not v}
report={'created_utc':datetime.now(timezone.utc).isoformat(),'unit_tests':result.testsRun,'unit_tests_passed':result.wasSuccessful(),'checks':checks,'all_checks_passed':True,'optimization_implemented':False,'npu_experiments_this_round':False,'full_final_pytorch_report_count':4,'usable_native_device_accounting_count':8,'strict_local_parse_concurrency_limit_met_for_entire_round':False,'concurrency_deviation':'Vendor aggregation pools found after initial local exports; disclosed in README/local-parsing.json and disabled in final scripts. No exact live peak process count retained.'}
(OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
public.append(OUT/'validation.json');hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(public)}
inputs={}
for case,stamp in [('b1-long','093151'),('b2-long','093306'),('b3-short','093327'),('mixed-b4','093352')]:
 for rank in (0,1):
  source=next((OUT/'raw/profiles').glob(f'*rank{rank}_*20261104{stamp}*'))
  kernel=source/'ASCEND_PROFILER_OUTPUT/kernel_details.csv' if case.startswith(('b1','b2')) else next(iter(list(source.glob('PROF*/mindstudio_profiler_output/op_summary*.csv'))+list(source.glob('PROF*/device_*/summary/op_summary*.csv'))))
  trace=source/'ASCEND_PROFILER_OUTPUT/trace_view.json' if case.startswith(('b1','b2')) else OUT/'raw'/f'{case}-rank{rank}-device-trace.json'
  inputs[f'{case}-rank{rank}']={'kernel_sha256':digest(kernel),'kernel_bytes':kernel.stat().st_size,'used_trace_sha256':digest(trace),'used_trace_bytes':trace.stat().st_size}
(OUT/'artifact-sha256.json').write_text(json.dumps({'public_artifacts':hashes,'anonymous_local_profile_inputs':inputs},indent=2)+'\n')
public.append(OUT/'artifact-sha256.json')
(OUT/'raw/public-delivery-files.json').write_text(json.dumps([str(p) for p in public],indent=2)+'\n')
print(json.dumps({'tests':result.testsRun,'checks_passed':sum(checks.values()),'public_artifacts':len(public),'all_passed':True}))
