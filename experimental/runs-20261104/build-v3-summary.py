#!/usr/bin/env python3
"""Build anonymous, numerical v3 tables from eight local device profiles."""
import csv,json,re,shutil
from pathlib import Path
ROOT=Path('results/decode-attribution-20261104');OUT=ROOT/'v3'
# Keep complete eager task-ID sets in ignored raw data; publish counted samples.
for path in (OUT/'profiles').glob('*.json'):
 profile=json.loads(path.read_text());changed=False
 for node in profile['fp16_module_attribution']['nodes']:
  ids=node['device_identifiers']
  if len(ids)>3:
   backup=OUT/'raw/detailed-attribution'/path.name;backup.parent.mkdir(exist_ok=True)
   if not backup.exists():shutil.copy2(path,backup)
   node['device_identifier_count']=len(ids);node['device_identifiers']=ids[:3]
   node['device_identifiers_are_samples']=True;changed=True
 if changed:path.write_text(json.dumps(profile,indent=2)+'\n')
weighted=json.loads((OUT/'weighted-composition.json').read_text());replay=json.loads((ROOT/'v2/replay-summary.json').read_text())
labels=['b1-long','b2-long','b3-short','mixed-b4'];profiles={label:[json.loads((OUT/f'profiles/{label}-rank{r}.json').read_text()) for r in (0,1)] for label in labels}
def dump(name,value):(OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def csvout(name,rows):
 with (OUT/name).open('w') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
rows=[];communications=[];fp16=[]
for b,label in enumerate(labels,1):
 for rank,p in enumerate(profiles[label]):
  cycle=p['window_ms_per_step']
  for category,x in p['categories'].items():rows.append({'batch':b,'rank':rank,'category':category,'steps':p['steps'],'cycle_ms':cycle,'kernel_union_ms_per_step':x['union_ms_per_step'],'fraction_of_case_cycle':x['union_ms_per_step']/cycle,'count_per_step':x['count']/p['steps']})
  for group,value in p['fp16_module_attribution']['groups_ms_per_step'].items():fp16.append({'batch':b,'rank':rank,'group':group,'ms_per_step':value,'fraction_of_case_cycle':value/cycle})
  graph=p['graph_reduction_node_ledger'];communications.append({'batch':b,'rank':rank,'cycle_ms':cycle,'graph_reduction_nodes':len(graph),'graph_reduction_payload_ms':sum(x['ms_per_step'] for x in graph),'graph_reduction_overlap_compute_ms':sum(x['overlap_compute_ms_per_step'] for x in graph),'all_81_reduce_payload_ms':p['hardware_tasks']['REDUCE_ASYNC_V2']['sum_ms_per_step'],'visible_allreduce_wrapper_ms':p['categories']['visible_allreduce']['union_ms_per_step'],'visible_allgather_wrapper_ms':p['categories']['visible_allgather']['union_ms_per_step'],'named_kernel_gap_ms':p['uncovered_by_named_kernel_ms_per_step'],'cpu_op_coverage_in_gap_ms':p['worker_cpu_ops_in_named_kernel_gap_ms_per_step'],'no_active_hardware_coverage_ms':p['uncovered_by_all_active_hardware_ms_per_step']})
csvout('per-case-composition.csv',rows);csvout('fp16-groups.csv',fp16);csvout('communication-host.csv',communications)
choices=[('moe-small-m','moe_grouped_w8a8',3,'大','仅替换80个/步QuantGroupedMatmulDequant的小M tiling，保持分组/scale/累加语义和通信不变','位序改变或逐位检查不一致则G1/G2；一致也需任务回归'),('qkvz-w8a8','linear_qkvz',2,'中','仅30个linear_attn.in_proj_qkvz从FP16改W8A8','必须G1/G2'),('lm-head','lm_head_fp16',2,'中','以生产FP16 NZ head为对照，只切换head实现；额外量化/分片成本计入','改变算术或量化则G1/G2'),('layout','layout_transdata',2,'中','仅复用经消费者格式校验的重复ND→NZ转换；当前尚未证明哪些节点可删','纯布局逐位一致；融合改算术则G1/G2')]
candidates=[]
for ident,key,cost,label,action,gate in choices:
 s=weighted['candidate_scenarios'][key];candidates.append({'id':ident,'component':key,'action':action,'cost':label,'cost_score':cost,'model_gate':gate,'assumed_kernel_speedup':2,'wall_reduction_scenario_pct':s['wall_reduction_fraction']*100,'decode_pool_zero_cost_ceiling_pct':s['removable_kernel_wall_fraction']*100,'scenario_saved_seconds':replay['wall_seconds']*s['wall_reduction_fraction'],'score_scenario_pct_per_cost':s['wall_reduction_fraction']*100/cost,'minimum_speedup_for_3pct_before_overheads':s['minimum_kernel_speedup_for_3pct_wall_ignoring_overheads']})
candidates.sort(key=lambda r:r['score_scenario_pct_per_cost'],reverse=True)
q=next(c for c in candidates if c['id']=='qkvz-w8a8');margin=q['scenario_saved_seconds']-.03*replay['wall_seconds']
dump('candidate-budgets.json',{'budget_scope':'Pure decode only: 438.268899s / 643.705051s. Prefill and mixed phase changes are unbudgeted. Ceilings are removable targeted kernel pools, not full-change all-phase ceilings or measured speedups. Fixed arrivals and pipeline dependencies can reduce realized savings.','cost_score_convention':'Medium=2; large=3; judgement, not measured engineering hours. Compare all candidates under the same unvalidated 2x scenario.','candidates':candidates,'qkvz_twofold_added_overhead_margin_to_3pct_seconds':margin,'qkvz_twofold_added_overhead_margin_ms_per_decode_step':margin*1000/11798})
transfer=[]
for f in (OUT/'raw').glob('*.rsync.log'):
 s=f.read_text();m=re.search(r'Total file size: ([\d,]+) bytes',s)
 if m:transfer.append(int(m[1].replace(',','')))
resources={};swaps=[]
for f in list((OUT/'raw').glob('cann-export-*.log'))+list((OUT/'raw').glob('*resource.txt'))+list((OUT/'raw').glob('analyze-*-rank*.log')):
 s=f.read_text();m=re.search(r'Maximum resident set size \(kbytes\): (\d+)',s)
 if m:resources[f.name]=int(m[1])
 m=re.search(r'\n\s*Swaps: (\d+)',s)
 if m:swaps.append(int(m[1]))
dump('local-parsing.json',{'profile_subdirectories_transferred':8,'profile_bytes_reported_by_rsync':sum(transfer),'profile_GiB_reported_by_rsync':sum(transfer)/1024**3,'transfer_bwlimit_KiB_per_second':20000,'remote_parsing_processes_this_round':0,'npu_workloads_this_round':0,'service_operations_this_round':0,'matched_cpu_exporter_cann_version':'9.1.0-beta.1','exporter_source':'Read-only copy of existing matching image layer; no container started','parse_address_space_limit_GiB_per_process':8,'final_export_affinity_cpus':2,'steady_state_heavy_parse_concurrency':1,'maximum_separate_heavy_parsers_running_together':2,'measured_peak_process_RSS_MiB':max(resources.values())/1024,'measured_peak_process_RSS_scope':'Per-process high-water value from time -v; not aggregate PSS or a cgroup peak','all_recorded_time_reports_swaps':max(swaps) if swaps else None,'whole_host_swap_io_measured':False,'initial_local_vendor_pool_deviation':'Initial CANN summary aggregation uses Pool(4); timeline aggregation can create 3 processes each with Pool(4). No live peak process count was saved. Top-level serialization alone did not enforce the requested <=2 ceiling. These aggregation paths were disabled after discovery; final device timelines were re-exported without them. Summary CSV contents are unchanged native output, but the initial concurrency violation remains.','raw_retention':'Original captures, journals, machine identities, copied vendor tools and full logs stay under gitignored raw directory; committed summaries contain none of those identifiers.'})
# Weighted auxiliary hardware quantities remain overlapping diagnostics, not extra rows in the closed composition.
aux={}
for key in ['raw_reduce_outside_compute_ms_per_step','uncovered_by_all_active_hardware_ms_per_step','worker_cpu_ops_in_named_kernel_gap_ms_per_step']:
 vals=[]
 for rank in (0,1):
  value=0.;coverage=0.
  for label,w in weighted['time_weights'].items():
   p=profiles[label][rank];v=p[key]
   if v is not None:value+=w*v/p['window_ms_per_step'];coverage+=w
  vals.append({'rank':rank,'decode_time_fraction':value,'observed_case_time_weight':coverage,'wall_fraction':value*replay['decode_only_fraction_of_wall']})
 aux[key]=vals
dump('overlapping-diagnostics.json',aux)
print('transfer GiB',sum(transfer)/1024**3,'peak MiB',max(resources.values())/1024,'QKVZ margin',margin)
