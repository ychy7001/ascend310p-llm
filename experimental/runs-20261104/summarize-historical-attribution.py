#!/usr/bin/env python3
"""Conditional budgets when only the historical B1 timing export is local.

These are deliberately NOT represented as the missing four-window measurements.
"""
import csv,json,pathlib
out=pathlib.Path('results/decode-attribution-20261104/v2')
r=json.loads((out/'replay-summary.json').read_text());profiles=[json.loads((out/f'historical-profile/profile-rank{i}.json').read_text()) for i in (0,1)]
n=sum(r['decode_batch_steps'].values());t=r['wall_seconds'];b1=r['decode_batch_seconds']['1'];mean=lambda xs:sum(xs)/len(xs)
groups={g:mean([p['fp16_module_attribution']['groups_ms_per_step'][g] for p in profiles]) for g in profiles[0]['fp16_module_attribution']['groups_ms_per_step']}
reductions=[]
ledger=json.loads((out/'module-shape-ledger.json').read_text())
for rank,p in enumerate(profiles):
 for observed,module in zip(p['graph_reduction_node_ledger'],ledger[rank]['reduction_ledger']):
  reductions.append({'rank':rank,**observed,'fx_node':module['fx_node'],'module':module['module'],'op':module['op'],'reaches_graph_output':module['reaches_graph_output'],'join_evidence':'ordered join of 80 graph tasks and 80 runtime FX reductions, exact count stable over 96 steps; payload duration is not complete collective wait'})
(out/'historical-profile/reduction-node-map.json').write_text(json.dumps(reductions,indent=2)+'\n')
rows=[]
for g,k in groups.items():
 rows.append({'group':g,'calls_per_step':ledger[0]['group_counts'][g],'historical_b1_rank0_ms':profiles[0]['fp16_module_attribution']['groups_ms_per_step'][g], 'historical_b1_rank1_ms':profiles[1]['fp16_module_attribution']['groups_ms_per_step'][g],'mean_ms':k,'constant_cost_decode_seconds':n*k/1000,'constant_cost_fraction_of_replay':n*k/1000/t})
with (out/'historical-profile/fp16-group-summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
cat=lambda key:mean([p['categories'][key]['sum_ms_per_step'] for p in profiles])
shared=groups['shared_gate_up']+groups['shared_down'];linear=sum(groups[k] for k in ['linear_qkvz','linear_ba','linear_out_proj'])
def item(name,k,cost,numeric,scope):
 pool=n*k/1000
 return {'candidate':name,'historical_b1_ms_per_step':k,'assumed_new_kernel_speedup':2,'constant_cost_removed_pool_seconds':pool,'constant_cost_zero_cost_ceiling_fraction':pool/t,'constant_cost_2x_scenario_seconds':pool/2,'constant_cost_2x_scenario_fraction':pool/2/t,'cost':cost,'numeric_gate_required':numeric,'scope':scope,'status':'conditional; B2/B3/B4 timings and mixed-window cost are missing, not a measured production gain'}
items=[item('linear_attention_and_shared_experts_w8a8',linear+shared,'中',True,'170/220 FP16 NZ calls; excludes router, full-attention o_proj and lm_head'),item('moe_grouped_kernel_same_precision',cat('moe_grouped_w8a8'),'大',True,'existing quantization unchanged; tile/kernel scheduling only'),item('linear_qkvz_only_w8a8',groups['linear_qkvz'],'中',True,'30 projections; nested subset of combined quantization candidate, cannot add gains'),item('lm_head_path',cat('lm_head_fp16'),'中',True,'production FP16 NZ baseline required; prior ND microbenchmark is not comparable'),item('layout_transdata',cat('layout_transdata'),'中',False,'pure layout equivalence must be bitwise; arithmetic fusion requires numeric gate'),item('shared_experts_only_w8a8',shared,'中',True,'80 small projections; subset of combined quantization candidate')]
result={'evidence':'Historical 2026-11-03 B1/32K 96-step dual-rank kernel times joined to new real B3 module/FX mapping. New four-window exports unavailable during experiment-host connectivity loss.', 'formula':'N_decode * historical_B1_kernel_ms / 1000 / replay_wall. 2x scenario halves the pool, with zero added quantization/layout/launch overhead.','assumptions':['Kernel cost is constant over batches 1..4 and contexts; unvalidated.','No savings counted for pure Prefill or mixed steps; quantization may also affect those phases, so this is not a whole-implementation ceiling.','Removed busy time need not reduce wall time equally with frozen external arrivals.','Cross-rank values averaged as a statistic, never summed.'],'decode_steps':n,'replay_wall_seconds':t,'new_weighted_profile_composition_available':False,'candidate_scenarios':items,'fp16_groups_mean_ms':groups,'graph_reduction_payload_ms_per_step':[sum(x['ms_per_step'] for x in p['graph_reduction_node_ledger']) for p in profiles], 'host_scheduler_zero_cost_ceiling_fraction':r['scheduler_cpu_seconds']/t, 'graph_14_historical_reweighted_favorable_scenario_fraction':r['decode_batch_seconds']['1']/t*(1-1/1.0018)+r['decode_batch_seconds']['3']/t*(1-1/1.0296)+r['decode_batch_seconds']['4']/t*(1-1/1.0543)}
(out/'candidate-scenarios.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
