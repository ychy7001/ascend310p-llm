#!/usr/bin/env python3
"""Validate available evidence; missing deliverables remain explicitly incomplete."""
import csv,gzip,hashlib,io,json,pathlib,re,subprocess,sys
root=pathlib.Path('results/decode-attribution-20261104');out=root/'v2'
public=[root/'README.md',root/'v1-README.md']+[out/n for n in ['site-summary.json','length-quantiles.csv','site-concurrency-histograms.csv','configuration-summary.json','frozen-replay-plan.json','replay-summary.json','replay-requests.csv','prefill-chunks.csv','replay-batch-weights.csv','decode-steps.csv.gz','module-shape-ledger.json','candidate-scenarios.json','recovery-and-limitations.json']]+[out/'historical-profile'/n for n in ['profile-rank0.json','profile-rank1.json','fp16-group-summary.csv','reduction-node-map.json']]
checks={}
r=json.loads((out/'replay-summary.json').read_text());s=json.loads((out/'site-summary.json').read_text());ledger=json.loads((out/'module-shape-ledger.json').read_text());profiles=[json.loads((out/f'historical-profile/profile-rank{i}.json').read_text()) for i in (0,1)]
checks['stage_wall_closes']=abs(sum(r['phase_seconds'].values())+r['dispatch_idle_and_unassigned_seconds']-r['wall_seconds'])<1e-7
checks['all_replay_steps_joined']=not r['validation']['unmatched_replay_worker_sequences']
checks['eleven_requests_completed']=r['requests']==11 and r['validation']['request_count_matches']
checks['token_counts_match_http']=r['validation']['http_server_token_counts_match']
checks['all_prompt_tokens_accounted']=r['validation']['prompt_tokens_accounted']
checks['cold_prefix_zero_hits']=r['prefix_counter_delta']['prefix_cache_hits_total']==0
checks['decode_weights_close']=sum(r['decode_batch_steps'].values())==r['phase_steps']['decode'] and abs(sum(r['decode_batch_seconds'].values())-r['phase_seconds']['decode'])<1e-6
checks['raw_synthetic_steps_preserved']=gzip.decompress((out/'decode-steps.csv.gz').read_bytes())==(out/'decode-steps.csv').read_bytes()
checks['all_phase_steps_present']=len(list(csv.DictReader(io.StringIO(gzip.decompress((out/'decode-steps.csv.gz').read_bytes()).decode()))))==sum(r['phase_steps'].values())
checks['real_request_pair_count']=sum(d['requests']['n'] for d in s['sites'])==s['pooled_requests']==1143
checks['proxy_count']=sum(d['meeting_length_proxy']['n'] for d in s['sites'])==s['meeting_proxy_n']==813
checks['six_real_mapping_forwards']=len(ledger)==2 and sum(d['mapped_forward_count'] for d in ledger)==6
checks['real_mapping_shapes_order_match']=all(d['identical_order_and_weight_shapes'] for d in ledger)
checks['mapping_220_fp16_80_reductions']=all(len(d['fp16_ledger'])==220 and len(d['reduction_ledger'])==80 for d in ledger)
checks['all_reduction_results_reach_output']=all(n['reaches_graph_output'] for d in ledger for n in d['reduction_ledger'])
checks['historical_window_96_each_rank']=all(d['steps']==96 for d in profiles)
checks['graph_fp16_220_real_device_ids']=all(d['graph_fp16_node_count']==220 and d['fp16_hardware_metadata_matches']==220*96 for d in profiles)
checks['graph_reductions_80_per_step']=all(d['graph_reduction_counts_by_step']=={'80':96} and len(d['graph_reduction_node_ledger'])==80 for d in profiles)
checks['fp16_budget_closes']=all(abs(d['fp16_module_attribution']['total_ms_per_step']-d['categories']['fp16_nz_projection']['sum_ms_per_step'])<1e-7 for d in profiles)
checks['graph_reduction_payload_does_not_overlap_named_compute']=all(sum(n['overlap_compute_ms_per_step'] for n in d['graph_reduction_node_ledger'])<1e-6 for d in profiles)
checks['frozen_plan_same_eleven_lengths']=len(json.loads((out/'frozen-replay-plan.json').read_text())['requests'])==11
private_patterns=[r'ychy',r'ails-n\d',r'tail5bdb4c',r'\b(?:\d{1,3}\.){3}\d{1,3}\b',r'/(?:home|srv|workspace|usr|etc)/']
leaks=[]
for path in public:
 text=(gzip.decompress(path.read_bytes()).decode() if path.suffix=='.gz' else path.read_text())
 for pattern in private_patterns:
  if re.search(pattern,text):leaks.append({'artifact':str(path.relative_to(root)),'pattern':pattern})
checks['public_summaries_no_site_host_ip_or_absolute_paths']=not leaks
p=subprocess.run(['git','check-ignore',str(out/'raw/site02-metadata.jsonl'),str(out/'raw/paired-lengths-arrivals.json')],capture_output=True,text=True)
checks['raw_site_inputs_gitignored']=p.returncode==0 and len(p.stdout.splitlines())==2
modules=['test_observe_decode','test_inference_numeric_metadata','test_analyze_decode_observation','test_decode_trace_utils','test_analyze_decode_profile','test_attribute_decode_kernels','test_map_decode_modules']
import os
env={**os.environ,'PYTHONPATH':'tools'};p=subprocess.run([sys.executable,'-m','unittest',*modules],env=env,capture_output=True,text=True)
(out/'raw/unit-tests.log').write_text(p.stdout+p.stderr);match=re.search(r'Ran (\d+) tests',p.stderr)
checks['reusable_tool_tests_pass']=p.returncode==0 and match is not None
hop=json.loads((out/'raw/hop-cpu-validation.json').read_text());checks['higher_order_cpu_bitwise_transparency']=hop['bitwise_equal'] and hop['input_unchanged'] and hop['cpu_only']
validation={'overall_task_complete':False,'available_evidence_checks':checks,'available_evidence_checks_passed':sum(checks.values()),'available_evidence_checks_total':len(checks),'unit_tests':int(match.group(1)) if match else None,'higher_order_cpu_regression':hop,'privacy_violations':leaks,'incomplete_deliverables':['new four-window dual-rank exported kernel/trace validation','new measured concurrency-weighted operator composition and final candidate ranking','complete collective wait/critical-path duration','current health after offline-export connectivity loss'],'interpretation':'Passing checks validate available data only; they do not complete the blocked deliverables.'}
(out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
public.append(out/'validation.json')
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
manifest={'artifacts':{str(p.relative_to(root)):sha(p) for p in public},'inputs':{'site01_numeric':sha(out/'raw/site01-metadata.jsonl'),'site02_numeric':sha(out/'raw/site02-metadata.jsonl'),'site03_numeric':sha(out/'raw/site03-metadata.jsonl'),'historical_kernel_rank0':sha(pathlib.Path('results/production-w8a8-20261103/kernel-details-decode-32768-after-first-output-rank0.csv.gz')),'historical_kernel_rank1':sha(pathlib.Path('results/production-w8a8-20261103/kernel-details-decode-32768-after-first-output-rank1.csv.gz')),'cold_replay_observation_archive':sha(out/'raw/replay-observation.tar.gz'),'module_observation_archive':sha(out/'raw/module-observation.tar.gz')},'raw_site_data_committed':False,'new_four_window_profile_exports_available':False}
(out/'artifact-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n');public.append(out/'artifact-sha256.json')
(out/'raw/public-artifact-paths.txt').write_text(''.join(str(p)+'\n' for p in public))
print(json.dumps(validation,indent=2))
assert all(checks.values()),checks
