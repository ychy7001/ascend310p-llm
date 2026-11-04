#!/usr/bin/env python3
"""只用已保存证据汇总本轮，不访问设备或改变阈值。"""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);a=p.parse_args();r=a.results;raw=r/'raw'
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
native=read(raw/'native-analysis/summary.json');full=read(raw/'native-gdn-analysis/summary.json');cann=read(raw/'cann-final-analysis/summary.json')
def compact(summary):
 result={}
 for key,value in summary.items():
  if not isinstance(value,dict):result[key]=value;continue
  result[key]={k:v for k,v in value.items() if k!='first_failure_by_rank'}
  if 'first_failure_by_rank' in value:
   result[key]['first_failure_by_rank']={rank:None if row is None else {'source':row['source'],'layer':row['metadata']['layer'],'call':row['metadata']['call'],'failed_outputs':[x for x in row['metrics'] if not x['passed']]} for rank,row in value['first_failure_by_rank'].items()}
 return result
native=compact(native);cann=compact(cann)
row=[json.loads(x) for x in (raw/'native-analysis/rows.jsonl').read_text().splitlines()][0]
variants=read(raw/'minimize-analysis.json')
report={'date':'2026-11-04','conclusion_zh':'原生WY偏差已单算子复现，暂不修改；B2真实快照门禁失败，不进入G1',
 'started_utc':'2026-11-04T13:21:16Z','deadline_utc':'2026-11-04T17:21:16Z',
 'model_probe':{'arm':'A / Torch WY','case':'replay-08','prompt_tokens':765,'output_tokens':1,'captured_calls':120,'calls_per_rank':60,'prefill_segments':[640,125],'padded_segments':[640,128],'cache_hit_tokens':0,'first_generated_token':101780,'matches_previous_A_and_A_prime_first_token':True,'previous_B_first_token':127326,'scope_zh':'一条观测请求，不是整模型G1/G2'},
 'provenance':{'image_id':'d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a','pr17640_commit':'18163f6b1d208ab3b4fb06e9752225611c37f40e','native_source':read(raw/'native-blob-verification.json'),'chunk_sha256':row['metadata']['chunk_sha256'],'original_G1_inputs_sha256':'e8fa2ed60bb717b2cad731988a0f85ba9dd07f39f0e614b2fb2e9b333b9e8c13','original_G1_rule_sha256':'42e2cf305e35b308ce75045bd91dc9dcd59d04daaa4cf7a2e4ef044ddcb8c67c','original_G1_thresholds_sha256':'2ca8e95fc5374825a5725bf525e85ce59be5fcf0dafebf521cd416a0e30e0426','localization_protocol_sha256':sha(raw/'localization-protocol.json'),'snapshot_manifest_sha256':sha(raw/'snapshot-manifest.json')},
 'thresholds':read(raw/'localization-protocol.json'),
 'first_exceedance':{'layer_zero_based':0,'wy_call_per_rank_one_based':1,'ranks':[0,1],'layer_name':row['metadata']['layer'],'first_bad_boundary':'u_kernel','raw_sequence_tokens':640,'padded_tokens':640,'chunks':10,'varlen_cu':[0,640],'initial_state_nonzero':False,'metadata':row['metadata'],'rank0_native_vs_torch':row['native_vs_torch'],'rank0_native_vs_cpu':row['native_vs_cpu'],'rank0_torch_vs_cpu':row['captured_torch_vs_cpu'],'full_GDN_first_bad_boundary':'final_state','full_GDN_output_passed_all_120_calls':True},
 'single_operator':{'native_WY':native,'full_GDN':full,'torch_replay_equal_capture':120,'native_bitwise_repeat_equal':120,'tensor_layout_qk_exact':True,'all_outputs_finite':True},
 'trigger':{'minimal':read(raw/'minimal-input-statistics.json'),'source_coordinate_zero_based':[0,14,209,79],'source_chunk_start':192,'minimal_coordinate':[0,0,17,79],'single_variable_results':[{ 'variant':v['metadata']['variant'],**{key:{'passed':all(x['passed'] for x in v[key]),'max_abs':{x['output']:x['max_abs'] for x in v[key]},'outside_tolerance':{x['output']:x['outside_tolerance'] for x in v[key]}} for key in ('native_vs_cpu','native_vs_torch','torch_vs_cpu')}} for v in variants], 'root_cause_zh':'较大幅度v、弱衰减与相关k组合中的混合精度/消去敏感；未定位唯一错误指令。该最小失败未触发Frobenius快速分支。','not_required':['long_sequence','multi_chunk','varlen','multi_sequence_batch','nonzero_initial_state','TP_communication','noncontiguous_stride']},
 'fix':{'made':False,'reason_zh':'基线也超差，原生误差未整体大于基线；没有已证实且有把握的小修复'},
 'B2':{'namespace':'cann_wy921::compute','vendor':'wy921_cann_transformer','gating_changed':False,'fallback':False,'thin_binding':read(raw/'cann-thin/provenance.json'),'synthetic_devices':[next(json.loads(line) for line in (raw/'cann-final'/('synthetic-d'+str(device)+'.jsonl')).read_text().splitlines() if json.loads(line)['kind']=='summary') for device in (0,1)],'synthetic_passed':30,'synthetic_total':30,'real_WY_snapshots':cann,'device_gate_passed':False,'G1_run':False,'G2_run':False,'reason_zh':'真实WY快照逐点门禁失败；不放宽阈值进入模型门禁'},
 'repro':read(r/'repro-manifest.json'),
 'validation':{'cpu_oracle_tests_passed':3,'no_native_full_model_G1_G2':True,'no_performance':True,'no_driver_CANN_changes':True,'production_image_changed':False,'other_experiment_machine_accessed':False,'remote_heavy_parsing':False,'maximum_concurrent_device_jobs':1,'maximum_CPU_workers':2},
 'limitations_zh':['仅采样一条765-token真实请求，未穷举35条或长32K调用。','首次超差按既有算子容差定义，未获得模型隐藏层/logits的逐层因果传播。','Torch基线也存在小幅逐点超差；尚未证明本轮小误差完整解释上一轮最大logprob差≥13。','CPU/NPU取样增加同步，可能改变时序；Torch首token与原A/A′一致，离线Torch输出与120份截获逐位一致。','快照保留原始shape/stride/format元数据，离线重建使用连续ND张量；最小连续输入仍可复现。','复现包不含二进制，需匹配9.2.1工具链重编译；完整快照和包不入git，仅提交清单。']}
for device in report['B2']['synthetic_devices']:
 device['loaded_apis']=[value.replace('/cann-wy/','<cann-wy>/').replace('/gdn-pair/','<gdn-pair>/') for value in device['loaded_apis']]
report['execution_issues_zh']=['原生设备重放v1因PYTHONPATH覆盖缺tbe，数值未执行；修正后v2完成。','B2 v1因vendor顺序缺_pattern，shell后续重放掩盖退出码；已修正失败即停。','B2 v2数值15项通过但H/O来源错误；v3恢复bootstrap后的vendor顺序，双卡30项及来源检查通过。']
recovery=raw/'final-independent-recovery.json'
if recovery.exists():
 from datetime import datetime,timezone
 report['restoration']={**read(recovery),'independent_new_SSH_session':True,'image_service_started_by_experiment':False}
 report['completed_utc']=read(recovery)['checked_at_utc']
 report['elapsed_minutes']=(datetime.fromisoformat(report['completed_utc'])-datetime.fromisoformat(report['started_utc'].replace('Z','+00:00'))).total_seconds()/60
(r/'adjudication.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
