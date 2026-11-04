#!/usr/bin/env python3
"""从本轮原始证据生成脱敏最终判定；不访问实验机或执行 NPU 请求。"""
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import statistics
import re


def read(path):return json.loads(path.read_text())


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]


def performance(raw):
    results={}
    for arm in ('P','A1','B','A2'):
        label='perf-'+arm;values={}
        for length in (1024,4096,16384,32768):
            data=[r for r in rows(raw/(label+'-prefill-'+str(length)+'.jsonl')) if r['kind']=='request']
            assert len(data)==3 and all(r['valid'] for r in data)
            values['cold_prefill_'+str(length)+'_tok_s']=statistics.median(r['prefill_tok_s'] for r in data)
        for c in (1,3,4):
            data=rows(raw/(label+'-stream-C'+str(c)+'.jsonl'))
            rounds=[r for r in data if r['kind']=='round'];requests=[r for r in data if r['kind']=='request']
            assert len(rounds)==3 and all(r['failed_requests']==0 for r in rounds)
            values['stream_C'+str(c)+'_output_tok_s']=statistics.median(r['aggregate_output_tok_s'] for r in rounds)
            values['stream_C'+str(c)+'_ttft_s']=statistics.median(r['ttft_s'] for r in requests)
        replay=read(raw/(label+'-replay.json'));assert len(replay['calls'])==26 and replay['cold_cache_verified']
        values['replay_26_elapsed_s']=replay['elapsed_s'];values['replay_output_tokens']=replay['output_tokens']
        results[arm]=values
    changes={}
    for key in results['P']:
        changes[key]={label:100*(results[x][key]/results[y][key]-1)
                      for label,x,y in [('B_vs_A_percent','B','A1'),('B_vs_P_percent','B','P'),('A2_vs_A1_drift_percent','A2','A1')]}
    return {'status':'complete','values':results,'changes':changes,
            'aggregation':'Median of three measured prefill/stream rounds; replay single wall-clock run per arm; positive throughput delta is faster; negative duration delta is faster.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();raw=a.raw;out=a.out
    status=read(raw/'status.json');rule=read(raw/'threshold-rule.json');inputs=read(raw/'inputs.json')
    g1=read(raw/'G1-adjudication.json') if (raw/'G1-adjudication.json').exists() else None
    g2=read(raw/'G2-adjudication.json') if (raw/'G2-adjudication.json').exists() else None
    if g2 is None:
        g2={'status':'incomplete','passed':None,'counts':{arm:read(raw/(arm+'-G2-counts.json')) for arm in ('P','A','B') if (raw/(arm+'-G2-counts.json')).exists()}}
    perf=performance(raw) if status.get('performance_complete') else {'status':'not_run','reason':'G1/G2 did not both pass or run stopped before performance'}
    complete=status['state']=='complete'
    if g1 is not None and (g1['passed'] is False or g2['passed'] is False):
        recommendation='不采用';reason='G1数值偏差超过A′噪声的1.5倍；G2通过，性能按门禁规则未执行。'
    elif complete and perf['status']=='complete':
        deltas=perf['changes']
        benefit=deltas['replay_26_elapsed_s']['B_vs_A_percent']<0 and deltas['replay_26_elapsed_s']['B_vs_P_percent']<0
        recommendation='建议采用' if benefit else '不采用'
        reason='双门禁通过，且业务回放相对A和P均缩短。' if benefit else '双门禁通过，但业务回放未同时优于A和P。'
    else:recommendation='待定';reason='计划外问题中止，尚无完整模型门禁证据。'
    evidence={}
    for path in sorted(raw.rglob('*')):
        if path.is_file() and 'cache' not in path.parts:
            evidence[str(path.relative_to(raw))]=sha(path)
    manifests={label:read(raw/label/'manifest.json') for label in ('P','A','A_prime','B') if (raw/label/'manifest.json').exists()}
    identity={}
    if 'A' in manifests and 'A_prime' in manifests:
        identity['A_and_A_prime_manifests_identical']=manifests['A']==manifests['A_prime']
        assert identity['A_and_A_prime_manifests_identical']
    if 'A' in manifests and 'B' in manifests:
        left,right=copy.deepcopy(manifests['A']),copy.deepcopy(manifests['B'])
        assert left['env'].pop('WY_GATE_ARM')=='A' and right['env'].pop('WY_GATE_ARM')=='B'
        identity['A_vs_B_only_manifest_difference']='WY_GATE_ARM'
        assert left==right
    configuration={}
    for label,m in manifests.items():
        command=m['command']
        configuration[label]={'image':m['image'],'tensor_parallel_size':int(command[command.index('--tensor-parallel-size')+1]),
                              'max_model_len':int(command[command.index('--max-model-len')+1]),
                              'max_num_seqs':int(command[command.index('--max-num-seqs')+1]),
                              'max_num_batched_tokens':int(command[command.index('--max-num-batched-tokens')+1]),
                              'compilation_config':json.loads(command[command.index('--compilation-config')+1]),
                              'prefix_caching':'--enable-prefix-caching' in command,
                              'MTP':0,'WY':'native' if label=='B' else 'torch'}
    details={name:read(raw/name) for name in ['A_prime-vs-A.json','B-vs-A.json','concrete-thresholds.json','observation-mode.json'] if (raw/name).exists()}
    closeout=read(out/'independent-closeout-v2.json')
    assert closeout['health_http_status']==200 and closeout['service_active_state']=='active'
    initial_manifest_sha=read(out/'preflight.json')['source_sha256']['live_service_manifest']
    assert initial_manifest_sha==closeout['live_service_manifest_sha256']
    restoration=read(raw/'service-restoration.json')
    assert restoration['start_returncode']==0 and restoration['health_200'] and restoration['active_state']=='active'
    offline=read(raw/'offline-adjudication.json') if (raw/'offline-adjudication.json').exists() else None
    concrete=read(raw/'concrete-thresholds.json')
    assert concrete['rule_sha256']==sha(raw/'threshold-rule.json')
    assert concrete['A_reference_sha256']==sha(raw/'A-G1.json')
    assert concrete['noise_sha256']==sha(raw/'A_prime-vs-A.json')
    assert concrete['thresholds']=={key:1.5*value for key,value in details['A_prime-vs-A.json']['metrics'].items()}
    b_started=read(raw/'B-container-inspect.json')[0]['State']['StartedAt']
    assert not concrete['B_started']
    assert concrete['frozen_at_unix_ns'] < int(datetime.datetime.fromisoformat(b_started.replace('Z','+00:00')).timestamp()*1e9)
    native_log=(raw/'B-server.log').read_text()
    native_ranks=sorted(set(re.findall(r'Worker_(TP[01])[^\n]*\[wy_gate_native_call\]',native_log)))
    assert native_ranks==['TP0','TP1']
    observations={}
    for label in ('A','A_prime','B'):
        capture=read(raw/(label+'-G1.json'))
        assert len(capture['cases'])==35 and all(len(case['greedy_ids'])==256 for case in capture['cases'])
        assert capture['prefix_cache_delta']['vllm:prefix_cache_hits_total']==0
        observations[label]={'cases':35,'generated_tokens':8960,'prefix_cache_delta':capture['prefix_cache_delta']}
    for label in ('A_prime','B'):
        compared=details[label+'-vs-A.json']
        indices=[case['greedy_first_divergence'] for case in compared['per_case'] if case['greedy_first_divergence'] is not None]
        observations[label].update(positions_with_identical_conditioning=compared['positions_compared'],
                                   greedy_divergent_cases=len(indices),first_greedy_divergence=min(indices) if indices else None)
    gates_done=g1 is not None and g1['passed'] is not None and g2['passed'] is not None
    result={'date':'2026-11-04','recommendation_zh':recommendation,'reason_zh':reason,
            'status':'gates_complete_g1_rejected' if gates_done and g1['passed'] is False else status['state'],
            'controller_status':status['state'],'execution_error':status.get('error'),
            'offline_adjudication':offline,'model':'Qwen3.6-35B-A3B W8A8 TP2',
            'input_design_zh':'设计方已允许合成数值输入：26条原样回放+5边界fixture+16K/32K各2条合成会议，共35条',
            'input_sha256':sha(raw/'inputs.json'),'threshold_rule_sha256':sha(raw/'threshold-rule.json'),
            'concrete_thresholds_sha256':sha(raw/'concrete-thresholds.json'),
            'threshold_freeze_proof':{'frozen_at_unix_ns':concrete['frozen_at_unix_ns'],'B_container_started_at':b_started,'frozen_before_B':True},
            'input_token_counts':[{'id':c['id'],'tokens':c['tokens'],'token_ids_sha256':c['token_ids_sha256']} for c in inputs['cases']],
            'threshold_rule':rule,'G1':g1,'G2':g2,'numerical_details':details,'performance':perf,
            'configuration':configuration,'arm_identity':identity,'observations':observations,
            'native_WY_execution':{'worker_ranks_observed':native_ranks,'fallback':False},
            'service_restoration':restoration,'live_service_manifest_unchanged':True,
            'independent_closeout':closeout,'raw_evidence_sha256':evidence,
            'limitations_zh':['仅35条固定输入及固定任务子集，不是完整业务分布；不迁移旧栈收益。',
                              'KL为共同top-k支持内归一化的条件KL，不覆盖支持外概率质量，不是完整词表KL。',
                              '生产参数下33024-token prompt_logprobs探测OOM，按已授权退路使用生成logprob；只比较相同生成条件前缀及首分歧位置，非完整teacher forcing。',
                              'B有7个A目标token落在B的top-20外；最大绝对logprob差只报保守下界，下界已超冻结阈值。',
                              'A′/B与A分别仅7368/1391个位置具备相同生成条件；后续不同上下文数值未混入KL和logprob比较。',
                              'A′自身有11/35条greedy首分歧；本轮使用该实测噪声预算，未完成算子根因归因。'],
            'preparation_issues_zh':['CPU预处理首轮镜像自动加载torch_npu时缺少驱动库；仅CPU预处理关闭设备后端自动加载。',
                                   'transformers新返回结构导致token长度断言失败；显式return_dict=False后，五条fixture的长度与既有SHA全部核对通过。'],
            'execution_issues_zh':['prompt_logprobs=20探测需额外3.55GiB而仅空闲1.79GiB，A引擎OOM；保留失败证据且finally恢复服务，未修改生产内存参数或引擎。',
                                   'A/A′汇总时FP32概率舍入使OTHER桶产生inf；B前改为共同显式top-k条件KL并冻结方法、具体阈值。',
                                   'B的全部G1/G2已采集完成；在线汇总因top-20遗漏A目标token而停止，离线以保守删失下界完成拒绝判定，阈值未改且未追加NPU请求。'],
            'validation_zh':['数值对照及既有stream/replay工具共13项CPU测试通过。','独立连接确认服务active、18081/health=200，无运行实验容器；现役服务manifest SHA与实验前一致。'],
            'independent_closeout_sha256':sha(out/'independent-closeout-v2.json'),
            'constraints':{'a1_accessed':False,'host_installs':False,'CANN_or_driver_changed':False,'pushed_or_published':False}}
    out.joinpath('adjudication.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'recommendation':recommendation,'reason':reason,'G1_passed':g1['passed'] if g1 else None,'G2_passed':g2['passed'] if g2 else None},ensure_ascii=False))


if __name__=='__main__':main()
