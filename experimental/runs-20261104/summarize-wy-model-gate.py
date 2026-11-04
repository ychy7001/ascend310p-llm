#!/usr/bin/env python3
"""从本轮原始证据生成脱敏最终判定；不访问实验机或执行 NPU 请求。"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


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
    perf=performance(raw) if status.get('performance_complete') else {'status':'not_run','reason':'G1/G2 did not both pass or run stopped before performance'}
    complete=status['state']=='complete'
    if g1 is not None and g2 is not None and not (g1['passed'] and g2['passed']):
        recommendation='不采用';reason='本轮模型数值或任务门禁未通过，性能按设计未执行。'
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
    result={'date':'2026-11-04','recommendation_zh':recommendation,'reason_zh':reason,
            'status':status['state'],'execution_error':status.get('error'),'model':'Qwen3.6-35B-A3B W8A8 TP2',
            'input_design_zh':'设计方已允许合成数值输入：26条原样回放+5边界fixture+16K/32K各2条合成会议，共35条',
            'input_sha256':sha(raw/'inputs.json'),'threshold_rule_sha256':sha(raw/'threshold-rule.json'),
            'input_token_counts':[{'id':c['id'],'tokens':c['tokens'],'token_ids_sha256':c['token_ids_sha256']} for c in inputs['cases']],
            'threshold_rule':rule,'G1':g1,'G2':g2,'numerical_details':details,'performance':perf,
            'configuration':configuration,'service_restoration':read(raw/'service-restoration.json'),
            'independent_closeout':closeout,'raw_evidence_sha256':evidence,
            'limitations_zh':['仅35条固定输入及固定任务子集，不是完整业务分布；不迁移旧栈收益。',
                              'KL为共同top-k token+OTHER桶的折叠分布，不是完整词表KL。',
                              '生成模式仅在相同生成条件前缀上比较逐位数值；若启用，不能冒充teacher forcing。'],
            'preparation_issues_zh':['CPU预处理首轮镜像自动加载torch_npu时缺少驱动库；仅CPU预处理关闭设备后端自动加载。',
                                   'transformers新返回结构导致token长度断言失败；显式return_dict=False后，五条fixture的长度与既有SHA全部核对通过。'],
            'constraints':{'a1_accessed':False,'host_installs':False,'CANN_or_driver_changed':False,'pushed_or_published':False}}
    out.joinpath('adjudication.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'recommendation':recommendation,'reason':reason,'G1_passed':g1['passed'] if g1 else None,'G2_passed':g2['passed'] if g2 else None},ensure_ascii=False))


if __name__=='__main__':main()
