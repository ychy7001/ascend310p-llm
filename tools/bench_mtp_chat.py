# SPDX-License-Identifier: Apache-2.0
"""固定业务近似chat输入：流式延迟、阶段指标和MTP接受率；不是质量评分。"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from bench_http_stream import post
from bench_server_phases import snapshot

PROMPTS=[('dialogue','用中文解释 INT4 权重、INT8 激活和 KV Cache 量化分别节省什么，并说明为什么它们不一定提高速度。分四点回答。'),
('meeting','请根据以下会议记录总结决定、待办事项、负责人和风险。不得补充记录中不存在的事实。\n' + '产品会议记录：李明提出单芯片48G方案，王芳负责测试112K上下文。团队决定先保留实验基线，验证精度后再发布。陈强负责整理性能数据。风险是长上下文速度下降和量化误差。下周一复核。\n'*16),
('code','Write a Python function that merges overlapping closed intervals. Explain the algorithm and give two examples.'),
('analysis','Explain why faster prefill kernels might improve time to first token without increasing decode speed. Include a concrete example and discuss memory bandwidth and launch overhead.')]

def metrics(base):
    with __import__('urllib.request',fromlist=['urlopen']).urlopen(base+'/metrics') as r: raw=r.read().decode()
    out={}
    for line in raw.splitlines():
        if line.startswith('vllm:spec_decode_') and '_total{' in line:
            key=line.split('{')[0];out[key]=out.get(key,0)+float(line.split()[-1])
    return out

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base-url',required=True);p.add_argument('--model',default='qwen36-w4a8-c8');p.add_argument('--label',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--repeats',type=int,default=3)
a=p.parse_args()
with a.out.open('x') as file:
    file.write(json.dumps({'kind':'config','label':a.label,'repeats':a.repeats,'warmup':1,'max_tokens':256,'temperature':0,'enable_thinking':False,'scope':__doc__})+'\n');file.flush()
    for label,prompt in PROMPTS:
        for repeat in range(-1,a.repeats):
            phase_before=snapshot(a.base_url);mtp_before=metrics(a.base_url)
            begin=time.perf_counter();first=None;events=[];text='';usage=None;finish=None
            with post(a.base_url,'/v1/chat/completions',{'model':a.model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':256,'chat_template_kwargs':{'enable_thinking':False},'stream':True,'stream_options':{'include_usage':True}}) as response:
                for line in response:
                    if not line.startswith(b'data: '):continue
                    data=line[6:].strip()
                    if data==b'[DONE]':break
                    payload=json.loads(data);now=time.perf_counter()
                    for choice in payload.get('choices',[]):
                        value=choice.get('delta',{}).get('content') or ''
                        if value:
                            if first is None:first=now
                            text+=value;events.append({'seconds':now-begin,'text':value})
                        finish=choice.get('finish_reason') or finish
                    usage=payload.get('usage') or usage
            end=time.perf_counter();phase_after=snapshot(a.base_url);mtp_after=metrics(a.base_url)
            delta={k:mtp_after[k]-mtp_before.get(k,0) for k in mtp_after}
            pre=phase_after['vllm:request_prefill_time_seconds_sum']-phase_before['vllm:request_prefill_time_seconds_sum'];dec=phase_after['vllm:request_decode_time_seconds_sum']-phase_before['vllm:request_decode_time_seconds_sum']
            count=usage['completion_tokens'];draft=delta.get('vllm:spec_decode_num_draft_tokens_total',0);accepted=delta.get('vllm:spec_decode_num_accepted_tokens_total',0);cycles=delta.get('vllm:spec_decode_num_drafts_total',0)
            row={'kind':'warmup' if repeat<0 else 'request','case':label,'repeat':repeat,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'usage':usage,'finish_reason':finish,'text':text,'events':events,'elapsed_s':end-begin,'ttft_s':first-begin if first else None,'mean_tpot_s':(end-first)/(count-1) if first and count>1 else None,'prefill_tok_s':usage['prompt_tokens']/pre if pre>0 else None,'decode_tok_s':(count-1)/dec if dec>0 else None,'mtp_delta':delta,'acceptance_rate':accepted/draft if draft else None,'mean_acceptance_length':1+accepted/cycles if cycles else None,'valid':bool(text) and phase_after['vllm:request_prefill_time_seconds_count']-phase_before['vllm:request_prefill_time_seconds_count']==1}
            assert row['valid'],row
            file.write(json.dumps(row,ensure_ascii=False)+'\n');file.flush()
            print(json.dumps({k:row[k] for k in ('case','repeat','decode_tok_s','acceptance_rate','mean_acceptance_length')}),flush=True)
