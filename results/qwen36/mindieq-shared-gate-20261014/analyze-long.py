# SPDX-License-Identifier: Apache-2.0
"""只接受完整112K三次持续生成及请求后状态；三次分位不作为产品尾延迟。"""
import json
from pathlib import Path
from statistics import median
root=Path(__file__).resolve().parent
run=root/'service'
def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
def quantile(values,q):
    values=sorted(values);position=(len(values)-1)*q
    lo=int(position);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(position-lo)
rows=read(run/'p114560-o128.jsonl');config=rows[0]
assert config['prompt_tokens']==114560 and config['output_tokens']==128
assert config['repeats']==3 and config['unique_prompts'] and config['spec_metrics']
assert len([r for r in rows if r['kind']=='warmup'])==1
requests=[r for r in rows if r['kind']=='request']
assert len(requests)==3 and all(r['valid'] for r in requests),'incomplete or invalid phase requests'
assert len({r['prompt_ids_sha256'] for r in requests})==3
assert all(r['response']['usage']['prompt_tokens']==114560 and r['response']['usage']['completion_tokens']==128 for r in requests)
assert (run/'long-state.txt').read_text().strip()=='DONE'
sanity=json.loads((run/'post-phases-sanity.json').read_text())
assert sanity['status']=='PASS' and sanity['cases']==5
result={'scope':'单流112K上限附近/合成输入/128持续输出，warmup1+三次；非全面长文质量、最大可支持上下文或生产P95','prompt_tokens':114560,'output_tokens':128,'total_tokens':114688,'prefill_tok_s_median':median(r['prefill_tok_s'] for r in requests),'decode_tok_s_median':median(r['decode_tok_s'] for r in requests),'samples':[{'repeat':r['repeat'],'prefill_tok_s':r['prefill_tok_s'],'decode_tok_s':r['decode_tok_s'],'ttft_s':r['response']['ttft_s'],'mean_tpot_s':r['response']['mean_tpot_s'],'elapsed_s':r['response']['elapsed_s'],'spec_decode':r['spec_decode']} for r in requests],'post_sanity':'PASS 5/5'}
for name,field in [('ttft_s','ttft_s'),('request_mean_tpot_s','mean_tpot_s')]:
    values=[r['response'][field] for r in requests]
    result[name]={'p50':quantile(values,.5),'p95_interpolated':quantile(values,.95),'n':3,'limitation':'三次请求描述统计，TPOT为请求均值而非逐token延迟；不代表生产P95'}
accepted=sum(r['spec_decode']['accepted_tokens'] for r in requests)
drafted=sum(r['spec_decode']['drafted_tokens'] for r in requests)
result['acceptance_rate']=accepted/drafted
(root/'long-phases-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
