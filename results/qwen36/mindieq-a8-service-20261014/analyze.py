# SPDX-License-Identifier: Apache-2.0
"""完成C3门禁后分析相同质量/计时协议；拒绝不完整结果。"""
import json
from pathlib import Path
from statistics import median

root=Path(__file__).resolve().parent
baseline=root.parent/'mindieq-paired-20261014'
def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
quality=read(root/'gsm64.jsonl');reference=read(baseline/'A1-gsm64.jsonl')
assert quality[-1]['kind']==reference[-1]['kind']=='summary'
for key in ('dataset_sha256','dataset_rows','indices','max_tokens','temperature','enable_thinking','prompt_suffix','protocol'):
    assert quality[0][key]==reference[0][key],key
qr={r['index']:r for r in quality if r['kind']=='answer'};rr={r['index']:r for r in reference if r['kind']=='answer'}
assert len(qr)==len(rr)==64
changes=[{'index':i,'gold':rr[i]['gold'],'A1':rr[i]['predicted'],'C3':qr[i]['predicted'],'A1_finish':rr[i]['finish_reason'],'C3_finish':qr[i]['finish_reason']} for i in rr if rr[i]['predicted']!=qr[i]['predicted'] or rr[i]['correct']!=qr[i]['correct']]
summary={'scope':'C3与同日A1旧结果对照，未进行新量化器交替A/B；单流/prefix off/warmup1/repeats3/phase counter',
    'quality':{'A1':reference[-1],'C3':quality[-1],'answer_changes':changes,'identical_texts':sum(rr[i]['text']==qr[i]['text'] for i in rr),'limitation':'64题固定零样本子集，非完整GSM8K或全面质量证明'},'performance':[]}
for tokens in (8192,32256):
    data={label:[r for r in read(path) if r['kind']=='request'] for label,path in [('A1',baseline/f'A1-p{tokens}-o128.jsonl'),('C3',root/f'p{tokens}-o128.jsonl')]}
    assert all(len(rows)==3 and all(r['valid'] for r in rows) for rows in data.values())
    assert [r['prompt_ids_sha256'] for r in data['A1']]==[r['prompt_ids_sha256'] for r in data['C3']]
    metrics={label:{key:median(r[key] for r in rows) for key in ('prefill_tok_s','decode_tok_s')} for label,rows in data.items()}
    for label,rows in data.items():
        accepted=sum(r['spec_decode']['accepted_tokens'] for r in rows);drafted=sum(r['spec_decode']['drafted_tokens'] for r in rows);drafts=sum(r['spec_decode']['drafts'] for r in rows)
        assert drafted>=accepted>=0 and drafts>0
        metrics[label]['acceptance_rate']=accepted/drafted
        metrics[label]['mean_acceptance_length']=1+accepted/drafts
    summary['performance'].append({'prompt_tokens':tokens,'output_tokens':128,'stats':metrics,'relative_change':{k:metrics['C3'][k]/metrics['A1'][k]-1 for k in ('prefill_tok_s','decode_tok_s')},'identical_texts':sum(a['response']['text']==b['response']['text'] for a,b in zip(data['A1'],data['C3']))})
needle=read(root/'needle114560.jsonl');old=read(root.parent/'pr17640-b4096-20261013'/'needle114560.jsonl')
assert needle[0]['prompt_ids_sha256']==old[0]['prompt_ids_sha256']
assert needle[-1]['kind']=='result' and needle[-1]['retrieved'] and needle[-1]['usage']['prompt_tokens']==114560
sanity=json.loads((root/'post-needle-sanity.json').read_text());assert sanity['status']=='PASS' and sanity['cases']==5
summary['long_request']={'C3':needle[-1],'A1_historical_elapsed_s':old[-1]['elapsed_s'],'relative_elapsed_change':needle[-1]['elapsed_s']/old[-1]['elapsed_s']-1,'limitation':'单次检索canary总时间，不是112K持续Decode吞吐或最长上下文证明','post_sanity':'PASS 5/5'}
(root/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))

chat=read(root/'chat.jsonl');prior=read(root.parent/'pr17640-mtp-20261013'/'mtp2-chat.jsonl')
for key in ('repeats','warmup','max_tokens','temperature','enable_thinking'):
    assert chat[0][key]==prior[0][key],key
business={'scope':'历史同配置业务输入对照，未交替重启；256输出上限，非质量评分','cases':[]}
for case in ('dialogue','meeting','code','analysis'):
    rows={label:[r for r in data if r['kind']=='request' and r['case']==case] for label,data in [('native',prior),('C3',chat)]}
    assert all(len(values)==3 and all(r['valid'] for r in values) for values in rows.values())
    assert [r['prompt_sha256'] for r in rows['native']]==[r['prompt_sha256'] for r in rows['C3']]
    stats={label:{key:median(r[key] for r in values) for key in ('ttft_s','decode_tok_s','elapsed_s','acceptance_rate')} for label,values in rows.items()}
    business['cases'].append({'case':case,'stats':stats,'relative_change':{key:stats['C3'][key]/stats['native'][key]-1 for key in ('ttft_s','decode_tok_s','elapsed_s')},'prompt_tokens':rows['C3'][0]['usage']['prompt_tokens'],'completion_tokens':{label:[r['usage']['completion_tokens'] for r in values] for label,values in rows.items()},'finish_reason':{label:[r['finish_reason'] for r in values] for label,values in rows.items()},'identical_texts':sum(a['text']==b['text'] for a,b in zip(rows['native'],rows['C3']))})
(root/'chat-comparison.json').write_text(json.dumps(business,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(business,ensure_ascii=False,indent=2))
