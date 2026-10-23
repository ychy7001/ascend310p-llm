# SPDX-License-Identifier: Apache-2.0
"""E5完成服务门禁后核对D4/C3同输入性能、全文、MTP及64题质量。"""
import json
from pathlib import Path
from statistics import median
root=Path(__file__).resolve().parent
run=root/'service'
def read(path):return [json.loads(line) for line in path.read_text().splitlines()]
assert (run/'state.txt').read_text().strip()=='DONE'
sanity=json.loads((run/'sanity.json').read_text());assert sanity['status']=='PASS' and sanity['cases']==5
result={'scope':'E5与前轮同配置D4及C3独立启动对照，非交替A/B；prefix off；warmup1/repeat3','performance':[],'business':[]}
for tokens,previous in [(8192,'mindieq-input-dispatch-20261014'),(32256,'mindieq-a8-service-20261014')]:
 data={label:[r for r in read(path) if r['kind']=='request'] for label,path in [('previous',root.parent/previous/f'p{tokens}-o128.jsonl'),('E5',run/f'p{tokens}-o128.jsonl')]}
 assert all(len(rows)==3 and all(r['valid'] for r in rows) for rows in data.values())
 assert [r['prompt_ids_sha256'] for r in data['previous']]==[r['prompt_ids_sha256'] for r in data['E5']]
 stats={label:{key:median(r[key] for r in rows) for key in ('prefill_tok_s','decode_tok_s')} for label,rows in data.items()}
 for label,rows in data.items():
  accepted=sum(r['spec_decode']['accepted_tokens'] for r in rows);drafted=sum(r['spec_decode']['drafted_tokens'] for r in rows)
  stats[label]['acceptance_rate']=accepted/drafted
 result['performance'].append({'tokens':tokens,'reference':previous,'stats':stats,'relative_change':{key:stats['E5'][key]/stats['previous'][key]-1 for key in ('prefill_tok_s','decode_tok_s')},'identical_texts':sum(a['response']['text']==b['response']['text'] for a,b in zip(data['previous'],data['E5']))})
chats={label:read(path) for label,path in [('D4',root.parent/'mindieq-input-dispatch-20261014'/'chat.jsonl'),('E5',run/'chat.jsonl')]}
for key in ('warmup','repeats','temperature','enable_thinking','max_tokens'):assert chats['D4'][0][key]==chats['E5'][0][key]
for case in ('dialogue','meeting','code','analysis'):
 rows={label:[r for r in data if r['kind']=='request' and r['case']==case] for label,data in chats.items()}
 assert all(len(data)==3 and all(r['valid'] for r in data) for data in rows.values())
 assert [r['prompt_sha256'] for r in rows['D4']]==[r['prompt_sha256'] for r in rows['E5']]
 stats={label:{key:median(r[key] for r in data) for key in ('ttft_s','decode_tok_s','acceptance_rate')} for label,data in rows.items()}
 result['business'].append({'case':case,'stats':stats,'identical_texts':sum(a['text']==b['text'] for a,b in zip(rows['D4'],rows['E5'])),'finish_reason':{label:[r['finish_reason'] for r in data] for label,data in rows.items()},'completion_tokens':{label:[r['usage']['completion_tokens'] for r in data] for label,data in rows.items()}})
quality={label:read(path) for label,path in [('D4',root.parent/'mindieq-input-dispatch-20261014'/'gsm64.jsonl'),('E5',run/'gsm64.jsonl')]}
for key in ('dataset_sha256','dataset_rows','indices','max_tokens','temperature','enable_thinking','prompt_suffix','protocol'):assert quality['D4'][0][key]==quality['E5'][0][key]
assert all(data[-1]['kind']=='summary' for data in quality.values())
answers={label:{r['index']:r for r in data if r['kind']=='answer'} for label,data in quality.items()}
assert all(len(data)==64 for data in answers.values())
result['quality']={'summaries':{label:data[-1] for label,data in quality.items()},'answer_changes':[i for i in answers['D4'] if answers['D4'][i]['predicted']!=answers['E5'][i]['predicted'] or answers['D4'][i]['correct']!=answers['E5'][i]['correct']],'identical_texts':sum(answers['D4'][i]['text']==answers['E5'][i]['text'] for i in answers['D4']),'limitation':'固定64题子集，非全面精度证明'}
(root/'service-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
