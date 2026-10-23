# SPDX-License-Identifier: Apache-2.0
"""只分析完整B/A/B对照；从仓库根目录运行。"""
import json
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
labels = ('B0', 'A1', 'B2')
def records(name, final_kind=None):
    rows = [json.loads(line) for line in (root / name).read_text().splitlines()]
    if not rows or rows[0]['kind'] != 'config':
        raise ValueError(f'Missing config: {name}')
    if final_kind and rows[-1]['kind'] != final_kind:
        raise ValueError(f'Incomplete result: {name}')
    return rows

summary = {'scope': '串行B0/A1/B2、相同制品和参数、单流、prefix off、phase counter；三次/条件，不能代表生产P95', 'performance': []}
for tokens in (8192, 32256):
    runs = {}
    for label in labels:
        raw = records(f'{label}-p{tokens}-o128.jsonl')
        rows = [r for r in raw if r['kind'] == 'request']
        if len(rows) != 3 or not all(r['valid'] for r in rows):
            raise ValueError(f'Incomplete phase measurements: {label}/{tokens}')
        if raw[0]['output_tokens'] != 128 or not raw[0]['spec_metrics']:
            raise ValueError('Unexpected benchmark protocol')
        for r in rows:
            spec = r['spec_decode']
            if not (spec['drafts'] > 0 and spec['drafted_tokens'] >= spec['accepted_tokens'] >= 0):
                raise ValueError('Invalid speculative decoding counters')
        runs[label] = rows
    ref = runs['A1']
    if not all([r['prompt_ids_sha256'] for r in runs[label]] == [r['prompt_ids_sha256'] for r in ref] for label in labels):
        raise ValueError('Prompt hashes differ')
    stats = {}
    for label, rows in runs.items():
        stats[label] = {k: median(r[k] for r in rows) for k in ('prefill_tok_s', 'decode_tok_s')}
        stats[label]['acceptance_rate'] = sum(r['spec_decode']['accepted_tokens'] for r in rows) / sum(r['spec_decode']['drafted_tokens'] for r in rows)
        stats[label]['mean_acceptance_length'] = 1 + sum(r['spec_decode']['accepted_tokens'] for r in rows) / sum(r['spec_decode']['drafts'] for r in rows)
        stats[label]['per_request'] = [{k: r[k] for k in ('repeat', 'prefill_tok_s', 'decode_tok_s', 'spec_decode')} for r in rows]
    summary['performance'].append({'prompt_tokens': tokens, 'output_tokens': 128, 'stats': stats,
        'candidate_relative_to_A1': {label: {k: stats[label][k] / stats['A1'][k] - 1 for k in ('prefill_tok_s', 'decode_tok_s')} for label in ('B0','B2')},
        'identical_texts_vs_A1': {label: sum(r['response']['text'] == a['response']['text'] for r,a in zip(runs[label],ref)) for label in ('B0','B2')}})

quality = {label: records(f'{label}-gsm64.jsonl', 'summary') for label in ('B0','A1')}
keys = ('dataset_sha256','dataset_rows','indices','max_tokens','temperature','enable_thinking','prompt_suffix','protocol')
if not all(quality['B0'][0][k] == quality['A1'][0][k] for k in keys):
    raise ValueError('Quality protocols differ')
answers = {label: {r['index']: r for r in rows if r['kind']=='answer'} for label,rows in quality.items()}
if not all(len(rows)==64 for rows in answers.values()):
    raise ValueError('Incomplete quality samples')
changes = []
for index in quality['A1'][0]['indices']:
    a,b = answers['A1'][index],answers['B0'][index]
    if a['gold'] != b['gold']:
        raise ValueError('Gold differs')
    if a['correct'] != b['correct'] or a['predicted'] != b['predicted']:
        changes.append({'index':index,'gold':a['gold'],'A1':a['predicted'],'B0':b['predicted'],'A1_finish':a['finish_reason'],'B0_finish':b['finish_reason']})
summary['quality'] = {'summary': {label: rows[-1] for label,rows in quality.items()},'answer_changes':changes,
    'identical_texts':sum(answers['A1'][i]['text']==answers['B0'][i]['text'] for i in answers['A1']),
    'limitation':'固定64题零样本子集，非完整GSM8K，不能证明全面精度等价'}
(root / 'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
