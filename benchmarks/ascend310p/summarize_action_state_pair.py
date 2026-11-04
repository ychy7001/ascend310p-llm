#!/usr/bin/env python3
"""按相同请求匹配字段回归，区分共同失败、候选新增失败和改善。"""
import argparse
import json
from pathlib import Path


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def index(rows, kind, fields):
    selected = [r for r in rows if r['kind'] == kind]
    result = {tuple(r[k] for k in fields): r for r in selected}
    if len(result) != len(selected): raise ValueError('Duplicate result keys')
    return result


def differences(row):
    parsed = row.get('parsed')
    if not isinstance(parsed, dict) or not isinstance(parsed.get('actions'), list):
        return {'format_or_execution': row.get('error', 'No parsed actions')}
    actual = parsed['actions']
    if any(not isinstance(x,dict) or 'task' not in x for x in actual):
        return {'format_or_execution': 'Invalid action schema'}
    mapping = {x['task']: x for x in actual}
    result = {}
    if row.get('error'):
        result['format_or_execution'] = row['error']
    choice = row.get('response', {}).get('choices', [{}])[0]
    if choice.get('finish_reason') != 'stop':
        result['finish_reason'] = choice.get('finish_reason')
    if len(actual) != 3 or len(mapping) != 3:
        result['action_count_or_duplicates'] = actual
    expected = {x['task']: x for x in row['expected']['actions']}
    for task in sorted(set(mapping) | set(expected)):
        if task not in expected: result['extra_task:'+str(task)] = mapping[task]
        elif task not in mapping: result['missing_task:'+task] = expected[task]
        else:
            for key, value in expected[task].items():
                if mapping[task].get(key) != value:
                    result[task+':'+key] = dict(expected=value, actual=mapping[task].get(key))
    return result


def summarize(directory):
    stages = {stage: read(directory/(stage+'-fields.jsonl')) for stage in ('A-production','B-pair921')}
    configs = [next(r for r in rows if r['kind']=='config') for rows in stages.values()]
    if configs[0]['fixtures_sha256'] != configs[1]['fixtures_sha256']: raise ValueError('Different fixtures')
    keys = ('id','concurrency_limit')
    a, b = [index(rows,'request',keys) for rows in stages.values()]
    fixtures = json.loads((directory/'fixtures.json').read_text())
    expected_keys = {(case['id'], c) for case in fixtures['cases'] for c in (1,3)}
    if set(a) != expected_keys or set(b) != expected_keys: raise ValueError('Incomplete paired requests')
    tokens = {stage:[r for r in rows if r['kind']=='tokenization'] for stage,rows in stages.items()}
    if any(len(rows)!=len(fixtures['cases']) or not all(r['matches'] for r in rows) for rows in tokens.values()):
        raise ValueError('Incomplete/mismatched runtime input tokenization')
    paired = []
    for key in sorted(a):
        old, new = a[key], b[key]
        if old['payload_sha256'] != new['payload_sha256'] or old['expected'] != new['expected']:
            raise ValueError('Request or independent expected value differs: '+repr(key))
        category = {(True,True):'both_pass',(False,False):'both_fail',
                    (True,False):'candidate_new_failure',(False,True):'candidate_fixed_failure'}[(old['correct'],new['correct'])]
        paired.append(dict(id=key[0],concurrency_limit=key[1],category=category,
            production_differences=differences(old),candidate_differences=differences(new),
            production_correct=old['correct'],candidate_correct=new['correct']))
    free = [index(read(directory/(stage+'-freeform.jsonl')),'freeform',('id','concurrency_limit','repeat'))
            for stage in stages]
    if set(free[0]) != set(free[1]) or len(free[0]) != 9: raise ValueError('Incomplete free-form replay')
    if any(free[0][key]['request_sha256'] != free[1][key]['request_sha256'] for key in free[0]):
        raise ValueError('Free-form replay payload differs')
    summary = dict(status='paired_diagnostics_complete_requires_manual_adjudication',
        comparisons=len(paired), categories={key:sum(x['category']==key for x in paired) for key in
            ('both_pass','both_fail','candidate_new_failure','candidate_fixed_failure')},
        stage_correct={stage:sum(r['correct'] for r in rows if r['kind']=='request') for stage,rows in stages.items()},
        long_correct={stage:sum(r['correct'] for r in rows if r['kind']=='request' and r['id'].startswith('long-')) for stage,rows in stages.items()},
        usage_token_count_mismatches={stage:sum(r.get('prompt_tokens_match_tokenize') is not True for r in rows if r['kind']=='request') for stage,rows in stages.items()},
        exact_input_tokenization=True,freeform_exactly_paired_requests=9,paired=paired,
        scope='Whole original production versus fixed9.2.1 + paired H/O, Torch WY unchanged; no causal attribution to CANN alone, no performance/logits equivalence')
    (directory/'paired-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);args=p.parse_args()
    result=summarize(args.directory)
    print(json.dumps({k:v for k,v in result.items() if k!='paired'},ensure_ascii=False,indent=2))
