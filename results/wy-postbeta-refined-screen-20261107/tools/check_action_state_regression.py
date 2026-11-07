#!/usr/bin/env python3
"""冻结负责人/期限/执行状态与长上下文用例；独立记录基线错误及候选回归。"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
from pathlib import Path
import time
import urllib.request

MODEL = 'Qwen3.6-35B-A3B-w8a8'
FIELDS = ('task', 'owner', 'due_date', 'execution_status')
STATUSES = ('未明确', '待执行', '已完成', '已取消')
INSTRUCTION = '''只根据以下记录提取全部三条行动项，输出符合schema的JSON。
逐字保留任务名称、负责人和期限；没有明确期限写“未明确”。
execution_status只允许：未明确、待执行、已完成、已取消。
只有明确确认完成才写已完成；计划/要求执行写待执行；明确取消写已取消。
仅列出负责人和期限，或提问“是否完成”，均不能推断已经完成，状态写未明确。
以最后明确更正为准，不把主持人当负责人，不输出背景记录。
'''
SCHEMA = {'type': 'object', 'properties': {'actions': {'type': 'array', 'minItems': 3, 'maxItems': 3,
    'items': {'type': 'object', 'properties': {key: ({'type': 'string', 'enum': list(STATUSES)}
        if key == 'execution_status' else {'type': 'string'}) for key in FIELDS},
        'required': list(FIELDS), 'additionalProperties': False}}},
    'required': ['actions'], 'additionalProperties': False}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def post(base, path, body):
    req = urllib.request.Request(base.rstrip('/')+path, json.dumps(body, ensure_ascii=False).encode(),
                                 {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)


def token_info(base, prompt, model=MODEL):
    d = post(base, '/tokenize', dict(model=model, messages=[dict(role='user', content=prompt)],
        add_generation_prompt=True, chat_template_kwargs={'enable_thinking': False}))
    assert len(d['tokens']) == d['count']
    return dict(count=d['count'], token_ids_sha256=digest(d['tokens']))


def make_case(index):
    names = ['周明', '刘洋', '何丽', '王敏']
    owners = names[index % 4:] + names[:index % 4]
    tasks = ['回复客户', '修复报表时区', '补充跨时区用例']
    dates = ['2026-11-11', '2026-11-12', '2026-11-13']
    status_sets = [(0, 0, 0), (1, 1, 1), (2, 1, 3), (1, 2, 0),
                   (0, 0, 0), (1, 1, 1), (1, 0, 2), (3, 2, 1)]
    statuses = [STATUSES[x] for x in status_sets[index % 8]]
    actions = [dict(task=t, owner=o, due_date=d, execution_status=s)
               for t, o, d, s in zip(tasks, owners, dates, statuses)]
    segments = []
    for action in actions:
        task, owner, date, status = (action[key] for key in FIELDS)
        base = f'任务名称：{task}；负责人：{owner}；期限：{date}。'
        if index % 8 == 4:
            line = base + '会上有人问“是否已完成”，没有人确认执行进度。'
        elif index % 8 == 5:
            line = base + '旧记录曾标为已完成；本次复核明确撤回，当前状态更正为待执行。'
        else:
            line = base + {'未明确': '未说明执行进度。', '待执行': '计划执行，目前待执行。',
                           '已完成': '会上明确确认已完成。', '已取消': '会议明确取消此任务。'}[status]
        if index % 8 == 3:
            line = f'旧负责人是{owners[3]}，此旧分工已撤销。' + line
        segments.append(line)
    return dict(id=f'short-{index}', expected={'actions': actions}, segments=segments,
                header=f'记录编号{index}。{owners[3]}仅主持会议，不执行上述行动项。',
                scope='Explicit structured extraction contract; not the unchanged free-form production prompt')


def prepare(base, output):
    cases = []
    for i in range(8):
        case = make_case(i)
        case['prompt'] = INSTRUCTION+case['header']+'\n'+'\n'.join(case.pop('segments'))
        case['token_info'] = token_info(base, case['prompt'])
        cases.append(case)
    for i, target in enumerate((4095, 4096, 4097, 16384, 32768)):
        case = make_case(2 if i % 2 == 0 else 7)
        case['id'] = f'long-{target}'
        segments = case.pop('segments')
        def prompt(n):
            left = n//2
            return (INSTRUCTION+case['header']+'\n[开头行动项]\n'+segments[0]+
                '\n[以下仅为无关背景，不含行动项]\n'+' background'*left+
                '\n[中间行动项]\n'+segments[1]+'\n[以下仅为无关背景，不含行动项]\n'+
                ' background'*(n-left)+'\n[结尾行动项]\n'+segments[2]+'\n请提取全部三条行动项。')
        n = max(0, target-token_info(base, prompt(0))['count'])
        for _ in range(5):
            text = prompt(n); info = token_info(base, text)
            if info['count'] == target: break
            n += target-info['count']
        else: raise RuntimeError('Could not construct exact chat token length')
        case.update(prompt=text, token_info=info, target_chat_tokens=target,
                    scope='Synthetic long retrieval; relevant actions at beginning/middle/end; no claim of real long-meeting quality')
        cases.append(case)
    for case in cases:
        case['prompt_sha256'] = hashlib.sha256(case['prompt'].encode()).hexdigest()
        case['case_sha256'] = digest(case)
    value = dict(protocol='action-state-v1', model=MODEL, schema=SCHEMA, cases=cases,
                 cache='Prefix caching remains enabled; each request has a distinct frozen salt',
                 normalization='Only reorder actions; all four fields, keys, action count and duplicates checked exactly',
                 tokenization='Chat /tokenize with enable_thinking=False; freeze token IDs hash and verify at each runtime')
    with output.open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')


def canonical(value):
    if not isinstance(value, dict) or set(value) != {'actions'}:
        raise ValueError('Unexpected top-level schema')
    actions = value['actions']
    if not isinstance(actions, list) or len(actions) != 3:
        raise ValueError('Expected exactly3 actions')
    for action in actions:
        if not isinstance(action, dict) or set(action) != set(FIELDS) or any(not isinstance(action[k], str) for k in FIELDS):
            raise ValueError('Unexpected action schema')
        if action['execution_status'] not in STATUSES: raise ValueError('Unexpected status')
    return sorted(tuple(action[key] for key in FIELDS) for action in actions)


def run_case(base, case, concurrency, protocol, model=MODEL):
    payload = dict(model=model, messages=[dict(role='user', content=case['prompt'])], temperature=0, seed=0,
        max_tokens=512, cache_salt=f'{protocol}-c{concurrency}-{case["id"]}',
        chat_template_kwargs={'enable_thinking': False}, response_format={'type': 'json_schema',
        'json_schema': {'name': 'action_state', 'strict': True, 'schema': SCHEMA}})
    row = dict(kind='request', id=case['id'], concurrency_limit=concurrency,
        prompt_sha256=case['prompt_sha256'], payload_sha256=digest(payload), expected=case['expected'],
        correct=False, numeric_scope='No logits or kernel equivalence claim')
    start = time.monotonic()
    try:
        response = post(base, '/v1/chat/completions', payload)
        row['response'] = response
        choice = response['choices'][0]
        parsed = json.loads(choice['message']['content'])
        row.update(parsed=parsed, correct=choice['finish_reason']=='stop' and canonical(parsed)==canonical(case['expected']))
        row['prompt_tokens_match_tokenize'] = response.get('usage', {}).get('prompt_tokens') == case['token_info']['count']
    except Exception as error:
        row.update(error_type=type(error).__name__, error=str(error))
    row['elapsed_s_diagnostic_only'] = time.monotonic()-start
    return row


def run(base, fixtures, output, model=MODEL):
    data = json.loads(fixtures.read_text()); cases = data['cases']
    assert data['schema'] == SCHEMA and data['model'] == MODEL
    with output.open('x') as f:
        f.write(json.dumps(dict(kind='config', fixtures_sha256=hashlib.sha256(fixtures.read_bytes()).hexdigest(),
            schema=data['schema'], served_model=model, fixture_source_model=data['model'], scope='Paired field correctness; not throughput measurement'), ensure_ascii=False)+'\n'); f.flush()
        for case in cases:
            info = token_info(base, case['prompt'], model)
            row = dict(kind='tokenization', id=case['id'], matches=info==case['token_info'], actual=info, expected=case['token_info'])
            f.write(json.dumps(row)+'\n'); f.flush()
            if not row['matches']: raise RuntimeError('Runtime chat tokenization differs: '+case['id'])
        rows = []
        for c in (1, 3):
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                futures = [pool.submit(run_case, base, case, c, data['protocol'], model) for case in cases]
                for future in futures:
                    row = future.result(); rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False)+'\n'); f.flush()
                    print(json.dumps({key:row[key] for key in ('id','concurrency_limit','correct')}, ensure_ascii=False), flush=True)
        summary = dict(kind='summary', requests=len(rows), correct=sum(r['correct'] for r in rows),
            usage_token_count_mismatches=sum(r.get('prompt_tokens_match_tokenize') is not True for r in rows),
            scope='Semantic diagnostics complete; errors preserved for paired analysis, do not infer acceptance from exit0')
        f.write(json.dumps(summary)+'\n'); print(json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--prepare', type=Path)
    p.add_argument('--model', default=MODEL, help='Evaluation endpoint alias; frozen source fixtures/expectations unchanged')
    p.add_argument('--fixtures', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    if a.prepare:
        if a.fixtures or a.out: p.error('prepare is separate from evaluation')
        if a.model != MODEL: p.error('prepare uses original source model; model override is evaluation only')
        prepare(a.base_url, a.prepare)
    else:
        if not a.fixtures or not a.out: p.error('fixtures and out required')
        run(a.base_url, a.fixtures, a.out, a.model)


if __name__ == '__main__': main()
