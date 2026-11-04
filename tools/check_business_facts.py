#!/usr/bin/env python3
"""固定中文角色/任务/期限/预算更正事实门禁；不替代完整业务语义评测。"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid


def make_case(index):
    names = ['王敏', '李强', '赵静', '陈洁']
    owners = names[index % 4:] + names[:index % 4]
    tasks = ['完成接口联调', '复测支付回调', '提交采购清单']
    dates = [f'2026-11-{20 + offset:02d}' for offset in range(3)]
    actions = [dict(owner=owner, task=task, due_date=date)
               for owner, task, date in zip(owners, tasks, dates)]
    old_budget, final_budget = 10000 + 100 * index, 25000 + 137 * index
    expected = dict(actions=actions, budget=final_budget, decision_status='待确认')
    transcript = ('本段只记录各人的发言和更正。' + '\n'.join(
        f'{action["owner"]}负责{action["task"]}，截止日期{action["due_date"]}。' for action in actions)
        + f'最初提出预算{old_budget}元，随后更正为{final_budget}元，旧数字作废。'
        + f'{owners[3]}仅主持会议，不承担以上任务。负责人尚未批准方案，最终状态是待确认。')
    prompt = ('根据记录提取全部三条行动项、最终预算和方案状态。逐字保留负责人、任务、日期；'
              '不要将主持人当作执行人。decision_status仅使用“待确认”或“已批准”。\n' + transcript)
    return dict(id=index, prompt=prompt, expected=expected,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


SCHEMA = {'type': 'object', 'properties': {
    'actions': {'type': 'array', 'items': {'type': 'object', 'properties': {
        key: {'type': 'string'} for key in ('owner', 'task', 'due_date')},
        'required': ['owner', 'task', 'due_date'], 'additionalProperties': False}},
    'budget': {'type': 'integer'}, 'decision_status': {'type': 'string'}},
    'required': ['actions', 'budget', 'decision_status'], 'additionalProperties': False}


def normalized(value):
    # 行动项顺序不影响正确性；重复/多余/缺失行动项仍不匹配。
    actions = [{**action, 'task': action['task'].removeprefix('负责').strip()} for action in value['actions']]
    return {**value, 'actions': sorted(actions, key=lambda action: tuple(action[k] for k in ('owner', 'task', 'due_date')))}


def run_case(base, model, case, salt):
    begin = time.perf_counter()
    payload = dict(model=model, messages=[dict(role='user', content=case['prompt'])],
                   temperature=0, seed=0, max_tokens=512, cache_salt=salt,
                   chat_template_kwargs={'enable_thinking': False},
                   response_format={'type': 'json_schema', 'json_schema': {
                       'name': 'meeting_facts', 'strict': True, 'schema': SCHEMA}})
    row = dict(id=case['id'], expected=case['expected'], prompt_sha256=case['prompt_sha256'],
               success=False, correct=False, cache_salt=salt)
    try:
        request = urllib.request.Request(base.rstrip('/') + '/v1/chat/completions',
            json.dumps(payload, ensure_ascii=False).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
        choice = result['choices'][0]
        text = choice['message']['content']
        parsed = json.loads(text)
        row.update(success=choice['finish_reason'] == 'stop', parsed=parsed, text=text,
                   usage=result.get('usage'), finish_reason=choice['finish_reason'])
        row['exact_correct'] = row['success'] and parsed == case['expected']
        row['correct'] = row['success'] and normalized(parsed) == normalized(case['expected'])
    except Exception as error:
        row.update(error_type=type(error).__name__, error=str(error))
    row['elapsed_s'] = time.perf_counter() - begin
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--cases', type=int, default=8)
    p.add_argument('--concurrencies', type=int, nargs='+', default=[1, 3])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if min([a.cases] + a.concurrencies) < 1:
        p.error('cases and concurrencies must be positive')
    cases = [make_case(index) for index in range(a.cases)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    run_id = uuid.uuid4().hex
    with a.out.open('x') as stream:
        stream.write(json.dumps(dict(kind='config', cases=cases, concurrencies=a.concurrencies,
                                     protocol=__doc__, task_normalization='only remove one leading 负责; retain all owner/date/budget/status fields'), ensure_ascii=False) + '\n')
        stream.flush()
        for concurrency in a.concurrencies:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(run_case, a.base_url, a.model, case,
                           f'{run_id}-c{concurrency}-i{case["id"]}') for case in cases]
                for future in futures:
                    row = dict(kind='request', concurrency=concurrency, **future.result())
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    failures += not row['correct']
                    print(json.dumps({key: row[key] for key in ('id', 'concurrency', 'success', 'correct', 'elapsed_s')}), flush=True)
        stream.write(json.dumps(dict(kind='summary', requests=len(cases)*len(a.concurrencies),
                                     failures=failures, all_correct=failures == 0)) + '\n')
    if failures:
        raise SystemExit(f'{failures} fact probes failed; inspect raw outputs')


if __name__ == '__main__':
    main()
