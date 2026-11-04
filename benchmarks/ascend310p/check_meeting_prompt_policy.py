#!/usr/bin/env python3
"""固定转录的小范围提示词敏感性诊断；保留全文，由人工按冻结准则审阅。"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

POLICY = ('\n额外规则：转录没有明确说“已完成”“已执行”“已回复”等完成证据时，'
          '不得把行动写成已经完成，执行状态写“未明确”；“当天”只保留为原文时间，'
          '不据此推断已执行。不补充、细化或推断截止时间；未提及写“未明确”。'
          '负责人可以是部门或角色，原文给出部门或角色就照录，不改为“未明确”，'
          '也不擅自增加“负责人”等身份。按原文分别保留每项行动的负责人、任务和时间对应关系。')
REMOTE = """import sys,json,urllib.request
x=json.load(sys.stdin)
r=urllib.request.Request(x['url'],json.dumps(x['payload']).encode(),{'Content-Type':'application/json'})
with urllib.request.urlopen(r,timeout=900) as s: print(s.read().decode())
"""

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--ssh-host')
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    source = [json.loads(line) for line in a.fixtures.read_text().splitlines()]
    a.out.mkdir(parents=True, exist_ok=False)
    cases = []
    for row in source:
        if row['id'] not in (4, 5, 18):
            continue
        for mode in ('original_system', 'explicit_policy'):
            req = copy.deepcopy(row)
            case_id = req.pop('id')
            req['model'] = a.model
            req['seed'] = 0
            req['cache_salt'] = f'meeting-policy-v1-{case_id}-{mode}'
            if mode == 'explicit_policy':
                assert req['messages'][0]['role'] == 'system'
                req['messages'][0]['content'] += POLICY
            assert req['messages'][1:] == row['messages'][1:]
            cases.append({'id': case_id, 'mode': mode, 'request': req})
    assert len(cases) == 6
    frozen = {'scope': '3 fixtures x 2 prompts, C1 once; prompt sensitivity only, not causal model/quant attribution or general acceptance',
              'source_sha256': hashlib.sha256(a.fixtures.read_bytes()).hexdigest(),
              'policy': POLICY, 'cases': cases,
              'criteria': {'4': '周明当天回复/刘洋周三修复/何丽补用例期限未明确，不补完成状态',
                           '5': 'IT管理员当天停用/所有部门下周一前核对/合规负责人确认季度复核期限未明确，不补完成状态',
                           '18': '销售每周一预测/产品周二需求/研发周三排期，保留三项关系'}}
    (a.out/'frozen-plan.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2))
    rows = []
    with (a.out/'responses.jsonl').open('x') as stream:
        for case in cases:
            row = copy.deepcopy(case)
            row.update(backend_url=a.base_url, ssh_host=a.ssh_host, ok=False)
            start = time.monotonic()
            try:
                url = a.base_url.rstrip('/')+'/v1/chat/completions'
                if a.ssh_host:
                    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', a.ssh_host, 'python3', '-c',
                                             __import__('shlex').quote(REMOTE)],
                                            input=json.dumps({'url': url, 'payload': case['request']}),
                                            text=True, capture_output=True, timeout=930)
                    if result.returncode:
                        raise RuntimeError(result.stderr)
                    response = json.loads(result.stdout)
                else:
                    req = urllib.request.Request(url, json.dumps(case['request']).encode(),
                                                 {'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=900) as res:
                        response = json.load(res)
                row['response'] = response
                row['ok'] = response['choices'][0]['finish_reason'] == 'stop'
            except Exception as exc:
                row['error'] = repr(exc)
            row['elapsed_s'] = time.monotonic()-start
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
            stream.flush()
            print(json.dumps({k:row[k] for k in ('id', 'mode', 'ok', 'elapsed_s')}), flush=True)
    (a.out/'status.json').write_text(json.dumps({'state': 'requires_manual_adjudication',
         'requests': len(rows), 'format_completed': sum(r['ok'] for r in rows)}, indent=2))

if __name__ == '__main__':
    main()
