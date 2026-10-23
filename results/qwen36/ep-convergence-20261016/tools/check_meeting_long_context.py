# SPDX-License-Identifier: Apache-2.0
"""合成会议决议更新 canary：不是完整会议质量评测。"""
import argparse
import gzip
import json
from pathlib import Path
import re
import time
import urllib.request

from bench_http_stream import post
from bench_server_phases import METRICS, SPEC_METRICS, snapshot, prompt_hash

EXPECTED = {'budget_yuan': 180000, 'owner': '周启', 'deadline': '2026-11-12', 'status': '延期上线'}
RECORDS = [
    '正式决议记录A：云桥项目原计划预算120000元，由李雯负责，2026-11-05上线。',
    '正式决议记录B：云桥项目预算更新为180000元，负责人改为周启；日期和状态暂时沿用原计划。B取代A中的预算和负责人。',
    '最终决议记录C：云桥项目延期上线，最新截止日期为2026-11-12；预算180000元、负责人周启保持不变。C中的日期和状态取代先前计划。',
]


def tokenize(base, model, text):
    with post(base, '/tokenize', {'model': model, 'prompt': text, 'add_special_tokens': False}) as response:
        return json.load(response)['tokens']


def make_prompt(base, model, count):
    # 与该服务 /tokenize(messages, enable_thinking=False) 的模板核对过。
    prefix = tokenize(base, model, '<|im_start|>user\n下面是合成会议纪要。普通讨论不是决议；有更新时以较新的正式决议为准。\n')
    suffix = tokenize(base, model, '\n问题：提取云桥项目最终预算、负责人、截止日期和状态。只返回JSON，字段必须是budget_yuan（整数）、owner、deadline（YYYY-MM-DD）、status，不要解释。\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    records = [tokenize(base, model, '\n' + value + '\n') for value in RECORDS]
    filler = tokenize(base, model, ''.join(
        f'普通讨论第{i}项：会议系统采集音频、核对设备、汇总反馈；本段不涉及云桥项目，也不改变任何正式决议。\n'
        for i in range(128)))
    budget = count - len(prefix) - len(suffix) - sum(map(len, records))
    if budget < 100:
        raise ValueError('prompt length too small')
    cuts = [int(budget * fraction) for fraction in (.1, .5, .9)]
    tokens = list(prefix)
    positions = []
    previous = 0
    for cut, record in zip(cuts, records):
        size = cut - previous
        tokens.extend((filler * ((size + len(filler) - 1) // len(filler)))[:size])
        positions.append(len(tokens))
        tokens.extend(record)
        previous = cut
    size = budget - previous
    tokens.extend((filler * ((size + len(filler) - 1) // len(filler)))[:size])
    tokens.extend(suffix)
    assert len(tokens) == count
    return tokens, positions


def score(text):
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {'passed': False, 'parsed': None, 'error': 'invalid JSON'}
    return {'passed': parsed == EXPECTED, 'parsed': parsed, 'expected': EXPECTED}


def complete(base, model, tokens):
    body = {'model': model, 'prompt': tokens, 'temperature': 0, 'max_tokens': 128,
            'stream': False, 'stop': ['\n\n']}
    req = urllib.request.Request(base.rstrip('/') + '/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'Connection': 'close'})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as response:
        result = json.load(response)
    return result, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mindieq-url', required=True)
    p.add_argument('--original-url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--prompt-tokens', type=int, default=32256)
    p.add_argument('--prompt-file', type=Path, help='复用此工具保存的 gzip token-ID 输入')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.prompt_file:
        with gzip.open(a.prompt_file, 'rt', encoding='utf-8') as prompt_file:
            saved = json.load(prompt_file)
        tokens = saved['prompt_ids']
        if len(tokens) != saved['prompt_tokens']:
            raise ValueError('stored prompt token count mismatch')
        positions = saved.get('record_positions')
    else:
        tokens, positions = make_prompt(a.mindieq_url, a.model, a.prompt_tokens)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(str(a.out) + '.prompt.json.gz', 'xt', encoding='utf-8') as f:
        json.dump({'prompt_tokens': len(tokens), 'prompt_ids': tokens, 'record_positions': positions}, f)
    with a.out.open('x', encoding='utf-8') as f:
        f.write(json.dumps({'kind': 'config', 'prompt_tokens': len(tokens), 'prompt_ids_sha256': prompt_hash(tokens),
                            'record_positions': positions, 'records': RECORDS, 'expected': EXPECTED,
                            'max_output_tokens': 128, 'protocol': 'synthetic meeting-update canary; warmed endpoints; prefix off'}, ensure_ascii=False) + '\n')
        f.flush()
        for name, base in [('mindieq', a.mindieq_url), ('original', a.original_url)]:
            before = snapshot(base, include_spec=True)
            result, elapsed = complete(base, a.model, tokens)
            after = snapshot(base, include_spec=True)
            delta = {key: after[key] - before[key] for key in before}
            if delta[METRICS[0]] != 1 or result['usage']['prompt_tokens'] != len(tokens):
                raise RuntimeError('phase/usage mismatch; endpoint must be exclusive')
            text = result['choices'][0].get('text', '')
            row = {'kind': 'result', 'service': name, 'elapsed_s': elapsed, 'text': text,
                   'usage': result.get('usage'), 'metric_delta': delta, 'score': score(text),
                   'finish_reason': result['choices'][0].get('finish_reason'),
                   'mtp_acceptance': delta[SPEC_METRICS[2]] / delta[SPEC_METRICS[1]] if delta[SPEC_METRICS[1]] else None}
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            print(json.dumps({'service': name, 'score': row['score'], 'mtp_acceptance': row['mtp_acceptance'], 'usage': row['usage']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
