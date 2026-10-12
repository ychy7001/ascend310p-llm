# SPDX-License-Identifier: Apache-2.0
"""Measure one cold prefix request followed by repeated prefix hits."""
import argparse
import json
import time
import urllib.request
from pathlib import Path


PHASE_METRICS = (
    'vllm:request_prefill_time_seconds_count',
    'vllm:request_prefill_time_seconds_sum',
    'vllm:request_decode_time_seconds_sum',
)


def post(base, path, body):
    request = urllib.request.Request(
        base.rstrip('/') + path,
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.load(response)


def metrics(base):
    with urllib.request.urlopen(base.rstrip('/') + '/metrics', timeout=30) as response:
        lines = response.read().decode().splitlines()
    values = {name: 0.0 for name in PHASE_METRICS}
    prefix = []
    for line in lines:
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split()[0]
        if name in values:
            values[name] += float(line.split()[-1])
        if 'prefix_cache' in name:
            prefix.append(line)
    return values, prefix


def request(base, model, tokens, output_tokens):
    before, prefix_before = metrics(base)
    start = time.perf_counter()
    result = post(base, '/v1/completions', {
        'model': model, 'prompt': tokens, 'temperature': 0,
        'max_tokens': output_tokens,
    })
    elapsed = time.perf_counter() - start
    after, prefix_after = metrics(base)
    delta = {name: after[name] - before[name] for name in PHASE_METRICS}
    usage = result.get('usage') or {}
    return {
        'wall_seconds': elapsed,
        'prefill_seconds': delta[PHASE_METRICS[1]],
        'decode_seconds': delta[PHASE_METRICS[2]],
        'prefill_tok_s': len(tokens) / delta[PHASE_METRICS[1]],
        'decode_tok_s': ((usage.get('completion_tokens', 0) - 1) /
                         delta[PHASE_METRICS[2]]),
        'usage': usage,
        'finish_reason': result['choices'][0].get('finish_reason'),
        'prefix_metrics_before': prefix_before,
        'prefix_metrics_after': prefix_after,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--prompt-tokens', type=int, default=8192)
    parser.add_argument('--output-tokens', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    prompt = ('Explain how quantization affects memory bandwidth and numerical '
              'precision. ' * (args.prompt_tokens + 1))
    tokens = post(args.base_url, '/tokenize', {
        'model': args.model, 'prompt': prompt, 'add_special_tokens': False,
    })['tokens'][:args.prompt_tokens]
    if len(tokens) != args.prompt_tokens:
        raise RuntimeError('Insufficient prompt tokens')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [request(args.base_url, args.model, tokens, args.output_tokens)
            for _ in range(args.repeats + 1)]
    result = {
        'protocol': 'first request is cold; following requests reuse identical token IDs',
        'model': args.model, 'prompt_tokens': len(tokens),
        'output_tokens': args.output_tokens, 'rows': rows,
    }
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    for index, row in enumerate(rows):
        print(json.dumps({'request': index, **{
            key: row[key] for key in ('wall_seconds', 'prefill_tok_s', 'decode_tok_s')
        }}), flush=True)


if __name__ == '__main__':
    main()
