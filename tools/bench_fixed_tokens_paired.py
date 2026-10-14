# SPDX-License-Identifier: Apache-2.0
"""Paired long-context benchmark using a pre-tokenized prompt file."""

import argparse
import gzip
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

METRICS = (
    'vllm:request_prefill_time_seconds_count',
    'vllm:request_prefill_time_seconds_sum',
    'vllm:request_decode_time_seconds_sum',
)


def snapshot(base):
    raw = subprocess.run(
        ['curl', '--fail', '--silent', '--show-error', '--max-time', '30',
         '-H', 'Connection: close', base.rstrip('/') + '/metrics'],
        check=True, capture_output=True, text=True).stdout
    values = {name: 0.0 for name in METRICS}
    for line in raw.splitlines():
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split()[0]
        if name in values:
            values[name] += float(line.split()[-1])
    return values


def complete(base, model, tokens, output_tokens):
    body = json.dumps({'model': model, 'prompt': tokens,
                       'max_tokens': output_tokens, 'temperature': 0,
                       'ignore_eos': True, 'stream': False}).encode()
    request = urllib.request.Request(
        base.rstrip('/') + '/v1/completions', data=body,
        headers={'Content-Type': 'application/json', 'Connection': 'close'})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=3600) as response:
        result = json.load(response)
    return result, time.perf_counter() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mindieq-url', required=True)
    p.add_argument('--original-url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--prompt-file', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--output-tokens', type=int, default=32)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--stream', action='store_true')
    p.add_argument('--prompt-limit', type=int)
    a = p.parse_args()
    opener = gzip.open if a.prompt_file.suffix == '.gz' else open
    with opener(a.prompt_file, 'rt', encoding='utf-8') as prompt_stream:
        config = json.loads(prompt_stream.readline())
    tokens = config['prompt_ids']
    if len(tokens) != int(config['prompt_tokens']):
        raise RuntimeError('prompt file token count mismatch')
    if a.prompt_limit is not None:
        if not 1 <= a.prompt_limit <= len(tokens):
            p.error('prompt-limit must be within the stored prompt length')
        tokens = tokens[:a.prompt_limit]
    prompt_hash = hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as out:
        out.write(json.dumps({'kind': 'config', 'prompt_tokens': len(tokens),
                              'output_tokens': a.output_tokens, 'repeats': a.repeats,
                              'prompt_ids_sha256': prompt_hash,
                              'protocol': 'fixed token IDs; alternating endpoints',
                              'stream': a.stream,
                              'phase_metrics': 'Prometheus deltas; streaming TTFT is first nonempty text event'}) + '\n')
        out.flush()
        for repeat in range(a.repeats):
            endpoints = [('mindieq', a.mindieq_url), ('original', a.original_url)]
            if repeat % 2:
                endpoints.reverse()
            for order, (name, base) in enumerate(endpoints):
                before = snapshot(base)
                if a.stream:
                    from bench_http_stream import request
                    streamed = request(base, a.model, tokens, a.output_tokens)
                    result = {'usage': streamed['usage'], 'choices': [{'text': streamed['text']}]}
                    elapsed = streamed['elapsed_s']
                else:
                    result, elapsed = complete(base, a.model, tokens, a.output_tokens)
                after = snapshot(base)
                delta = {key: after[key] - before[key] for key in METRICS}
                if delta[METRICS[0]] != 1 or delta[METRICS[1]] <= 0 or delta[METRICS[2]] <= 0:
                    raise RuntimeError(f'invalid phase metric delta: {delta}')
                usage = result.get('usage') or {}
                row = {'kind': 'request', 'service': name, 'repeat': repeat,
                       'order': order, 'prompt_tokens': len(tokens),
                       'output_tokens': a.output_tokens,
                       'prompt_ids_sha256': prompt_hash,
                       'prefill_tok_s': len(tokens) / delta[METRICS[1]],
                       'decode_tok_s': (a.output_tokens - 1) / delta[METRICS[2]],
                       'elapsed_s': elapsed, 'usage': usage,
                       'text_sha256': hashlib.sha256(
                           (result['choices'][0].get('text') or '').encode()).hexdigest()}
                if usage.get('prompt_tokens') != len(tokens) or usage.get('completion_tokens') != a.output_tokens:
                    raise RuntimeError(f'usage mismatch: {usage}')
                row['phase_delta'] = delta
                if a.stream:
                    row['streaming'] = streamed
                out.write(json.dumps(row) + '\n')
                out.flush()
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
