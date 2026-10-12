# SPDX-License-Identifier: Apache-2.0
"""独占服务的分阶段计时；Prometheus 请求阶段耗时并非纯 kernel 时间。"""

import argparse
import json
from pathlib import Path
import urllib.request

from bench_http_stream import post, request


METRICS = (
    'vllm:request_prefill_time_seconds_count',
    'vllm:request_prefill_time_seconds_sum',
    'vllm:request_decode_time_seconds_sum',
)


def snapshot(base):
    with urllib.request.urlopen(base.rstrip('/') + '/metrics', timeout=30) as response:
        raw = response.read().decode()
    values = {name: 0.0 for name in METRICS}
    found = set()
    for line in raw.splitlines():
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split()[0]
        if name in values:
            values[name] += float(line.split()[-1])
            found.add(name)
    if found != set(METRICS):
        raise RuntimeError(f'Missing phase metrics: {set(METRICS) - found}')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prompt-tokens', type=int, default=3072)
    parser.add_argument('--output-tokens', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument(
        '--unique-prompts', action='store_true',
        help='Use a different token prefix for warmup and every measured request; '
             'required for cold-Prefill measurements when prefix caching is enabled')
    args = parser.parse_args()
    if min(args.prompt_tokens, args.output_tokens, args.repeats) < 1:
        parser.error('token counts and repeats must be positive')
    def tokenize(variant):
        prompt = (
            f'Benchmark cold-prefill variant {variant}. '
            'Explain how quantization affects memory bandwidth and numerical precision. '
            * (args.prompt_tokens + 1)
        )
        with post(args.base_url, '/tokenize', {
            'model': args.model,
            'prompt': prompt,
            'add_special_tokens': False,
        }) as response:
            tokens = json.load(response)['tokens'][:args.prompt_tokens]
        if len(tokens) != args.prompt_tokens:
            raise RuntimeError('Insufficient prompt tokens')
        return tokens

    prompt_sets = [tokenize(i) for i in range(args.repeats + 1)] if args.unique_prompts else None
    tokens = prompt_sets[0] if prompt_sets else tokenize(0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        output.write(json.dumps({
            'kind': 'config',
            **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'prompt_ids': tokens,
            'protocol': 'exclusive endpoint; warmup=1; request phase counters, not pure kernel time',
            'unique_prompts': args.unique_prompts,
        }) + '\n')
        warmup_tokens = prompt_sets[0] if prompt_sets else tokens
        warmup = request(args.base_url, args.model, warmup_tokens, min(32, args.output_tokens))
        output.write(json.dumps({'kind': 'warmup', **warmup}) + '\n')
        output.flush()
        for repeat in range(args.repeats):
            request_tokens = prompt_sets[repeat + 1] if prompt_sets else tokens
            before = snapshot(args.base_url)
            result = request(args.base_url, args.model, request_tokens, args.output_tokens)
            after = snapshot(args.base_url)
            delta = {name: after[name] - before[name] for name in METRICS}
            prefill = delta[METRICS[1]]
            decode = delta[METRICS[2]]
            usage = result['usage']
            valid = (delta[METRICS[0]] == 1 and prefill > 0 and decode >= 0
                     and usage is not None and usage['prompt_tokens'] == len(request_tokens)
                     and usage['completion_tokens'] == args.output_tokens)
            row = {'kind': 'request', 'repeat': repeat, 'valid': valid,
                   'metrics_before': before, 'metrics_after': after,
                   'prefill_seconds': prefill, 'decode_seconds': decode,
                   'prefill_tok_s': len(request_tokens) / prefill if valid else None,
                   'decode_tok_s': (usage['completion_tokens'] - 1) / decode
                   if valid and decode > 0 and usage['completion_tokens'] > 1 else None,
                   'response': result}
            output.write(json.dumps(row) + '\n')
            output.flush()
            print(json.dumps({k: v for k, v in row.items()
                              if k not in ('metrics_before', 'metrics_after', 'response')}), flush=True)
            if not valid:
                raise RuntimeError('Counter/usage mismatch; check endpoint exclusivity and metric resets')


if __name__ == '__main__':
    main()
