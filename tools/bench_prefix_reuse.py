# SPDX-License-Identifier: Apache-2.0
"""Measure one cold prefix request followed by repeated prefix hits."""
import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path
from bench_http_stream import post as stream_post


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
    found = set()
    for line in lines:
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split()[0]
        if name in values:
            values[name] += float(line.split()[-1])
            found.add(name)
        if 'prefix_cache' in name:
            prefix.append(line)
    if found != set(PHASE_METRICS):
        raise RuntimeError(f'Missing phase metrics: {set(PHASE_METRICS) - found}')
    return values, prefix


def request(base, model, tokens, output_tokens, cache_salt=None, *, ignore_eos=False, seed=0):
    before, prefix_before = metrics(base)
    start = time.perf_counter()
    first = None
    text = ''
    usage = None
    finish = None
    events = []
    with stream_post(base, '/v1/completions', {
        'model': model, 'prompt': tokens, 'temperature': 0, 'seed': seed, 'ignore_eos': ignore_eos,
        'max_tokens': output_tokens, 'stream': True,
        **({'cache_salt': cache_salt} if cache_salt is not None else {}),
        'stream_options': {'include_usage': True},
    }) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            payload = json.loads(data)
            now = time.perf_counter()
            for choice in payload.get('choices', []):
                chunk = choice.get('text') or ''
                if chunk:
                    first = now if first is None else first
                    text += chunk
                    events.append({'seconds': now - start, 'text': chunk})
                finish = choice.get('finish_reason') or finish
            usage = payload.get('usage') or usage
            if usage is not None:
                break
    end = time.perf_counter()
    elapsed = end - start
    if usage is None or usage.get('prompt_tokens') != len(tokens):
        raise RuntimeError(f'Incomplete prefix benchmark response: {usage}')
    if ignore_eos and usage.get('completion_tokens') != output_tokens:
        raise RuntimeError(f'Incomplete fixed output: {usage}')
    after, prefix_after = metrics(base)
    delta = {name: after[name] - before[name] for name in PHASE_METRICS}
    return {
        'wall_seconds': elapsed,
        'ttft_s': first - start if first else None,
        'mean_tpot_s': (end - first) / (usage['completion_tokens'] - 1) if first and usage['completion_tokens'] > 1 else None,
        'events': events,
        'prefill_seconds': delta[PHASE_METRICS[1]],
        'decode_seconds': delta[PHASE_METRICS[2]],
        'logical_prefill_tok_s': len(tokens) / delta[PHASE_METRICS[1]] if delta[PHASE_METRICS[1]] > 0 else None,
        'decode_tok_s': ((usage.get('completion_tokens', 0) - 1) /
                         delta[PHASE_METRICS[2]]) if delta[PHASE_METRICS[2]] > 0 else None,
        'usage': usage,
        'text': text,
        'valid': delta[PHASE_METRICS[0]] == 1 and usage.get('prompt_tokens') == len(tokens),
        'finish_reason': finish,
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
    parser.add_argument('--prompt-ids-file', type=Path, help='Reuse exact prompt_ids from a baseline JSONL config')
    parser.add_argument('--ignore-eos', action='store_true', help='Require exact output token count')
    parser.add_argument('--cache-salt', required=False,
                        help='Use the same salt for cold and hit requests; default random run ID')
    args = parser.parse_args()
    if min(args.prompt_tokens, args.output_tokens, args.repeats) < 1:
        parser.error('token counts and repeats must be positive')
    if args.cache_salt is None:
        import uuid
        args.cache_salt = uuid.uuid4().hex
    if args.prompt_ids_file:
        with args.prompt_ids_file.open() as source:
            config = json.loads(source.readline())
        tokens = config['prompt_ids']
        if not tokens or any(type(token) is not int or token < 0 for token in tokens):
            raise RuntimeError('Invalid prompt IDs in baseline config')
    else:
        prompt = ('Explain how quantization affects memory bandwidth and numerical '
                  'precision. ' * 4)
        unit = post(args.base_url, '/tokenize', {
            'model': args.model, 'prompt': prompt, 'add_special_tokens': False,
        })['tokens']
        if not unit:
            raise RuntimeError('Tokenizer returned empty benchmark unit')
        tokens = (unit * ((args.prompt_tokens + len(unit) - 1) // len(unit)))[:args.prompt_tokens]
    if len(tokens) != args.prompt_tokens:
        raise RuntimeError('Insufficient prompt tokens')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = [request(args.base_url, args.model, tokens, args.output_tokens,
                    args.cache_salt, ignore_eos=args.ignore_eos)
            for _ in range(args.repeats + 1)]
    result = {
        'protocol': 'first request is cold; following requests reuse identical token IDs',
        'model': args.model, 'prompt_tokens': len(tokens),
        'cache_salt': args.cache_salt, 'ignore_eos': args.ignore_eos, 'sampling_seed': 0,
        'prompt_ids_file': str(args.prompt_ids_file) if args.prompt_ids_file else None,
        'output_tokens': args.output_tokens, 'rows': rows,
        'prompt_ids': tokens,
        'prompt_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        'all_outputs_equal': len({row['text'] for row in rows}) == 1,
        'all_valid': all(row['valid'] for row in rows),
    }
    with args.out.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    for index, row in enumerate(rows):
        print(json.dumps({'request': index, **{
            key: row[key] for key in ('wall_seconds', 'logical_prefill_tok_s', 'decode_tok_s')
        }}), flush=True)


if __name__ == '__main__':
    main()
