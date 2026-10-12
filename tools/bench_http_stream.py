# SPDX-License-Identifier: Apache-2.0
"""固定 token 输入的 HTTP streaming 基准；保留逐请求原始时间和服务端 usage。"""

import argparse
import concurrent.futures
import json
import time
import urllib.request
from pathlib import Path


def post(base, path, body):
    return urllib.request.urlopen(
        urllib.request.Request(
            base.rstrip('/') + path,
            data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'},
        ), timeout=1800,
    )


def request(base, model, tokens, output_tokens):
    start = time.perf_counter()
    first = None
    events = []
    usage = None
    text = ''
    with post(base, '/v1/completions', {
        'model': model, 'prompt': tokens, 'max_tokens': output_tokens,
        'temperature': 0, 'ignore_eos': True, 'stream': True,
        'stream_options': {'include_usage': True},
    }) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                break
            event = json.loads(payload)
            now = time.perf_counter()
            chunk = ''.join(c.get('text', '') for c in event.get('choices', []))
            if chunk:
                first = now if first is None else first
                text += chunk
                events.append({'seconds': now - start, 'text': chunk})
            if event.get('usage'):
                usage = event['usage']
    end = time.perf_counter()
    count = usage['completion_tokens'] if usage else None
    return {
        'elapsed_s': end - start,
        'ttft_s': first - start if first else None,
        'mean_tpot_s': (end - first) / (count - 1) if first and count and count > 1 else None,
        'usage': usage, 'events': events, 'text': text,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prompt-tokens', type=int, default=512)
    p.add_argument('--output-tokens', type=int, default=128)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    with post(a.base_url, '/tokenize', {
        'model': a.model,
        'prompt': ('Explain how quantization affects memory bandwidth and numerical precision. ' * (a.prompt_tokens + 1)),
        'add_special_tokens': False,
    }) as response:
        tokens = json.load(response)['tokens'][:a.prompt_tokens]
    if len(tokens) != a.prompt_tokens:
        raise RuntimeError('tokenizer returned insufficient prompt tokens')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        f.write(json.dumps({'kind': 'config', **{k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}, 'prompt_ids': tokens}) + '\n')
        warmup = request(a.base_url, a.model, tokens, min(32, a.output_tokens))
        f.write(json.dumps({'kind': 'warmup', **warmup}) + '\n')
        f.flush()
        for repeat in range(a.repeats):
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
                futures = [pool.submit(request, a.base_url, a.model, tokens, a.output_tokens) for _ in range(a.concurrency)]
                rows = [future.result() for future in futures]
            elapsed = time.perf_counter() - start
            for i, row in enumerate(rows):
                f.write(json.dumps({'kind': 'request', 'repeat': repeat, 'request': i, **row}) + '\n')
            summary = {'kind': 'round', 'repeat': repeat, 'elapsed_s': elapsed,
                       'aggregate_output_tok_s': sum(r['usage']['completion_tokens'] for r in rows) / elapsed}
            f.write(json.dumps(summary) + '\n')
            f.flush()
            print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
