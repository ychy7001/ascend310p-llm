# SPDX-License-Identifier: Apache-2.0
"""固定 token 输入的 HTTP streaming 基准；保留逐请求原始时间和服务端 usage。"""

import argparse
import concurrent.futures
import json
import time
import uuid
import urllib.request
from pathlib import Path


def post(base, path, body):
    return urllib.request.urlopen(
        urllib.request.Request(
            base.rstrip('/') + path,
            data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Connection': 'close'},
        ), timeout=1800,
    )


def request(base, model, tokens, output_tokens, *, cache_salt=None, seed=0):
    start = time.perf_counter()
    first = None
    events = []
    usage = None
    text = ''
    body = {
        'model': model, 'prompt': tokens, 'max_tokens': output_tokens,
        'temperature': 0, 'seed': seed, 'ignore_eos': True, 'stream': True,
        'stream_options': {'include_usage': True},
    }
    if cache_salt is not None:
        body['cache_salt'] = cache_salt
    with post(base, '/v1/completions', body) as response:
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
                # include_usage 的最终事件已包含完整输出。显式关闭响应，
                # 避免实验镜像在 usage 后不发送 DONE/EOF 导致无限等连接。
                break
    end = time.perf_counter()
    if (usage is None or usage.get('completion_tokens') != output_tokens
            or usage.get('prompt_tokens') != len(tokens)):
        raise RuntimeError(
            f'Incomplete benchmark response after {end - start:.3f}s: '
            f'expected prompt/output={len(tokens)}/{output_tokens}, usage={usage}')
    count = usage['completion_tokens'] if usage else None
    return {
        'elapsed_s': end - start,
        'ttft_s': first - start if first else None,
        'mean_tpot_s': (end - first) / (count - 1) if first and count and count > 1 else None,
        'usage': usage, 'events': events, 'text': text,
    }


def measured_request(*args, **kwargs):
    """保留失败请求与耗时，避免只对成功请求报告成功率。"""
    start = time.perf_counter()
    try:
        return {'success': True, **request(*args, **kwargs)}
    except Exception as exc:
        return {'success': False, 'elapsed_s': time.perf_counter() - start,
                'error_type': type(exc).__name__, 'error': str(exc)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prompt-tokens', type=int, default=512)
    p.add_argument('--output-tokens', type=int, default=128)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--seed', type=int, default=310)
    p.add_argument('--cache-salt-prefix', default=None,
                   help='Default: random run ID; every request gets a distinct salt')
    a = p.parse_args()
    if min(a.prompt_tokens, a.output_tokens, a.concurrency, a.repeats) < 1:
        p.error('token counts, concurrency and repeats must be positive')
    a.cache_salt_prefix = a.cache_salt_prefix or uuid.uuid4().hex
    with post(a.base_url, '/tokenize', {
        'model': a.model,
        'prompt': ('Explain how quantization affects memory bandwidth and numerical precision. ' * 4),
        'add_special_tokens': False,
    }) as response:
        unit = json.load(response)['tokens']
        if not unit:
            raise RuntimeError('Tokenizer returned empty benchmark unit')
        tokens = (unit * ((a.prompt_tokens + len(unit) - 1) // len(unit)))[:a.prompt_tokens]
    if len(tokens) != a.prompt_tokens:
        raise RuntimeError('tokenizer returned insufficient prompt tokens')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    with a.out.open('x') as f:
        f.write(json.dumps({'kind': 'config', **{k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}, 'prompt_ids': tokens}) + '\n')
        warmup = request(a.base_url, a.model, tokens, min(32, a.output_tokens),
                         cache_salt=f'{a.cache_salt_prefix}-warmup', seed=a.seed)
        f.write(json.dumps({'kind': 'warmup', **warmup}) + '\n')
        f.flush()
        for repeat in range(a.repeats):
            start = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
                futures = [pool.submit(measured_request, a.base_url, a.model, tokens, a.output_tokens,
                                       cache_salt=f'{a.cache_salt_prefix}-{repeat}-{i}', seed=a.seed)
                           for i in range(a.concurrency)]
                rows = [future.result() for future in futures]
            elapsed = time.perf_counter() - start
            for i, row in enumerate(rows):
                f.write(json.dumps({'kind': 'request', 'repeat': repeat, 'request': i, **row}) + '\n')
            summary = {'kind': 'round', 'repeat': repeat, 'elapsed_s': elapsed,
                       'successful_requests': sum(r['success'] for r in rows),
                       'failed_requests': sum(not r['success'] for r in rows),
                       'aggregate_output_tok_s': sum(r['usage']['completion_tokens'] for r in rows
                                                      if r['success']) / elapsed}
            failed += summary['failed_requests']
            f.write(json.dumps(summary) + '\n')
            f.flush()
            print(json.dumps(summary), flush=True)
    if failed:
        raise SystemExit(f'{failed} measured requests failed; see raw JSONL')


if __name__ == '__main__':
    main()
