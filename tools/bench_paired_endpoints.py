# SPDX-License-Identifier: Apache-2.0
"""交替测量两个独占 HTTP 端点的长 Prefill。

两个端点分别占用不同的同型号 NPU；相同 token ids 在端点之间交替发送，
并同时保存 Prometheus 阶段计时和原始流式响应，避免把单次长请求当成 P50。
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

from bench_http_stream import request, post
from bench_server_phases import METRICS, snapshot, prompt_hash


def tokens_for(base, model, count):
    # 约 8～10 tokens/句；不要按 token 数直接重复，否则 64K 请求会先
    # 生成数十万到百万 token 的 tokenizer 输入，掩盖真正的 benchmark。
    repeats = count // 8 + 256
    text = (
        "Benchmark cold long-prefill. Explain how quantization affects memory "
        "bandwidth and numerical precision. " * repeats
    )
    with post(base, '/tokenize', {
        'model': model, 'prompt': text, 'add_special_tokens': False,
    }) as response:
        tokens = json.load(response)['tokens'][:count]
    if len(tokens) != count:
        raise RuntimeError(f'tokenizer returned {len(tokens)} tokens, expected {count}')
    return tokens


def one(base, model, tokens, output_tokens, service, repeat, order):
    before = snapshot(base)
    result = request(base, model, tokens, output_tokens)
    after = snapshot(base)
    delta = {name: after[name] - before[name] for name in METRICS}
    usage = result.get('usage') or {}
    if usage.get('prompt_tokens') != len(tokens) or usage.get('completion_tokens') != output_tokens:
        raise RuntimeError(f'{service} usage mismatch: {usage}')
    if delta[METRICS[0]] != 1 or delta[METRICS[1]] <= 0 or delta[METRICS[2]] < 0:
        raise RuntimeError(f'{service} metric mismatch: {delta}')
    return {
        'kind': 'request', 'service': service, 'repeat': repeat, 'order': order,
        'prompt_tokens': len(tokens), 'output_tokens': output_tokens,
        'prompt_ids_sha256': prompt_hash(tokens),
        'prefill_s': delta[METRICS[1]], 'decode_s': delta[METRICS[2]],
        'prefill_tok_s': len(tokens) / delta[METRICS[1]],
        'decode_tok_s': (output_tokens - 1) / delta[METRICS[2]] if output_tokens > 1 else None,
        'elapsed_s': result['elapsed_s'], 'ttft_s': result['ttft_s'],
        'usage': usage, 'text_sha256': hashlib.sha256(result.get('text', '').encode()).hexdigest(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mindieq-url', required=True)
    p.add_argument('--original-url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prompt-tokens', type=int, required=True)
    p.add_argument('--output-tokens', type=int, default=128)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    if min(a.prompt_tokens, a.output_tokens, a.repeats) < 1:
        p.error('token counts and repeats must be positive')
    tokens = tokens_for(a.mindieq_url, a.model, a.prompt_tokens)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        config = {'kind': 'config', **vars(a), 'out': str(a.out),
                  'prompt_ids_sha256': prompt_hash(tokens),
                  'protocol': 'warmup each endpoint; alternate order; Prometheus phase counters'}
        f.write(json.dumps(config) + '\n')
        for name, base in (('mindieq', a.mindieq_url), ('original', a.original_url)):
            warm = request(base, a.model, tokens, min(32, a.output_tokens))
            f.write(json.dumps({'kind': 'warmup', 'service': name,
                                'usage': warm.get('usage'), 'elapsed_s': warm['elapsed_s']}) + '\n')
        f.flush()
        for repeat in range(a.repeats):
            endpoints = [('mindieq', a.mindieq_url), ('original', a.original_url)]
            if repeat % 2:
                endpoints.reverse()
            for order, (name, base) in enumerate(endpoints):
                row = one(base, a.model, tokens, a.output_tokens, name, repeat, order)
                f.write(json.dumps(row) + '\n')
                f.flush()
                print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
