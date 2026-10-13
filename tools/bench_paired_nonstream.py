# SPDX-License-Identifier: Apache-2.0
"""Long-context fallback benchmark using non-streaming completions.

Some 310P experimental servers leave a long streaming response open after the
usage event. This variant uses the JSON completion endpoint and therefore
records reliable wall time and usage; it is a fallback, not a replacement for
the phase-counter benchmark.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from bench_http_stream import post
from bench_server_phases import METRICS, snapshot, prompt_hash


def tokenize(base, model, count):
    text = ("Benchmark cold long-prefill. Explain quantization and memory bandwidth. "
            * (count // 8 + 256))
    with post(base, '/tokenize', {'model': model, 'prompt': text,
                                 'add_special_tokens': False}) as response:
        tokens = json.load(response)['tokens'][:count]
    if len(tokens) != count:
        raise RuntimeError(f'expected {count} tokens, got {len(tokens)}')
    return tokens


def complete(base, model, tokens, output_tokens):
    body = json.dumps({'model': model, 'prompt': tokens, 'max_tokens': output_tokens,
                       'temperature': 0, 'ignore_eos': True, 'stream': False}).encode()
    req = urllib.request.Request(base.rstrip('/') + '/v1/completions', data=body,
                                 headers={'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as response:
        result = json.load(response)
    result['_elapsed_s'] = time.perf_counter() - start
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mindieq-url', required=True); p.add_argument('--original-url', required=True)
    p.add_argument('--model', required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--prompt-tokens', type=int, required=True); p.add_argument('--output-tokens', type=int, default=128)
    p.add_argument('--repeats', type=int, default=3); a = p.parse_args()
    tokens = tokenize(a.mindieq_url, a.model, a.prompt_tokens)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
        f.write(json.dumps({'kind': 'config', **config, 'prompt_ids_sha256': prompt_hash(tokens),
                            'protocol': 'non-streaming completion; alternating endpoints; wall time plus phase counters'}) + '\n')
        for repeat in range(a.repeats):
            endpoints = [('mindieq', a.mindieq_url), ('original', a.original_url)]
            if repeat % 2: endpoints.reverse()
            for order, (name, base) in enumerate(endpoints):
                before = snapshot(base)
                result = complete(base, a.model, tokens, a.output_tokens)
                after = snapshot(base)
                delta = {k: after[k] - before[k] for k in METRICS}
                usage = result.get('usage') or {}
                if usage.get('prompt_tokens') != a.prompt_tokens or usage.get('completion_tokens') != a.output_tokens:
                    raise RuntimeError(f'usage mismatch: {usage}')
                row = {'kind': 'request', 'service': name, 'repeat': repeat, 'order': order,
                       'prompt_tokens': a.prompt_tokens, 'output_tokens': a.output_tokens,
                       'prompt_ids_sha256': prompt_hash(tokens), 'phase_delta': delta,
                       'prefill_tok_s': a.prompt_tokens / delta[METRICS[1]],
                       'decode_tok_s': (a.output_tokens - 1) / delta[METRICS[2]],
                       'elapsed_s': result['_elapsed_s'], 'usage': usage,
                       'text_sha256': hashlib.sha256((result['choices'][0].get('text') or '').encode()).hexdigest()}
                f.write(json.dumps(row) + '\n'); f.flush(); print(json.dumps(row), flush=True)


if __name__ == '__main__': main()
