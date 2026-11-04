# SPDX-License-Identifier: Apache-2.0
"""首个流式输出后采集Decode；可能遗漏初始Decode步，耗时非benchmark。"""
import argparse
import json
import time
import uuid
from pathlib import Path
from bench_http_stream import post
from bench_server_phases import prefix_snapshot

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base-url', required=True)
p.add_argument('--model', required=True)
p.add_argument('--prompt-tokens', type=int, default=8192)
p.add_argument('--output-tokens', type=int, default=128)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
with post(a.base_url, '/tokenize', {'model': a.model, 'prompt':
    'Explain how quantization affects memory bandwidth and numerical precision. ' * 4,
    'add_special_tokens': False}) as response:
    unit = json.load(response)['tokens']
    if not unit:
        raise RuntimeError('Tokenizer returned empty benchmark unit')
    tokens = (unit * ((a.prompt_tokens + len(unit) - 1) // len(unit)))[:a.prompt_tokens]
assert len(tokens) == a.prompt_tokens
salt = uuid.uuid4().hex
before = prefix_snapshot(a.base_url)
started = False
begin = time.perf_counter()
with a.out.open('x') as output:
    def emit(row):
        output.write(json.dumps(row) + '\n')
        output.flush()
    emit({'kind': 'config', 'prompt_tokens': a.prompt_tokens, 'output_tokens': a.output_tokens, 'prompt_ids': tokens,
          'cache_salt': salt, 'seed': 0,
          'scope': __doc__, 'warmup': 'no request warmup; startup Graph capture; diagnostic only'})
    usage = None
    try:
        with post(a.base_url, '/v1/completions', {'model': a.model, 'prompt': tokens,
            'max_tokens': a.output_tokens, 'temperature': 0, 'seed': 0,
            'cache_salt': salt, 'ignore_eos': True,
            'stream': True, 'stream_options': {'include_usage': True}}) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    break
                event = json.loads(payload)
                text = ''.join(c.get('text', '') for c in event.get('choices', []))
                if text and not started:
                    emit({'kind': 'first_output', 'elapsed_s': time.perf_counter() - begin})
                    with post(a.base_url, '/start_profile', {}) as control:
                        control.read()
                    started = True
                    emit({'kind': 'profile_started', 'elapsed_s': time.perf_counter() - begin})
                if event.get('usage'):
                    usage = event['usage']
                emit({'kind': 'stream', 'elapsed_s': time.perf_counter() - begin, 'event': event})
                if usage:
                    break
    finally:
        if started:
            with post(a.base_url, '/stop_profile', {}) as control:
                control.read()
            emit({'kind': 'profile_stopped', 'elapsed_s': time.perf_counter() - begin})
    assert started and usage and usage['prompt_tokens'] == a.prompt_tokens
    assert usage['completion_tokens'] == a.output_tokens
    after = prefix_snapshot(a.base_url)
    hits = after['vllm:prefix_cache_hits_total'] - before['vllm:prefix_cache_hits_total']
    assert hits == 0, 'Decode profile unexpectedly hit the prefix cache'
    emit({'kind': 'summary', 'status': 'PASS', 'usage': usage, 'prefix_cache_hits': hits})
