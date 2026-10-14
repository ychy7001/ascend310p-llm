# SPDX-License-Identifier: Apache-2.0
"""Capture a complete HTTP Decode request, including the first Decode step.

The Ascend profiler may use its own bounded trace window.  The output records
the full HTTP scope, but callers must compare the exported device span and
event count before treating it as a full-device trace.
"""
import argparse
import json
import time
from pathlib import Path
from bench_http_stream import post


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base-url', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--prompt-tokens', type=int, default=8192)
parser.add_argument('--output-tokens', type=int, default=256)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()

with post(args.base_url, '/tokenize', {
    'model': args.model,
    'prompt': 'Explain how quantization affects memory bandwidth and numerical precision. '
              * (args.prompt_tokens + 1),
    'add_special_tokens': False,
}) as response:
    tokens = json.load(response)['tokens'][:args.prompt_tokens]
assert len(tokens) == args.prompt_tokens

args.out.parent.mkdir(parents=True, exist_ok=True)
begin = time.perf_counter()
with args.out.open('x') as output:
    def emit(row):
        output.write(json.dumps(row) + '\n')
        output.flush()

    emit({'kind': 'config', 'prompt_tokens': args.prompt_tokens,
          'output_tokens': args.output_tokens, 'scope': __doc__,
          'profile_scope': 'request start through usage event'})
    with post(args.base_url, '/start_profile', {}) as control:
        control.read()
    emit({'kind': 'profile_started', 'elapsed_s': time.perf_counter() - begin})
    usage = None
    try:
        with post(args.base_url, '/v1/completions', {
            'model': args.model, 'prompt': tokens,
            'max_tokens': args.output_tokens, 'temperature': 0,
            'ignore_eos': True, 'stream': True,
            'stream_options': {'include_usage': True},
        }) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                payload = line[6:].strip()
                if payload == b'[DONE]':
                    break
                event = json.loads(payload)
                if event.get('usage'):
                    usage = event['usage']
                emit({'kind': 'stream', 'elapsed_s': time.perf_counter() - begin,
                      'event': event})
                if usage:
                    break
    finally:
        with post(args.base_url, '/stop_profile', {}) as control:
            control.read()
        emit({'kind': 'profile_stopped', 'elapsed_s': time.perf_counter() - begin})
    assert usage and usage['prompt_tokens'] == args.prompt_tokens
    assert usage['completion_tokens'] == args.output_tokens
    emit({'kind': 'summary', 'status': 'PASS', 'usage': usage,
          'elapsed_s': time.perf_counter() - begin,
          'trace_scope_warning': 'HTTP scope is complete; device profiler scope must be validated separately'})
