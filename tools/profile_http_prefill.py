# SPDX-License-Identifier: Apache-2.0
"""独占端点采集一次Prefill；profile请求耗时不能作为常规性能结果。"""
import argparse
import json
import uuid
from pathlib import Path

from bench_http_stream import post, request
from bench_server_phases import prefix_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prompt-tokens', type=int, default=3072)
    args = parser.parse_args()
    with post(args.base_url, '/tokenize', {
        'model': args.model,
        'prompt': 'Explain how quantization affects memory bandwidth and numerical precision. ' * (args.prompt_tokens + 1),
        'add_special_tokens': False,
    }) as response:
        tokens = json.load(response)['tokens'][:args.prompt_tokens]
    if len(tokens) != args.prompt_tokens:
        raise RuntimeError('Insufficient prompt tokens')
    with args.out.open('x') as output:
        output.write(json.dumps({'kind': 'config', 'prompt_ids': tokens,
                                 'model': args.model, 'base_url': args.base_url,
                                 'protocol': 'one warmup then one profiled output=1 request; not throughput benchmark'}) + '\n')
        salt = uuid.uuid4().hex
        warmup = request(args.base_url, args.model, tokens, 1, cache_salt=salt + '-warmup')
        output.write(json.dumps({'kind': 'warmup', **warmup}) + '\n')
        output.flush()
        with post(args.base_url, '/start_profile', {}) as response:
            response.read()
        try:
            before = prefix_snapshot(args.base_url)
            result = request(args.base_url, args.model, tokens, 1, cache_salt=salt + '-profile')
            after = prefix_snapshot(args.base_url)
            result['prefix_cache_hits'] = after['vllm:prefix_cache_hits_total'] - before['vllm:prefix_cache_hits_total']
            assert result['prefix_cache_hits'] == 0, 'Prefill profile unexpectedly hit the prefix cache'
            output.write(json.dumps({'kind': 'profiled_request', **result}) + '\n')
            output.flush()
        finally:
            with post(args.base_url, '/stop_profile', {}) as response:
                response.read()
        print(json.dumps({'profile_stopped': True, 'usage': result['usage']}), flush=True)


if __name__ == '__main__':
    main()
