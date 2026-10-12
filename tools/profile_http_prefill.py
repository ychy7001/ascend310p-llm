# SPDX-License-Identifier: Apache-2.0
"""独占端点采集一次Prefill；profile请求耗时不能作为常规性能结果。"""
import argparse
import json
from pathlib import Path

from bench_http_stream import post, request


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
        warmup = request(args.base_url, args.model, tokens, 1)
        output.write(json.dumps({'kind': 'warmup', **warmup}) + '\n')
        output.flush()
        with post(args.base_url, '/start_profile', {}) as response:
            response.read()
        try:
            result = request(args.base_url, args.model, tokens, 1)
            output.write(json.dumps({'kind': 'profiled_request', **result}) + '\n')
            output.flush()
        finally:
            with post(args.base_url, '/stop_profile', {}) as response:
                response.read()
        print(json.dumps({'profile_stopped': True, 'usage': result['usage']}), flush=True)


if __name__ == '__main__':
    main()
