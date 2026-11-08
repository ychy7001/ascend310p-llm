#!/usr/bin/env python3
"""Measure the frozen MoE experiment's RAG and general load profiles.

Phase metrics are server request spans, not kernel time. Each prompt has a cold
salt; prompt IDs and seed match the attribution replay's synthetic unit.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import statistics
import time
import uuid

from bench_http_stream import request
from bench_server_phases import snapshot
from replay_frozen_arrivals import UNIT, post, prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with post(args.base_url, '/tokenize', dict(model=args.model, prompt=UNIT,
                                             add_special_tokens=False)) as response:
        unit = json.load(response)['tokens']
    if not unit:
        raise ValueError('Empty tokenizer unit')
    cases = [('general', n, c, 256) for n in (1024,4096,16384,32768) for c in (1,4)]
    cases += [('RAG', n, c, 32) for n in (8192,16384,32768) for c in (1,4)]
    salt = uuid.uuid4().hex
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as output:
        def emit(event):
            output.write(json.dumps(event, ensure_ascii=False)+'\n')
            output.flush()
        emit(dict(kind='config', cases=cases, repeats=3, seed=17, unit_token_ids=unit,
                  cache_salt_prefix=salt, protocol=__doc__))
        for profile, n, c, count in cases:
            label = f'{profile}-{n}-C{c}'
            tokens = (unit*((n+len(unit)-1)//len(unit)))[:n]
            # One short generation warms each shape, with an independent salt.
            request(args.base_url,args.model,tokens,32,cache_salt=f'{salt}:{label}:warmup',seed=17)
            for repeat in range(3):
                cache_before, before = prefix(args.base_url), snapshot(args.base_url)
                begin = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                    tasks = [pool.submit(request,args.base_url,args.model,tokens,count,
                              cache_salt=f'{salt}:{label}:{repeat}:{i}',seed=17) for i in range(c)]
                    responses = [task.result() for task in tasks]
                elapsed = time.perf_counter()-begin
                after, cache_after = snapshot(args.base_url), prefix(args.base_url)
                delta = {key:after[key]-before[key] for key in before}
                cache_delta = {key:cache_after[key]-cache_before[key] for key in cache_before}
                valid = (delta['vllm:request_prefill_time_seconds_count']==c and
                         delta['vllm:request_prefill_time_seconds_sum']>0 and
                         cache_delta['prefix_cache_hits_total']==0)
                row = dict(kind='round',profile=profile,prompt_tokens=n,concurrency=c,
                           output_tokens=count,repeat=repeat,valid=valid,elapsed_s=elapsed,
                           aggregate_output_tok_s=c*count/elapsed,
                           median_ttft_s=statistics.median(r['ttft_s'] for r in responses),
                           median_tpot_s=statistics.median(r['mean_tpot_s'] for r in responses),
                           request_prefill_tok_s=c*n/delta['vllm:request_prefill_time_seconds_sum'],
                           metrics_delta=delta,prefix_delta=cache_delta,responses=responses)
                emit(row)
                print(json.dumps({k:v for k,v in row.items() if k not in ('responses','metrics_delta','prefix_delta')}),flush=True)
                if not valid:
                    raise RuntimeError(f'{label}: counter or prefix mismatch')


if __name__ == '__main__':
    main()
