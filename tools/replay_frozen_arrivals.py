#!/usr/bin/env python3
"""Replay frozen synthetic token lengths at fixed external arrival offsets.

Uses the attribution replay's synthetic tokenizer unit and seed, cold salts,
and streaming usage checks. No generation warmup or occupancy-driven admission.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request
import uuid

UNIT = '会议记录：项目进度、风险和行动项。 Alice reviews the project schedule and decisions. '


def post(base, path, body):
    return urllib.request.urlopen(urllib.request.Request(
        base.rstrip('/') + path, data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'}), timeout=1800)


def prefix(base):
    with urllib.request.urlopen(base.rstrip('/') + '/metrics', timeout=10) as response:
        lines = response.read().decode().splitlines()
    values = {}
    for line in lines:
        for name in ('prefix_cache_hits_total', 'prefix_cache_queries_total'):
            if line.startswith('vllm:' + name + '{'):
                values[name] = values.get(name, 0) + float(line.split()[-1])
    if len(values) != 2:
        raise ValueError('Missing prefix counters')
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    rows = json.loads(a.plan.read_text())['requests']
    if not rows or len({r['label'] for r in rows}) != len(rows):
        p.error('Plan needs nonempty unique request labels')
    if any(r['arrival_seconds'] < 0 or min(r['input_tokens'], r['output_tokens']) <= 0 for r in rows):
        p.error('Invalid arrival or token count')
    with post(a.base_url, '/tokenize', {'model': a.model, 'prompt': UNIT,
                                      'add_special_tokens': False}) as response:
        unit = json.load(response)['tokens']
    if not unit:
        raise ValueError('Empty tokenizer unit')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    salt = uuid.uuid4().hex
    with a.out.open('x') as output:
        def emit(kind, **fields):
            event = dict(kind=kind, mono=time.monotonic(), wall=time.time(), **fields)
            with lock:
                output.write(json.dumps(event, ensure_ascii=False) + '\n')
                output.flush()

        emit('config', plan_sha256=hashlib.sha256(a.plan.read_bytes()).hexdigest(),
             unit_token_ids=unit, seed=17, cache_salt_prefix=salt,
             protocol='fixed external arrivals; cold process; no generation warmup')
        before = prefix(a.base_url)
        started = time.monotonic()
        emit('replay_start', prefix=before)

        def invoke(row):
            label = row['label']
            tokens = (unit * ((row['input_tokens'] + len(unit)-1)//len(unit)))[:row['input_tokens']]
            begin = time.monotonic()
            emit('request_start', label=label, prompt_tokens=len(tokens),
                 output_tokens=row['output_tokens'], planned_arrival_seconds=row['arrival_seconds'],
                 arrival_lateness_seconds=begin-started-row['arrival_seconds'])
            first = last = usage = None
            identified = False
            try:
                with post(a.base_url, '/v1/completions', dict(
                    model=a.model, prompt=tokens, max_tokens=row['output_tokens'],
                    temperature=0, seed=17, ignore_eos=True, cache_salt=salt+':'+label,
                    stream=True, stream_options={'include_usage': True})) as response:
                    for line in response:
                        if not line.startswith(b'data: '):
                            continue
                        raw = line[6:].strip()
                        if raw == b'[DONE]':
                            break
                        event = json.loads(raw)
                        if 'error' in event:
                            raise RuntimeError(event['error'])
                        if event.get('id') and not identified:
                            identified = True
                            emit('request_identity', label=label, ids=[
                                hashlib.sha256((event['id']+suffix).encode()).hexdigest()[:20]
                                for suffix in ('', '-0')])
                        if any(c.get('text') for c in event.get('choices', [])):
                            last = time.monotonic()
                            if first is None:
                                first = last
                                emit('first_output', label=label, elapsed=first-begin)
                        if event.get('usage'):
                            usage = event['usage']
                            break
                if (not usage or usage['prompt_tokens'] != len(tokens) or
                        usage['completion_tokens'] != row['output_tokens'] or first is None):
                    raise RuntimeError(f'{label}: incomplete response {usage}')
            except Exception as error:
                emit('request_error', label=label, error_type=type(error).__name__, error=str(error))
                raise
            end = time.monotonic()
            emit('request_end', label=label, start=begin, first=first, last=last, end=end,
                 usage=usage, ttft_s=first-begin,
                 mean_tpot_s=(last-first)/(row['output_tokens']-1) if row['output_tokens']>1 else None)
            return usage

        # Submission itself follows the frozen offsets, regardless of completions.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(rows)) as pool:
            futures = []
            for row in sorted(rows, key=lambda r:r['arrival_seconds']):
                remaining = started + row['arrival_seconds'] - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                futures.append(pool.submit(invoke, row))
            completed = [f.result() for f in futures]
        finished = time.monotonic()
        after = prefix(a.base_url)
        emit('replay_end', elapsed=finished-started, requests=len(completed), prefix=after,
             input_tokens=sum(r['prompt_tokens'] for r in completed),
             output_tokens=sum(r['completion_tokens'] for r in completed))
        if after['prefix_cache_hits_total'] != before['prefix_cache_hits_total']:
            raise RuntimeError('Frozen replay requires zero prefix-cache hits')
        print(json.dumps({'wall_s': finished-started, 'requests': len(completed),
                          'output_tokens': sum(r['completion_tokens'] for r in completed)}), flush=True)


if __name__ == '__main__':
    main()
