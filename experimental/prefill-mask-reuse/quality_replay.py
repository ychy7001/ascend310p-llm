# SPDX-License-Identifier: Apache-2.0
"""Replay fixed arrivals, collect token-level evidence outside the same-call audit."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import sys
import time
import uuid

from bench_http_stream import post
from bench_server_phases import prefix_snapshot, prompt_hash


def completion(base, model, prompt, count, salt, request_id, observe=True):
    body = dict(model=model, prompt=prompt, max_tokens=count, temperature=0, seed=17,
                ignore_eos=True, stream=True, stream_options={'include_usage': True},
                cache_salt=salt, request_id=request_id)
    if observe:
        body.update(logprobs=5, return_token_ids=True, return_tokens_as_token_ids=True)
    usage = None
    positions = []
    text = []
    response_ids = set()
    with post(base, '/v1/completions', body) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            payload = line[6:].strip()
            if payload == b'[DONE]':
                break
            event = json.loads(payload)
            if 'error' in event:
                raise RuntimeError(event['error'])
            response_ids.add(event['id'])
            for choice in event['choices']:
                text.append(choice['text'])
                if observe:
                    ids = choice['token_ids'] or []
                    lp = choice['logprobs']
                    for tok, rendered, selected, top in zip(ids, lp['tokens'], lp['token_logprobs'],
                                                           lp['top_logprobs'], strict=True):
                        assert rendered == f'token_id:{tok}' and selected is not None and top
                        positions.append(dict(token_id=tok, selected_logprob=selected,
                                              top={int(k.split(':')[1]):v for k,v in top.items()}))
            if event.get('usage'):
                usage = event['usage']
                break
    assert usage and usage['prompt_tokens'] == len(prompt) and usage['completion_tokens'] == count, (request_id, usage)
    assert len(response_ids) == 1 and (not observe or len(positions) == count), (request_id, len(positions))
    return dict(usage=usage, text=''.join(text), positions=positions, response_id=response_ids.pop())


def main(base, model, out, control, plan_path, only=None):
    out, control, plan_path = map(Path, (out, control, plan_path))
    plan = json.loads(plan_path.read_text())
    slots = plan['requests']
    shared = plan['shared_prefix_tokens']

    def tokenize(text, n):
        with post(base, '/tokenize', {'model':model, 'prompt':text, 'add_special_tokens':False}) as response:
            unit = json.load(response)['tokens']
        assert unit
        return (unit * ((n + len(unit) - 1) // len(unit)))[:n]

    common = tokenize('会议记录：项目进度、风险、决定和行动项。 Alice reviews the project schedule. ', shared)
    prime = common + tokenize('Summarize the current meeting and list key decisions. ', plan['warm_input_tokens'] - shared)
    prompts = {}
    for slot in slots:
        pre = common if slot['shared'] else []
        label = slot['label']
        prompts[label] = pre + tokenize(f'{label}: 提取负责人、截止日期和行动项。 Review the notes and identify commitments. ',
                                        slot['input'] - len(pre))
    windows = ['fresh', 'reuse', 'fresh', 'reuse', 'audit']
    hits = 'vllm:prefix_cache_hits_total'
    with (out/'quality.jsonl').open('x', buffering=1) as file, ThreadPoolExecutor(max_workers=4) as pool:
        def emit(row):
            file.write(json.dumps(row, allow_nan=False) + '\n')

        emit(dict(kind='config', windows=windows, slots=slots,
                  prompt_sha256={k:prompt_hash(v) for k,v in prompts.items()}, prime_sha256=prompt_hash(prime)))

        def run(slot, start, namespace, prefix):
            label = slot['label']
            salt = namespace if slot['shared'] else namespace+'-'+label
            request_id = prefix+'-'+label
            begin = time.monotonic()
            result = completion(base, model, prompts[label], slot['output'], salt, request_id)
            cached = (result['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0)
            assert cached == (11520 if slot['shared'] else 0), (label, cached)
            return dict(label=label, request_id=request_id, cached_tokens=cached,
                        lateness_s=begin-start-slot['arrival_s'], **result)

        for window, mode in enumerate(windows):
            if only is not None and mode != only:
                continue
            prefix = f'mq-w{window}-{mode}'
            namespace = uuid.uuid4().hex
            temp = control/'mode.next'
            temp.write_text(mode+'\n')
            temp.replace(control/'mode')
            before = prefix_snapshot(base)[hits]
            completion(base, model, prime, 1, namespace, prefix+'-prime', observe=False)
            after = prefix_snapshot(base)[hits]
            assert before == after, (before, after)
            start = time.monotonic()
            futures = []
            for slot in slots:
                time.sleep(max(0, start+slot['arrival_s']-time.monotonic()))
                futures.append(pool.submit(run, slot, start, namespace, prefix))
            for future in futures:
                result = future.result()
                assert result['lateness_s'] < 0.1
                emit(dict(kind='request', window=window, mode=mode, **result))
            print(f'{prefix} passed HTTP/token/cache checks', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
