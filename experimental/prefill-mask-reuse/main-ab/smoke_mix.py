import json
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from bench_http_stream import post, request
from bench_server_phases import prefix_snapshot, prompt_hash

base, model, path = sys.argv[1:]
out = Path(path)
salt = uuid.uuid4().hex
HITS = 'vllm:prefix_cache_hits_total'

def ids(text, n):
    with post(base, '/tokenize', {'model':model,'prompt':text,'add_special_tokens':False}) as f:
        unit = json.load(f)['tokens']
    assert unit
    return (unit*((n+len(unit)-1)//len(unit)))[:n]

common = ids('Reference text. The following numbers identify individual kernel requests. ', 4096)
# The observed 310P hybrid cache aligns Mamba states to 640 tokens.
# End each warmup just beyond the last shared boundary (3840), rather
# than filling the final aligned state with a different suffix.
prime = common[:3841]
start = prefix_snapshot(base)[HITS]
for i in (0, 3):
    primed = request(base, model, prime, 1, cache_salt=f'{salt}-{i}')
    assert prefix_snapshot(base)[HITS] == start, primed['usage']
barrier = threading.Barrier(4)

def run(i, length, warm):
    pre = common if warm else []
    prompt = pre + ids(f'Synthetic request {i}. Compute the output using the preceding details. ',length-len(pre))
    barrier.wait(30)
    response = request(base, model, prompt, 32, cache_salt=f'{salt}-{i}')
    cached = (response['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0)
    return dict(slot=i, shape=(length,32), warm=warm, cached=cached, prompt_sha=prompt_hash(prompt), **response)

with ThreadPoolExecutor(max_workers=4) as pool:
    fs = [pool.submit(run,i,n,w) for i,(n,w) in enumerate(((6656,True),(8192,False),(1024,False),(7168,True)))]
    rows = [f.result() for f in fs]
(out/'mixed.json').write_text(json.dumps(rows, indent=2))
assert all((r['cached'] > 0) == r['warm'] for r in rows), [(r['slot'], r['warm'], r['cached']) for r in rows]
assert prefix_snapshot(base)[HITS] > start
needle = [json.loads(l) for l in (out/'needle.jsonl').read_text().splitlines()][-1]
assert needle['retrieved'], needle
