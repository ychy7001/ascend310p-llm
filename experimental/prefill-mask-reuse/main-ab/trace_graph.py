"""Request enough output for MTP and record actual speculative counters."""
import json
import sys
import uuid
from pathlib import Path

from bench_http_stream import post, request
from bench_server_phases import SPEC_METRICS, snapshot

base, model, out = sys.argv[1:]
with post(base, '/tokenize', {'model': model, 'prompt': 'List ten prime numbers and explain how to test primality.',
                             'add_special_tokens': False}) as stream:
    unit = json.load(stream)['tokens']
assert unit
prompt = (unit * (3072 // len(unit) + 1))[:3072]
before = snapshot(base, include_spec=True)
result = request(base, model, prompt, 64, cache_salt=uuid.uuid4().hex)
after = snapshot(base, include_spec=True)
delta = {key: after[key] - before[key] for key in SPEC_METRICS}
(Path(out) / 'graph-spec.json').write_text(json.dumps({'counters': delta, 'result': result}, indent=2) + '\n')
assert all(delta[k] > 0 for k in SPEC_METRICS[:2]), delta
