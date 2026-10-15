#!/usr/bin/env python3
"""统一部署验收：固定token阶段计时、112K会议canary及请求状态复用。"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

from check_meeting_long_context import make_prompt, complete, score
from bench_server_phases import snapshot, METRICS, prompt_hash

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--base-url', required=True)
p.add_argument('--model', required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
base = a.base_url.rstrip('/')

def status(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)

def invoke(script, *arguments):
    subprocess.run(['python3', str(Path(__file__).parent / script), *arguments], check=True)

status('waiting for health')
deadline = time.monotonic() + 1500
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(base + '/health', timeout=3) as r:
            assert r.status == 200
        break
    except Exception:
        time.sleep(5)
else:
    raise TimeoutError('health timeout')
with urllib.request.urlopen(base+'/v1/models', timeout=30) as r:
    (a.out/'models.json').write_text(json.dumps(json.load(r), indent=2))
status('health=200; quality before long requests')
invoke('check_cache_request_reuse.py', '--base-url', base, '--model', a.model,
       '--out', str(a.out/'quality-before.json'))
quality = json.loads((a.out/'quality-before.json').read_text())
assert quality['strict_matches'] == 8 and quality['reuse']['identical_text'] and not quality['reuse']['canary_leaked']
for tokens, repeats in [(32768, 2), (112000, 1)]:
    status(f'fixed output benchmark {tokens}, warmup1 + repeats{repeats}')
    invoke('bench_server_phases.py', '--base-url', base, '--model', a.model,
           '--out', str(a.out/f'p{tokens}-o128.jsonl'), '--prompt-tokens', str(tokens),
           '--output-tokens', '128', '--repeats', str(repeats), '--unique-prompts',
           '--spec-metrics', '--cache-salt-prefix', f'unified2048-{tokens}')
status('112K meeting decision canary')
tokens, positions = make_prompt(base, a.model, 112000)
before = snapshot(base, True)
response, elapsed = complete(base, a.model, tokens)
after = snapshot(base, True)
text = response['choices'][0]['text']
scored = score(text)
delta = {k:after[k]-before[k] for k in before}
usage = response['usage']
record = {'prompt_tokens':len(tokens), 'prompt_hash':prompt_hash(tokens), 'positions':positions,
          'response':response, 'score':scored, 'elapsed_s':elapsed, 'metrics_delta':delta,
          'prefill_tok_s':len(tokens)/delta[METRICS[1]],
          'decode_tok_s':(usage['completion_tokens']-1)/delta[METRICS[2]] if delta[METRICS[2]]>0 else None}
(a.out/'meeting-112k.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
assert scored['passed'] and usage['prompt_tokens']==112000, record
status('quality after long requests')
invoke('check_cache_request_reuse.py', '--base-url', base, '--model', a.model,
       '--out', str(a.out/'quality-after.json'))
quality = json.loads((a.out/'quality-after.json').read_text())
assert quality['strict_matches'] == 8 and quality['reuse']['identical_text'] and not quality['reuse']['canary_leaked']
(a.out/'passed.json').write_text(json.dumps({'passed': True, 'protocol':'health;8 toy questions+A/B/A;32K warmup+2;112K warmup+1;112K meeting;8 toy questions+A/B/A'},indent=2))
status('all validation gates passed')
