#!/usr/bin/env python3
"""在远端独立目录顺序执行门禁与基线，保留每步退出码。"""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
BASE = 'http://127.0.0.1:18103'
MODEL = 'qwen36-w4a8-c8'
STATUS = ROOT / 'status.json'
steps = []


def status(state, **extra):
    STATUS.write_text(json.dumps({'state': state, 'steps': steps, **extra}, indent=2) + '\n')


def run(label, script, *args):
    status('running', current=label)
    started = time.time()
    with (ROOT / (label + '.log')).open('x') as output:
        code = subprocess.run(['python3', str(ROOT / script), *map(str, args)],
                              stdout=output, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    steps.append({'label': label, 'returncode': code, 'elapsed_s': time.time() - started})
    if code:
        status('failed', current=label)
        raise SystemExit(code)


status('waiting_for_health')
for attempt in range(180):
    try:
        with urllib.request.urlopen(BASE + '/health', timeout=3) as response:
            if response.status == 200:
                break
    except Exception:
        pass
    time.sleep(5)
else:
    status('failed', current='startup_health_timeout')
    raise SystemExit(1)

common = ['--base-url', BASE, '--model', MODEL]
run('quality-before', 'check_cache_request_reuse.py', *common, '--out', ROOT / 'quality-before.json')
quality = json.loads((ROOT / 'quality-before.json').read_text())
if (quality['strict_matches'] != 8 or not quality['reuse']['identical_text']
        or quality['reuse']['canary_leaked']):
    status('failed', current='quality-before-gate')
    raise SystemExit(1)
for length in (1024, 4096, 16384, 32768):
    label = f'c1-p{length}-o256'
    run(label, 'bench_server_phases.py', *common, '--prompt-tokens', length,
        '--output-tokens', 256, '--repeats', 3, '--spec-metrics',
        '--cache-salt-prefix', f'baseline-20261103-{length}', '--out', ROOT / (label + '.jsonl'))
run('quality-after-long', 'check_cache_request_reuse.py', *common, '--out', ROOT / 'quality-after-long.json')
quality = json.loads((ROOT / 'quality-after-long.json').read_text())
if (quality['strict_matches'] != 8 or not quality['reuse']['identical_text']
        or quality['reuse']['canary_leaked']):
    status('failed', current='quality-after-long-gate')
    raise SystemExit(1)
dataset = Path('/path/to/lab-runtime/qwen36-public-release-20261012/gsm8k-test.jsonl')
expected = '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'
if hashlib.sha256(dataset.read_bytes()).hexdigest() != expected:
    status('failed', current='gsm-dataset-hash')
    raise SystemExit(1)
run('gsm64', 'eval_gsm8k_http.py', *common, '--label', 'frozen-c16-mtp2',
    '--dataset', dataset, '--limit', 64, '--seed', 310, '--max-tokens', 512,
    '--out', ROOT / 'gsm64.jsonl')
status('complete')
