#!/usr/bin/env python3
"""基线结束后在同一芯片顺序采集，Profile计时不进入Benchmark。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

root = Path(__file__).resolve().parent
profile = root / 'profile-c16'
profile.mkdir(exist_ok=True)
state_path = profile / 'status.json'
container = 'qwen36-profile-20261103'
steps = []


def state(value, **kwargs):
    state_path.write_text(json.dumps({'state': value, 'steps': steps, **kwargs}, indent=2) + '\n')


def run(label, command, **kwargs):
    state('running', current=label)
    with (profile / (label + '.log')).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    steps.append({'label': label, 'returncode': result.returncode})
    if result.returncode:
        raise RuntimeError(f'{label}: returncode={result.returncode}')


state('waiting_for_baseline')
for _ in range(1440):
    current = json.loads((root / 'status.json').read_text())
    if current['state'] == 'failed':
        state('blocked', reason='baseline runner failed; no automatic restart')
        raise SystemExit(1)
    if current['state'] == 'complete':
        break
    time.sleep(5)
else:
    state('blocked', reason='baseline wait exceeded two hours')
    raise SystemExit(1)
try:
    # Only this session's isolated container is stopped.
    run('baseline-log', ['podman', 'logs', 'qwen36-baseline-20261103'])
    run('checkpoint-hashes', ['python3', str(root / 'validate_checkpoint.py'),
        '/path/to/lab-runtime/qwen36-public-release-20261012/model', '--verify-hashes'])
    run('stop-baseline', ['podman', 'stop', '-t', '90', 'qwen36-baseline-20261103'])
    shutil.copy2(root / 'start-experiment.sh', profile / 'start-experiment.sh')
    environment = dict(os.environ, EXPERIMENT_DIR=str(profile), CONTAINER_NAME=container,
                       PORT='18104', EXPERIMENT_START_SCRIPT='1', EXPERIMENT_PROFILE='1',
                       EXPERIMENT_MTP_TOKENS='2')
    run('launch', ['bash', str(root / 'launch-frozen.sh')], env=environment)
    base = 'http://127.0.0.1:18104'
    for _ in range(180):
        try:
            with urllib.request.urlopen(base + '/health', timeout=3) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        time.sleep(5)
    else:
        raise RuntimeError('profile container health timeout')
    common = ['--base-url', base, '--model', 'qwen36-w4a8-c8']
    run('quality-before', ['python3', str(root / 'check_cache_request_reuse.py'), *common,
                          '--out', str(profile / 'quality-before.json')])
    quality = json.loads((profile / 'quality-before.json').read_text())
    if quality['strict_matches'] != 8 or not quality['reuse']['identical_text'] or quality['reuse']['canary_leaked']:
        raise RuntimeError('profile container quality gate failed')
    run('prefill-4096', ['python3', str(root / 'profile_http_prefill.py'), *common,
                        '--prompt-tokens', '4096', '--out', str(profile / 'prefill-4096.jsonl')])
    run('request-4096-128', ['python3', str(root / 'profile_http_decode_full.py'), *common,
                            '--prompt-tokens', '4096', '--output-tokens', '128',
                            '--out', str(profile / 'request-4096-128.jsonl')])
    run('decode-32768-128', ['python3', str(root / 'profile_http_decode.py'), *common,
                            '--prompt-tokens', '32768', '--output-tokens', '128',
                            '--out', str(profile / 'decode-32768-128.jsonl')])
    run('quality-after', ['python3', str(root / 'check_cache_request_reuse.py'), *common,
                         '--out', str(profile / 'quality-after.json')])
    exported = list((profile / 'profiles').rglob('kernel_details.csv'))
    if not exported:
        raise RuntimeError('profiler produced no kernel_details.csv; inspect export logs')
    for i, path in enumerate(exported):
        run(f'analyze-{i}', ['python3', str(root / 'analyze_decode_kernels.py'),
                            '--input', str(path), '--out', str(profile / f'kernel-summary-{i}.json')])
    state('complete', kernel_files=[str(p) for p in exported])
except Exception as exc:
    state('failed', error=str(exc))
    raise
finally:
    with (profile / 'service-final.log').open('w') as log:
        subprocess.run(['podman', 'logs', container], stdout=log, stderr=subprocess.STDOUT)
    subprocess.run(['podman', 'stop', '-t', '90', container], stdout=subprocess.DEVNULL)
