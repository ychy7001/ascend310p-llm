#!/usr/bin/env python3
"""评测器误报后仅继续未运行阶段；原始失败状态与已有效性能数据保留。"""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request

root = Path(__file__).resolve().parent
output = root / 'production-w8a8'
state_path = output / 'status.json'
steps = []
container = 'qwen36-production-replica-20261103'
base = 'http://127.0.0.1:18106'
model = 'Qwen3.6-35B-A3B-w8a8'
common = ['--base-url', base, '--model', model]
fixtures = Path('/path/to/lab-runtime/a1-310p-tune-20261101')


def state(value, **extra):
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'state': value, 'steps': steps, **extra}, indent=2) + '\n')
    temporary.replace(state_path)


def run(label, command, **kwargs):
    state('running', current=label)
    begin = time.monotonic()
    with (output / (label + '.log')).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    steps.append({'label': label, 'returncode': result.returncode,
                  'elapsed_s': time.monotonic() - begin})
    if result.returncode:
        raise RuntimeError(f'{label}: returncode={result.returncode}')


def quality(label):
    run(label, ['python3', str(root / 'check_cache_request_reuse.py'), *common,
                '--out', str(output / (label + '.json'))])
    result = json.loads((output / (label + '.json')).read_text())
    if result['strict_matches'] != 8 or not result['reuse']['identical_text'] or result['reuse']['canary_leaked']:
        raise RuntimeError(f'{label}: short gate failed')


original_state = json.loads((output / 'status-initial-fact-gate-failed.json').read_text())
steps = original_state['steps'] + [dict(label='business-facts-regrade', returncode=0,
    note='14/16 exact; 16/16 facts after allowing one leading 负责 in task text; raw outputs retained')]
try:
    run('restart-after-evaluator-fix', ['podman', 'start', container])
    state('waiting_for_health')
    for _ in range(240):
        try:
            with urllib.request.urlopen(base + '/health', timeout=3) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        inspect = subprocess.run(['podman', 'inspect', '--format', '{{.State.Status}}', container],
                                 capture_output=True, text=True)
        if inspect.returncode == 0 and inspect.stdout.strip() == 'exited':
            raise RuntimeError('replica exited during startup; inspect service-final.log')
        time.sleep(5)
    else:
        raise RuntimeError('replica health timeout')
    quality('quality-after-evaluator-fix')
    # One representative full workload establishes a business wall-clock baseline.
    run('business-replay-long', ['python3', str(root / 'replay_stage_workload.py'),
                                 '--workload', str(fixtures / 'a1-310p-tune.workload.json'),
                                 '--recording', 'chinese_company_annual_meeting_planning',
                                 '--urls', base, '--model', model,
                                 '--prompts', str(output / 'replay-prompts.json'),
                                 '--out', str(output / 'business-replay-long.json')])
    dataset = Path('/path/to/lab-runtime/qwen36-public-release-20261012/gsm8k-test.jsonl')
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14':
        raise RuntimeError('GSM dataset hash mismatch')
    run('gsm64', ['python3', str(root / 'eval_gsm8k_http.py'), *common,
                   '--dataset', str(dataset), '--limit', '64', '--seed', '310', '--max-tokens', '512',
                   '--label', 'production-w8a8-tp2', '--out', str(output / 'gsm64.jsonl')])
    # Profiler runs after ordinary performance windows.
    run('profile-prefill-4096', ['python3', str(root / 'profile_http_prefill.py'), *common,
                                '--prompt-tokens', '4096', '--out', str(output / 'profile-prefill-4096.jsonl')])
    run('profile-decode-32768', ['python3', str(root / 'profile_http_decode.py'), *common,
                                '--prompt-tokens', '32768', '--output-tokens', '128',
                                '--out', str(output / 'profile-decode-32768.jsonl')])
    quality('quality-after-profile')
    exported = sorted((output / 'profiles').rglob('kernel_details.csv'))
    if len(exported) != 4:
        run('profile-export-controller', ['python3', str(root / 'export-production-profiles.py'),
            '--out', str(output), '--manifest', str(root / 'production-manifest.json'),
            '--container', container + '-offline-export', '--label', 'profile-offline-export'])
        exported = sorted((output / 'profiles').rglob('kernel_details.csv'))
    if len(exported) != 4:
        raise RuntimeError('Expected 2 scopes × 2 ranks kernel exports')
    for index, path in enumerate(exported):
        run(f'analyze-{index}', ['python3', str(root / 'analyze_decode_kernels.py'),
                                '--input', str(path), '--out', str(output / f'kernel-summary-{index}.json')])
    state('complete', kernel_files=[str(path) for path in exported])
except Exception as exc:
    state('failed', error=str(exc))
    raise
finally:
    with (output / 'service-final.log').open('w') as log:
        subprocess.run(['podman', 'logs', container], stdout=log, stderr=subprocess.STDOUT)
    subprocess.run(['podman', 'stop', '-t', '90', container], stdout=subprocess.DEVNULL)
