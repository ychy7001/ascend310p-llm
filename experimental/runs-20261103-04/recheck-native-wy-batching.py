#!/usr/bin/env python3
"""A′ 完成后拆分数值路径，复测原三并发纪要序列，最后单独采集候选 Profile。"""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
root = a.root.resolve()
control = root / 'production-wy-control-recheck'
native = root / 'native-gdn'
b = root / 'production-native-wy'
out = root / 'production-wy-batching-recheck'
out.mkdir(exist_ok=False)
container = 'qwen36-production-native-wy-20261103'
base = 'http://127.0.0.1:18108'
model = 'Qwen3.6-35B-A3B-w8a8'
steps = []
paused = False
started = False
finished = False


def state(value, **extra):
    f = out / 'status.tmp'
    f.write_text(json.dumps(dict(state=value, steps=steps, **extra), indent=2) + '\n')
    f.replace(out / 'status.json')


def run(label, command):
    state('running', current=label)
    begin = time.monotonic()
    with (out / (label + '.log')).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    steps.append(dict(label=label, returncode=result.returncode, elapsed_s=time.monotonic()-begin))
    if result.returncode:
        raise RuntimeError(label + ' failed')


def idle():
    count = 0
    for _ in range(90):
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True)
        count = count + 1 if 'No running processes found' in result.stdout else 0
        if count >= 2:
            return
        time.sleep(2)
    raise RuntimeError('NPU cleanup incomplete')


try:
    state('waiting_control')
    for _ in range(1440):
        j = json.loads((control / 'status.json').read_text())
        if j['state'] in ('failed', 'not_started'):
            state('not_started', reason='Previous control did not complete')
            raise SystemExit(0)
        if j['state'] == 'complete':
            break
        time.sleep(5)
    else:
        raise TimeoutError('control')
    paused = True
    run('pause-a2-temporary-unit', ['sudo', '-n', 'systemctl', 'stop', 'llm-backend-primary.service'])
    idle()
    command = json.loads((native / 'validation-command.json').read_text())
    command[command.index('--name')+1] = 'qwen36-native-wy-isolate-stages-20261103'
    command[-1] = '''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="/work/opp/vendors/hyperqwen_wy_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"
cd /work/tools
python3 isolate_wy_stages.py --work /work --out-dir /work/stage-isolation
'''
    run('isolate-wy-stages', command)
    idle()
    started = True
    run('restart-candidate', ['podman', 'start', container])
    state('waiting_health')
    for _ in range(240):
        try:
            with urllib.request.urlopen(base + '/health', timeout=3) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        r = subprocess.run(['podman', 'inspect', '--format', '{{.State.Status}}', container],
                           capture_output=True, text=True)
        if r.stdout.strip() == 'exited':
            raise RuntimeError('Candidate exited during startup')
        time.sleep(5)
    else:
        raise TimeoutError('candidate health')
    # The original quality run uses three workers. Serial case4 requests alone
    # cannot reproduce its batch/scheduling history, so repeat the same full suite.
    for repeat in range(3):
        run(f'business-quality-{repeat}', ['python3', str(root / 'check_business_quality.py'),
            '--url', base, '--model', model,
            '--fixtures-dir', '/path/to/lab-runtime/a1-310p-tune-20261101',
            '--out', str(out / f'business-quality-{repeat}.jsonl')])
    common = ['--base-url', base, '--model', model]
    run('profile-prefill-4096', ['python3', str(root / 'profile_http_prefill.py'), *common,
        '--prompt-tokens', '4096', '--out', str(out / 'profile-prefill-4096.jsonl')])
    run('profile-decode-32768', ['python3', str(root / 'profile_http_decode.py'), *common,
        '--prompt-tokens', '32768', '--output-tokens', '128',
        '--out', str(out / 'profile-decode-32768.jsonl')])
    run('quality-after-profile', ['python3', str(root / 'check_cache_request_reuse.py'), *common,
        '--out', str(out / 'quality-after-profile.json')])
    q = json.loads((out / 'quality-after-profile.json').read_text())
    if q['strict_matches'] != 8 or not q['reuse']['identical_text'] or q['reuse']['canary_leaked']:
        raise RuntimeError('Post-profile short quality failed')
    run('stop-candidate', ['podman', 'stop', '--time', '90', container])
    started = False
    idle()
    # The existing container's profiler mount points to the original B directory.
    run('profile-export', ['python3', str(root / 'export-production-profiles.py'),
        '--out', str(b), '--manifest', str(root / 'production-manifest.json'),
        '--container', 'qwen36-native-wy-profile-export-20261103', '--label', 'wy-profile-offline-export'])
    files = sorted((b / 'profiles').rglob('kernel_details.csv'))
    artifacts = []
    for index, path in enumerate(files):
        run(f'analyze-{index}', ['python3', str(root / 'analyze_decode_kernels.py'),
            '--input', str(path), '--out', str(out / f'kernel-summary-{index}.json')])
        artifacts.append(dict(index=index, csv=str(path), summary=f'kernel-summary-{index}.json'))
    (out / 'profile-artifacts.json').write_text(json.dumps(artifacts, indent=2) + '\n')
    finished = True
except Exception as e:
    state('failed', error=str(e))
    raise
finally:
    if started:
        subprocess.run(['podman', 'stop', '--time', '90', container], capture_output=True, text=True)
    if paused:
        with (out / 'candidate-server.log').open('w') as log:
            subprocess.run(['podman', 'logs', container], stdout=log, stderr=subprocess.STDOUT)
        r = subprocess.run(['podman', 'inspect', '--format', '{{.State.Status}}', container],
                           capture_output=True, text=True)
        if r.stdout.strip() != 'exited':
            finished = False
            state('failed', error='Candidate cleanup incomplete')
        with (out / 'restore-a2-service.log').open('x') as log:
            r = subprocess.run(['sudo', '-n', 'systemctl', 'start', 'llm-backend-primary.service'],
                               stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            finished = False
            state('failed', error='Could not restore a2 temporary service')
    if finished:
        state('complete', scope='Numerical stage diagnostics, three concurrent business suites and separate profile; semantic review required')
