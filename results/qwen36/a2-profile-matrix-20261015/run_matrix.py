#!/usr/bin/env python3
"""在a2串行执行2×2参数对照，退出时恢复托管服务。"""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path('/home/ychy/services/qwen36/c16-20261015/repo')
OUT = Path('/home/ychy/ascend-lab-runtime/a2-profile-matrix-20261015')
BASE = 'http://127.0.0.1:18024'
NAME = 'qwen36-profile-matrix'

def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)

def status(message):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)

def wait_health(process=None):
    deadline = time.monotonic() + 1000
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f'candidate exited {process.returncode}')
        try:
            with urllib.request.urlopen(BASE + '/health', timeout=3) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError('health timeout')

OUT.mkdir(parents=True, exist_ok=True)
try:
    run(['systemctl', '--user', 'stop', 'qwen36-a2.service'])
    for drafts, batch in [(2,4096), (1,2048), (1,4096), (2,2048)]:
        label = f'mtp{drafts}-b{batch}'
        dest = OUT / label
        dest.mkdir()
        env = os.environ.copy()
        env.update(QWEN36_PROFILE='compat' if drafts == 1 else 'long-prefill',
                   QWEN36_C8_ENABLED='0', QWEN36_MINDIEQ_ENABLED='1',
                   QWEN36_MINDIEQ_CUSTOM_QUANT='1', QWEN36_MINDIEQ_SHARED_GATE='1',
                   QWEN36_MINDIEQ_MIN_TOKENS='128', MAX_BATCHED_TOKENS=str(batch),
                   MAX_MODEL_LEN='114688', MEMORY_UTILIZATION='0.95', MAX_NUM_SEQS='1',
                   ENABLE_PREFIX_CACHING='0')
        command = 'source deploy/a2.env; export CONTAINER_NAME=qwen36-profile-matrix; exec bash scripts/run-candidate.sh'
        with (dest / 'service.log').open('w') as log:
            status(f'{label}: starting')
            process = subprocess.Popen(['bash','-c',command], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_health(process)
                status(f'{label}: ready')
                with (dest/'container.json').open('w') as f:
                    run(['podman','inspect',NAME], stdout=f)
                with (dest/'npu-before.txt').open('w') as f:
                    run(['npu-smi','info'],stdout=f)
                for tokens in (8192,32768):
                    status(f'{label}: benchmark {tokens}')
                    run(['python3',str(OUT/'bench_server_phases.py'), '--base-url',BASE,
                         '--model','qwen36-w4a8-c8', '--out',str(dest/f'p{tokens}-o128.jsonl'),
                         '--prompt-tokens',str(tokens), '--output-tokens','128', '--repeats','2',
                         '--unique-prompts','--spec-metrics', '--cache-salt-prefix',f'matrix-{label}-{tokens}'])
                with (dest/'npu-after.txt').open('w') as f:
                    run(['npu-smi','info'],stdout=f)
                status(f'{label}: complete')
            finally:
                subprocess.run(['podman','stop','-t','90',NAME],check=False)
                process.wait(timeout=120)
finally:
    status('restoring qwen36-a2.service')
    managed_name = 'qwen36-tp1-48g-112k'
    if subprocess.run(['podman', 'container', 'exists', managed_name]).returncode == 0:
        saved = run(['podman', 'inspect', managed_name], capture_output=True, text=True).stdout
        info = json.loads(saved)[0]
        if info['State']['Status'] == 'exited' and not info['State']['Running']:
            (OUT / 'managed-stopped-before-recovery.json').write_text(saved)
            with (OUT / 'managed-stopped-before-recovery.txt').open('w') as log:
                run(['podman', 'logs', managed_name], stdout=log, stderr=subprocess.STDOUT)
            run(['podman', 'rm', managed_name])
    run(['systemctl','--user','start','qwen36-a2.service'])
    wait_health()
    status('managed service restored')
