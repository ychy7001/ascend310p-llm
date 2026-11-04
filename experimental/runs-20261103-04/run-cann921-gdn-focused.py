#!/usr/bin/env python3
"""扩展门禁发现T512偏差后，独立进程复核生产及Dense头数；不加载模型。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
a = p.parse_args(); root = a.root.resolve()
out = root / 'cann921-upgrade/gdn-focused-v1'; out.mkdir(exist_ok=False)
shutil.copy2(root / 'native-gdn/tools/repro_chunk_gdn_nondeterminism.py', out / 'probe.py')
shutil.copy2(__file__, out / 'supervisor-used.py')
(out / 'freeze.json').write_text(json.dumps({'probe_sha256': hashlib.sha256((out / 'probe.py').read_bytes()).hexdigest()}, indent=2) + '\n')
prior = root / 'cann921-upgrade/a3b-quality-v1'
if json.loads((prior / 'status.json').read_text())['state'] != 'failed' or not (prior / 'restore-a2-service.log').exists():
    raise RuntimeError('Previous controller cleanup incomplete')
steps = []; paused = False; container = None

def state(value, **extra):
    tmp = out / 'status.tmp'; tmp.write_text(json.dumps(dict(state=value, steps=steps, **extra), indent=2) + '\n'); tmp.replace(out / 'status.json')
try:
    paused = True
    subprocess.run(['sudo', '-n', 'systemctl', 'stop', 'llm-backend-primary.service'], check=True)
    idle = 0
    for _ in range(90):
        r = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True)
        idle = idle + 1 if 'No running processes found' in r.stdout else 0
        if idle >= 2: break
        time.sleep(2)
    else: raise RuntimeError('NPU not idle')
    template = json.loads((root / 'cann921-upgrade/operator-screen-v3/gdn-device0-command.json').read_text())
    for hv in (16, 24):
        label = f't512-hv{hv}-seed318'
        cmd = template.copy(); container = 'cann921-focused-' + label + '-' + str(time.time_ns())
        cmd[cmd.index('--name') + 1] = container
        idx = cmd.index('d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a')
        cmd[idx:idx] = ['-v', str(out) + ':/focused:rw']
        cmd[-1] = cmd[-1][:cmd[-1].index('python3 ')] + f'python3 /focused/probe.py --out /focused/{label}.jsonl --repeats 48 --tokens 512 --seed 318 --value-heads {hv}\n'
        (out / (label + '-command.json')).write_text(json.dumps(cmd, indent=2) + '\n')
        state('running', current=label)
        begin = time.monotonic()
        try:
            with (out / (label + '.log')).open('x') as log:
                r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=240)
        except subprocess.TimeoutExpired:
            subprocess.run(['podman', 'stop', '--time', '5', container], capture_output=True, timeout=30)
            raise
        steps.append(dict(label=label, returncode=r.returncode, elapsed_s=time.monotonic() - begin))
        if r.returncode: raise RuntimeError(label + ' did not finish diagnostic')
    state('diagnostics_complete_requires_adjudication')
except Exception as error:
    state('failed', error=str(error)); raise
finally:
    if container: subprocess.run(['podman', 'stop', '--time', '5', container], capture_output=True, timeout=30)
    if paused:
        with (out / 'restore-a2-service.log').open('x') as log:
            subprocess.run(['sudo', '-n', 'systemctl', 'reset-failed', 'llm-backend-primary.service'], stdout=log, stderr=subprocess.STDOUT)
            r = subprocess.run(['sudo', '-n', 'systemctl', 'start', 'llm-backend-primary.service'], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode: state('failed', error='Could not restore a2 temporary service')
