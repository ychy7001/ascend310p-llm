#!/usr/bin/env python3
"""固定9.2.1完整镜像的默认算子筛选；不改宿主栈，不加载权重。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--image-id', required=True)
p.add_argument('--attempt', default='v1', help='distinct result directory; never overwrite an earlier attempt')
a = p.parse_args()
if not re.fullmatch(r'[0-9a-f]{64}', a.image_id):
    p.error('image-id must be the complete immutable image ID')
root = a.root.resolve()
if not re.fullmatch(r'[a-zA-Z0-9_-]+', a.attempt):
    p.error('invalid attempt label')
out = root / 'cann921-upgrade' / ('operator-screen-' + a.attempt)
out.mkdir(exist_ok=False)
(out / 'cache').mkdir()
for name in ('repro_chunk_gdn_nondeterminism.py', 'replay_dense_quant_capture.py'):
    shutil.copy2(root / 'native-gdn/tools' / name, out / name)
shutil.copy2(__file__, out / 'supervisor-used.py')
(out / 'probe-freeze.json').write_text(json.dumps({f.name: hashlib.sha256(f.read_bytes()).hexdigest()
    for f in out.glob('*.py')}, indent=2) + '\n')
steps = []
paused = False
container = None

def state(value, **extra):
    tmp = out / 'status.tmp'
    tmp.write_text(json.dumps(dict(state=value, steps=steps, **extra), indent=2) + '\n')
    tmp.replace(out / 'status.json')

def interrupted(signum, frame):
    raise RuntimeError(f'Supervisor interrupted: {signum}')

signal.signal(signal.SIGTERM, interrupted)
try:
    inspected = json.loads(subprocess.check_output(['podman', 'image', 'inspect', a.image_id], text=True))
    if inspected[0]['Id'] != a.image_id:
        raise RuntimeError('Image identity mismatch')
    (out / 'image-inspect.json').write_text(json.dumps(inspected, indent=2) + '\n')
    inventory = json.loads((root / 'cann921-upgrade' / 'runtime-inventory.json').read_text())
    for name, version in {'vllm': '0.30.0', 'vllm-ascend': '0.30.0', 'torch-npu': '2.10.0.post7'}.items():
        if inventory['packages'][name].split('+', 1)[0] != version:
            raise RuntimeError(f'Unexpected package: {name}={inventory["packages"][name]}')
    if not inventory['packages']['torch'].startswith('2.10.0'):
        raise RuntimeError('Unexpected torch version')
    if not any('9.2.1' in value for value in inventory['version_files'].values()):
        raise RuntimeError('Missing CANN 9.2.1 inventory evidence')
    state('checking_idle')
    paused = True
    subprocess.run(['sudo', '-n', 'systemctl', 'stop', 'llm-backend-primary.service'], check=True)
    idle = 0
    for _ in range(90):
        info = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True)
        idle = idle + 1 if 'No running processes found' in info.stdout else 0
        if idle >= 2:
            break
        time.sleep(2)
    else:
        raise RuntimeError('NPU not idle')
    cases = [('gdn-device0', None, 0, None), ('gdn-device1', None, 1, None)]
    cases += [(f'dense-seed{seed}-{allocator}', seed, 0, allocator)
              for seed in range(310, 315) for allocator in ('False', 'True')]
    for label, seed, device, allocator in cases:
        container = 'cann921-' + label + '-' + str(time.time_ns())
        command = ['podman', 'run', '--name', container, '--pull=never', '--network', 'none',
                   '--entrypoint', '/bin/bash', '--group-add', 'keep-groups', '--shm-size', '8g',
                   '-v', str(out) + ':/screen:rw', '-v', str(out / 'cache') + ':/root/.cache:rw']
        for name, value in {'OMP_NUM_THREADS': '1', 'SOC_VERSION': 'ascend310p1', 'TASK_QUEUE_ENABLE': '1',
                            'ASCEND_RT_VISIBLE_DEVICES': str(device), 'ASCEND_VISIBLE_DEVICES': str(device),
                            'PYTHONPATH': ''}.items():
            command += ['-e', f'{name}={value}']
        # Devices stay visible as 0,1; the probe selects the logical device explicitly.
        for flag in ('ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES'):
            command[command.index(f'{flag}={device}')] = f'{flag}=0,1'
        for node in ('davinci0', 'davinci1', 'davinci_manager', 'devmm_svm', 'hisi_hdc'):
            command += ['--device', '/dev/' + node]
        for path in ('/usr/local/Ascend/driver/lib64', '/usr/local/Ascend/driver/version.info',
                     '/usr/local/dcmi', '/etc/ascend_install.info'):
            command += ['-v', f'{path}:{path}:ro']
        setup = '''set -eo pipefail
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then source /usr/local/Ascend/ascend-toolkit/set_env.sh; else source /usr/local/Ascend/cann/set_env.sh; fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then source /usr/local/Ascend/nnal/atb/set_env.sh; fi
unset ACLNN_CACHE_LIMIT ASCEND_CUSTOM_OPP_PATH
'''
        if seed is None:
            setup += f'python3 /screen/repro_chunk_gdn_nondeterminism.py --device {device} --out /screen/{label}.jsonl --repeats 48\n'
        else:
            setup += f'export ASCEND_LAUNCH_BLOCKING=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:{allocator}\n'
            setup += f'python3 /screen/replay_dense_quant_capture.py --synthetic --seed {seed} --device {device} --out /screen/{label}.json\n'
        command += [a.image_id, '--noprofile', '--norc', '-c', setup]
        (out / (label + '-command.json')).write_text(json.dumps(command, indent=2) + '\n')
        state('running', current=label)
        begin = time.monotonic()
        timed_out = False
        with (out / (label + '.log')).open('x') as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240)
                code = result.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                code = None
                subprocess.run(['podman', 'stop', '--time', '5', container], capture_output=True, timeout=30)
        steps.append(dict(label=label, returncode=code, timed_out=timed_out, elapsed_s=time.monotonic() - begin))
        if timed_out:
            raise TimeoutError(label + ' exceeded240s; distinguish init failure from operator execution')
        marker = out / (label + ('.jsonl' if seed is None else '.json'))
        if not marker.exists():
            raise RuntimeError(label + ' failed before diagnostic output; stop further workloads')
    state('diagnostics_complete_requires_adjudication', scope='No model, performance or production acceptance')
except Exception as error:
    state('failed', error=str(error))
    raise
finally:
    if container:
        subprocess.run(['podman', 'stop', '--time', '5', container], capture_output=True, timeout=30)
    if paused:
        with (out / 'restore-a2-service.log').open('x') as log:
            subprocess.run(['sudo', '-n', 'systemctl', 'reset-failed', 'llm-backend-primary.service'], stdout=log, stderr=subprocess.STDOUT)
            restored = subprocess.run(['sudo', '-n', 'systemctl', 'start', 'llm-backend-primary.service'], stdout=log, stderr=subprocess.STDOUT)
        if restored.returncode:
            state('failed', error='Temporary service restoration failed')
