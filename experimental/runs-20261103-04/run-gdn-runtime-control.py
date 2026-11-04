#!/usr/bin/env python3
"""Compare complete runtime images using unchanged production GDN and CPU oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
root = a.root.resolve()
out = root / 'gdn-runtime-control'
out.mkdir(exist_ok=False)
shutil.copy2(__file__, out / 'supervisor-used.py')
for name in ('repro_chunk_gdn_nondeterminism.py', 'probe_gdn_intermediates.py'):
    shutil.copy2(root / 'native-gdn/tools' / name, out / name)
(out / 'probe-freeze.json').write_text(json.dumps({name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('repro_chunk_gdn_nondeterminism.py', 'probe_gdn_intermediates.py')}, indent=2)+'\n')
steps = []
paused = False
finished = False
primary_error = None


def state(value, **extra):
    temporary = out / 'status.tmp'
    temporary.write_text(json.dumps(dict(state=value, steps=steps, **extra), indent=2)+'\n')
    temporary.replace(out / 'status.json')


try:
    previous = json.loads((root/'dense-quant-captured-replay-synthetic-allocator-paired/status.json').read_text())
    if previous['state'] not in ('complete', 'failed') or not (root/'dense-quant-captured-replay-synthetic-allocator-paired/restore-a2-service.log').exists():
        raise RuntimeError('Synthetic replay controller has not completed cleanup')
    state('checking_idle')
    paused = True
    subprocess.run(['sudo', '-n', 'systemctl', 'stop', 'llm-backend-primary.service'], check=True)
    idle = 0
    for _ in range(90):
        info = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True)
        idle = idle+1 if 'No running processes found' in info.stdout else 0
        if idle >= 2:
            break
        time.sleep(2)
    else:
        raise RuntimeError('NPU not idle')
    template = json.loads((root/'native-gdn/validation-command.json').read_text())
    original = json.loads((root/'production-manifest.json').read_text())['image']
    post4 = 'b381d8e5ed45c11633a7bd07a50a0f5fcf8cf71bc797f1b543d589ab07dd759d'
    inspected = json.loads(subprocess.check_output(['podman', 'image', 'inspect', post4], text=True))
    if inspected[0]['Id'] != post4:
        raise RuntimeError('Compatibility image identity mismatch')
    for label, image in [('production-post2', original), ('audited-post4', post4)]:
        command = template.copy()
        container = 'gdn-runtime-control-'+label+'-'+str(time.time_ns())
        command[command.index('--name')+1] = container
        index = command.index(original)
        command[index] = image
        command[index:index] = ['-v',str(out)+':/runtime-control:rw','-e','PYTHONPATH=','-e','QWEN36_C8_ENABLED=0']
        command[-1] = f'''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
unset ACLNN_CACHE_LIMIT PYTORCH_NPU_ALLOC_CONF
python3 /runtime-control/repro_chunk_gdn_nondeterminism.py --out /runtime-control/{label}-repro.jsonl --repeats 48
python3 /runtime-control/probe_gdn_intermediates.py --work /work --out-dir /runtime-control/{label}-chain
'''
        (out/(label+'-command.json')).write_text(json.dumps(command,indent=2)+'\n')
        state('running', current=label)
        begin = time.monotonic()
        try:
            with (out/(label+'.log')).open('x') as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=240)
        except subprocess.TimeoutExpired:
            subprocess.run(['podman','stop','--time','5',container],capture_output=True,timeout=30)
            raise TimeoutError(label+' exceeded240s; no quality conclusion')
        steps.append(dict(label=label, returncode=result.returncode, elapsed_s=time.monotonic()-begin))
        if result.returncode:
            raise RuntimeError(label+' failed to finish diagnostic')
    finished = True
except Exception as error:
    primary_error = str(error)
    state('failed', error=primary_error)
    raise
finally:
    if paused:
        with (out/'restore-a2-service.log').open('x') as log:
            result = subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            finished = False
            state('failed',error='Could not restore a2 temporary unit',primary_error=primary_error)
    if finished:
        state('complete',scope='Whole-image compatibility diagnostic only; inspect independent CPU-oracle metrics, not process exit codes')
