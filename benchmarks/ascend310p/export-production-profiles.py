#!/usr/bin/env python3
"""同镜像离线解析daemon worker数据；不加载模型，msprof需驱动设备元数据。"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--container', required=True)
p.add_argument('--label', default='offline-export')
a = p.parse_args()
image = json.loads(a.manifest.read_text())['image']
code = 'from torch_npu.profiler.profiler import analyse; analyse("/experiment/profiles")'
command = ['podman', 'run', '--rm', '--name', a.container, '--network', 'none',
           '--entrypoint', '/bin/bash', '--group-add', 'keep-groups',
           '-e', 'OMP_NUM_THREADS=1', '-v', f'{a.out.resolve()}:/experiment:rw',
           '-v', '/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro']
# msprof初始化平台时会访问驱动，即使只执行离线export。
for device in ('davinci0', 'davinci1', 'davinci_manager', 'devmm_svm', 'hisi_hdc'):
    command += ['--device', '/dev/' + device]
for source in ('/etc/ascend_install.info', '/usr/local/Ascend/driver/version.info', '/usr/local/dcmi'):
    command += ['-v', f'{source}:{source}:ro']
command += [image, '--noprofile', '--norc', '-c',
            'source /usr/local/Ascend/ascend-toolkit/set_env.sh && python3 -c ' + shlex.quote(code)]
with (a.out / f'{a.label}-command.json').open('x') as stream:
    json.dump(command, stream, indent=2)
begin = time.monotonic()
with (a.out / f'{a.label}.log').open('x') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
files = sorted((a.out / 'profiles').rglob('kernel_details.csv'))
report = dict(returncode=result.returncode, elapsed_s=time.monotonic()-begin,
              kernel_files=[str(path) for path in files],
              scope='offline CPU analysis of captured data, same image/CANN; driver metadata accessible, no model inference')
with (a.out / f'{a.label}-result.json').open('x') as stream:
    json.dump(report, stream, indent=2)
if result.returncode or len(files) != 4 or any(path.stat().st_size == 0 for path in files):
    raise RuntimeError(f'Expected four nonempty kernel exports (2 scopes × 2 ranks): {report}')
print(json.dumps(report), flush=True)
