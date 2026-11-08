"""One isolated device window, one fresh Python process per arithmetic case."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/ychy/ascend-lab-runtime/hyperqwen-20261103')
sys.path.insert(0, str(ROOT))
import run_w8_mask_fix_20261108 as common
work = ROOT / 'graph-localize/pr18730-candidate-context'
out = ROOT / 'graph-localize/pr18730-slot-mul'
out.mkdir()
start = time.monotonic()
name = 'pr18730-slot-mul'
with open('/tmp/a310-npu.lock', 'a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = common.pdf.snapshot()
    common.write(out / 'window-start.json', before)
    assert before['rootless']['stdout'] == before['rootful']['stdout'] == ''
    assert 'No running processes' in before['npu']['stdout']
    cmd = ['podman', 'run', '--name', name, '--pull=never', '--network=none',
           '--group-add', 'keep-groups', '--shm-size', '2g', '--entrypoint', 'bash',
           '-e', 'TORCH_DEVICE_BACKEND_AUTOLOAD=0', '-e', 'ATEN_CPU_CAPABILITY=default']
    for d in ['davinci0', 'davinci_manager', 'devmm_svm', 'hisi_hdc']:
        cmd += ['--device', '/dev/' + d]
    for path in ['/usr/local/Ascend/driver/lib64', '/usr/local/Ascend/driver/version.info',
                 '/usr/local/dcmi', '/etc/ascend_install.info']:
        cmd += ['-v', path + ':' + path + ':ro']
    cmd += ['-v', str(work) + ':/audit:ro', '-v', str(out) + ':/out:rw',
            'localhost/qwen36-w8-wy-mtp-pr18730:20261108', '-c',
            'source /usr/local/Ascend/ascend-toolkit/set_env.sh; python3 /audit/run_slot_cases.py']
    common.write(out / 'command.json', cmd)
    code = None
    error = None
    try:
        with (out / 'repro.log').open('w') as log:
            code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=360).returncode
    except Exception as exc:
        error = str(exc)
    finally:
        subprocess.run(['podman', 'rm', '-f', name], capture_output=True, timeout=45)
        after = common.pdf.snapshot()
        clean = after['rootless']['stdout'] == after['rootful']['stdout'] == '' and 'No running processes' in after['npu']['stdout']
        common.write(out / 'status.json', {'exit_code': code, 'error': error,
            'elapsed_s': time.monotonic() - start, 'cleanup_passed': clean,
            'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [work / 'repro_slot_mul.py', work / 'run_slot_cases.py', Path(__file__)]},
            'snapshot': after})
