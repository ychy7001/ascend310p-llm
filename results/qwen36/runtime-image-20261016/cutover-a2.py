"""Promote the validated digest to a standalone user unit; roll back on failure."""
import argparse
import gzip
import json
from pathlib import Path
import re
import runpy
import subprocess

ROOT = Path('/home/ychy/ascend-lab-runtime/runtime-image-20261016')
api = runpy.run_path(str(ROOT / 'run-a2.py'))
run, health, quality = (api[k] for k in ('run', 'health', 'quality'))
BASE = api['BASE']
parser = argparse.ArgumentParser()
parser.add_argument('image')
args = parser.parse_args()
assert re.fullmatch(r'registry\.cn-hangzhou\.aliyuncs\.com/meetai/llm-service-vllm-ascend@sha256:[0-9a-f]{64}', args.image)
assert json.loads((ROOT / 'outcome.json').read_text())['passed']
config = Path.home() / '.config/qwen36-image.env'
unit = Path.home() / '.config/systemd/user/qwen36-image.service'
assert not config.exists() and not unit.exists(), 'refuse to overwrite deployment configuration'
run('podman', 'image', 'inspect', args.image, capture_output=True)
verified = run('podman', 'run', '--rm', '--entrypoint', 'python3', args.image,
               '-S', '/opt/qwen36/verify-image.py', '--base', capture_output=True)
(ROOT / 'published-image-verify.json').write_text(verified.stdout)
old_enabled = subprocess.run(['systemctl', '--user', 'is-enabled', 'qwen36-a2.service'], text=True, capture_output=True).stdout.strip()
(ROOT / 'pre-cutover.json').write_text(json.dumps({'old_unit_enabled': old_enabled,
    'old_unit': run('systemctl', '--user', 'cat', 'qwen36-a2.service', capture_output=True).stdout}, indent=2))
values = {
    'IMAGE': args.image,
    'MODEL_DIR': '/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012/model',
    'PORT': '18024', 'QWEN36_PROFILE': 'long-prefill', 'QWEN36_C8_ENABLED': '0',
    'NUM_SPECULATIVE_TOKENS': '2', 'MAX_MODEL_LEN': '114688', 'MAX_BATCHED_TOKENS': '2048',
    'MEMORY_UTILIZATION': '0.95', 'MAX_NUM_SEQS': '1', 'ENABLE_PREFIX_CACHING': '0',
    'ASYNC_SCHEDULING': '1', 'LANGUAGE_MODEL_ONLY': '0', 'QWEN36_SYNC_METADATA': '0', 'OMP_NUM_THREADS': '1',
}
with config.open('x') as stream:
    stream.write(''.join(f'{k}={v}\n' for k, v in values.items()))
config.chmod(0o600)
with unit.open('xb') as stream:
    stream.write((ROOT / 'qwen36-image.service').read_bytes())
run('systemd-analyze', '--user', 'verify', str(unit))
run('systemctl', '--user', 'daemon-reload')
result = {'passed': False, 'image_reference': args.image, 'environment': values}
try:
    run('systemctl', '--user', 'disable', '--now', 'qwen36-a2.service')
    run('systemctl', '--user', 'enable', '--now', 'qwen36-image.service')
    health(BASE)
    quality(BASE, ROOT / 'managed-quality.json')
    meta = json.loads(run('podman', 'inspect', 'qwen36-image-managed', capture_output=True).stdout)[0]
    assert {m['Destination'] for m in meta['Mounts']} == {'/model', '/usr/local/Ascend/driver/lib64',
        '/usr/local/Ascend/driver/version.info', '/usr/local/dcmi', '/etc/ascend_install.info'}
    result.update(passed=True, health_status=200, image_id=meta['Image'], mounts=meta['HostConfig']['Binds'])
except Exception as error:
    result['error'] = repr(error)
    run('systemctl', '--user', 'disable', '--now', 'qwen36-image.service')
    if old_enabled == 'enabled':
        run('systemctl', '--user', 'enable', 'qwen36-a2.service')
    run('systemctl', '--user', 'start', 'qwen36-a2.service')
    health(BASE)
    quality(BASE, ROOT / 'cutover-rollback-quality.json')
    result['rolled_back'] = True
finally:
    result['running_containers'] = run('podman', 'ps', '--format', '{{.Names}}', capture_output=True).stdout.splitlines()
    (ROOT / 'cutover-outcome.json').write_text(json.dumps(result, indent=2))
    logs = run('journalctl', '--user', '-u', 'qwen36-image.service', '--no-pager', '-n', '500', capture_output=True)
    with gzip.open(ROOT / 'managed-service.log.gz', 'wt') as stream:
        stream.write(logs.stdout + logs.stderr)
    print(json.dumps(result, indent=2), flush=True)
