"""Publish the qualified image, using one temporary registry-scoped auth entry."""
import json
from pathlib import Path
import shlex
import subprocess

HOST = 'registry.cn-hangzhou.aliyuncs.com'
REMOTE = 'ychy-a2'
ROOT = '/home/ychy/ascend-lab-runtime/hyperqwen-20261103'
BASE = ROOT + '/graph-localize'
IMAGE = 'localhost/qwen36-w8-wy-mtp-main-graph:20261108'
IMAGE_ID = 'bb0fa66f1c715a94abf56afc17a6bd02e3f993c4804e200dfd229e22b0d5cab4'
TAG = HOST + '/meetai/llm-service-vllm-ascend:qwen36-tp2-w8a8-wy-mtp2-262k-maingraph-20261108-bb0fa66f'
OUT = BASE + '/main-graph-release-registry'


def remote(command, **kwargs):
    return subprocess.check_output(['ssh', REMOTE, command], **kwargs)


closure = json.loads(remote('cat ' + shlex.quote(BASE + '/main-graph-release-closure.json')))
assert closure['state'] == 'complete' and closure['same_image_and_entry']
assert closure['packaged_entry_unmounted'] and closure['packaged_audit_disabled']
assert closure['image_id'] == IMAGE_ID
actual_id = remote('podman image inspect --format ' + shlex.quote('{{.Id}}') + ' ' + shlex.quote(IMAGE), text=True).strip()
assert actual_id == IMAGE_ID
config = json.loads((Path.home() / '.config/containers/auth.json').read_text())
entry = config.get('auths', {}).get(HOST)
assert entry and entry.get('auth'), 'No standard registry auth entry available'
payload = json.dumps({'auths': {HOST: entry}}).encode()
create = "import os,sys,tempfile; fd,path=tempfile.mkstemp(prefix='310p-main-graph-auth-',suffix='.json'); f=os.fdopen(fd,'wb'); f.write(sys.stdin.buffer.read()); f.close(); print(path)"
auth_path = remote('python3 -c ' + shlex.quote(create), input=payload).decode().strip()
try:
    q = shlex.quote
    script = '\n'.join([
        'set -euo pipefail',
        'mkdir -p ' + q(OUT),
        'podman push --authfile ' + q(auth_path) + ' --digestfile ' + q(OUT + '/digest.txt') + ' ' + q(IMAGE) + ' ' + q('docker://' + TAG) + ' > ' + q(OUT + '/push.log') + ' 2>&1',
        'python3 ' + q(ROOT + '/wy-mtp-production-20261108/verify_registry.py') + ' --auth-file ' + q(auth_path) + ' --digest-file ' + q(OUT + '/digest.txt') + ' --image-id ' + q(IMAGE_ID) + ' --out ' + q(OUT),
    ]) + '\n'
    subprocess.run(['ssh', REMOTE, 'bash', '-s'], input=script, text=True, check=True, timeout=1800)
finally:
    subprocess.run(['ssh', REMOTE, 'rm', '-f', auth_path], check=True)
print(json.dumps({'registry_tag': TAG, 'image_id': IMAGE_ID, 'temporary_credentials_removed': True}))
