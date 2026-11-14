#!/usr/bin/env python3
"""Build the released eager-mask runtime on the exact internal WY base, offline."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

BASE = 'bb0fa66f1c715a94abf56afc17a6bd02e3f993c4804e200dfd229e22b0d5cab4'
SOURCE_SHA = '04540ffa5eca866d7f3960dc987bc721757f90e76f62daa427aed70af97c145a'
HOOK_SHA = '6c9aa557b44b001761f7d8584d147f79af641151e97180f21c87e67d015eddb2'
ENTRY_SHA = 'ca8391c85f1a89dae6f71eb61233c06eec0f849dedd0be0453b800cb4f60388a'
PATCHED_HOOK_SHA = '723d02a91575ede78c41f62fc3960dbdcb3470562608c25725b0af51a18621f1'

def run(*args):
    return subprocess.check_output(['podman', *args])

def sha(data):
    return hashlib.sha256(data).hexdigest()

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[3] / 'runtime/wy_compat/eager_mask_reuse.py')
p.add_argument('--tag', default='localhost/qwen36-wy-mask-reuse:20261114')
p.add_argument('--manifest', type=Path, required=True)
a = p.parse_args()
assert sha(a.source.read_bytes()) == SOURCE_SHA, 'Requires the qualified reuse implementation'
base = json.loads(run('image', 'inspect', BASE))[0]
hook = run('run', '--rm', '--network=none', '--entrypoint', 'cat', BASE, '/opt/qwen36/wy-compat/sitecustomize.py')
assert sha(hook) == HOOK_SHA
entry = run('run', '--rm', '--network=none', '--entrypoint', 'cat', BASE, '/opt/qwen36/start-wy-mtp.sh')
assert sha(entry) == ENTRY_SHA
needle = b'    graph_safe_splitfuse.install()\n'
assert hook.count(needle) == 1
extra = b"    if os.environ.get('QWEN36_EAGER_MASK_REUSE') == '1':\n        import eager_mask_reuse\n        eager_mask_reuse.install()\n"
patched = hook.replace(needle, needle + extra)
assert sha(patched) == PATCHED_HOOK_SHA
with tempfile.TemporaryDirectory(prefix='wy-mask-package-') as tmp:
    root = Path(tmp)
    shutil.copyfile(Path(__file__).with_name('Containerfile'), root/'Containerfile')
    (root/'eager_mask_reuse.py').write_bytes(a.source.read_bytes())
    (root/'sitecustomize.py').write_bytes(patched)
    subprocess.run(['podman', 'build', '--pull=never', '--network=none', '--format=docker',
                    '--build-arg', f'BASE_IMAGE={BASE}', '-t', a.tag, str(root)], check=True)
image = json.loads(run('image', 'inspect', a.tag))[0]
for key in ('Entrypoint', 'Cmd', 'WorkingDir', 'User'):
    assert image['Config'].get(key) == base['Config'].get(key), key
env = image['Config']['Env']
assert [value for value in env if not value.startswith('QWEN36_EAGER_MASK_REUSE=')] == base['Config']['Env']
assert [value for value in env if value.startswith('QWEN36_EAGER_MASK_REUSE=')] == ['QWEN36_EAGER_MASK_REUSE=1']
assert image['RootFS']['Layers'][:-1] == base['RootFS']['Layers']
assert run('run', '--rm', '--network=none', '--entrypoint', 'cat', a.tag,
           '/opt/qwen36/start-wy-mtp.sh') == entry
assert run('run', '--rm', '--network=none', '--entrypoint', 'cat', a.tag,
           '/opt/qwen36/wy-compat/eager_mask_reuse.py') == a.source.read_bytes()
assert run('run', '--rm', '--network=none', '--entrypoint', 'cat', a.tag,
           '/opt/qwen36/wy-compat/sitecustomize.py') == patched
manifest = dict(base_image_id=BASE, image_id=image['Id'], tag=a.tag,
                source_sha256=SOURCE_SHA, entry_sha256=ENTRY_SHA, hook_sha256=sha(patched),
                rollback_env={'QWEN36_EAGER_MASK_REUSE':'0'}, runtime_smoke='pending')
a.manifest.parent.mkdir(parents=True, exist_ok=True)
a.manifest.write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest, indent=2))
