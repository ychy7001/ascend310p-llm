# SPDX-License-Identifier: Apache-2.0
"""Verify packaged files without importing torch or initializing the NPU."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--base', action='store_true')
args = parser.parse_args()
manifest = json.loads(Path('/opt/qwen36/manifest.json').read_text())
architecture = {'x86_64': 'amd64', 'aarch64': 'arm64'}[platform.machine()]
assert architecture == manifest['base']['architecture'], 'host architecture mismatch'
if args.base:
    for directory, expected in manifest['base']['sources'].items():
        actual = subprocess.check_output(['git', '-C', directory, 'rev-parse', 'HEAD'], text=True).strip()
        assert actual == expected, (directory, actual, expected)
        subprocess.run(['git', '-C', directory, 'diff', '--exit-code', 'HEAD'], check=True)
for name, expected in manifest['files'].items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    assert actual == expected, f'packaged file mismatch: {name}'
print(json.dumps({'verified': True, 'source_commit': manifest['source_commit'],
                  'architecture': architecture, 'files': len(manifest['files'])}))
