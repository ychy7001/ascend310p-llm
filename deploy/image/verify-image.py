# SPDX-License-Identifier: Apache-2.0
"""Verify packaged files without importing torch or initializing the NPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--base', action='store_true')
parser.add_argument('--production-base', type=Path)
args = parser.parse_args()
if args.production_base:
    os.environ['GIT_OPTIONAL_LOCKS'] = '0'
    lock = json.loads(args.production_base.read_text())
    assert platform.machine() == 'x86_64', 'production image requires x86_64'
    source_root = args.production_base.resolve().parents[2]
    for name, expected in lock['patches'].items():
        actual = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        assert actual == expected, f'patch mismatch: {name}'
    for directory, expected in lock['sources'].items():
        git = ['git', '-c', f'safe.directory={directory}', '-C', directory]
        actual = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
        assert actual == expected, (directory, actual, expected)
        status = subprocess.check_output(git + ['status', '--porcelain=v1', '--untracked-files=all'], text=True)
        assert not status, (directory, status)
    print(json.dumps({'verified': True, 'sources': lock['sources']}))
    raise SystemExit(0)
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
