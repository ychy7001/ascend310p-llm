# SPDX-License-Identifier: Apache-2.0
"""Compare published file sizes and remote SHA256 to local release files.

Requires the isolated modelscope-hub CLI environment; never prints credentials.
Does not download all weights again. This is a remote-manifest check, not a
fresh-download inference test.
"""
import argparse
import hashlib
import json
from pathlib import Path
from modelscope_hub import HubApi


def verify(root, repo):
    remote = {f.path: f for f in HubApi().list_repo_files(repo, 'model') if f.type != 'tree'}
    checked = 0
    hashed = 0
    for file in sorted(root.iterdir()):
        if not file.is_file():
            continue
        record = remote.get(file.name)
        if record is None:
            raise ValueError(f'Remote file missing: {file.name}')
        if record.size != file.stat().st_size:
            raise ValueError(f'Remote file size mismatch: {file.name}')
        if record.sha256:
            with file.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != record.sha256:
                raise ValueError(f'Remote SHA256 mismatch: {file.name}')
            hashed += 1
        elif file.suffix == '.safetensors':
            raise ValueError(f'Remote weight SHA256 unavailable: {file.name}')
        checked += 1
    result = {'status': 'PASS', 'repo': repo, 'files_checked': checked,
              'remote_sha256_matched': hashed, 'remote_files': len(remote),
              'verification': 'remote size/hash manifest; not fresh-download inference'}
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('repo')
    args = parser.parse_args()
    verify(args.model, args.repo)
