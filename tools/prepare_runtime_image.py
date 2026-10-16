#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export a committed runtime and hash-locked tested binaries to a new build context.

This packages existing artifacts; it does not claim to rebuild CANN or kernels.
Adapted from llm-service/scripts/build-qwen36-tp1.py, with a runtime allowlist,
base/source lock and no business-specific serving arguments.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tarfile


ADAPTERS = (
    'c8_backend_contract.py', 'c8_cache_310.py', 'c8_vllm_adapter.py',
    'direct_graph_310.py', 'mindieq_activation.py', 'mindieq_prefill_adapter.py',
    'pr17640_gdn_adapter.py', 'rc_blocking_mm_h2d.py', 'route_occupancy_trace.py',
    'sync_metadata_310.py',
)
ARTIFACTS = {
    'EXTENSION': 'custom/qwen36_int4_custom.so',
    'C8_LIBRARY': 'custom/libqwen36_c8.so',
    'GDN_LIBRARY': 'custom/pr17640_gdn.so',
    'MINDIEQ_LIBRARY': 'custom/mindieq_grouped.so',
    'GDN_SOURCE': 'gdn-source.py',
    'OPP_DIR': 'opp', 'GDN_OPP_DIR': 'gdn-opp',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='HEAD')
    parser.add_argument('--base-lock', default='deploy/image/base-amd64.json')
    parser.add_argument('--mindieq-license', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', args.revision + '^{commit}'], text=True).strip()
    # Read the base lock from the same commit as the runtime, not the worktree.
    base = json.loads(subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{args.base_lock}']))
    if '@sha256:' not in base['image']:
        raise SystemExit('Base lock must use an immutable registry digest')
    machine = {'amd64': 62, 'arm64': 183}[base['architecture']]
    paths = ['runtime', 'bootstrap', 'experimental/c8_bootstrap', 'configs/tp1-profile.sh',
             'deploy/image', 'LICENSE', 'NOTICE', *('experimental/' + p for p in ADAPTERS)]
    archive = subprocess.check_output(['git', '-C', str(repo), 'archive', commit, '--', *paths])
    args.output.mkdir(parents=True, exist_ok=False)
    export = args.output / 'source'
    export.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(export, filter='data')
    rootfs = args.output / 'rootfs'
    rootfs.mkdir()
    shutil.copytree(export / 'runtime', rootfs / 'w4only')
    shutil.copytree(export / 'bootstrap', rootfs / 'bootstrap')
    shutil.copytree(export / 'experimental/c8_bootstrap', rootfs / 'c8bootstrap')
    adapter = rootfs / 'c8adapter'
    adapter.mkdir()
    for name in ADAPTERS:
        shutil.copy2(export / 'experimental' / name, adapter / name)
    product = rootfs / 'opt/qwen36'
    (product / 'configs').mkdir(parents=True)
    shutil.copy2(export / 'configs/tp1-profile.sh', product / 'configs')
    for name in ('start.sh', 'verify-image.py'):
        shutil.copy2(export / 'deploy/image' / name, product / name)
    licenses = product / 'licenses'
    licenses.mkdir()
    for name in ('LICENSE', 'NOTICE'):
        shutil.copy2(export / name, licenses / name)
    shutil.copy2(args.mindieq_license, licenses / 'CANN-Open-Software-License.txt')
    artifact_sources = {}
    for key, name in ARTIFACTS.items():
        src = Path(os.environ[key]).resolve(strict=True)
        dest = rootfs / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dest, symlinks=False)
        else:
            shutil.copy2(src, dest)
        artifact_sources[key] = str(src)
    files = {}
    for path in sorted(rootfs.rglob('*')):
        if not path.is_file():
            continue
        data = path.read_bytes()
        # Device .o kernels are not host ELF; check only host shared libraries.
        if '.so' in path.name and data[:4] == b'\x7fELF':
            endian = '<' if data[5] == 1 else '>'
            actual = struct.unpack_from(endian + 'H', data, 18)[0]
            if actual != machine:
                raise SystemExit(f'Host ELF architecture mismatch: {path}: {actual}, expected {machine}')
        files['/' + str(path.relative_to(rootfs))] = hashlib.sha256(data).hexdigest()
    manifest = {'schema_version': 1, 'source_commit': commit, 'base': base,
                'build_type': 'packaged-tested-binaries', 'artifact_sources': artifact_sources,
                'files': files, 'scope': 'TP1 runtime; weights and host driver excluded; no MTP2 diagnostic patches'}
    (product / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    shutil.copy2(export / 'deploy/image/Containerfile', args.output / 'Containerfile')
    # Exported source is an intermediate owned by this tool, outside rootfs.
    shutil.rmtree(export)
    print(json.dumps({'context': str(args.output), 'source_commit': commit,
                      'base_image': base['image'], 'files': len(files)}, indent=2))


if __name__ == '__main__':
    main()
