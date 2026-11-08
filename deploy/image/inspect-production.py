#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Read image provenance without importing torch, plugins, or Ascend libraries.

Run with python3 -S and no device mounts. JSON is evidence, not a build input.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


def run(args):
    result = subprocess.run(args, capture_output=True, text=True)
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def main():
    result = {'python': sys.version, 'sources': {}}
    for name in ('vllm', 'vllm-ascend'):
        path = Path('/vllm-workspace') / name
        source = {'exists': path.exists(), 'git_exists': (path / '.git').exists()}
        result['sources'][str(path)] = source
        if source['git_exists']:
            git = ['git', '-c', f'safe.directory={path}', '-C', str(path)]
            for key, command in (
                ('head', ['rev-parse', 'HEAD']),
                ('status', ['status', '--porcelain=v1', '--untracked-files=all']),
                ('diff', ['diff', 'HEAD', '--binary']),
                ('remotes', ['remote', '-v']),
            ):
                source[key] = run(git + command)
    site = Path(sys.executable).resolve().parents[1] / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    result['site_packages'] = str(site)
    result['editable_installs'] = {
        str(path): path.read_text() for path in sorted(site.glob('*.pth'))
    }
    result['package_paths'] = {
        str(path): os.readlink(path) if path.is_symlink() else 'directory'
        for path in site.iterdir() if path.name in ('vllm', 'vllm_ascend')
    }
    # -S avoids executing sitecustomize and editable/plugin hooks. pip still reads
    # the same installed distribution metadata, including editable direct_url.
    result['pip_freeze'] = run([
        sys.executable, '-S', '-c',
        f'import sys; sys.path.insert(0, {str(site)!r}); '
        'from pip._internal.cli.main import main; sys.exit(main(["freeze", "--all"]))',
    ])
    result['versions'] = {}
    for package in ('torch', 'torch_npu', 'vllm', 'vllm_ascend'):
        for path in sorted(site.glob(f'{package}-*.dist-info/METADATA')):
            result['versions'][str(path)] = '\n'.join(
                line for line in path.read_text().splitlines()
                if line.startswith(('Name:', 'Version:'))
            )
    result['cann'] = {}
    toolkit = Path('/usr/local/Ascend/ascend-toolkit/latest')
    for relative in ('version.cfg', 'version.info', 'x86_64-linux/ascend_toolkit_install.info'):
        path = toolkit / relative
        if path.is_file():
            result['cann'][str(path)] = path.read_text()
    result['entrypoint'] = {
        path: Path(path).read_text()
        for path in ('/usr/local/bin/entrypoint.sh', '/root/.bashrc', '/etc/profile')
        if Path(path).is_file()
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
