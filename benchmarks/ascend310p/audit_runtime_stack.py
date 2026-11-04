#!/usr/bin/env python3
"""只读记录容器用户态版本及默认GDN源文件；不初始化NPU。"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

result = {'packages': {}, 'version_files': {}, 'git': {}, 'gdn_sources': {},
          'scope': 'Read-only container inventory; no NPU initialization or custom patch'}
for name in ('torch', 'torch-npu', 'vllm', 'vllm-ascend', 'transformers', 'triton', 'triton-ascend'):
    try:
        result['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        result['packages'][name] = None
for root in ('/usr/local/Ascend/ascend-toolkit', '/usr/local/Ascend/cann', '/usr/local/Ascend/nnal'):
    directory = Path(root)
    if not directory.exists():
        continue
    for pattern in ('**/ascend_toolkit_install.info', '**/ascend_install.info', '**/version.info', '**/version.cfg'):
        for path in directory.glob(pattern):
            if path.is_file() and path.stat().st_size < 16384:
                result['version_files'][str(path)] = path.read_text(errors='replace')
for root in ('/vllm-workspace/vllm', '/vllm-workspace/vllm-ascend'):
    r = subprocess.run(['git', '-C', root, 'rev-parse', 'HEAD'], capture_output=True, text=True)
    result['git'][root] = {'returncode': r.returncode, 'head': r.stdout.strip(), 'error': r.stderr.strip()}
for path in Path('/vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops').rglob('*gated_delta*'):
    if path.is_file() and path.suffix == '.py':
        result['gdn_sources'][str(path)] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                          'source': path.read_text()}
result['environment'] = {key: os.environ.get(key) for key in
                         ('ASCEND_HOME_PATH', 'ASCEND_OPP_PATH', 'ATB_HOME_PATH', 'ASCEND_CUSTOM_PATH', 'PYTHONPATH')}
print(json.dumps(result, ensure_ascii=False, indent=2))
