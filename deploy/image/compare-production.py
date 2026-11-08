#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare CPU image inspection results and complete filesystem hash inventories."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--production-files', type=Path, required=True)
    parser.add_argument('--candidate-files', type=Path, required=True)
    parser.add_argument('--production-inspect', type=Path, required=True)
    parser.add_argument('--candidate-inspect', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    production = json.loads(args.production.read_text())
    candidate = json.loads(args.candidate.read_text())
    left = json.loads(args.production_files.read_text())
    right = json.loads(args.candidate_files.read_text())

    def cache(path):
        return ('/__pycache__/' in path or path.endswith('.pyc')
                or path.startswith(('/root/.cache/', '/var/cache/')))

    def source(path):
        return (path.startswith(('/vllm-workspace/vllm/', '/vllm-workspace/vllm-ascend/'))
                and '/.git/' not in path) or ('/site-packages/' in path and 'vllm' in path)

    production_config = json.loads(args.production_inspect.read_text())[0]['Config']
    candidate_config = json.loads(args.candidate_inspect.read_text())[0]['Config']
    changes = []
    for path in sorted(left.keys() | right.keys()):
        if left.get(path) != right.get(path):
            changes.append({'path': path, 'production': left.get(path),
                            'candidate': right.get(path), 'cache': cache(path)})
    source_files = [path for path, value in left.items()
                    if source(path) and not cache(path) and value['type'] == 'file']
    result = {
        'image_config_equal': production_config == candidate_config,
        'environment_equal': production_config.get('Env') == candidate_config.get('Env'),
        'entrypoint_config_equal': all(production_config.get(key) == candidate_config.get(key)
                                       for key in ('Entrypoint', 'Cmd', 'WorkingDir', 'User')),
        'source_files_compared': len(source_files),
        'source_differences': [item for item in changes if source(item['path']) and not item['cache']],
        'pip_freeze_equal': production['pip_freeze']['returncode'] == 0
            and candidate['pip_freeze']['returncode'] == 0
            and production['pip_freeze']['stdout'] == candidate['pip_freeze']['stdout'],
        'entrypoint_equal': production['entrypoint'] == candidate['entrypoint'],
        'editable_installs_equal': production['editable_installs'] == candidate['editable_installs'],
        'versions_equal': production['versions'] == candidate['versions'],
        'noncache_differences': [item for item in changes if not item['cache']],
        'cache_differences': [item for item in changes if item['cache']],
    }
    result['passed'] = all(result[key] for key in (
        'pip_freeze_equal', 'entrypoint_equal', 'editable_installs_equal', 'versions_equal',
        'environment_equal', 'entrypoint_config_equal', 'image_config_equal'
    )) and not result['noncache_differences']
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: len(value) if isinstance(value, list) else value
                      for key, value in result.items()}))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
