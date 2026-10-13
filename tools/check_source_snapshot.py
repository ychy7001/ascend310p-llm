# SPDX-License-Identifier: Apache-2.0
"""校验显式指定的源码快照；历史清单和当前清单不可混用。"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    files = manifest.get('files', manifest)
    failures = []
    for relative, expected in files.items():
        path = args.root / relative
        if not path.is_file():
            failures.append({'path': relative, 'reason': 'missing'})
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append({'path': relative, 'reason': 'hash mismatch'})
    print(json.dumps({'checked': len(files), 'failures': failures,
                      'status': 'FAIL' if failures else 'PASS'}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
