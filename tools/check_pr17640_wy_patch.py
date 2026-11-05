# SPDX-License-Identifier: Apache-2.0
"""CPU-only integration checks against the pinned upstream kernel checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from patch_pr17640_wy_fp32 import KERNEL, apply, patch_text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    args = p.parse_args()
    original = (args.source_root / KERNEL).read_bytes()
    patched = patch_text(original)
    # This is the source identity of the CPU-built candidate, not a numerical
    # oracle. Numerical acceptance still requires the independent G0 gates.
    assert hashlib.sha256(patched).hexdigest() == '3ec7d34f7b3f43e142e69ecdb4e2dfaf0c5f9dd469b5682b8d3d3fe6b3e2dda8'
    for rejected in (original + b'\n', patched):
        try:
            patch_text(rejected)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed/already-patched source was accepted')
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        target = root / KERNEL
        target.parent.mkdir(parents=True)
        target.write_bytes(original)
        sentinel = root / 'gating-untouched'
        sentinel.write_bytes(b'unchanged')
        apply(root)
        assert target.read_bytes() == patched
        assert sentinel.read_bytes() == b'unchanged'
        assert sorted(str(x.relative_to(root)) for x in root.rglob('*') if x.is_file()) == sorted([str(KERNEL), 'gating-untouched'])
    print(json.dumps({'checks': 4, 'passed': 4, 'scope': ['exact compiled source', 'reject changed source', 'reject reapply', 'only WY target changed']}))


if __name__ == '__main__':
    main()
