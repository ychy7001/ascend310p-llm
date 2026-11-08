#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hash one mounted image tree serially, retaining modes, owners, and symlinks.

Run inside podman unshare so uid/gid values use the container's mapping.
Timestamps are deliberately excluded. No file contents or symlinks are executed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    rows = {}
    for directory, dirs, files in os.walk(args.root, followlinks=False):
        for name in sorted(dirs + files):
            path = Path(directory) / name
            info = path.lstat()
            item = {'mode': stat.S_IMODE(info.st_mode), 'uid': info.st_uid, 'gid': info.st_gid}
            if stat.S_ISLNK(info.st_mode):
                item.update(type='symlink', target=os.readlink(path))
            elif stat.S_ISDIR(info.st_mode):
                item['type'] = 'directory'
            elif stat.S_ISREG(info.st_mode):
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                item.update(type='file', size=info.st_size, sha256=digest.hexdigest())
            else:
                item.update(type='special', rdev=info.st_rdev)
            rows['/' + str(path.relative_to(args.root))] = item
    args.out.write_text(json.dumps(rows, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'entries': len(rows), 'output': str(args.out)}))


if __name__ == '__main__':
    main()
