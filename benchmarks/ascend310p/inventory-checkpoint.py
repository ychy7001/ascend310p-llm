#!/usr/bin/env python3
"""纯stdlib校验ModelSlim分片/index与张量边界，并冻结每文件SHA256。"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('model', type=Path)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
indexes = list(a.model.glob('*.safetensors.index.json'))
if len(indexes) != 1:
    raise RuntimeError('Expected one safetensors index')
index = json.loads(indexes[0].read_text())['weight_map']
expected_files = set(index.values())
if any(Path(name).name != name for name in expected_files):
    raise RuntimeError('Unsafe/non-flat weight path in index')
seen = set()
dtypes = Counter()
files = []
for path in sorted(a.model.iterdir()):
    if not path.is_file():
        continue
    if path.is_symlink():
        raise RuntimeError(f'Unexpected checkpoint symlink: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    files.append(dict(name=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest()))
    if path.name not in expected_files:
        continue
    with path.open('rb') as stream:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < header_size < path.stat().st_size - 8:
            raise RuntimeError(f'Invalid safetensors header: {path}')
        header = json.loads(stream.read(header_size))
    end = 0
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        if name in seen or index.get(name) != path.name:
            raise RuntimeError(f'Tensor/index mismatch: {name}')
        left, right = tensor['data_offsets']
        if not 0 <= left <= right <= path.stat().st_size - 8 - header_size:
            raise RuntimeError(f'Tensor offset out of bounds: {name}')
        seen.add(name)
        end = max(end, right)
        dtypes[tensor['dtype']] += 1
    if end != path.stat().st_size - 8 - header_size:
        raise RuntimeError(f'Weight file size mismatch: {path}')
if seen != set(index):
    raise RuntimeError(f'Missing {len(set(index) - seen)} tensors')
result = dict(status='PASS', scope='destination checkpoint index/header integrity and SHA256 fingerprint; not a separate source weight hash comparison',
              shards=len(expected_files), tensors=len(seen), dtypes=dict(dtypes), files=files,
              total_bytes=sum(row['bytes'] for row in files))
with a.out.open('x') as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({key: value for key, value in result.items() if key != 'files'}))
