# SPDX-License-Identifier: Apache-2.0
"""Export indexed payloads into a new self-contained checkpoint.

Input may contain source/sidecar symlinks. Output contains regular files only;
unreferenced original floating-point experts are omitted. No tensor arithmetic.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import struct
from validate_checkpoint import read_header


def digest_region(path, start, length):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        stream.seek(start)
        while length:
            block = stream.read(min(length, 8 * 1024 * 1024))
            if not block:
                raise ValueError(f'Truncated file: {path}')
            digest.update(block)
            length -= len(block)
    return digest.hexdigest()


def export(source, destination):
    source = source.resolve()
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    groups = defaultdict(list)
    for name, shard in index['weight_map'].items():
        if Path(shard).name != shard:
            raise ValueError('Shard must be a basename')
        groups[shard].append(name)
    destination.mkdir(parents=True, exist_ok=False)
    total = 0
    for shard, names in sorted(groups.items()):
        original, base = read_header(source / shard)
        header = {'__metadata__': original['__metadata__']} if '__metadata__' in original else {}
        offset = 0
        for name in names:
            info = dict(original[name])
            start, stop = info['data_offsets']
            if not 0 <= start <= stop <= (source / shard).stat().st_size - base:
                raise ValueError(f'Invalid payload: {name}')
            info['data_offsets'] = [offset, offset + stop - start]
            header[name] = info
            offset += stop - start
        raw = json.dumps(header, separators=(',', ':')).encode()
        raw += b' ' * (-len(raw) % 8)
        out_base = 8 + len(raw)
        hashes = {}
        with (source / shard).open('rb') as src, (destination / shard).open('xb') as dst:
            dst.write(struct.pack('<Q', len(raw)))
            dst.write(raw)
            for name in names:
                start, stop = original[name]['data_offsets']
                src.seek(base + start)
                left = stop - start
                digest = hashlib.sha256()
                while left:
                    block = src.read(min(left, 8 * 1024 * 1024))
                    if not block:
                        raise ValueError(f'Truncated tensor: {name}')
                    dst.write(block)
                    digest.update(block)
                    left -= len(block)
                hashes[name] = digest.hexdigest()
        for name in names:
            start, stop = header[name]['data_offsets']
            if digest_region(destination / shard, out_base + start, stop - start) != hashes[name]:
                raise ValueError(f'Copied payload mismatch: {name}')
        total += offset
        print(json.dumps({'shard': shard, 'tensors': len(names), 'payload_bytes': offset,
                          'payload_hashes_verified': True}), flush=True)
    for file in source.iterdir():
        if file.name in {'model.safetensors.index.json', 'deployment-sha256.json'}:
            continue
        if file.name in groups:
            continue
        if file.is_file() and (file.suffix in {'.json', '.jinja', '.model', '.txt'}
                               or file.name in {'LICENSE', 'NOTICE'}):
            shutil.copyfile(file, destination / file.name)
    index.setdefault('metadata', {})['total_size'] = total
    (destination / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2) + '\n')
    hashes = {}
    for file in sorted(destination.iterdir()):
        with file.open('rb') as stream:
            hashes[file.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (destination / 'deployment-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({'complete': True, 'tensors': len(index['weight_map']),
                      'tensor_bytes': total}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    export(args.source, args.destination)
