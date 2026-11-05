#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create a base OCI manifest from the locked production image's first 19 layers.

Input is a podman-save OCI directory. Existing blobs are retained; only small
config/manifest/index JSON files are written. This does not recover the lost
original official config or its registry digest.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layout', type=Path)
    args = parser.parse_args()
    lock = json.loads(Path(__file__).with_name('production-forensics.json').read_text())
    blobs = args.layout / 'blobs/sha256'

    def read(descriptor):
        raw = (blobs / descriptor['digest'].removeprefix('sha256:')).read_bytes()
        if 'sha256:' + hashlib.sha256(raw).hexdigest() != descriptor['digest']:
            raise ValueError('OCI blob digest mismatch')
        return json.loads(raw)

    def write(payload, media_type):
        raw = json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode()
        digest = hashlib.sha256(raw).hexdigest()
        (blobs / digest).write_bytes(raw)
        return {'mediaType': media_type, 'digest': 'sha256:' + digest, 'size': len(raw)}

    index = json.loads((args.layout / 'index.json').read_text())
    if len(index['manifests']) != 1:
        raise ValueError('Expected one production image in OCI layout')
    manifest = read(index['manifests'][0])
    config = read(manifest['config'])
    expected = [layer['diff_id'] for layer in lock['layers']]
    if config['rootfs']['diff_ids'] != expected:
        raise ValueError('Image does not match locked production filesystem layers')
    if config['architecture'] != 'amd64' or len(manifest['layers']) != 22:
        raise ValueError('Expected amd64 production image with 22 layers')
    base = copy.deepcopy(config)
    base['rootfs']['diff_ids'] = expected[:19]
    # Keep history through the official CMD, excluding the six overlay entries.
    history = base['history'][:-len(lock['overlay_history'])]
    if sum(not h.get('empty_layer', False) for h in history) != 19:
        raise ValueError('Base history does not describe 19 filesystem layers')
    base['history'] = history
    base['created'] = history[-1]['created']
    for label in ('org.opencontainers.image.revision', 'org.opencontainers.image.title'):
        base.get('config', {}).get('Labels', {}).pop(label, None)
    manifest['config'] = write(base, 'application/vnd.oci.image.config.v1+json')
    manifest['layers'] = manifest['layers'][:19]
    descriptor = write(manifest, 'application/vnd.oci.image.manifest.v1+json')
    descriptor['annotations'] = {'org.opencontainers.image.ref.name': 'a310-production-base'}
    index['manifests'] = [descriptor]
    (args.layout / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'manifest': descriptor['digest'], 'config': manifest['config']['digest'],
                      'layers': 19, 'provenance': 'recovered-from-production'}, indent=2))


if __name__ == '__main__':
    main()
