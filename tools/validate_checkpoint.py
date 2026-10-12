# SPDX-License-Identifier: Apache-2.0
"""Validate a standalone Qwen3.6 W4 checkpoint without Torch or CANN."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


def read_header(path):
    with path.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f'Truncated header: {path.name}')
        size = struct.unpack('<Q', raw)[0]
        if size > 128 * 1024 * 1024 or size > path.stat().st_size - 8:
            raise ValueError(f'Invalid header size: {path.name}')
        return json.loads(stream.read(size)), 8 + size


def validate(root, verify_hashes=False):
    root = root.resolve()
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    text = config.get('text_config', config)
    metadata = text['ascend_expert_quantization']
    if (metadata['format'] != 'qwen36_modelslim_ssz_w4a8_v1'
            or metadata['activation_quantization'] != 'int8_per_token'
            or metadata['scale_layout'] != 'per_output_channel_repeated_g128'):
        raise ValueError('Unsupported quantization contract')
    layers, experts = text['num_hidden_layers'], text['num_experts']
    hidden, intermediate = text['hidden_size'], text['moe_intermediate_size']
    required = {
        f'model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}'
        for layer in range(layers) for expert in range(experts)
        for proj in ('gate_proj', 'up_proj', 'down_proj')
        for kind in ('weight', 'weight_scale', 'weight_offset')
    }
    weight_map = index['weight_map']
    if not required <= weight_map.keys():
        raise ValueError(f'Missing expert tensors: {len(required - weight_map.keys())}')
    for file in root.iterdir():
        if file.is_symlink():
            raise ValueError(f'Standalone checkpoint contains symlink: {file.name}')
    headers = {}
    for shard in sorted(set(weight_map.values())):
        if Path(shard).name != shard:
            raise ValueError('Shard must be a basename')
        path = root / shard
        header, base = read_header(path)
        size = path.stat().st_size
        spans = []
        for name, info in header.items():
            if name == '__metadata__':
                continue
            start, stop = info['data_offsets']
            if not 0 <= start <= stop <= size - base:
                raise ValueError(f'Invalid data offsets: {name}')
            spans.append((start, stop, name))
        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            if previous[1] > current[0]:
                raise ValueError(f'Overlapping tensor payload: {current[2]}')
        headers[shard] = header
    total = 0
    mtp = []
    pattern = re.compile(r'\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale|weight_offset)$')
    for name, shard in weight_map.items():
        info = headers[shard][name]
        start, stop = info['data_offsets']
        total += stop - start
        match = pattern.search(name)
        if match:
            proj, kind = match.groups()
            n, k = (hidden, intermediate) if proj == 'down_proj' else (intermediate, hidden)
            want = [n, k // (2 if kind == 'weight' else 128)]
            if info['shape'] != want or info['dtype'] != ('I8' if kind == 'weight' else 'F16'):
                raise ValueError(f'Expert shape/dtype mismatch: {name}')
        if name.startswith('mtp.'):
            if info['dtype'] != 'BF16':
                raise ValueError(f'MTP must preserve BF16: {name}')
            mtp.append(name)
    if len(mtp) != 19 or not {
        'mtp.layers.0.mlp.experts.gate_up_proj',
        'mtp.layers.0.mlp.experts.down_proj',
    } <= set(mtp):
        raise ValueError('Incomplete Qwen3.6 BF16 MTP tensors')
    if total != index['metadata']['total_size']:
        raise ValueError('Index total_size mismatch')
    if verify_hashes:
        hashes = json.loads((root / 'deployment-sha256.json').read_text())
        for name, expected in hashes.items():
            if Path(name).name != name:
                raise ValueError('Hash entry must be a basename')
            with (root / name).open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected:
                raise ValueError(f'SHA256 mismatch: {name}')
    result = {'status': 'PASS', 'tensors': len(weight_map),
              'expert_tensors': len(required), 'mtp_bf16_tensors': len(mtp),
              'shards': len(headers), 'tensor_bytes': total,
              'sha256_verified': verify_hashes, 'symlinks': 0}
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--verify-hashes', action='store_true')
    args = parser.parse_args()
    validate(args.model, args.verify_hashes)
