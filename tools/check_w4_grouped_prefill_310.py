# SPDX-License-Identifier: Apache-2.0
"""Real layer-0 W4 weights: grouped Prefill correctness and synchronized latency.

Synthetic activations and fixed expert boundaries; no dynamic router, Attention
or HCCL. CPU reference uses exact packed activation integers and dequantized
checkpoint weights, separating kernel error from checkpoint quantization error.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from statistics import median

import torch
import torch_npu
from safetensors import safe_open

from check_w4_repeatability_310 import load_bank


def signed_nibbles(packed):
    packed = packed.to(torch.int16)
    values = torch.stack((packed & 15, (packed >> 4) & 15), -1).flatten(-2)
    return torch.where(values >= 8, values - 16, values).float()


def reference_bank(root, projections):
    index = json.loads((root / 'model.safetensors.index.json').read_text())['weight_map']
    weights = []
    for expert in range(8):
        parts = []
        for projection in projections:
            tensors = []
            for kind in ('weight', 'weight_scale', 'weight_offset'):
                key = f'model.language_model.layers.0.mlp.experts.{expert}.{projection}.{kind}'
                with safe_open(root / index[key], framework='pt', device='cpu') as source:
                    tensors.append(source.get_tensor(key))
            codes = signed_nibbles(tensors[0])
            scale, offset = [value.float().repeat_interleave(128, dim=-1) for value in tensors[1:]]
            parts.append((codes - offset) * scale)
        weights.append(torch.cat(parts))
    return torch.stack(weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--library', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--repeats', type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    torch.set_num_threads(4)
    torch.manual_seed(177)
    torch.ops.load_library(args.library)
    with args.out.open('x') as output:
        output.write(json.dumps({'kind': 'config', 'label': args.label,
            'seed': 177, 'warmup': 5, 'repeats': args.repeats,
            'library_sha256': hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
            'model_index_sha256': hashlib.sha256((args.model / 'model.safetensors.index.json').read_bytes()).hexdigest(),
            'scope': __doc__, 'tolerance': {'rtol': .002, 'atol': .002}}) + '\n')
        output.flush()
        for name, projections in (('gate_up', ('gate_proj', 'up_proj')), ('down', ('down_proj',))):
            bank = load_bank(args.model, projections)
            weights = reference_bank(args.model, projections)
            n, k = weights.shape[1:]
            for rows in (129, 257, 1024):
                for routing in ('balanced', 'one_expert'):
                    x = (torch.randn(rows, k) * .2).half().npu()
                    packed = torch.ops.qwen36_int4.pack(x)
                    # Metadata has eight broadcast lanes; decode the exact NPU
                    # integer pack rather than approximating its rounding on CPU.
                    low, high, scales, _ = [value.cpu() for value in packed]
                    quant = signed_nibbles(low) + 16 * signed_nibbles(high) + 8
                    dequant = quant * scales[:, :, 0].repeat_interleave(128, dim=-1)
                    ends = ([rows] * 8 if routing == 'one_expert'
                            else [rows * (expert + 1) // 8 for expert in range(8)])
                    reference = torch.empty(rows, n)
                    start = 0
                    for expert, end in enumerate(ends):
                        reference[start:end] = dequant[start:end] @ weights[expert].T
                        start = end
                    boundaries = torch.tensor(ends, dtype=torch.int64).npu()
                    def operation():
                        return torch.ops.qwen36_int4.matmul(*packed, *bank, boundaries)
                    for _ in range(5):
                        operation()
                    torch.npu.synchronize()
                    actual = operation().cpu().float()
                    torch.testing.assert_close(actual, reference, rtol=.002, atol=.002)
                    timings = []
                    for _ in range(args.repeats):
                        torch.npu.synchronize()
                        begin = time.perf_counter()
                        operation()
                        torch.npu.synchronize()
                        timings.append((time.perf_counter() - begin) * 1000)
                    row = {'kind': 'measurement', 'projection': name, 'rows': rows,
                        'n': n, 'k': k, 'experts': 8, 'routing': routing,
                        'group_ends': ends, 'max_abs': (actual - reference).abs().max().item(),
                        'rmse': (actual - reference).square().mean().sqrt().item(),
                        'latency_ms': timings, 'median_latency_ms': median(timings),
                        'timing_scope': 'synchronized host call; pack excluded; not pure kernel latency',
                        'status': 'PASS'}
                    output.write(json.dumps(row) + '\n'); output.flush()
                    print(json.dumps({key: value for key, value in row.items()
                                      if key != 'latency_ms'}), flush=True)
        output.write(json.dumps({'kind': 'summary', 'status': 'PASS', 'cases': 12}) + '\n')


if __name__ == '__main__':
    main()
