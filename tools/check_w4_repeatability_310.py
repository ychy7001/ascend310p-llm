# SPDX-License-Identifier: Apache-2.0
"""真实 layer-0 sidecar 权重，固定输入/路由，分阶段定位重复波动。

不含 Attention、共享专家、动态 router 或 HCCL。synthetic hidden states。
同时测试每次同步与连续入队；只把逐位一致性作为重复性判断。
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch_npu
from safetensors import safe_open


def native_weight(weight):
    n, k2 = weight.shape
    nibbles = torch.stack((weight & 15, (weight >> 4) & 15), -1).flatten(-2)
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    sums = signed.reshape(n, k2//64, 128).sum(-1).half()
    packed = weight.reshape(n//16, 16, k2//64, 2, 32).permute(0, 2, 3, 1, 4).contiguous().view_as(weight)
    return packed, native_metadata(sums)


def native_metadata(tensor):
    return tensor.reshape(tensor.shape[0]//16, 16, -1).transpose(1, 2).contiguous().view_as(tensor)


def load_bank(root, projections, experts=8):
    index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    columns = [[], [], [], []]
    for expert in range(experts):
        parts = []
        for projection in projections:
            tensors = []
            for kind in ('weight', 'weight_scale', 'weight_offset'):
                key = f'model.language_model.layers.0.mlp.experts.{expert}.{projection}.{kind}'
                with safe_open(root/index[key], framework='pt', device='cpu') as f:
                    tensors.append(f.get_tensor(key))
            packed, sums = native_weight(tensors[0])
            parts.append([packed, native_metadata(tensors[1]), native_metadata(tensors[2]), sums])
        for column, group in zip(columns, zip(*parts)):
            column.append(torch.cat(group, dim=0))
    return [torch.stack(column).npu() for column in columns]


def snapshot(output):
    tensors = output if isinstance(output, tuple) else (output,)
    return tuple(x.cpu().clone() for x in tensors)


def check(name, operation, repeats):
    for _ in range(5):
        operation()
    torch.npu.synchronize()
    reference = snapshot(operation())
    rows = []
    for mode in ('synchronize_each', 'queued'):
        if mode == 'queued':
            pending = [operation() for _ in range(repeats)]
            torch.npu.synchronize()
            actual = [snapshot(value) for value in pending]
        else:
            actual = []
            for _ in range(repeats):
                value = operation()
                torch.npu.synchronize()
                actual.append(snapshot(value))
        mismatches = [0]*len(reference)
        max_abs = [0.0]*len(reference)
        differing_elements = [0]*len(reference)
        for value in actual:
            for i, (a, b) in enumerate(zip(reference, value)):
                if not torch.isfinite(b.float()).all():
                    raise ValueError(f'nonfinite output: {name}/{mode}/{i}')
                count = int((a != b).sum())
                mismatches[i] += count > 0
                differing_elements[i] += count
                max_abs[i] = max(max_abs[i], float((a.float()-b.float()).abs().max()))
        rows.append({'mode':mode, 'repeats':repeats, 'mismatching_runs':mismatches,
                     'differing_elements_total':differing_elements, 'max_abs':max_abs})
    return {'stage':name, 'reference_sha256':[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in reference],
            'outputs':[{'shape':list(x.shape), 'dtype':str(x.dtype)} for x in reference],
            'status':'PASS' if all(not any(row['mismatching_runs']) for row in rows) else 'FAIL',
            'measurements':rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--library', default='/custom/qwen36_int4_custom.so')
    parser.add_argument('--out', required=True)
    parser.add_argument('--repeats', type=int, default=40)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(177)
    torch.ops.load_library(args.library)
    gate_bank = load_bank(args.model, ('gate_proj', 'up_proj'))
    down_bank = load_bank(args.model, ('down_proj',))
    rows = []
    for m in (1, 2, 4, 6, 8):
        x = (torch.randn(m, 2048)*0.2).half().npu()
        ids = torch.arange(8, dtype=torch.int32).repeat(m).npu()
        packs = torch.ops.qwen36_int4.pack(x)
        gate = torch.ops.qwen36_int4.matmul(*packs, *gate_bank, ids)
        downpacks = torch.ops.qwen36_int4.swiglu_pack(gate)
        operations = [
            ('pack', lambda:torch.ops.qwen36_int4.pack(x)),
            ('gate_fixed_pack', lambda:torch.ops.qwen36_int4.matmul(*packs, *gate_bank, ids)),
            ('swiglu_pack_fixed_gate', lambda:torch.ops.qwen36_int4.swiglu_pack(gate)),
            ('down_fixed_pack', lambda:torch.ops.qwen36_int4.matmul(*downpacks, *down_bank, ids)),
        ]
        def full_mlp():
            a = torch.ops.qwen36_int4.pack(x)
            b = torch.ops.qwen36_int4.matmul(*a, *gate_bank, ids)
            c = torch.ops.qwen36_int4.swiglu_pack(b)
            return torch.ops.qwen36_int4.matmul(*c, *down_bank, ids)
        operations.append(('routed_mlp_no_combine', full_mlp))
        for name, operation in operations:
            record = dict(m=m, routes=m*8, **check(name, operation, args.repeats))
            rows.append(record)
            print(json.dumps(record), flush=True)
    result = {'status':'PASS' if all(row['status']=='PASS' for row in rows) else 'FAIL',
              'scope':'real layer-0 expert0..7 weights; synthetic fixed hidden states; no Attention/router/HCCL',
              'rows':rows}
    with open(args.out, 'x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({'status':result['status'], 'stages':len(rows)}), flush=True)


if __name__ == '__main__':
    main()
