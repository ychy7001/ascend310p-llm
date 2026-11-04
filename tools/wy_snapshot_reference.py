# SPDX-License-Identifier: Apache-2.0
"""CPU FP32 WY oracle using triangular solves, independent of the Torch row update."""
import torch

OUTPUT_NAMES = ('q', 'k', 'w', 'u', 'g')


def reference(inputs, chunk_size=64):
    q, k, v, g, beta = inputs
    if any(x.device.type != 'cpu' for x in inputs):
        raise ValueError('CPU tensors required')
    batch, tokens, hk, kd = k.shape
    hv, vd = v.shape[2:]
    if tokens % chunk_size or hv % hk:
        raise ValueError('padded chunks and divisible head ratio required')
    key = k.float().repeat_interleave(hv // hk, dim=2).transpose(1, 2)
    key = key.reshape(batch, hv, -1, chunk_size, kd)
    value = v.float().transpose(1, 2).reshape(batch, hv, -1, chunk_size, vd)
    gate = g.float().transpose(1, 2).reshape(batch, hv, -1, chunk_size).cumsum(-1)
    weight = beta.float().transpose(1, 2).reshape(batch, hv, -1, chunk_size)
    decay = torch.exp(torch.tril(gate.unsqueeze(-1) - gate.unsqueeze(-2))).tril()
    lower = torch.tril((key * weight.unsqueeze(-1)) @ key.transpose(-1, -2) * decay, diagonal=-1)
    matrix = lower + torch.eye(chunk_size, dtype=torch.float32)
    w = torch.linalg.solve_triangular(matrix, key * weight.unsqueeze(-1) * gate.exp().unsqueeze(-1), upper=False, unitriangular=True)
    u = torch.linalg.solve_triangular(matrix, value * weight.unsqueeze(-1), upper=False, unitriangular=True)
    return (q.transpose(1, 2).contiguous().float(), k.transpose(1, 2).contiguous().float(),
            w.reshape(batch, hv, tokens, kd), u.reshape(batch, hv, tokens, vd),
            gate.reshape(batch, hv, tokens))


def metrics(actual, expected, *, atol=0.0005, rtol=0.04, max_relative_rmse=0.02):
    records = []
    for name, a, e in zip(OUTPUT_NAMES, actual, expected, strict=True):
        a, e = a.detach().cpu().float(), e.detach().cpu().float()
        if a.shape != e.shape:
            raise ValueError(f'{name}: shape mismatch')
        delta = a - e
        finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
        error = delta.abs()
        outside = error > atol + rtol * e.abs()
        relative = (delta.square().mean().sqrt() / e.square().mean().sqrt().clamp_min(1e-30)).item()
        exact = name in ('q', 'k')
        passed = finite and (torch.equal(a, e) if exact else (not bool(outside.any()) and relative <= max_relative_rmse))
        flat = error.flatten()
        where = int(flat.argmax())
        records.append({'output': name, 'finite': finite, 'max_abs': flat[where].item(),
                        'reference_at_max': e.flatten()[where].item(),
                        'actual_at_max': a.flatten()[where].item(),
                        'relative_rmse': relative, 'outside_tolerance': int(outside.sum()),
                        'elements': e.numel(), 'exact_required': exact, 'passed': bool(passed)})
    return records
