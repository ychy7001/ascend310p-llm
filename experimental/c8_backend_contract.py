# SPDX-License-Identifier: Apache-2.0
"""Opt-in contract between a vLLM paged cache manager and the 310P C8 ops.

This module intentionally contains no vLLM monkey patch and no attention call.
It validates/lowers scheduler metadata so a future C8 backend can keep dynamic
block-table handling outside the captured scatter/dequant kernels.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class C8PageSpec:
    block_size: int
    kv_heads: int
    head_dim: int
    scale_width: int = 16

    def validate(self):
        if self.block_size <= 0 or self.block_size % 16:
            raise ValueError('block_size must be a positive multiple of 16')
        if not 1 <= self.kv_heads <= 16 or self.head_dim <= 0 or self.head_dim % 32:
            raise ValueError('unsupported 310P C8 head shape')
        if self.kv_heads * self.head_dim > 32768 or self.scale_width != 16:
            raise ValueError('C8 row or scale alignment exceeds kernel contract')


def lower_block_table(block_table, sequence_length, spec: C8PageSpec):
    """Return physical token slots for one request, including no padded slots."""
    spec.validate()
    table = torch.as_tensor(block_table, dtype=torch.int64, device='cpu').flatten()
    if sequence_length < 0 or sequence_length > table.numel() * spec.block_size:
        raise ValueError('sequence length exceeds block table')
    positions = torch.arange(sequence_length, dtype=torch.int64)
    physical = table[positions // spec.block_size]
    slots = physical * spec.block_size + positions.remainder(spec.block_size)
    if (physical < 0).any() or (slots > torch.iinfo(torch.int32).max).any():
        raise ValueError('block table contains an invalid physical block')
    return slots.to(torch.int32)


def pad_scale_rows(scales, spec: C8PageSpec):
    """Pad per-token/per-KV-head FP16 scales to the aligned C8 row."""
    spec.validate()
    x = torch.as_tensor(scales)
    if x.ndim != 2 or x.shape[1] != spec.kv_heads:
        raise ValueError('expected [tokens, kv_heads] scales')
    out = torch.zeros((x.shape[0], spec.scale_width), dtype=torch.float16,
                      device=x.device)
    out[:, :spec.kv_heads] = x.to(torch.float16)
    return out.contiguous()


def validate_cache_tensors(k_cache, v_cache, k_scale, v_scale, capacity, spec):
    """Validate the ND C8 storage contract before launching a device op."""
    spec.validate()
    expected = (capacity, spec.kv_heads, spec.head_dim)
    if tuple(k_cache.shape) != expected or tuple(v_cache.shape) != expected:
        raise ValueError(f'expected cache shape {expected}')
    if tuple(k_scale.shape) != (capacity, spec.scale_width) or tuple(v_scale.shape) != (capacity, spec.scale_width):
        raise ValueError('scale cache must be [capacity, 16]')
    if k_cache.dtype != torch.int8 or v_cache.dtype != torch.int8:
        raise TypeError('C8 K/V cache must be int8')
    if k_scale.dtype != torch.float16 or v_scale.dtype != torch.float16:
        raise TypeError('C8 scales must be float16')
    tensors = (k_cache, v_cache, k_scale, v_scale)
    if any(t.device != k_cache.device or not t.is_contiguous() for t in tensors):
        raise ValueError('C8 storage must be contiguous and share a device')
