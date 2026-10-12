# SPDX-License-Identifier: Apache-2.0
"""Shape-only contracts matching the independent C++ W4 extension."""
import torch


def _packed(x, width):
    rows = x.shape[0]
    low = x.new_empty((rows, width // 2), dtype=torch.int8)
    scale = x.new_empty((rows, width // 128, 8), dtype=torch.float32)
    return low, torch.empty_like(low), scale, torch.empty_like(scale)


@torch.library.register_fake('qwen36_int4::pack')
def pack(x):
    return _packed(x, x.shape[1])


@torch.library.register_fake('qwen36_int4::swiglu_pack')
def swiglu_pack(gate_up):
    return _packed(gate_up, gate_up.shape[1] // 2)


@torch.library.register_fake('qwen36_int4::matmul')
def matmul(low, high, activation_scale, activation_sum, codes, scale,
           offset, weight_sum, group_ends):
    rows = group_ends.shape[0] if group_ends.dtype == torch.int32 else low.shape[0]
    return low.new_empty((rows, codes.shape[1]), dtype=torch.float16)


@torch.library.register_fake('qwen36_int4::down_reduce')
def down_reduce(low, high, activation_scale, activation_sum, codes, scale,
                offset, weight_sum, route_ids, route_weights):
    return low.new_empty((route_weights.shape[0], 2560), dtype=torch.float32)
