# SPDX-License-Identifier: Apache-2.0
"""MindIEQ Gate/Up准备：每token量化一次，再按设备路由复制量化码与scale。"""
import torch


def quantize_shared_routes(inputs, sorted_tokens):
    q, scale = torch.ops.qwen36_mindieq_grouped.quantize(inputs)
    return torch.ops.qwen36_mindieq_grouped.gather(q, scale, sorted_tokens)
