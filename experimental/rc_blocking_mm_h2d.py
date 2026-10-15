# SPDX-License-Identifier: Apache-2.0
"""显式 RC 兼容适配：仅改变多模态输入 H2D 的阻塞语义。

orangepi/24.1.t52.b060+CANN9.1 的独立探针中，非阻塞4MiB H2D
重复等待设备队列，阻塞1KiB/4/16/64MiB逐字节回读通过。
不修改普通Tensor.to、模型精度、KV或Decode计算。
"""
import torch
from vllm.multimodal import inputs
from vllm.utils.jsontree import json_map_leaves


def blocking_h2d(tensors, device):
    if device is None:
        return tensors
    return json_map_leaves(
        lambda value: value.to(device=device, non_blocking=False)
        if isinstance(value, torch.Tensor) else value,
        tensors,
    )


inputs._nested_tensors_h2d = blocking_h2d
print('RC_BLOCKING_MM_H2D enabled: multimodal input transfers only', flush=True)
