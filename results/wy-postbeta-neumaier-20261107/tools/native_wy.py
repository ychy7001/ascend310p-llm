# SPDX-License-Identifier: Apache-2.0
"""实验性 WY 单算子接入；复用 GruntovDima 的 Ascend PR17640 固定源码。

仅替换生产 chunk 模块的 WY helper。FwdH/FwdO、padding、状态管理保持原调用。
外部 OPP 必须只打包 ComputeWy，禁止使用含同名 FwdH/FwdO 的完整 PR 包。
"""
import hashlib
import importlib
import os
from pathlib import Path

import torch

PRODUCTION_CHUNK_SHA256 = 'ab2b739bb6630ebe9fcb1d7af820f93adffce79668e32247bbb869119a0d6c69'


def supported(q, k, v, g, beta, chunk_size):
    # 与固定 PR18163f6b 的 binding/tiling 契约对齐；空输入交给原路径。
    if chunk_size != 64 or q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        return False
    if g.ndim != 3 or beta.shape != g.shape:
        return False
    if q.device.type != 'npu' or any(x.device != q.device for x in (k, v, g, beta)):
        return False
    if any(x.dtype != torch.float16 for x in (q, k, v, beta)) or g.dtype != torch.float32:
        return False
    b, t, hk, kd = q.shape
    hv, vd = v.shape[2:]
    return (0 < b <= 32 and t > 0 and t % 64 == 0 and hk > 0 and
            0 < hv <= 64 and hv % hk == 0 and kd in (64, 128) and vd in (64, 128) and
            v.shape[:2] == q.shape[:2] and g.shape == (b, t, hv))


def install(library=None):
    module = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
    if hasattr(module, '_production_wy_before_native'):
        return module
    actual = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if actual != PRODUCTION_CHUNK_SHA256:
        raise RuntimeError(f'Production chunk source changed: {actual}')
    torch.ops.load_library(library or os.environ['NATIVE_WY_LIBRARY'])
    original = module._compute_kernel_inputs_from_torch_wy

    def compute(q, k, v, g, beta, chunk_size):
        if supported(q, k, v, g, beta, chunk_size):
            return torch.ops.pr17640_wy.compute(
                q.contiguous(), k.contiguous(), v.contiguous(), g.contiguous(),
                beta.contiguous(), chunk_size)
        return original(q, k, v, g, beta, chunk_size)

    module._production_wy_before_native = original
    module._compute_kernel_inputs_from_torch_wy = compute
    print('[native_wy] only WY helper replaced; production FwdH/FwdO retained', flush=True)
    return module
