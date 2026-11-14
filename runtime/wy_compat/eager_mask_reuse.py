# SPDX-License-Identifier: Apache-2.0
"""Reuse one legacy causal mask within a target-model eager forward."""
from contextvars import ContextVar
from functools import wraps
import os

_scope = ContextVar('eager_splitfuse_mask', default=None)


def scoped_forward(forward):
    @wraps(forward)
    def run(*args, **kwargs):
        token = _scope.set({})
        try:
            return forward(*args, **kwargs)
        finally:
            _scope.reset(token)
    return run


def one_mask(key, build):
    scope = _scope.get()
    if scope is None:
        return build()
    if scope.get('key') != key:
        mask = build()
        scope.clear()
        scope.update(key=key, mask=mask)
    return scope['mask']


def install():
    if os.getenv('QWEN36_EAGER_MASK_REUSE') != '1':
        return

    import torch
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm_ascend.attention.attention_v1 import AscendAttentionState
    from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310

    original = AttentionMaskBuilder310.get_splitfuse_mask
    audited = False

    @classmethod
    def get_splitfuse_mask(cls, metadata, device):
        nonlocal audited

        def build():
            return original(metadata, device)

        scope = _scope.get()
        if scope is None or metadata.attn_state not in (
            AscendAttentionState.ChunkedPrefill, AscendAttentionState.PrefillCacheHit
        ):
            return build()
        context = get_forward_context()
        if context.capturing or context.cudagraph_runtime_mode != CUDAGraphMode.NONE:
            return build()
        starts = tuple(metadata.query_start_loc.to('cpu', dtype=torch.int32).tolist())
        lengths = tuple(metadata.seq_lens.to('cpu', dtype=torch.int32).tolist())
        key = (torch.device(device), torch.float16, 'legacy_splitfuse_nz', starts, lengths)
        if not audited and scope.get('key') == key and starts[-1] - starts[0] >= 1024:
            print(f'[eager-mask-reuse] first hit starts={starts} lengths={lengths}', flush=True)
            audited = True
        return one_mask(key, build)

    NPUModelRunner310._model_forward = scoped_forward(NPUModelRunner310._model_forward)
    AttentionMaskBuilder310.get_splitfuse_mask = get_splitfuse_mask
    print('[eager-mask-reuse] enabled for eager target forward', flush=True)
