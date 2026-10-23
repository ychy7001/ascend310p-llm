# SPDX-License-Identifier: Apache-2.0
"""Opt-in blocking upload of the tiny MTP rejection-count tensor on 310P.

Keep the installed sampler algorithm unchanged. Some 310P deployments read
invalid counts after uploading a temporary CPU tensor with non_blocking=True.
Fail on an unknown source signature rather than silently patching another ABI.
"""
import inspect
import logging


def install():
    import vllm_ascend.sample.rejection_sampler as sampler

    original = sampler.rejection_greedy_sample_pytorch
    if getattr(original, '_qwen36_blocking_metadata', False):
        return
    source = inspect.getsource(original)
    old = 'draft_tokens_per_req = torch.tensor(draft_tokens_per_req).to(device, non_blocking=True)'
    if source.count(old) != 1:
        raise RuntimeError('Unsupported rejection sampler source for blocking count upload')
    namespace = {}
    exec(compile(source.replace(old, old.replace('True', 'False')),
                 inspect.getfile(original), 'exec'), original.__globals__, namespace)
    patched = namespace[original.__name__]
    patched._qwen36_blocking_metadata = True
    sampler.rejection_greedy_sample_pytorch = patched
    logging.warning('Qwen36/310P: rejection sampler count upload is blocking (opt-in)')
