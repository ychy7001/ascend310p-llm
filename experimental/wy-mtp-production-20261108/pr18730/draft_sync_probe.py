"""Diagnostic only: synchronize eager draft operations, preserving Graph capture."""
import functools
import json
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from vllm.forward_context import get_forward_context
from vllm_ascend._310p.spec_decode.llm_base_proposer_310 import AscendSpecDecodeBaseProposer310
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def tensors(value):
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'stride': list(value.stride()), 'offset': value.storage_offset(),
                'device': str(value.device)}
    if isinstance(value, (list, tuple)):
        return [tensors(item) for item in value]
    if isinstance(value, dict):
        return {key: tensors(item) for key, item in value.items()}
    return value if isinstance(value, (int, float, str, bool, type(None))) else type(value).__name__


class SyncDraftOps(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        try:
            result = func(*args, **(kwargs or {}))
            torch.npu.synchronize()
            return result
        except Exception:
            print('[draft-sync-failed] ' + json.dumps({'op': str(func), 'args': tensors(args),
                                                      'kwargs': tensors(kwargs or {})}), flush=True)
            raise


_original = AscendSpecDecodeBaseProposer310._run_merged_draft
_seen_modes = set()


@functools.wraps(_original)
def probe(self, *args, **kwargs):
    context = get_forward_context()
    mode = (bool(context.capturing), bool(self.use_cuda_graph))
    if mode not in _seen_modes:
        _seen_modes.add(mode)
        print('[draft-sync-mode] ' + json.dumps({'capturing': mode[0], 'draft_graph': mode[1],
                                               'class': type(self).__name__}), flush=True)
    if context.capturing or self.use_cuda_graph:
        return _original(self, *args, **kwargs)
    print('[draft-sync-enter] ' + json.dumps({'num_input_tokens': kwargs.get('num_input_tokens'),
                                            'batch_size': kwargs.get('batch_size')}), flush=True)
    try:
        torch.npu.synchronize()
    except Exception:
        print('[draft-sync-before-failed] pending work failed before eager draft', flush=True)
        raise
    with SyncDraftOps():
        return _original(self, *args, **kwargs)


AscendSpecDecodeBaseProposer310._run_merged_draft = probe
# patch_idex_310 copies this method to the base class during worker setup.
# Patch both locations so imports that already ran cannot retain the old method.
AscendSpecDecodeBaseProposer._run_merged_draft = probe
print('[draft-sync-installed] diagnostic only; capture skipped', flush=True)
