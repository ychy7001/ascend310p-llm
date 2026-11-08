"""Diagnostic only: synchronize eager draft input preparation op by op."""
import functools
import json
import traceback
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer
from draft_sync_probe import tensors


class SyncPrepareOps(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        try:
            result = func(*args, **(kwargs or {}))
            torch.npu.synchronize()
            return result
        except Exception:
            print('[prepare-sync-failed] ' + json.dumps({
                'op': str(func), 'args': tensors(args),
                'kwargs': tensors(kwargs or {}), 'stack': traceback.format_stack(limit=12),
            }), flush=True)
            raise


_original = AscendSpecDecodeBaseProposer._propose
_seen = False


@functools.wraps(_original)
def probe(self, *args, **kwargs):
    global _seen
    if self.use_cuda_graph:
        return _original(self, *args, **kwargs)
    if not _seen:
        _seen = True
        print('[prepare-sync-enter] eager preparation per-op synchronization active', flush=True)
    torch.npu.synchronize()
    with SyncPrepareOps():
        return _original(self, *args, **kwargs)


AscendSpecDecodeBaseProposer._propose = probe
print('[prepare-sync-installed] diagnostic only; draft Graph skipped', flush=True)
