# SPDX-License-Identifier: Apache-2.0
"""Quality probe: compare two masks on the same real SplitFuse operation."""
from contextvars import ContextVar
from functools import wraps
import json
from pathlib import Path

_step = ContextVar('mask_quality_step', default=None)
_mode = Path('/mask-quality-control/mode')


def install():
    import torch
    import torch_npu
    import eager_mask_reuse
    from vllm.config.compilation import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm_ascend._310p.attention.attention_mask import AttentionMaskBuilder310
    from vllm_ascend._310p.model_runner_310p import NPUModelRunner310

    original_mask = AttentionMaskBuilder310.get_splitfuse_mask
    eager_mask_reuse.install()
    reused_mask = AttentionMaskBuilder310.get_splitfuse_mask
    forward = NPUModelRunner310._model_forward
    execute = NPUModelRunner310.execute_model
    schedule = ContextVar('mask_quality_schedule', default=None)

    @wraps(execute)
    def audited_execute(self, scheduler_output, *args, **kwargs):
        token = schedule.set(scheduler_output.num_scheduled_tokens)
        try:
            return execute(self, scheduler_output, *args, **kwargs)
        finally:
            schedule.reset(token)

    @wraps(forward)
    def audited_forward(self, *args, **kwargs):
        counts = schedule.get()
        if not counts:
            return forward(self, *args, **kwargs)
        ctx = get_forward_context()
        mode = _mode.read_text().strip()
        assert mode in ('fresh', 'reuse', 'audit'), mode
        eager = ctx.cudagraph_runtime_mode == CUDAGraphMode.NONE and not ctx.capturing
        trace = mode == 'audit' or sum(counts.values()) >= 32
        state = dict(mode=mode, last=None, pending=None, checks=0, layouts={})
        token = _step.set(state if eager else None)
        op = None
        if eager and mode == 'audit':
            op = torch_npu._npu_paged_attention_splitfuse
            torch_npu._npu_paged_attention_splitfuse = make_audited_op(op)
        try:
            result = forward(self, *args, **kwargs)
            assert state['pending'] is None, 'SplitFuse mask hit was not seen by the op audit'
            if trace and (eager or mode == 'audit'):
                payload = dict(mode=mode, eager=eager, counts=list(counts.items()),
                               order=list(self.input_batch.req_ids), checks=state['checks'],
                               layouts=[[list(a),list(b),n] for (a,b),n in state['layouts'].items()])
                print('[mask-quality-step] '+json.dumps(payload, separators=(',', ':')), flush=True)
            return result
        finally:
            if op is not None:
                torch_npu._npu_paged_attention_splitfuse = op
            _step.reset(token)

    @classmethod
    def audited_mask(cls, metadata, device):
        step = _step.get()
        if step is None or step['mode'] == 'fresh':
            return original_mask(metadata, device)
        mask = reused_mask(metadata, device)
        if step['mode'] == 'audit' and mask is step['last']:
            key = tuple(metadata.query_start_loc.cpu().tolist()), tuple(metadata.seq_lens.cpu().tolist())
            step['pending'] = original_mask(metadata, device), mask, key
        step['last'] = mask
        return mask

    def make_audited_op(op):
        @wraps(op)
        def audited_op(*args, **kwargs):
            step = _step.get()
            if step is None or step['pending'] is None:
                return op(*args, **kwargs)
            fresh, reused, key = step['pending']
            step['pending'] = None
            assert kwargs['mask'] is reused
            out = kwargs['out']
            op(*args, **{**kwargs, 'mask': fresh})
            expected = out.clone()
            result = op(*args, **kwargs)
            if not torch.equal(expected.cpu().view(torch.int16), out.cpu().view(torch.int16)):
                max_delta = (expected.float() - out.float()).abs().max().item()
                raise RuntimeError(f'Same-call SplitFuse mismatch: {key}, max_abs={max_delta}')
            step['checks'] += 1
            step['layouts'][key] = step['layouts'].get(key, 0) + 1
            return result
        return audited_op

    NPUModelRunner310.execute_model = audited_execute
    NPUModelRunner310._model_forward = audited_forward
    AttentionMaskBuilder310.get_splitfuse_mask = audited_mask
