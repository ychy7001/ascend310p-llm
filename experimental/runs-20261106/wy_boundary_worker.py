# SPDX-License-Identifier: Apache-2.0
"""Diagnostic A/B worker: preserve actual outputs, capture bounded WY/H/O samples."""
import importlib
import inspect
import json
import os
from pathlib import Path
from wy_gate_worker import WYGateWorker


class WYBoundaryWorker(WYGateWorker):
    def init_device(self):
        super().init_device()
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
        arm = os.environ['WY_GATE_ARM']
        root = Path('/experiment')
        captures = root / 'captures'
        captures.mkdir(exist_ok=True)
        module = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
        active = module._compute_kernel_inputs_from_torch_wy
        torch_wy = getattr(module, '_production_wy_before_native', active)
        counter = 0

        def context():
            marker = root / 'capture-request.json'
            if not marker.exists():
                return None
            request = json.loads(marker.read_text())
            frame = inspect.currentframe().f_back
            layer = None
            while frame is not None:
                owner = frame.f_locals.get('self')
                if frame.f_code.co_name == '_forward_core' and hasattr(owner, 'prefix'):
                    layer = owner.prefix
                    break
                frame = frame.f_back
            del frame
            if layer is None:
                raise RuntimeError('Diagnostic boundary has no identifiable layer')
            return {'request': request, 'layer': layer, 'rank': rank, 'arm': arm}

        def sample(x):
            if not isinstance(x, torch.Tensor):
                return x
            flat = x.detach().reshape(-1)
            n = flat.numel()
            starts = sorted({0, max(0, n // 2 - 2048), max(0, n - 4096)})
            data = torch.cat([flat[s:min(s+4096, n)] for s in starts]).cpu()
            return {'shape': list(x.shape), 'dtype': str(x.dtype), 'starts': starts, 'data': data}

        def save(stage, ctx, record):
            nonlocal counter
            counter += 1
            info = {**ctx, 'stage': stage, 'call': counter}
            torch.save({'metadata': info, **record}, captures / f'rank{rank}-{counter:04d}-{stage}.pt')
            with (captures / f'rank{rank}.jsonl').open('a') as stream:
                stream.write(json.dumps(info) + '\n')
            print('[wy_boundary] ' + json.dumps(info), flush=True)

        def observe_wy(q, k, v, g, beta, chunk_size):
            ctx = context()
            result = active(q, k, v, g, beta, chunk_size)
            if ctx is None:
                return result
            chunks = q.shape[1] // chunk_size
            selected = sorted({0, chunks // 2, chunks - 1})
            indices = torch.tensor([i for c in selected for i in range(c*chunk_size, (c+1)*chunk_size)], device=q.device)
            small = [x.index_select(1, indices).contiguous() for x in (q, k, v, g, beta)]
            actual = [x.index_select(2, indices).detach().cpu() for x in result]
            # WY chunks are independent; the alternate calculation never enters the model.
            alternate = torch_wy(*small, chunk_size) if arm == 'B' else None
            save('WY', ctx, {'chunk_size': chunk_size, 'chunk_indices': selected,
                 'full_shapes': [list(x.shape) for x in (q, k, v, g, beta)],
                 'inputs': [x.detach().cpu() for x in small], 'actual': actual,
                 'same_input_torch': None if alternate is None else [x.detach().cpu() for x in alternate]})
            return result

        module._compute_kernel_inputs_from_torch_wy = observe_wy
        for name, stage in [('chunk_gated_delta_rule_fwd_h_vllm', 'H'), ('chunk_fwd_o_vllm', 'O')]:
            original = getattr(torch.ops._C_ascend, name)
            def wrap(original=original, stage=stage):
                def observe(*args, **kwargs):
                    ctx = context()
                    before = None if ctx is None else {'args': [sample(x) for x in args],
                              'kwargs': {k: sample(v) for k, v in kwargs.items()}}
                    result = original(*args, **kwargs)
                    if ctx is not None:
                        values = result if isinstance(result, (tuple, list)) else [result]
                        save(stage, ctx, {**before, 'outputs': [sample(x) for x in values]})
                    return result
                return observe
            setattr(torch.ops._C_ascend, name, wrap())
        print('[wy_boundary_ready] ' + json.dumps({'rank': rank, 'arm': arm}), flush=True)
