# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only timing ablation; no tensor substitution or adoption path."""
import importlib
import json
from pathlib import Path
from wy_gate_worker import WYGateWorker


class WYTimingWorker(WYGateWorker):
    def init_device(self):
        super().init_device()
        import torch
        module = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
        active = module._compute_kernel_inputs_from_torch_wy
        torch_wy = getattr(module, '_production_wy_before_native', active)
        marker = Path('/experiment/timing-mode.json')

        def mode():
            return json.loads(marker.read_text())['mode'] if marker.exists() else 'plain'

        def observe(q, k, v, g, beta, chunk_size):
            selected_mode = mode()
            result = active(q, k, v, g, beta, chunk_size)
            if selected_mode == 'sync_WY':
                torch.npu.synchronize()
            elif selected_mode == 'torch_side':
                chunks = q.shape[1] // chunk_size
                selected = sorted({0, chunks // 2, chunks - 1})
                indices = torch.tensor([i for c in selected for i in range(c*chunk_size, (c+1)*chunk_size)], device=q.device)
                small = [x.index_select(1, indices).contiguous() for x in (q, k, v, g, beta)]
                # Deliberately discarded, never enters actual H/O.
                torch_wy(*small, chunk_size)
            elif selected_mode not in ('plain', 'sync_H', 'sync_O'):
                raise ValueError('Unknown diagnostic mode: ' + selected_mode)
            return result

        module._compute_kernel_inputs_from_torch_wy = observe
        for name, stage in [('chunk_gated_delta_rule_fwd_h_vllm', 'H'), ('chunk_fwd_o_vllm', 'O')]:
            original = getattr(torch.ops._C_ascend, name)
            def wrap(original=original, stage=stage):
                def call(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if mode() == 'sync_' + stage:
                        torch.npu.synchronize()
                    return result
                return call
            setattr(torch.ops._C_ascend, name, wrap())
        print('[wy_timing_ready] diagnostic_only', flush=True)
