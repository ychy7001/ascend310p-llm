# SPDX-License-Identifier: Apache-2.0
"""New candidate: fixed layer cutoff for native WY, identical for every request."""
import importlib
import inspect
import json
import os
import re
from wy_gate_worker import WYGateWorker


class WYLayerHybridWorker(WYGateWorker):
    def init_device(self):
        super().init_device()
        arm = os.environ['WY_GATE_ARM']
        if arm == 'A':
            print('[wy_layer_hybrid] control_torch_all_layers', flush=True)
            return
        cutoff = int(os.environ['WY_NATIVE_FROM_LAYER'])
        if cutoff not in (20, 28):
            raise ValueError('Only the two predeclared diagnostic policies are allowed')
        module = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
        native = module._compute_kernel_inputs_from_torch_wy
        original = module._production_wy_before_native
        observed = set()

        def dispatch(q, k, v, g, beta, chunk_size):
            frame = inspect.currentframe().f_back
            layer = None
            while frame is not None:
                owner = frame.f_locals.get('self')
                if frame.f_code.co_name == '_forward_core' and hasattr(owner, 'prefix'):
                    match = re.search(r'\.layers\.(\d+)\.', owner.prefix)
                    if match:
                        layer = int(match.group(1))
                    break
                frame = frame.f_back
            del frame
            if layer is None:
                raise RuntimeError('Cannot identify model layer for frozen WY policy')
            enabled = layer >= cutoff
            if layer not in observed:
                observed.add(layer)
                print('[wy_layer_hybrid] ' + json.dumps({'layer': layer, 'native': enabled, 'native_from_layer': cutoff}), flush=True)
            return (native if enabled else original)(q, k, v, g, beta, chunk_size)

        module._compute_kernel_inputs_from_torch_wy = dispatch
        print('[wy_layer_hybrid_policy] ' + json.dumps({'native_from_layer': cutoff, 'request_independent': True,
              'candidate': 'H' + str(cutoff), 'kernel_artifacts_unchanged': True}), flush=True)
