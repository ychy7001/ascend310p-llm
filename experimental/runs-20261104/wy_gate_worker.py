# SPDX-License-Identifier: Apache-2.0
"""同一已验收 H/O worker，仅由 WY_GATE_ARM 决定是否替换 WY。"""
import hashlib
import importlib
import json
import os
from pathlib import Path
from cann921_gdn_pair_worker import CANN921GDNPairWorker


class WYGateWorker(CANN921GDNPairWorker):
    def init_device(self):
        super().init_device()
        import torch
        from native_wy import supported
        root=Path('/native-wy')
        for name,expected in json.loads((root/'operator-artifacts.json').read_text()).items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
        arm=os.environ['WY_GATE_ARM'];assert arm in ('A','B')
        module=importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
        if arm=='B':
            torch.ops.load_library(str(root/'extension/pr17640_wy.so'))
            original=module._compute_kernel_inputs_from_torch_wy
            shapes=set()
            def native(q,k,v,g,beta,chunk_size):
                if not supported(q,k,v,g,beta,chunk_size):
                    raise RuntimeError('Native WY unsupported real model inputs; fallback prohibited')
                shape=(tuple(q.shape),tuple(v.shape),str(q.dtype),str(g.dtype))
                result=torch.ops.pr17640_wy.compute(q.contiguous(),k.contiguous(),v.contiguous(),
                                                     g.contiguous(),beta.contiguous(),chunk_size)
                if shape not in shapes:
                    shapes.add(shape)
                    loaded=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                   if 'libcust_opapi.so' in line})
                    print('[wy_gate_native_call] '+json.dumps({'rank':os.environ.get('LOCAL_RANK'),
                          'q_shape':list(q.shape),'v_shape':list(v.shape),'loaded_apis':loaded}),flush=True)
                return result
            module._production_wy_before_native=original
            module._compute_kernel_inputs_from_torch_wy=native
        print('[wy_gate_worker] '+json.dumps({'arm':arm,'wy':'native' if arm=='B' else 'torch',
              'fallback':False,'wy_artifact_manifest_sha256':hashlib.sha256((root/'operator-artifacts.json').read_bytes()).hexdigest()}),flush=True)
