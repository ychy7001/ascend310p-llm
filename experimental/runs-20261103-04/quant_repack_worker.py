# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: rebuild down_proj NZ storage on device, preserve logical values."""
import hashlib
import json
from pathlib import Path

import torch
import torch_npu
from vllm_ascend._310p.worker_310p import NPUWorker310


class QuantRepackWorker(NPUWorker310):
    def init_device(self):
        super().init_device()
        import vllm_ascend._310p.quantization.methods.w8a8_dynamic as method
        digest = hashlib.sha256(Path(method.__file__).read_bytes()).hexdigest()
        if digest != '3314bba76317196e52a32506d29436bd5d88922cc4b68443d3878ea111ed04cf':
            raise RuntimeError('Repack diagnostic requires audited production method')
        original = torch_npu.npu_quant_matmul
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else self.local_rank
        log = Path(f'/experiment/quant-repack-rank{rank}.jsonl')
        calls = 0

        def record(payload):
            with log.open('a') as stream:
                stream.write(json.dumps(payload) + '\n')

        def metadata(tensor):
            return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                        format=int(torch_npu.get_npu_format(tensor)),
                        offset=tensor.storage_offset(), storage_bytes=tensor.untyped_storage().nbytes())

        def repacked(x, weight, scale, *args, **kwargs):
            nonlocal calls
            if tuple(weight.shape) != (8704, 5120):
                return original(x, weight, scale, *args, **kwargs)
            calls += 1
            record(dict(kind='before_repack', call=calls, weight=metadata(weight), x=metadata(x)))
            # Recover physical [N,K] base orientation, rebuild ND, then NZ and transpose.
            dense = torch_npu.npu_format_cast(weight.transpose(0, 1), 2).contiguous().clone()
            rebuilt = torch_npu.npu_format_cast(dense, 29).transpose(0, 1)
            # Strict exact-value gate precedes every diagnostic Matmul.
            equal = torch.equal(weight.cpu(), rebuilt.cpu())
            record(dict(kind='repacked', call=calls, equal=equal, weight=metadata(rebuilt)))
            if not equal:
                raise RuntimeError('Repack changed logical int8 weights')
            result = original(x, rebuilt, scale, *args, **kwargs)
            torch.npu.synchronize()
            record(dict(kind='matmul_complete', call=calls))
            return result

        torch_npu.npu_quant_matmul = repacked
        print('[quant-repack] diagnostic only; all K8704 N5120 weights rebuilt with exact CPU equality gate', flush=True)
