# SPDX-License-Identifier: Apache-2.0
"""Replace exactly the 30 production FP16 QKVZ weights after normal model loading."""
import hashlib
import json
from pathlib import Path

import torch
import torch_npu
from safetensors.torch import load_file
from vllm_ascend._310p.worker_310p import NPUWorker310


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class QKVZMethod:
    def apply(self, layer, x, bias=None):
        assert bias is None
        quant, scale = torch_npu.npu_dynamic_quant(x)
        return torch_npu.npu_quant_matmul(quant, layer.weight.T, layer.qkvz_scale,
                                         pertoken_scale=scale, output_dtype=x.dtype)


class QKVZW8A8Worker(NPUWorker310):
    def load_model(self):
        super().load_model()
        root = Path('/qkvz-weights')
        manifest = json.loads((root/'manifest.json').read_text())
        rank = torch.distributed.get_rank()
        entries = [r for r in manifest['rows'] if r['rank'] == rank]
        assert len(entries) == 30
        modules = dict(self.model_runner.model.named_modules())
        changed = []
        before = torch.npu.memory_allocated()
        for row in entries:
            name = row['module'].replace('model.language_model.layers.', 'language_model.model.layers.')
            layer = modules[name]
            assert layer.weight.shape == (6144, 2048) and layer.weight.dtype == torch.float16
            assert torch_npu.get_npu_format(layer.weight) == 29
            assert getattr(layer, 'bias', None) is None
            # This also checks the CPU-prepared FP16 bits against actual production loading.
            assert digest(layer.weight) == row['fp16_sha256'], name
            tensors = load_file(root/row['file'])
            for key in ('fp16', 'int8', 'scale'):
                assert digest(tensors[key]) == row[key+'_sha256'], (name, key)
            zero = tensors['fp16'].float().abs().amax(dim=1) == 0
            assert int(zero.sum()) == row['fp16_zero_channels']
            assert (tensors['int8'][zero] == 0).all() and (tensors['scale'][zero] == 1).all()
            layer.weight = torch.nn.Parameter(
                torch_npu.npu_format_cast(tensors['int8'].npu(), 29), requires_grad=False)
            layer.register_parameter('qkvz_scale', torch.nn.Parameter(tensors['scale'].npu(), requires_grad=False))
            layer.quant_method = QKVZMethod()
            changed.append({'module': name, 'fp16_source_matches_production': True,
                            'zero_channels': row['fp16_zero_channels']})
            del tensors
        assert len(changed) == 30
        torch.npu.synchronize()
        delta = torch.npu.memory_allocated() - before
        self.model_runner.model_memory_usage += delta
        result = {'rank': rank, 'changed': changed, 'parameter_memory_delta_bytes': delta,
                  'manifest_sha256': hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()}
        Path(f'/experiment/qkvz-worker-rank{rank}.json').write_text(json.dumps(result, indent=2)+'\n')
