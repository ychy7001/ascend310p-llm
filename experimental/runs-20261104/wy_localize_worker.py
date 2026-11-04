# SPDX-License-Identifier: Apache-2.0
"""实验容器观测：保存 Torch WY 边界，返回原 Torch 张量，不执行候选。"""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from cann921_gdn_pair_worker import CANN921GDNPairWorker


class WYLocalizeWorker(CANN921GDNPairWorker):
    def init_device(self):
        super().init_device()
        import torch
        import torch_npu
        from vllm.distributed import get_tensor_model_parallel_rank
        torch.set_num_threads(1)
        rank = get_tensor_model_parallel_rank()
        module = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
        original = module._compute_kernel_inputs_from_torch_wy
        count = 0
        layer_counts = {}
        out = Path('/experiment/captures')
        out.mkdir(exist_ok=True)

        def metadata(x):
            return {'shape': list(x.shape), 'stride': list(x.stride()),
                    'dtype': str(x.dtype), 'contiguous': x.is_contiguous(),
                    'storage_offset': x.storage_offset(),
                    'format': int(torch_npu.get_npu_format(x))}

        def observe(q, k, v, g, beta, chunk_size):
            nonlocal count
            marker = Path('/experiment/capture-request.json')
            if not marker.exists():
                return original(q, k, v, g, beta, chunk_size)
            request = json.loads(marker.read_text())
            count += 1
            frame = inspect.currentframe().f_back
            chunk_context = None
            layer = None
            while frame is not None:
                if frame.f_code.co_name == 'chunk_gated_delta_rule_310':
                    chunk_context = frame.f_locals.copy()
                owner = frame.f_locals.get('self')
                if frame.f_code.co_name == '_forward_core' and hasattr(owner, 'prefix'):
                    layer = owner.prefix
                    break
                frame = frame.f_back
            del frame
            if layer is None or chunk_context is None:
                raise RuntimeError('WY call has no identifiable layer/chunk context')
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
            inputs = [x.detach().cpu() for x in (q, k, v, g, beta)]
            state = chunk_context['initial_state']
            state_cpu = None if state is None else state.detach().cpu()
            cu = chunk_context['cu_seqlens']
            cu_cpu = None if cu is None else cu.detach().cpu()
            info = {'rank': rank, 'call': count, 'layer': layer,
                    'layer_call': layer_counts[layer], 'request': request,
                    'chunk_size': chunk_size, 'chunks': q.shape[1] // chunk_size,
                    'inputs': [metadata(x) for x in (q, k, v, g, beta)],
                    'raw_q_shape': list(chunk_context['q'].shape),
                    'cu_seqlens': None if cu_cpu is None else cu_cpu.tolist(),
                    'initial_state': None if state is None else metadata(state),
                    'initial_state_nonzero': bool(state_cpu is not None and torch.count_nonzero(state_cpu)),
                    'qk_l2norm': chunk_context['use_qk_l2norm_in_kernel'],
                    'chunk_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
            result = original(q, k, v, g, beta, chunk_size)
            record = {'metadata': info, 'inputs': inputs,
                      'torch_outputs': [x.detach().cpu() for x in result],
                      'initial_state': state_cpu, 'cu_seqlens': cu_cpu}
            target = out / f'rank{rank}-call{count:04d}.pt'
            torch.save(record, target)
            with (out / f'rank{rank}.jsonl').open('a') as stream:
                stream.write(json.dumps(info) + '\n')
            print('[wy_localize] ' + json.dumps({'rank': rank, 'call': count,
                  'layer': layer, 'q_shape': list(q.shape), 'request': request}), flush=True)
            return result

        module._compute_kernel_inputs_from_torch_wy = observe
        print('[wy_localize_ready] ' + json.dumps({'rank': rank, 'path': 'Torch WY unchanged'}), flush=True)
