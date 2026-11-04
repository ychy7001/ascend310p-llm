# SPDX-License-Identifier: Apache-2.0
"""仅诊断：保存首个Dense down_proj量化Matmul的实际输入；不测性能、不替换算法。"""
import hashlib
import json
from pathlib import Path
import torch
import torch_npu
from vllm_ascend._310p.worker_310p import NPUWorker310

class QuantTraceWorker(NPUWorker310):
    def init_device(self):
        super().init_device()
        import vllm_ascend._310p.quantization.methods.w8a8_dynamic as method
        digest=hashlib.sha256(Path(method.__file__).read_bytes()).hexdigest()
        if digest!='3314bba76317196e52a32506d29436bd5d88922cc4b68443d3878ea111ed04cf':
            raise RuntimeError('Quant trace requires audited production dynamic method')
        original=torch_npu.npu_quant_matmul
        rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else self.local_rank
        log=Path(f'/experiment/quant-trace-rank{rank}.jsonl');saved=False
        def traced(x,weight,scale,*args,**kwargs):
            nonlocal saved
            if not saved and tuple(weight.shape)==(8704,5120):
                saved=True
                tensors=dict(x=x,weight=weight,scale=scale,**{k:v for k,v in kwargs.items() if isinstance(v,torch.Tensor)})
                meta={k:dict(shape=list(v.shape),stride=list(v.stride()),dtype=str(v.dtype),format=int(torch_npu.get_npu_format(v))) for k,v in tensors.items()}
                with log.open('a') as f:f.write(json.dumps(dict(kind='before_capture',tensors=meta,output_dtype=str(kwargs.get('output_dtype'))))+'\n')
                cpu={k:v.detach().cpu() for k,v in tensors.items()}
                torch.save(dict(tensors=cpu,metadata=meta,output_dtype=kwargs.get('output_dtype')),f'/experiment/quant-matmul-case-rank{rank}.pt')
                summary={k:dict(finite=bool(torch.isfinite(v).all()),min=v.min().item(),max=v.max().item()) for k,v in cpu.items()}
                with log.open('a') as f:f.write(json.dumps(dict(kind='capture_complete',statistics=summary))+'\n')
            return original(x,weight,scale,*args,**kwargs)
        torch_npu.npu_quant_matmul=traced
        print('[quant-trace] capture enabled for first K8704 N5120 call; host synchronization added; diagnostic only',flush=True)
