# SPDX-License-Identifier: Apache-2.0
"""Repeated-request diagnostic with compact boundaries and full layer-zero chunks."""
import importlib
import inspect
import json
from pathlib import Path
from wy_gate_worker import WYGateWorker


class WYRepeatWorker(WYGateWorker):
    def init_device(self):
        super().init_device()
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
        root = Path('/experiment');target=root/'captures';target.mkdir(exist_ok=True)
        module=importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
        count=0

        def context():
            marker=root/'capture-request.json'
            if not marker.exists():return None
            request=json.loads(marker.read_text());frame=inspect.currentframe().f_back
            layer=None
            while frame is not None:
                owner=frame.f_locals.get('self')
                if frame.f_code.co_name=='_forward_core' and hasattr(owner,'prefix'):
                    layer=owner.prefix;break
                frame=frame.f_back
            del frame
            if layer is None:raise RuntimeError('No layer for repeat boundary')
            return {'request':request,'rank':rank,'layer':layer}

        def sample(x):
            if not isinstance(x,torch.Tensor):return x
            flat=x.detach().reshape(-1);n=flat.numel()
            starts=sorted({0,max(0,n//2-256),max(0,n-512)})
            return {'shape':list(x.shape),'dtype':str(x.dtype),'starts':starts,
                    'data':torch.cat([flat[s:min(s+512,n)] for s in starts]).cpu()}

        def save(stage,ctx,record):
            nonlocal count
            count+=1;meta={**ctx,'stage':stage,'call':count}
            torch.save({'metadata':meta,**record},target/f'rank{rank}-{count:05d}-{stage}.pt')
            with (target/f'rank{rank}.jsonl').open('a') as f:f.write(json.dumps(meta)+'\n')

        active=module._compute_kernel_inputs_from_torch_wy
        def wy(q,k,v,g,beta,chunk_size):
            ctx=context();before=None if ctx is None else [sample(x) for x in (q,k,v,g,beta)]
            result=active(q,k,v,g,beta,chunk_size)
            if ctx is not None:save('WY',ctx,{'inputs':before,'outputs':[sample(x) for x in result]})
            return result
        module._compute_kernel_inputs_from_torch_wy=wy
        for name,stage in [('chunk_gated_delta_rule_fwd_h_vllm','H'),('chunk_fwd_o_vllm','O')]:
            original=getattr(torch.ops._C_ascend,name)
            def wrap(original=original,stage=stage):
                def call(*args,**kwargs):
                    ctx=context();record=None
                    if ctx is not None:
                        record={'args':[sample(x) for x in args],'kwargs':{k:sample(v) for k,v in kwargs.items()}}
                        if '.layers.0.' in ctx['layer'] and stage=='H':
                            record['initial_state_full']=kwargs['initial_state'].detach().cpu()
                    result=original(*args,**kwargs)
                    if ctx is not None:
                        values=result if isinstance(result,(tuple,list)) else [result]
                        record['outputs']=[sample(x) for x in values]
                        if '.layers.0.' in ctx['layer']:
                            # Single request, non-transposed [B,H,NT,K,V] NZ h contract.
                            cs=kwargs['chunk_size'];tokens=args[0].shape[2];chunks=tokens//cs
                            assert kwargs['cu_seqlens']==[0,tokens],kwargs['cu_seqlens']
                            selected=sorted({0,chunks-1});records=[]
                            for i in selected:
                                sl=slice(i*cs,(i+1)*cs)
                                if stage=='H':
                                    k,w,u=args[:3];h,vn,final=result
                                    after=h[:,:,i+1].detach().cpu() if i+1<chunks else final.detach().cpu()
                                    records.append({'index':i,'k':k[:,:,sl].detach().cpu(),'w':w[:,:,sl].detach().cpu(),
                                      'u':u[:,:,sl].detach().cpu(),'g':kwargs['g'][:,:,sl].detach().cpu(),
                                      'state_nz':h[:,:,i].detach().cpu(),'v_new':vn[:,:,sl].detach().cpu(),
                                      'next_state':after,'next_state_layout':'NZ' if i+1<chunks else 'ND'})
                                else:
                                    q,k,vn,h,scale=args[:5]
                                    records.append({'index':i,'q':q[:,:,sl].detach().cpu(),'k':k[:,:,sl].detach().cpu(),
                                      'v_new':vn[:,:,sl].detach().cpu(),'g':kwargs['g'][:,:,sl].detach().cpu(),
                                      'state_nz':h[:,:,i].detach().cpu(),'scale':scale,'out':result[:,:,sl].detach().cpu()})
                            record['layer_zero_chunks']=records
                        save(stage,ctx,record)
                    return result
                return call
            setattr(torch.ops._C_ascend,name,wrap())
        print('[wy_repeat_ready] diagnostic_only',flush=True)
