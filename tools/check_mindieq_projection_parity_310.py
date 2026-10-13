# SPDX-License-Identifier: Apache-2.0
"""真实专家逐投影数值定位：native/custom A8代码、整数dot+FP32 scale参考。
合成激活及固定路由，无router/attention/模型质量/性能外推。
"""
import argparse,json
from itertools import accumulate
from pathlib import Path
import torch
import torch_npu
from safetensors import safe_open
from check_w4_repeatability_310 import load_bank
from check_w4_grouped_prefill_310 import signed_nibbles

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--native-library',required=True)
p.add_argument('--grouped-library',required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(310)
torch.ops.load_library(a.native_library);torch.ops.load_library(a.grouped_library)
index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
def codes(byte):
    return signed_nibbles(byte).to(torch.int32)
def error(actual,ref):
    d=actual.float()-ref.float()
    return {'differing_elements':int((d!=0).sum()),'elements':d.numel(),'max_abs':float(d.abs().max()),'relative_l2':float(d.norm()/ref.float().norm().clamp_min(1e-12)),'finite':bool(torch.isfinite(actual).all())}
with a.out.open('x') as f:
    def emit(row):
        f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
    emit({'kind':'config','scope':__doc__,'seed':310,'experts':8,'reference':'CPU INT32 matmul, FP32 weight scale then activation scale, FP16 rounding'})
    for names in [('gate_proj','up_proj'),('down_proj',)]:
        bank=load_bank(a.model,names)
        wc=[];sc=[]
        for expert in range(8):
            parts=[];scales=[]
            for name in names:
                key=f'model.language_model.layers.0.mlp.experts.{expert}.{name}.weight'
                skey=key.replace('.weight','.weight_scale')
                with safe_open(a.model/index[key],framework='pt',device='cpu') as sf:parts.append(codes(sf.get_tensor(key)))
                with safe_open(a.model/index[skey],framework='pt',device='cpu') as sf:
                    v=sf.get_tensor(skey);assert torch.equal(v,v[:,:1].expand_as(v));scales.append(v[:,0].float())
            wc.append(torch.cat(parts));sc.append(torch.cat(scales))
        n,k=wc[0].shape
        weight=bank[0].view(torch.int32).reshape(8,-1);scale=torch.stack(sc).npu()
        for m in (1,17,128,512,4096):
            x=(torch.randn(m,k)*.2).half();x[0]*=.0001
            ends=list(accumulate([m//8+(e<m%8) for e in range(8)]));groups=torch.tensor(ends,dtype=torch.int64).npu();device=x.npu()
            native_pack=torch.ops.qwen36_int4.pack(device)
            q,s=torch.ops.qwen36_mindieq_grouped.quantize(device)
            native_output=torch.ops.qwen36_int4.matmul(*native_pack,*bank,groups).cpu()
            grouped_output=torch.ops.qwen36_mindieq_grouped.matmul(q,s,weight,scale,groups).cpu()
            low,high,nscale,_=[v.cpu() for v in native_pack]
            nq=codes(low)+16*codes(high)+8
            cq=q.cpu()[:m].int();cs=s.cpu()[:m]
            ns=nscale.reshape(m,k//128,8)[:,0,0]
            assert torch.equal(nscale.reshape(m,k//128,8),ns[:,None,None].expand(m,k//128,8))
            maximum=x.float().abs().amax(1);rs=torch.where(maximum==0,torch.ones_like(maximum),maximum/127)
            rq=(x.float()/rs[:,None]).round().clamp(-127,127).int()
            reference=[];variants={name:[] for name in ("activation_first","combined_scale","fp64_scale","group128_scaled_sum")};start=0
            for e,end in enumerate(ends):
                dot=rq[start:end]@wc[e].T
                reference.append(((dot.float()*sc[e][None,:])*rs[start:end,None]).half())
                variants['activation_first'].append(((dot.float()*rs[start:end,None])*sc[e][None,:]).half())
                variants['combined_scale'].append((dot.float()*(sc[e][None,:]*rs[start:end,None])).half())
                grouped=torch.zeros_like(dot,dtype=torch.float32)
                for first in range(0,k,128):
                    partial=rq[start:end,first:first+128]@wc[e][:,first:first+128].T
                    grouped+=((partial.float()*sc[e][None,:])*rs[start:end,None])
                variants['group128_scaled_sum'].append(grouped.half())
                variants['fp64_scale'].append((dot.double()*sc[e][None,:].double()*rs[start:end,None].double()).half())
                start=end
            ref=torch.cat(reference)
            row={'kind':'case','projection':'+'.join(names),'m':m,'n':n,'k':k,'native_codes_vs_cpu':int((nq!=rq).sum()),'custom_codes_vs_cpu':int((cq!=rq).sum()),'custom_vs_native_codes':int((cq!=nq).sum()),'native_scale_vs_cpu':error(ns,rs),'custom_scale_vs_cpu':error(cs,rs),'native_vs_cpu':error(native_output,ref),'mindieq_vs_cpu':error(grouped_output,ref),'mindieq_vs_native':error(grouped_output,native_output)}
            row['native_vs_scale_variants']={name:error(native_output,torch.cat(values)) for name,values in variants.items()}
            row['mindieq_vs_scale_variants']={name:error(grouped_output,torch.cat(values)) for name,values in variants.items()}
            assert row['native_vs_cpu']['finite'] and row['mindieq_vs_cpu']['finite']
            emit(row)
    emit({'kind':'summary','status':'completed','cases':10})
