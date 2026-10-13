# SPDX-License-Identifier: Apache-2.0
"""八expert真实权重MLP诊断：固定分组，无router/共享专家/Attention/route combine。
计入激活量化、Gate/Up、SwiGLU、Down、Python分组调用；不外推模型吞吐。
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path
from statistics import median

import torch
import torch_npu
from safetensors import safe_open
from check_w4_repeatability_310 import load_bank
from check_w4_grouped_prefill_310 import signed_nibbles


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--library',required=True)
    p.add_argument('--upstream-test',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--synchronize-producers',action='store_true')
    p.add_argument('--experts',type=int,choices=[8,64,256],default=8)
    p.add_argument('--rows',type=int,nargs='+',default=[8,128,512,1024,4096])
    a=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(310)
    torch.ops.load_library(a.library)
    spec=importlib.util.spec_from_file_location('external_qmm',a.upstream_test)
    external=importlib.util.module_from_spec(spec);spec.loader.exec_module(external)
    index=json.loads((a.model/'model.safetensors.index.json').read_text())['weight_map']
    native=[load_bank(a.model,names) for names in [('gate_proj','up_proj'),('down_proj',)]]
    other=[]
    for names in [('gate_proj','up_proj'),('down_proj',)]:
        banks=[]
        for expert in range(8):
            parts=[]
            for name in names:
                ts=[]
                for suffix in ['weight','weight_scale','weight_offset']:
                    key=f'model.language_model.layers.0.mlp.experts.{expert}.{name}.{suffix}'
                    with safe_open(a.model/index[key],framework='pt',device='cpu') as f:ts.append(f.get_tensor(key))
                parts.append(ts)
            codes,scales,offsets=[torch.cat(c) for c in zip(*parts)]
            assert torch.count_nonzero(offsets)==0
            assert torch.equal(scales,scales[:,:1].expand_as(scales))
            weight=signed_nibbles(codes).to(torch.int8);n,k=weight.shape
            packed=torch.from_numpy(external.pack_biased_nz_int32_vec(weight.T.contiguous().numpy(),k,n).copy()).npu()
            w8=torch_npu.npu_format_cast(weight.npu(),29).T
            banks.append((packed,w8,scales[:,0].float().npu()))
        other.append(banks)
    if a.experts!=8:
        native=[[t.repeat(a.experts//8,1,1) for t in bank] for bank in native]
        other=[bank*(a.experts//8) for bank in other]
    with a.out.open('x') as f:
        f.write(json.dumps({'kind':'config','scope':__doc__,'warmup':5,'repeats':20,'experts':a.experts,
                            'weight_scope':'real expert0..7 repeated if experts>8; synthetic occupancy',
                            'activations':'native fused pack vs torch_npu.npu_dynamic_quant',
                            'synchronize_producers':a.synchronize_producers,
                            'non_native_swiglu':'FP32 silu and multiply -> FP16 -> dynamic quant'})+'\n');f.flush()
        for rows in a.rows:
            for routing in ['balanced','one_expert']:
                ends=[rows*(i+1)//a.experts for i in range(a.experts)] if routing=='balanced' else [rows]*a.experts
                boundaries=torch.tensor(ends,dtype=torch.int64).npu()
                x=(torch.randn(rows,2048)*.2).half().npu()
                def native_mlp():
                    pack=torch.ops.qwen36_int4.pack(x)
                    gate=torch.ops.qwen36_int4.matmul(*pack,*native[0],boundaries)
                    if rows<=128:
                        pack=torch.ops.qwen36_int4.swiglu_pack(gate)
                    else:
                        g,u=gate.float().chunk(2,dim=-1)
                        pack=torch.ops.qwen36_int4.pack((torch.nn.functional.silu(g)*u).half())
                    return torch.ops.qwen36_int4.matmul(*pack,*native[1],boundaries)
                def alternate_mlp(backend):
                    outputs=[];start=0
                    # Each expert slice resident; counts fixed on host for this diagnostic.
                    for expert,end in enumerate(ends):
                        if end==start:continue
                        q,s=torch_npu.npu_dynamic_quant(x[start:end])
                        s=s.float().contiguous()
                        q=torch_npu.npu_format_cast(q,2).contiguous()
                        w,w8,ws=other[0][expert]
                        if backend=='mindieq' and a.synchronize_producers:torch.npu.synchronize()
                        gate=external.OP(q,s,w,ws) if backend=='mindieq' else torch_npu.npu_quant_matmul(q,w8,ws,pertoken_scale=s,output_dtype=torch.float16)
                        g,u=gate.float().chunk(2,dim=-1)
                        hidden=(torch.nn.functional.silu(g)*u).half()
                        q,s=torch_npu.npu_dynamic_quant(hidden)
                        s=s.float().contiguous()
                        q=torch_npu.npu_format_cast(q,2).contiguous()
                        w,w8,ws=other[1][expert]
                        if backend=='mindieq' and a.synchronize_producers:torch.npu.synchronize()
                        out=external.OP(q,s,w,ws) if backend=='mindieq' else torch_npu.npu_quant_matmul(q,w8,ws,pertoken_scale=s,output_dtype=torch.float16)
                        outputs.append(out);start=end
                    return torch.cat(outputs)
                funcs={'native_w4':native_mlp,'mindieq_w4':lambda:alternate_mlp('mindieq'),'w8':lambda:alternate_mlp('w8')}
                reference=native_mlp().cpu().float();errors={}
                for name,fn in funcs.items():
                    output=fn().cpu().float()
                    errors[name]={'max_abs':(output-reference).abs().max().item(),
                                  'relative_l2':((output-reference).norm()/reference.norm().clamp_min(1e-12)).item(),
                                  'finite':bool(torch.isfinite(output).all())}
                    assert errors[name]['finite'], (rows,routing,name,errors[name])
                    # Independent per-token quantizers can differ at rounding boundaries.
                    assert errors[name]['relative_l2']<.03,errors[name]
                    for _ in range(5):fn()
                torch.npu.synchronize();samples={name:[] for name in funcs}
                for repeat in range(20):
                    order=list(funcs);order=order[repeat%3:]+order[:repeat%3]
                    for name in order:
                        begin=time.perf_counter();funcs[name]();torch.npu.synchronize()
                        samples[name].append((time.perf_counter()-begin)*1000)
                row={'kind':'measurement','rows':rows,'routing':routing,'group_ends':ends,
                     'status':'PASS','errors_vs_native':errors,'latency_ms':samples,
                     'median_ms':{name:median(v) for name,v in samples.items()},
                     'scope':'synchronized full MLP fixed grouping; no router/combine'}
                f.write(json.dumps(row)+'\n');f.flush()
                print(json.dumps({k:v for k,v in row.items() if k!='latency_ms'}),flush=True)
        f.write(json.dumps({'kind':'summary','status':'PASS','cases':2*len(a.rows)})+'\n')


if __name__=='__main__':main()
