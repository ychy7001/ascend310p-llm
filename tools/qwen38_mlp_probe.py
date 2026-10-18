# SPDX-License-Identifier: Apache-2.0
"""Real experts 0/256, fused gate/up -> SwiGLU pack -> down on 310P.
Two synthetic routes per token, no router/combine/HCCL/Attention.
"""
import argparse,json
from pathlib import Path
import torch,torch_npu
from safetensors.torch import load_file
from check_w4_repeatability_310 import native_weight,native_metadata
p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(38110);torch.ops.load_library(a.library);torch.npu.set_device(0)
tensors=load_file(a.fixture/'experts.safetensors');bank={};float_weights={}
for name,projs in [('gate_up',('gate_proj','up_proj')),('down',('down_proj',))]:
    columns=[[],[],[],[]];fw=[]
    for expert in (0,256):
        parts=[];weights=[]
        for proj in projs:
            stem=f'model.language_model.layers.0.mlp.experts.{expert}.{proj}.'
            w,s,z=[tensors[stem+k] for k in ('weight','weight_scale','weight_offset')]
            native,sums=native_weight(w);parts.append((native,native_metadata(s),native_metadata(z.half()),sums))
            n,k2=w.shape;u=torch.stack((w&15,(w>>4)&15),-1).flatten(-2);q=torch.where(u>=8,u-16,u).float()
            weights.append(((q.reshape(n,-1,128)-z.float().unsqueeze(-1))*s.float().unsqueeze(-1)).flatten(1))
        for c,part in zip(columns,zip(*parts)):c.append(torch.cat(part))
        fw.append(torch.cat(weights))
    bank[name]=[torch.stack(c).npu() for c in columns];float_weights[name]=fw

def quant(x):
    g=x.float().reshape(x.shape[0],-1,128);s=g.abs().amax(-1,keepdim=True)/127;s=torch.where(s==0,torch.ones_like(s),s)
    return ((g/s).round().clamp(-127,127)*s).flatten(1)
rows=[]
for m in (1,2,4,8):
    x=(torch.randn(m,2560)*.2).half();ids=torch.tensor([0,1]*m,dtype=torch.int32).npu();xn=x.npu()
    def operation():
        gu=torch.ops.qwen36_int4.matmul(*torch.ops.qwen36_int4.pack(xn),*bank['gate_up'],ids)
        return torch.ops.qwen36_int4.matmul(*torch.ops.qwen36_int4.swiglu_pack(gu),*bank['down'],ids)
    refx=quant(x);expected=[]
    for token in range(m):
        for e in (0,1):
            gu=(refx[token]@float_weights['gate_up'][e].T).half().float()
            hidden=(torch.nn.functional.silu(gu[:640])*gu[640:]).half().unsqueeze(0)
            expected.append((quant(hidden)[0]@float_weights['down'][e].T).half())
    ref=torch.stack(expected).float();out=operation().cpu().float();err=float((out-ref).norm()/ref.norm())
    assert torch.isfinite(out).all() and err<.01,(m,err)
    for _ in range(10):torch.testing.assert_close(operation().cpu().float(),out,rtol=0,atol=0)
    rows.append({'tokens':m,'routes':m*2,'rel_l2_vs_cpu_w4a8_mlp':err,'max_abs':float((out-ref).abs().max()),'repeat_bitwise_equal_runs':10,'status':'PASS'})
    print(json.dumps(rows[-1]),flush=True)
a.output.write_text(json.dumps({'status':'PASS','scope':__doc__,'rows':rows},indent=2))
