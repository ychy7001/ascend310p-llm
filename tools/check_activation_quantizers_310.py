# SPDX-License-Identifier: Apache-2.0
"""在独占310P比较native per-token pack与torch-npu dynamic_quant；合成输入诊断。"""
import argparse
import json
from pathlib import Path
import torch
import torch_npu

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--fp32-input',action='store_true',help='仅dynamic_quant采用FP32输入，native仍为FP16')
p.add_argument('--pre-scale',type=float,default=1.,help='诊断FP16输入预放大，输出scale除以该倍数')
a=p.parse_args()
if not a.pre_scale > 0: p.error('--pre-scale must be positive')
torch.set_num_threads(4)
torch.manual_seed(310)
torch.ops.load_library(a.library)

def unpack(packed):
    data=packed.to(torch.int16)
    nibble=torch.stack((data & 15, (data >> 4) & 15),dim=-1).flatten(1)
    return torch.where(nibble>=8,nibble-16,nibble)

rows=[]
for width in (512,2048):
    x=(torch.randn(16,width)*.2).half()
    x[0].zero_()
    x[1].fill_(1)
    x[2].fill_(-1)
    x[3]=torch.linspace(-1,1,width).half()
    x[4]=torch.linspace(0,1,width).half()
    x[5]=torch.linspace(-1,0,width).half()
    x[6]=(torch.arange(width)%255-127).half()/127
    x[7]*=.0001
    device=x.npu()
    lo,hi,scale,_=torch.ops.qwen36_int4.pack(device)
    dynamic_input=device.float() if a.fp32_input else device
    q,s=torch_npu.npu_dynamic_quant(dynamic_input*a.pre_scale if a.pre_scale!=1. else dynamic_input)
    s=s.float()/a.pre_scale
    torch.npu.synchronize()
    native=unpack(lo.cpu())+16*unpack(hi.cpu())+8
    ns=scale.cpu().reshape(16,-1)[:,0]
    q=q.cpu().to(torch.int16);s=s.cpu().float().reshape(16,-1)[:,0]
    maximum=x.float().abs().amax(1)
    refscale=torch.where(maximum==0,torch.ones_like(maximum),maximum/127)
    ref=(x.float()/refscale[:,None]).round().clamp(-127,127).to(torch.int16)
    if not torch.isfinite(ns).all() or not torch.isfinite(s).all():
        raise RuntimeError('Nonfinite activation scales')
    for i in range(16):
        nd=native[i].float()*ns[i];dd=q[i].float()*s[i]
        rows.append({'width':width,'row':i,'max_abs_input':float(maximum[i]),
            'native_scale':float(ns[i]),'dynamic_scale':float(s[i]),
            'native_code_range':[int(native[i].min()),int(native[i].max())],
            'dynamic_code_range':[int(q[i].min()),int(q[i].max())],
            'native_vs_dynamic_code_mismatches':int((native[i]!=q[i]).sum()),
            'native_vs_symmetric127_mismatches':int((native[i]!=ref[i]).sum()),
            'native_max_abs_reconstruction_error':float((nd-x[i].float()).abs().max()),
            'dynamic_max_abs_reconstruction_error':float((dd-x[i].float()).abs().max()),
            'reconstruction_difference_l2':float(torch.linalg.vector_norm(nd-dd))})
result={'scope':__doc__,'seed':310,'pre_scale':a.pre_scale,'dynamic_input_dtype':'float32' if a.fp32_input else 'float16','rows':rows,'limitation':'合成激活、单次诊断；不能代表模型质量或性能'}
with a.out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False))
