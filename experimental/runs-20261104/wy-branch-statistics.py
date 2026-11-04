#!/usr/bin/env python3
"""CPU 统计原生快速分支触发条件，便于与设备误差定位交叉核对。"""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2)
saved=torch.load(a.input,weights_only=True);q,k,v,g,beta=saved['inputs']
b,t,hk,kd=k.shape;hv=v.shape[2]
key=k.float().repeat_interleave(hv//hk,dim=2).transpose(1,2).reshape(b,hv,-1,64,kd)
beta=beta.float().transpose(1,2).reshape(b,hv,-1,64)
g=g.float().transpose(1,2).reshape(b,hv,-1,64).cumsum(-1)
lower=torch.tril((key*beta[...,None])@key.transpose(-1,-2)*torch.exp(torch.tril(g[...,None]-g[...,None,:])).tril(),diagonal=-1)
rows=lower.abs().sum(-1).amax(-1);frob=lower.square().sum((-2,-1));gate=(rows>=.75)&(frob<7.502183356513852)
value={'source':a.input.name,'max_row_sum':rows.tolist(),'frobenius_squared':frob.tolist(),'original_bypasses_fp32':gate.tolist()}
a.out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'source':a.input.name,'affected_heads_chunks':gate.nonzero().tolist(),'max_row_sum':rows.max().item(),'frob_max':frob.max().item()}))
