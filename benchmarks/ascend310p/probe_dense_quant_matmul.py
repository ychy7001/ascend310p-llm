# SPDX-License-Identifier: Apache-2.0
"""无模型权重的310P动态W8A8矩阵乘诊断；每进程一个形状/布局，严格对照采样CPU整数积。"""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--m',type=int,required=True);p.add_argument('--k',type=int,required=True);p.add_argument('--n',type=int,required=True)
p.add_argument('--layout',choices=['production','transpose-then-nz','nd'],required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(310)
torch.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False)
row=dict(m=a.m,k=a.k,n=a.n,layout=a.layout,scope='synthetic operator diagnostic; no model or performance claim',torch=torch.__version__,torch_npu=torch_npu.__version__)
a.out.write_text(json.dumps(dict(row,status='preparing'))+'\n')
try:
 x=torch.randint(-10,11,(a.m,a.k),dtype=torch.int8)
 w=torch.randint(-8,9,(a.n,a.k),dtype=torch.int8)
 xn=x.npu();wn=w.npu()
 if a.layout=='production':wn=torch_npu.npu_format_cast(wn,29).transpose(0,1)
 elif a.layout=='transpose-then-nz':wn=torch_npu.npu_format_cast(wn.transpose(0,1).contiguous(),29)
 else:wn=wn.transpose(0,1).contiguous()
 scale=torch.full((a.n,),.001,dtype=torch.float32,device='npu')
 token=torch.full((a.m,),.01,dtype=torch.float32,device='npu')
 row.update(weight_shape=list(wn.shape),weight_stride=list(wn.stride()),weight_format=torch_npu.get_npu_format(wn))
 a.out.write_text(json.dumps(dict(row,status='calling'))+'\n');print(json.dumps(dict(row,status='calling')),flush=True)
 y=torch_npu.npu_quant_matmul(xn,wn,scale,pertoken_scale=token,output_dtype=torch.float16).cpu()
 rid=torch.tensor(sorted({0,a.m//2,a.m-1}));cid=torch.arange(0,a.n,max(1,a.n//32))[:32]
 expected=(x[rid].int()@w[cid].int().T).float()*.001*.01
 actual=y[rid[:,None],cid].float()
 torch.testing.assert_close(actual,expected,atol=3e-5,rtol=.005)
 assert torch.isfinite(y).all()
 row.update(status='PASS',max_sample_abs=(actual-expected).abs().max().item(),sample_elements=actual.numel())
except Exception as error:
 row.update(status='FAIL',error=str(error));raise
finally:
 a.out.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
