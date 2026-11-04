# SPDX-License-Identifier: Apache-2.0
"""单进程重放捕获的Dense量化Matmul；不加载模型，CPU采样参考独立计算。"""
import argparse
import json
import os
from pathlib import Path
import torch
import torch_npu
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--case',type=Path);p.add_argument('--synthetic',action='store_true');p.add_argument('--seed',type=int,default=310);p.add_argument('--out',type=Path,required=True)
p.add_argument('--device',type=int,choices=[0,1],default=0);p.add_argument('--rows',type=int);p.add_argument('--activation-layout',choices=['captured','nd','nz'],default='captured')
a=p.parse_args()
if bool(a.case)==a.synthetic:p.error('choose exactly one of --case and --synthetic')
torch.set_num_threads(4);torch.npu.set_device(a.device);torch_npu.npu.set_compile_mode(jit_compile=False)
from torch_npu.npu._format import Format
if a.synthetic:
 torch.manual_seed(a.seed)
 tensors={'x':torch.zeros((4096,8704),dtype=torch.int8),
          'weight':torch.randint(-128,128,(5120,8704),dtype=torch.int8).T,
          'scale':torch.rand(5120,dtype=torch.float32)*.0074+.00028,
          'pertoken_scale':torch.zeros(4096,dtype=torch.float32)}
 metadata={k:dict(shape=list(v.shape),stride=list(v.stride()),dtype=str(v.dtype),format=29 if k=='weight' else 2) for k,v in tensors.items()}
 saved=dict(tensors=tensors,metadata=metadata,output_dtype=torch.float16)
else:
 with torch.serialization.safe_globals([Format]):
  saved=torch.load(a.case,weights_only=True)
cpu=saved['tensors'];meta=saved['metadata']
if a.rows:
 cpu['x']=cpu['x'][:a.rows].contiguous()
 if 'pertoken_scale' in cpu:cpu['pertoken_scale']=cpu['pertoken_scale'][:a.rows].contiguous()

def restore(name,value):
 fmt=meta[name]['format']
 if name=='x' and a.activation_layout!='captured':fmt=29 if a.activation_layout=='nz' else 2
 if value.ndim==2 and meta[name]['stride']==[1,meta[name]['shape'][0]]:
  device=value.T.contiguous().npu()
  if fmt not in (0,2):device=torch_npu.npu_format_cast(device,fmt)
  return device.T
 device=value.npu()
 if fmt not in (0,2):device=torch_npu.npu_format_cast(device,fmt)
 return device
row=dict(allocator=os.environ.get('PYTORCH_NPU_ALLOC_CONF'),device=a.device,capture_file=f'synthetic-seed-{a.seed}' if a.synthetic else a.case.name,rows=cpu['x'].shape[0],activation_layout=a.activation_layout,metadata=meta,scope='captured-input diagnostic; no model/performance claim')
a.out.write_text(json.dumps(dict(row,status='preparing'))+'\n')
try:
 devices={name:restore(name,value) for name,value in cpu.items()}
 row['replay_formats']={k:torch_npu.get_npu_format(v) for k,v in devices.items()}
 a.out.write_text(json.dumps(dict(row,status='calling'))+'\n');print(json.dumps(dict(row,status='calling')),flush=True)
 output=torch_npu.npu_quant_matmul(devices['x'],devices['weight'],devices['scale'],
  pertoken_scale=devices.get('pertoken_scale'),bias=devices.get('bias'),output_dtype=saved['output_dtype']).cpu()
 m,n=output.shape;ri=torch.tensor(sorted({0,m//2,m-1}));ci=torch.arange(0,n,max(1,n//32))[:32]
 expected=(cpu['x'][ri].int()@cpu['weight'][:,ci].int()).float()
 if 'bias' in cpu and cpu['bias'].dtype==torch.int32:expected+=cpu['bias'][ci].float()
 expected*=cpu['scale'][ci].float()
 if 'pertoken_scale' in cpu:expected*=cpu['pertoken_scale'][ri,None].float()
 if 'bias' in cpu and cpu['bias'].dtype!=torch.int32:expected+=cpu['bias'][ci].float()
 actual=output[ri[:,None],ci].float()
 torch.testing.assert_close(actual,expected,atol=.003,rtol=.01)
 assert torch.isfinite(output).all()
 if a.synthetic:assert torch.count_nonzero(output).item()==0
 row.update(status='PASS',max_sample_abs=(actual-expected).abs().max().item(),sample_elements=actual.numel())
except Exception as error:row.update(status='FAIL',error=str(error));raise
finally:a.out.write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
