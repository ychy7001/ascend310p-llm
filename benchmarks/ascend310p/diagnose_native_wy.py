# SPDX-License-Identifier: Apache-2.0
"""失败门禁诊断；完整保留 baseline/candidate/reference 指标，不签发 PASS。"""
import argparse
import importlib
import json
from pathlib import Path

import torch
import torch_npu
from check_native_wy import recurrent_reference
from native_wy import install

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(4)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
torch.npu.set_device(0)
torch_npu.npu.set_compile_mode(jit_compile=False)
m=importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
original=m._compute_kernel_inputs_from_torch_wy
install(a.library)
native=m._compute_kernel_inputs_from_torch_wy

def metrics(actual,expected):
 result=[]
 for x,y in zip(actual,expected):
  x,y=x.detach().cpu().float(),y.detach().cpu().float()
  d=(x-y).abs()
  result.append(dict(finite=bool(torch.isfinite(x).all()),max_abs=d.max().item(),
      relative_rmse=((x-y).square().mean().sqrt()/y.square().mean().sqrt().clamp(min=1e-30)).item(),
      cosine=torch.nn.functional.cosine_similarity(x.flatten().double(),y.flatten().double(),dim=0).item(),
      outside_tolerance=int((d>(.0002+.04*y.abs())).sum()),elements=y.numel()))
 return result
with a.out.open('x') as f:
 for idx,(tokens,hk,hv,decay,cu) in enumerate([(64,8,16,.01,None),(111,8,16,1.,None),
     (128,16,32,1.,None),(512,8,16,8.,None),(177,8,16,1.,[0,1,65,177]),(128,8,24,1.,None)]):
  torch.manual_seed(310+idx)
  q=torch.nn.functional.normalize(torch.randn(1,tokens,hk,128),dim=-1).half()
  k=torch.nn.functional.normalize(torch.randn_like(q).float(),dim=-1).half()
  v=(torch.randn(1,tokens,hv,128)*.1).half();g=-torch.rand(1,tokens,hv)*decay
  beta=(.05+.9*torch.rand(1,tokens,hv)).half();cu=cu or [0,tokens]
  state=(torch.randn(len(cu)-1,hv,128,128)*.01).half()
  expected=recurrent_reference(q,k,v,g,beta,state,cu)
  inputs=[x.npu() for x in (q,k,v,g,beta)]
  kw=dict(initial_state=state.npu(),output_final_state=True,head_first=False,use_qk_l2norm_in_kernel=False)
  if len(cu)>2:kw['cu_seqlens']=torch.tensor(cu,dtype=torch.int64,device='npu')
  m._compute_kernel_inputs_from_torch_wy=original
  baseline=m.chunk_gated_delta_rule_310(*inputs,**kw)
  m._compute_kernel_inputs_from_torch_wy=native
  candidate=m.chunk_gated_delta_rule_310(*inputs,**kw)
  row=dict(case=idx,tokens=tokens,hk=hk,hv=hv,decay=decay,cu=cu,status='DIAGNOSTIC_ONLY',
      baseline_vs_reference=metrics(baseline,expected),candidate_vs_reference=metrics(candidate,expected),
      candidate_vs_baseline=metrics(candidate,baseline))
  torch.save(dict(cpu_inputs=[q,k,v,g,beta,state],cu=cu,baseline=[x.cpu() for x in baseline],
      candidate=[x.cpu() for x in candidate],reference=expected),str(a.out.parent/f'diagnostic-case-{idx}.pt'))
  f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
