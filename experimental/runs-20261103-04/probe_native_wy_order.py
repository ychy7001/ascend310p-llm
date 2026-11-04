# SPDX-License-Identifier: Apache-2.0
"""111-token 失败样本交替复测；只诊断，不放宽门限。"""
import argparse
import importlib
import json
from pathlib import Path
import torch
import torch_npu
from native_wy import install
from check_native_wy import compare
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--work',type=Path,required=True)
p.add_argument('--out-dir',type=Path)
a=p.parse_args()
out=a.out_dir or a.work
if a.out_dir:out.mkdir(exist_ok=False)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
torch.set_num_threads(4);torch.npu.set_device(0);torch_npu.npu.set_compile_mode(jit_compile=False)
m=importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
original=m._compute_kernel_inputs_from_torch_wy
install(str(a.work/'wy-extension/pr17640_wy.so'))
native=m._compute_kernel_inputs_from_torch_wy
saved=torch.load(a.work/'diagnostic-case-1.pt',weights_only=True)
q,k,v,g,beta,state=saved['cpu_inputs'];inputs=[x.npu() for x in (q,k,v,g,beta)]
kw=dict(initial_state=state.npu(),output_final_state=True,head_first=False,use_qk_l2norm_in_kernel=False)
with (out/'order-diagnostic.jsonl').open('x') as f:
 for repeat in range(12):
  for name,helper in (('native',native),('torch',original)) if repeat%2 else (('torch',original),('native',native)):
   m._compute_kernel_inputs_from_torch_wy=helper
   actual=m.chunk_gated_delta_rule_310(*inputs,**kw)
   row=dict(repeat=repeat,mode=name,status='DIAGNOSTIC_ONLY')
   try: row['metrics']=compare(actual,saved['reference']);row['within_tolerance']=True
   except AssertionError as e:row['within_tolerance']=False;row['error']=str(e)
   row['output_max_abs']=(actual[0].cpu().float()-saved['reference'][0]).abs().max().item()
   torch.save([x.cpu() for x in actual],out/f'order-{repeat}-{name}.pt')
   f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
