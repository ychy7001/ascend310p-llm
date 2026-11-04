# SPDX-License-Identifier: Apache-2.0
"""完整调用顺序下保留WY/FwdH/FwdO边界；所有CPU取样延至输出完成之后。"""
import argparse
import importlib
import json
from pathlib import Path
import torch
import torch_npu
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--work',type=Path,required=True)
p.add_argument('--out-dir',type=Path,required=True)
p.add_argument('--maskfix-library','--fwd-o-library',dest='maskfix_library',type=Path)
p.add_argument('--fwd-o-namespace',default='maskfix_fwd_o')
p.add_argument('--fwd-o-entry',default='run')
p.add_argument('--fwd-h-library',type=Path)
p.add_argument('--fwd-h-namespace',default='pr17640_gdn_pair')
p.add_argument('--wy-library',type=Path)
p.add_argument('--repeats',type=int,default=24)
p.add_argument('--device',type=int,choices=(0,1),default=0)
a=p.parse_args();a.out_dir.mkdir(exist_ok=False)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
torch.set_num_threads(4);torch.npu.set_device(a.device)
torch_npu.npu.set_compile_mode(jit_compile=False)
m=importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
saved=torch.load(a.work/'diagnostic-case-1.pt',weights_only=True)
q,k,v,g,beta,state=saved['cpu_inputs'];inputs=[x.npu() for x in (q,k,v,g,beta)]
kw=dict(initial_state=state.npu(),output_final_state=True,head_first=False,use_qk_l2norm_in_kernel=False)
if a.wy_library:
 from native_wy import install
 install(str(a.wy_library))
original_wy=m._compute_kernel_inputs_from_torch_wy
if a.fwd_h_library:torch.ops.load_library(str(a.fwd_h_library))
original_h=getattr(torch.ops,a.fwd_h_namespace).fwd_h if a.fwd_h_library else torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h
if a.maskfix_library:torch.ops.load_library(str(a.maskfix_library))
original_o=getattr(getattr(torch.ops,a.fwd_o_namespace),a.fwd_o_entry) if a.maskfix_library else torch.ops._C_ascend.chunk_fwd_o
capture={};first={};frozen=None

def wy(*args):
 result=original_wy(*args) if mode=='torch_wy' else tuple(x.clone() for x in frozen)
 capture['wy']=result
 return result

def fwd_h(*args,**kw):
 result=original_h(*args,**kw);capture['fwd_h']=result;return result

def fwd_o(q,k,v,h,scale,**kw):
 capture['fwd_o_inputs']=(q,k,v,h,kw['g'])
 result=original_o(q,k,v,h,scale,**kw)
 capture['fwd_o_raw']=(result,)
 return result

m._compute_kernel_inputs_from_torch_wy=wy
torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h=fwd_h
torch.ops._C_ascend.chunk_fwd_o=fwd_o
rows=[]
with (a.out_dir/'intermediates.jsonl').open('x') as log:
 for mode in ('torch_wy','frozen_wy_clone'):
  for repeat in range(a.repeats):
   capture.clear()
   result=m.chunk_gated_delta_rule_310(*inputs,**kw)
   output=result[0].cpu()
   final_state=result[1].cpu()
   # No host synchronization was added at the interfaces before FwdO above.
   cpu={name:[x.cpu() for x in tensors] for name,tensors in capture.items()}
   if mode not in first:
    first[mode]=cpu
    if frozen is None:frozen=tuple(x.clone() for x in capture['wy'])
    torch.save(cpu,a.out_dir/(mode+'-first.pt'))
   raw_output=cpu['fwd_o_raw'][0].transpose(1,2)[:,:111].contiguous()
   delta=output.float()-saved['reference'][0]
   finite=bool(torch.isfinite(output).all())
   relative_rmse=(delta.square().mean().sqrt()/saved['reference'][0].square().mean().sqrt().clamp_min(1e-30)).item()
   state_delta=final_state.float()-saved['reference'][1]
   state_bad=state_delta.abs()>(.0002+.04*saved['reference'][1].abs())
   state_relative_rmse=(state_delta.square().mean().sqrt()/saved['reference'][1].square().mean().sqrt().clamp_min(1e-30)).item()
   state_valid=bool(torch.isfinite(final_state).all()) and not bool(state_bad.any()) and state_relative_rmse<=.02
   bad=delta.abs()>(.0002+.04*saved['reference'][0].abs())
   where=bad.nonzero()
   equal={name:[bool(torch.equal(x,y)) for x,y in zip(tensors,first[mode][name])] for name,tensors in cpu.items()}
   row=dict(mode=mode,repeat=repeat,interfaces_equal_first=equal,
    max_abs=delta.abs().max().item(),outside_tolerance=int(bad.sum()),
    finite=finite,relative_rmse=relative_rmse,
    within_tolerance=finite and not bool(bad.any()) and relative_rmse<=.02,
    state_within_tolerance=state_valid,state_max_abs=state_delta.abs().max().item(),state_relative_rmse=state_relative_rmse,
    raw_matches_final=bool(torch.equal(raw_output,output)),
    raw_max_abs=(raw_output.float()-saved['reference'][0]).abs().max().item(),
    affected_tokens=where[:,1].unique().tolist() if len(where) else [],
    status='DIAGNOSTIC_ONLY')
   if bool(bad.any()):torch.save(dict(interfaces=cpu,output=output),a.out_dir/f'{mode}-bad-{repeat}.pt')
   rows.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
 summary={mode:dict(tested=sum(r['mode']==mode for r in rows),
  outside_tolerance_runs=sum(r['mode']==mode and not r['within_tolerance'] for r in rows),
  state_outside_tolerance_runs=sum(r['mode']==mode and not r['state_within_tolerance'] for r in rows),
  raw_output_differs_final_cpu=sum(r['mode']==mode and not r['raw_matches_final'] for r in rows),
  interface_variation={name:sum(r['mode']==mode and not all(r['interfaces_equal_first'][name]) for r in rows)
   for name in ('wy','fwd_h','fwd_o_inputs')}) for mode in ('torch_wy','frozen_wy_clone')}
 (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
