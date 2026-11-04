# SPDX-License-Identifier: Apache-2.0
"""生产 GDN 完整调用链单变量上下文对照；不改 kernel，不加载权重。"""
import argparse
import importlib
import json
from pathlib import Path
import torch
import torch_npu

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=Path, required=True)
p.add_argument('--out-dir', type=Path, required=True)
p.add_argument('--context', choices=('original', 'sync-before-o', 'clone-before-o', 'retain-interfaces'), required=True)
p.add_argument('--device', type=int, choices=(0,1), default=0)
p.add_argument('--repeats', type=int, default=48)
a = p.parse_args()
a.out_dir.mkdir(exist_ok=False)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
torch.set_num_threads(4)
torch.npu.set_device(a.device)
torch_npu.npu.set_compile_mode(jit_compile=False)
m = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
from native_wy import PRODUCTION_CHUNK_SHA256
import hashlib
assert hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() == PRODUCTION_CHUNK_SHA256
saved = torch.load(a.work/'diagnostic-case-1.pt', weights_only=True)
q,k,v,g,beta,state = saved['cpu_inputs']
inputs = [x.npu() for x in (q,k,v,g,beta)]
kw = dict(initial_state=state.npu(),output_final_state=True,head_first=False,use_qk_l2norm_in_kernel=False)
original_wy = m._compute_kernel_inputs_from_torch_wy
original_h = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h
original_o = torch.ops._C_ascend.chunk_fwd_o
capture = {}
retained = []

def wy(*args):
    result = original_wy(*args)
    capture['wy'] = result
    return result

def fwd_h(*args, **kwargs):
    result = original_h(*args, **kwargs)
    capture['h'] = result
    return result

def fwd_o(q,k,v,h,scale,**kwargs):
    capture['o_inputs'] = (q,k,v,h,kwargs['g'])
    if a.context == 'sync-before-o':
        torch.npu.synchronize()
    if a.context == 'clone-before-o':
        q,k,v,h,g = (x.clone() for x in capture['o_inputs'])
        kwargs['g'] = g
    capture['o_actual_inputs'] = (q,k,v,h,kwargs['g'])
    result = original_o(q,k,v,h,scale,**kwargs)
    capture['raw'] = (result,)
    return result

def compare(actual, expected):
    actual,expected = actual.float(),expected.float()
    delta = actual-expected
    finite = bool(torch.isfinite(actual).all())
    bad = delta.abs() > (2e-4+.04*expected.abs())
    relative_rmse = (delta.square().mean().sqrt()/expected.square().mean().sqrt().clamp_min(1e-30)).item()
    where = bad.nonzero()
    return dict(within_tolerance=finite and not bool(bad.any()) and relative_rmse <= .02,
                finite=finite, max_abs=delta.abs().max().item(), relative_rmse=relative_rmse,
                outside_tolerance=int(bad.sum()), affected_tokens=where[:,1].unique().tolist() if len(where) else [])

def metadata(x):
    return dict(shape=list(x.shape),stride=list(x.stride()),storage_offset=x.storage_offset(),
                data_ptr=x.data_ptr(),storage_nbytes=x.untyped_storage().nbytes(),format=int(torch_npu.get_npu_format(x)))

m._compute_kernel_inputs_from_torch_wy = wy
torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h = fwd_h
torch.ops._C_ascend.chunk_fwd_o = fwd_o
first = None
rows = []
with (a.out_dir/'calls.jsonl').open('x') as log:
    for repeat in range(a.repeats):
        capture.clear()
        result = m.chunk_gated_delta_rule_310(*inputs,**kw)
        # 接口处无CPU取样；original和retain组也没有新增主机同步。
        output, final_state = (x.cpu() for x in result)
        cpu = {key:tuple(x.cpu() for x in tensors) for key,tensors in capture.items()}
        if first is None:
            first = dict(cpu=cpu,output=output,state=final_state)
        row = dict(repeat=repeat,context=a.context,device=a.device,output=compare(output,saved['reference'][0]),
                   state=compare(final_state,saved['reference'][1]),
                   interfaces_equal_first={key:[bool(torch.equal(x,y)) for x,y in zip(tensors,first['cpu'][key])] for key,tensors in cpu.items()},
                   output_equal_first=bool(torch.equal(output,first['output'])),state_equal_first=bool(torch.equal(final_state,first['state'])),
                   o_input_metadata=[metadata(x) for x in capture['o_actual_inputs']],
                   raw_matches_final=bool(torch.equal(cpu['raw'][0].transpose(1,2)[:,:q.shape[1]].contiguous(),output)),
                   status='DIAGNOSTIC_ONLY')
        rows.append(row)
        log.write(json.dumps(row)+'\n');log.flush()
        print(json.dumps(row),flush=True)
        if a.context == 'retain-interfaces':
            retained.append(dict(capture))
    summary = dict(context=a.context,device=a.device,tested=len(rows),
                   output_within_tolerance=sum(r['output']['within_tolerance'] for r in rows),
                   state_within_tolerance=sum(r['state']['within_tolerance'] for r in rows),
                   output_equal_first=sum(r['output_equal_first'] for r in rows),
                   state_equal_first=sum(r['state_equal_first'] for r in rows),
                   raw_matches_final=sum(r['raw_matches_final'] for r in rows),
                   interface_variation={key:sum(not all(r['interfaces_equal_first'][key]) for r in rows) for key in first['cpu']},
                   scope='Single-variable diagnostic only; a passing context is not a kernel fix or performance acceptance')
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
