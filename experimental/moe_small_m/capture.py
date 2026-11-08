"""Eager observation hook: save real grouped inputs, banks, and routing counts.

Enabled only inside the disposable observation replica. Captures are for
operator replay; this execution mode is never used as a performance result.
"""
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys


def patch(module):
    import torch
    import torch_npu
    original = module.quant_apply_mlp
    bank_ids = {}
    captured = {}
    directory = Path('/experiment/capture')
    directory.mkdir(exist_ok=True)
    pid = os.getpid()

    @functools.wraps(original)
    def observed(hidden_states, w1, w1_scale, w2, w2_scale, group_list, group_list_type=1):
        m = int(hidden_states.shape[0])
        arm = directory / 'arm.json'
        if m not in (8,16,24,32) or not arm.exists():
            return original(hidden_states,w1,w1_scale,w2,w2_scale,group_list,group_list_type)
        run = json.loads(arm.read_text())
        if m != run['m']:
            return original(hidden_states,w1,w1_scale,w2,w2_scale,group_list,group_list_type)
        bank_key = w1.data_ptr()
        if bank_key not in bank_ids:bank_ids[bank_key] = len(bank_ids)
        layer = bank_ids[bank_key]
        ends = torch.cumsum(group_list,dim=0) if group_list_type == 1 else group_list
        key = (run['label'],m)
        sample = captured.get(key,0)
        if sample >= 4:
            return original(hidden_states,w1,w1_scale,w2,w2_scale,group_list,group_list_type)
        endscpu = ends.cpu()
        counts = torch.diff(endscpu,prepend=torch.zeros(1,dtype=torch.int64))
        with (directory / f'routes-{pid}.jsonl').open('a') as f:
            f.write(json.dumps({'run':run['label'],'pid':pid,'layer_index':layer,'m':m,
                'counts':counts.tolist(),'last_end':endscpu[-1].item()})+'\n')
        # Sample the first encountered layer per rank, with three distinct
        # stable decode steps per request. All layers' routing stays recorded.
        save = layer == 0 and sample < 3
        if save:
            for kind,weight,scale in [('up',w1,w1_scale),('down',w2,w2_scale)]:
                bank = directory / f'bank-{pid}-{kind}.pt'
                if not bank.exists():
                    logical = torch_npu.npu_format_cast(weight,2).cpu()
                    torch.save({'weight':logical,'scale':scale.cpu()},bank)
            def snapshot(x,kind):
                name=f'input-{pid}-{run["label"]}-{sample}-{kind}.pt'
                torch.save({'x':x.cpu(),'ends':endscpu,'bank':f'bank-{pid}-{kind}.pt'}, directory/name)
            snapshot(hidden_states,'up')
        gate = torch_npu.npu_quant_grouped_matmul_dequant(hidden_states,w1,w1_scale,ends,quant_mode='pertoken')
        act = torch_npu.npu_swiglu(gate)
        if save:snapshot(act,'down')
        result = torch_npu.npu_quant_grouped_matmul_dequant(act,w2,w2_scale,ends,quant_mode='pertoken')
        if layer == 0:captured[key] = sample+1
        return result
    module.quant_apply_mlp = observed


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path,target=None):
        if fullname != 'vllm_ascend._310p.fused_moe.moe_mlp':return None
        spec = importlib.machinery.PathFinder.find_spec(fullname,path)
        loader = spec.loader
        class Loader(importlib.abc.Loader):
            def create_module(self,spec):return loader.create_module(spec)
            def exec_module(self,module):
                loader.exec_module(module)
                patch(module)
        spec.loader = Loader()
        return spec


def install():sys.meta_path.insert(0,Finder())
