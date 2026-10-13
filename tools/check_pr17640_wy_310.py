# SPDX-License-Identifier: Apache-2.0
"""Standalone PR WY correctness/latency on Qwen3.6 GVA shapes, external CPU oracle."""
import argparse
import ast
import json
import time
from statistics import median
from pathlib import Path
import torch
import torch_npu
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source', required=True)
p.add_argument('--library', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
torch.set_num_threads(4)
tree = ast.parse(Path(a.source).read_text())
names = {'_expand_qk_to_v_heads', '_compute_kernel_inputs_from_torch_wy'}
selected = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
ns = {'torch':torch}
exec(compile(selected, a.source, 'exec'),ns)
ref = ns['_compute_kernel_inputs_from_torch_wy']
torch.ops.load_library(a.library)
with open(a.out, 'x') as out:
    for tokens in (128,512,1024):
        for decay in (.01,1.):
            torch.manual_seed(tokens)
            q=(torch.randn(1,tokens,16,128)*.01).half()
            k=(torch.randn_like(q)*.01).half()
            v=(torch.randn(1,tokens,32,128)*.01).half()
            g=-torch.rand(1,tokens,32)*decay
            beta=(.1+.2*torch.rand(1,tokens,32)).half()
            cpu=(q,k,v,g,beta)
            expected=ref(*cpu,64)
            device=tuple(x.npu() for x in cpu)
            def op(): return torch.ops.pr17640_wy.compute(*device,64)
            actual=op()
            errors=[]
            for i,(value,target) in enumerate(zip(actual,expected)):
                value=value.cpu().float(); target=target.float()
                if i in (0,1): torch.testing.assert_close(value,target,rtol=0,atol=0)
                else: torch.testing.assert_close(value,target,rtol=.03,atol=.0002)
                errors.append((value-target).abs().max().item())
            def bench(fn):
                for _ in range(5): fn()
                torch.npu.synchronize()
                times=[]
                for _ in range(20):
                    start=time.perf_counter();fn();torch.npu.synchronize()
                    times.append((time.perf_counter()-start)*1e3)
                return times
            row={'tokens':tokens,'q_heads':16,'v_heads':32,'head_dim':128,'decay':decay,'status':'PASS','max_abs_by_output':errors,
                 'timing_scope':'synchronized host call; not pure kernel', 'warmup':5,'repeats':20}
            row['wy_ms']=bench(op)
            row['fallback_ms']=bench(lambda:ref(*device,64))
            row['wy_median_ms']=median(row['wy_ms']);row['fallback_median_ms']=median(row['fallback_ms'])
            row['speedup']=row['fallback_median_ms']/row['wy_median_ms']
            out.write(json.dumps(row)+'\n');out.flush()
            print(json.dumps({k:v for k,v in row.items() if k not in ('wy_ms','fallback_ms')}),flush=True)
