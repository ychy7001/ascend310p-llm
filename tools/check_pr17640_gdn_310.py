# SPDX-License-Identifier: Apache-2.0
"""完整 PR GDN：生产 head 形状、尾块、初始状态、CPU 数值参考和重复性。"""
import argparse
import json
import time
from pathlib import Path
from statistics import median
import torch
import torch_npu
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',required=True)
p.add_argument('--library',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
torch.set_num_threads(4)
torch.ops.load_library(a.library)
ns={'__name__':'standalone_pr17640'}
source=Path(a.source).read_text().replace('torch.ops._C_ascend.chunk_gated_delta_rule_compute_wy', 'torch.ops.pr17640_gdn.compute').replace('"_C_ascend"','"pr17640_gdn"').replace('torch.ops._C_ascend.','torch.ops.pr17640_gdn.')
# Source remains external; only operator namespace is redirected for isolation.
exec(compile(source,a.source,'exec'),ns)
ns['enable_custom_op']=lambda:True
# Capability probe and call must refer to the same independent registration.
ns['_can_use_npu_compute_wy'] = lambda *args, **kwargs: True
can_use=ns['_can_use_npu_compute_wy']

def cosine(x,y):
    return torch.nn.functional.cosine_similarity(x.flatten().double(),y.flatten().double(),dim=0).item()

def bench(fn):
    for _ in range(5):fn()
    torch.npu.synchronize()
    times=[]
    for _ in range(20):
        start=time.perf_counter();fn();torch.npu.synchronize()
        times.append((time.perf_counter()-start)*1000)
    return times

with open(a.out,'x') as file:
    for tokens in (111,128,512,1024):
        torch.manual_seed(tokens)
        q=torch.nn.functional.normalize(torch.randn(1,tokens,16,128),dim=-1).half()
        k=torch.nn.functional.normalize(torch.randn_like(q).float(),dim=-1).half()
        v=(torch.randn(1,tokens,32,128)*.1).half()
        g=-torch.rand(1,tokens,32)
        beta=(.1+.8*torch.rand(1,tokens,32)).half()
        state=(torch.randn(1,32,128,128)*.01).half()
        expected=ns['_torch_chunk_gated_delta_rule_chunked'](q.repeat_interleave(2,dim=2),k.repeat_interleave(2,dim=2),v,g,beta,initial_state=state,output_final_state=True)
        device=tuple(x.npu() for x in (q,k,v,g,beta,state))
        def run():
            return ns['chunk_gated_delta_rule_310'](*device[:5],initial_state=device[5],output_final_state=True,head_first=False,use_qk_l2norm_in_kernel=False)
        actual=run()
        cpu=tuple(x.cpu().float() for x in actual)
        similarities=[cosine(x,y.float()) for x,y in zip(cpu,expected)]
        assert min(similarities)>.99, similarities
        errors=[(x-y.float()).abs().max().item() for x,y in zip(cpu,expected)]
        for _ in range(5):
            repeated=run()
            for x,y in zip(repeated,cpu):torch.testing.assert_close(x.cpu().float(),y,rtol=0,atol=0)
        times=bench(run)
        row={'tokens':tokens,'q_heads':16,'v_heads':32,'head_dim':128,'normalized_qk':True,'nonzero_initial_state':True,'status':'PASS','cosine_output_state':similarities,'max_abs_output_state':errors,'deterministic_repeats':5,'warmup':5,'repeats':20,'gdn_ms':times,'median_gdn_ms':median(times),'timing_scope':'full PR GDN synchronized host call; not model throughput'}
        # Keep same FwdH/FwdO and change only WY; this isolates the WY contribution.
        ns['_can_use_npu_compute_wy']=lambda *args,**kwargs:False
        fallback=run()
        for x,y in zip(fallback,cpu): assert cosine(x.cpu().float(),y)>.99
        row['torch_wy_gdn_ms']=bench(run)
        row['median_torch_wy_gdn_ms']=median(row['torch_wy_gdn_ms'])
        row['wy_contribution_speedup']=row['median_torch_wy_gdn_ms']/row['median_gdn_ms']
        ns['_can_use_npu_compute_wy']=can_use
        file.write(json.dumps(row)+'\n');file.flush()
        print(json.dumps({k:v for k,v in row.items() if not k.endswith('_ms') or k.startswith('median')}),flush=True)
