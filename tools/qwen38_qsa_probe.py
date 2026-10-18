# SPDX-License-Identifier: Apache-2.0
"""Real 310P QSA kernel probe at TP2 local Hq=12,Hkv=1,D=256.
Synthetic Q/K/V and selection; no model projections, router, or HCCL.
"""
import argparse,json,time
from pathlib import Path
import torch,torch_npu
p=argparse.ArgumentParser();p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(38310);torch.ops.load_library(a.library);torch.npu.set_device(0)
rows=[]
for tokens in (1,2,8,128):
    q=(torch.randn(tokens,12,256)*.2).half();k=(torch.randn(512,1,256)*.2).half();v=torch.randn_like(k)
    block_table=torch.tensor([[3,1,7,4,2,0,6,5]],dtype=torch.int32);cache=[]
    for dense in (k,v):
        nz=torch.zeros(8,16,64,16,dtype=torch.float16)
        for token in range(512):nz[block_table[0,token//64],:,token%64,:]=dense[token].reshape(16,16)
        cache.append(nz.npu())
    indices=torch.arange(0,32,2,dtype=torch.int32).repeat(tokens,1)
    selected=[i*4+j for i in indices[0].tolist() for j in range(4)]+[508,509,510]
    args=[q.npu(),*cache,indices.npu(),torch.full((tokens,),16,dtype=torch.int32).npu(),torch.full((tokens,),508,dtype=torch.int32).npu(),torch.full((tokens,),3,dtype=torch.int32).npu(),block_table.npu(),torch.tensor([0,tokens],dtype=torch.int32).npu()]
    def op():return torch.ops.qwen38_probe.qsa(*args,1/16,4,1)
    ref=torch.einsum('ths,shd->thd',torch.softmax(torch.einsum('thd,shd->ths',q.float(),k[selected].float())/16,dim=-1),v[selected].float()).half()
    for _ in range(5):op()
    torch.npu.synchronize();t=time.perf_counter()
    for _ in range(20):out=op()
    torch.npu.synchronize();elapsed=(time.perf_counter()-t)*1000/20;actual=out.cpu()
    torch.testing.assert_close(actual,ref,rtol=.005,atol=.003)
    rows.append({'tokens':tokens,'local_query_heads':12,'local_kv_heads':1,'head_dim':256,'selected_tokens':len(selected),'max_abs':float((actual-ref).abs().max()),'relative_l2':float((actual.float()-ref.float()).norm()/ref.float().norm()),'host_dispatch_sync_average_ms':elapsed,'status':'PASS'})
    print(json.dumps(rows[-1]),flush=True)
a.output.write_text(json.dumps({'status':'PASS','scope':__doc__,'rows':rows},indent=2))
