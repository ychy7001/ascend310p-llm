# SPDX-License-Identifier: Apache-2.0
"""Expose chunk_fwd_o's 64x64 causal mask as its output, without GDN/WY.
q,k share one unit coordinate; V contains an identity basis; H=0, g=0.
The exact first 64 output columns must therefore equal tril(ones(64,64)).
"""
import json
from pathlib import Path
import torch


def run(rank, output):
    q=torch.zeros(1,8,64,128,dtype=torch.float16);q[...,0]=1
    k=q.clone()
    v=torch.zeros(1,24,64,128,dtype=torch.float16)
    v[0,:,:,:64]=torch.eye(64,dtype=torch.float16)
    h=torch.zeros(1,24,1,128,128,dtype=torch.float16)
    g=torch.zeros(1,24,64,dtype=torch.float32)
    q,k,v,h,g=[t.to(f'npu:{rank}') for t in (q,k,v,h,g)]
    expected=torch.zeros(1,24,64,128,dtype=torch.float16)
    expected[:,:,:,:64]=torch.ones(64,64,dtype=torch.float16).tril()
    records=[]
    for repeat in range(3):
        out=torch.ops._C_ascend.chunk_fwd_o_vllm(q,k,v,h,1.0,g=g,g_gamma=None,cu_seqlens=[0,64],chunk_indices=[0,0],chunk_size=64,transpose_state_layout=False)
        torch.npu.synchronize()
        got=out.cpu(); bad=got!=expected
        records.append({'repeat':repeat,'bad_elements':bad.sum().item(),'bad_rows':bad.any(-1).any(1).nonzero().tolist(),'max_abs_error':(got-expected).abs().max().item(),'row56_head0':got[0,0,56,:64].tolist()})
    result={'rank':rank,'scope':'exact isolated causal mask test','cases':records}
    (Path(output)/f'chunk-causal-rank{rank}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return all(c['bad_elements']==0 for c in records)
