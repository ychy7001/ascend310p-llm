# SPDX-License-Identifier: Apache-2.0
"""Warmup/repeated device latency of chunk_fwd_o, not end-to-end prefill."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False)
torch.set_num_threads(1)
records=[]
for tokens in (64,128,256):
    q=torch.zeros(1,8,tokens,128,dtype=torch.float16);q[...,0]=1
    k=q.clone();v=torch.zeros(1,24,tokens,128,dtype=torch.float16)
    for start in range(0,tokens,64):v[0,:,start:start+64,:64]=torch.eye(64,dtype=torch.float16)
    h=torch.zeros(1,24,tokens//64,128,128,dtype=torch.float16)
    g=torch.zeros(1,24,tokens,dtype=torch.float32)
    q,k,v,h,g=[t.npu() for t in (q,k,v,h,g)]
    def run():
        return torch.ops._C_ascend.chunk_fwd_o_vllm(q,k,v,h,1.0,g=g,g_gamma=None,cu_seqlens=[0,tokens],chunk_indices=[j for c in range(tokens//64) for j in (0,c)],chunk_size=64,transpose_state_layout=False)
    for _ in range(10):out=run()
    torch.npu.synchronize()
    pairs=[];start=time.perf_counter()
    for _ in range(50):
        begin,end=torch.npu.Event(enable_timing=True),torch.npu.Event(enable_timing=True)
        begin.record();out=run();end.record();pairs.append((begin,end))
    torch.npu.synchronize();wall=time.perf_counter()-start
    raw=[begin.elapsed_time(end) for begin,end in pairs]
    record={'tokens':tokens,'q_heads':8,'v_heads':24,'head_dim':128,'warmup':10,'repeats':50,'device_ms':raw,'median_device_ms':statistics.median(raw),'p95_device_ms':sorted(raw)[47],'mean_host_wall_ms':wall*1000/50}
    records.append(record);print(json.dumps(record),flush=True)
a.output.write_text(json.dumps({'scope':'single native GDN output kernel only; synthetic inputs; excludes WY and other model layers','cases':records},indent=2))
