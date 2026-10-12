# SPDX-License-Identifier: Apache-2.0
"""Fused/reference write parity on independent caches, with mutable graph inputs."""
import argparse
import json
import torch
import torch_npu
import c8_cache_310 as module
from c8_cache_310 import C8Cache310

p=argparse.ArgumentParser();p.add_argument('--library',required=True);a=p.parse_args()
torch.ops.load_library(a.library)
torch.npu.set_device(0);torch.npu.set_compile_mode(jit_compile=False)
torch.manual_seed(753)
for heads,dim,count in [(1,128,1),(2,256,2),(16,256,65),(2,256,4096)]:
    capacity=max(128,count+64)
    reference=C8Cache310(capacity//64,64,heads,dim)
    fused=C8Cache310(capacity//64,64,heads,dim)
    key=torch.randn(count,heads,dim).half().npu();value=torch.randn_like(key)
    if count>1: key[0].zero_();value[0].mul_(1e-6)
    slots=torch.arange(count,dtype=torch.int32,device='npu').flip(0).contiguous()
    if count>2:slots[-1]=-1
    def verify():
        torch.npu.synchronize()
        for actual,expected in [(fused.k,reference.k),(fused.v,reference.v)]:
            error=(actual.cpu().short()-expected.cpu().short()).abs().max().item()
            assert error<=1,error
        for actual,expected in [(fused.ks,reference.ks),(fused.vs,reference.vs)]:
            torch.testing.assert_close(actual.cpu(),expected.cpu(),atol=2e-5,rtol=.002)
        print(json.dumps({'status':'PASS','heads':heads,'dim':dim,'tokens':count,
            'int8_allowed_bin_difference':1,'scope':'random finite FP16, padded/shuffled slots'}),flush=True)
    module._FUSED_WRITE=False;reference.write(key,value,slots)
    module._FUSED_WRITE=True;fused.write(key,value,slots);verify()
    if count==2:
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph):fused.write(key,value,slots)
        for factor in (0.01,4.0):
            key.mul_(factor);value.add_(.1);slots.copy_(slots.flip(0))
            module._FUSED_WRITE=False;reference.write(key,value,slots)
            graph.replay();verify()
        print(json.dumps({'status':'PASS','mutable_graph_inputs_scale_slots':True}),flush=True)
