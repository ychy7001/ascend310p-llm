# SPDX-License-Identifier: Apache-2.0
"""Real PA probe: future draft slots are invisible and rejected slots can be overwritten.

Tests C8 storage/PA semantics, not the model scheduler or linear-state rollback.
"""
import argparse
import json
import math
import torch
import torch_npu
from c8_cache_310 import C8Cache310

parser=argparse.ArgumentParser()
parser.add_argument('--library',required=True)
args=parser.parse_args()
torch.ops.load_library(args.library)
torch.manual_seed(214)
table=torch.tensor([2,0,3],dtype=torch.int32).npu()
query=torch.randn(1,16,256).half().npu()


def slots(start,count):
    positions=torch.arange(start,start+count,dtype=torch.int32,device='npu')
    return table[(positions//64).long()]*64+positions%64


def attend(cache,length):
    pages=math.ceil(length/64)
    shape=(pages,32,64,16)
    k=torch_npu.empty_with_format(size=shape,dtype=torch.float16,device='npu',acl_format=29)
    v=torch_npu.empty_with_format(size=shape,dtype=torch.float16,device='npu',acl_format=29)
    torch.ops.npu.c8_cache_to_nz(cache.k,cache.v,cache.ks,cache.vs,k,v,64,table[:pages].contiguous())
    output=torch.empty_like(query)
    torch_npu._npu_paged_attention(query=query,key_cache=k,value_cache=v,
        num_kv_heads=2,num_heads=16,scale_value=1/16,
        block_table=torch.arange(pages,dtype=torch.int32,device='npu').reshape(1,-1),
        context_lens=torch.tensor([length],dtype=torch.int32,device='npu'),out=output)
    torch.npu.synchronize()
    return output.cpu().float()


for prefix in (63,64,127):
    candidate=C8Cache310(4,64,2,256)
    fresh=C8Cache310(4,64,2,256)
    key=torch.randn(prefix+1,2,256).half().npu()
    value=torch.randn_like(key)
    candidate.write(key[:prefix],value[:prefix],slots(0,prefix))
    before=attend(candidate,prefix)
    # Two speculative positions, with conspicuously different values/scales.
    wrong_key=torch.full((2,2,256),32,dtype=torch.float16,device='npu')
    wrong_value=torch.full_like(wrong_key,-64)
    candidate.write(wrong_key,wrong_value,slots(prefix,2))
    future=attend(candidate,prefix)
    torch.testing.assert_close(future,before,atol=0,rtol=0)
    # Reject draft and overwrite only the next committed position.
    candidate.write(key[-1:],value[-1:],slots(prefix,1))
    fresh.write(key,value,slots(0,prefix+1))
    actual=attend(candidate,prefix+1)
    reference=attend(fresh,prefix+1)
    error=(actual-reference).abs().max().item()
    torch.testing.assert_close(actual,reference,atol=.002,rtol=.002)
    print(json.dumps({'status':'PASS','prefix':prefix,'future_tail_invisible':True,
        'rejected_slot_overwrite':True,'max_abs_vs_fresh_c8':error,
        'scope':'C8 storage plus native PA; not scheduler/linear state rollback'}),flush=True)
