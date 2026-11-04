#!/usr/bin/env python3
"""CPU-only compatibility check against the exact compiled higher-order dispatcher."""
import json,sys
sys.path.insert(0,'/observe')
import torch
from observe_decode import shape_dispatch_mode
from torch._higher_order_ops.auto_functionalize import auto_functionalized_v2

@torch.library.custom_op('decode_observation_test::copy_add',mutates_args={'out'})
def copy_add(x:torch.Tensor,out:torch.Tensor)->None:
    out.copy_(x+1)

x=torch.arange(12,dtype=torch.float32).reshape(3,4)
base=torch.zeros_like(x)
def run():
    return auto_functionalized_v2(torch.ops.decode_observation_test.copy_add.default,
        x=x,_out_base_index=0,_all_bases=[base])
reference=run()
with shape_dispatch_mode([]):observed=run()
assert len(reference)==len(observed)
for left,right in zip(reference,observed):
    if isinstance(left,torch.Tensor):assert torch.equal(left,right)
    else:assert left==right
assert torch.equal(base,torch.zeros_like(base))
print(json.dumps({'cpu_only':True,'higher_order':'auto_functionalized_v2','bitwise_equal':True,'input_unchanged':True,'torch_version':torch.__version__}))
