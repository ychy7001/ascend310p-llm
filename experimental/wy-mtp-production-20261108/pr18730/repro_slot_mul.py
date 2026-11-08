"""Independent scalar-integer slot arithmetic checks on the current 310P stack."""
import argparse
import json
import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument('--dtype', choices=['int32', 'int64'], default='int32')
p.add_argument('--length', type=int, default=2)
p.add_argument('--method', choices=['mul', 'shift', 'wide', 'padded'], default='mul')
p.add_argument('--jit', action='store_true')
a = p.parse_args()
torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=a.jit)
dtype = getattr(torch, a.dtype)
reference = torch.arange(1, a.length + 1, dtype=dtype)
x = reference.npu()
torch.npu.synchronize()
print(json.dumps({'case': vars(a), 'stage': 'before'}), flush=True)
for _ in range(3):
    if a.method == 'shift':
        result = torch.bitwise_left_shift(x, 7)
    elif a.method == 'wide':
        result = (x.to(torch.int64) * 128).to(dtype)
    elif a.method == 'padded':
        padded = torch.zeros(16, dtype=dtype, device=x.device)
        padded[:a.length].copy_(x)
        result = (padded * 128)[:a.length]
    else:
        result = x * 128
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), reference * 128, rtol=0, atol=0)
print(json.dumps({'case': vars(a), 'stage': 'passed'}), flush=True)
