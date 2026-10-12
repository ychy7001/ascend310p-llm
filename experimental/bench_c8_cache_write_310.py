# SPDX-License-Identifier: Apache-2.0
"""Separate dynamic C8 write/launch cost from historical attention conversion."""
import argparse
import json
import statistics
import time
import torch
import torch_npu
from c8_cache_310 import C8Cache310
import c8_cache_310 as cache_module

p=argparse.ArgumentParser()
p.add_argument('--library',required=True)
p.add_argument('--profile-dir')
p.add_argument('--compare-fused',action='store_true')
a=p.parse_args()
torch.ops.load_library(a.library)
torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)
torch.manual_seed(910)
paths=[]
for count in (1,2,256,4096):
    key=torch.randn(count,2,256).half().npu()
    value=torch.randn_like(key)
    slots=torch.arange(count,dtype=torch.int32,device='npu')
    cache=C8Cache310(64,64,2,256)
    shape=(64,32,64,16)
    k16=torch_npu.empty_with_format(size=shape,dtype=torch.float16,device='npu',acl_format=29)
    v16=torch_npu.empty_with_format(size=shape,dtype=torch.float16,device='npu',acl_format=29)
    def c8(key=key,value=value,slots=slots,cache=cache):
        cache.write(key,value,slots)
    def native(key=key,value=value,slots=slots,k16=k16,v16=v16):
        torch_npu._npu_reshape_and_cache(key=key,value=value,
            key_cache=k16,value_cache=v16,slot_indices=slots)
    def select(mode,fn):
        def call():
            cache_module._FUSED_WRITE=mode
            fn()
        return call
    candidates=([('c8_unfused_write',select(False,c8)),('c8_fused_write',select(True,c8))]
                if a.compare_fused else [('c8_dynamic_write',c8)])
    candidates.append(('fp16_reshape_cache',native))
    for name,fn in candidates:
        for _ in range(5): fn()
        torch.npu.synchronize()
        samples=[]
        for _ in range(20):
            start=time.perf_counter();fn();torch.npu.synchronize()
            samples.append((time.perf_counter()-start)*1e6)
        print(json.dumps({'path':name,'tokens':count,'median_us':statistics.median(samples),
            'samples_us':samples,'warmup':5,'repeats':20,
            'timing':'host launch+synchronize; unfused C8 allocates intermediates, fused uses preallocated caches'}),flush=True)
        if count==1: paths.append((name,fn))
if a.profile_dir:
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(a.profile_dir)) as profiler:
        for _ in range(5):
            for name,fn in paths:
                with torch.profiler.record_function(name): fn()
        torch.npu.synchronize();profiler.step()
