"""同代码测 NPU copy、MatMul、eager/Graph 主机与设备时间；非LLM benchmark。"""
import argparse
import json
import statistics
import time
import torch
import torch_npu

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--label', required=True)
a = p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(42)
torch.npu.set_compile_mode(jit_compile=False)

def measure(name, fn, *, traffic=None, flops=None, graph=False):
    for _ in range(5): fn()
    torch.npu.synchronize()
    if graph:
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g): held = fn()
        fn = g.replay
        for _ in range(5): fn()
        torch.npu.synchronize()
    host, device = [], []
    for _ in range(30):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start = time.perf_counter()
        begin.record(); result = fn(); end.record()
        torch.npu.synchronize()
        host.append((time.perf_counter() - start) * 1000)
        device.append(begin.elapsed_time(end))
    h, d = statistics.median(host), statistics.median(device)
    row = dict(label=a.label, name=name, mode='graph' if graph else 'eager',
               warmup=5, repeats=30, host_ms=host, device_ms=device,
               median_host_ms=h, median_device_ms=d,
               scope='event bracket includes device queue/dependencies; host includes events/sync')
    if traffic: row['effective_read_write_GB_s'] = traffic / (d * 1e6)
    if flops: row['effective_TFLOP_s'] = flops / (d * 1e9)
    output.write(json.dumps(row)+'\n'); output.flush()
    print(json.dumps({k:v for k,v in row.items() if k not in ('host_ms','device_ms')}), flush=True)

with open(a.out, 'x') as output:
    output.write(json.dumps(dict(kind='config', label=a.label, torch=torch.__version__,
        torch_npu=torch_npu.__version__, device=str(torch.npu.get_device_name()),
        num_threads=torch.get_num_threads(), semantics='copy bandwidth counts read+write bytes; not HBM counter'))+'\n')
    for mib in (1,16,128):
        src = torch.ones(mib*1024*1024, dtype=torch.int8).npu()
        dst = torch.empty_like(src)
        for graph in (False,True):
            measure(f'copy_{mib}MiB', lambda:dst.copy_(src), traffic=2*src.numel(), graph=graph)
        assert int(dst[0].cpu()) == 1 and int(dst[-1].cpu()) == 1
        del src,dst
    for m,k,n in ((1,2048,1024),(128,2048,1024),(1024,2048,1024)):
        x=(torch.randn(m,k)*.1).half();w=(torch.randn(k,n)*.1).half()
        expected=(x.float() @ w.float()).half()
        x=x.npu();w=w.npu(); out=torch.empty((m,n),dtype=torch.float16,device='npu')
        fn=lambda:torch.mm(x,w,out=out)
        fn();torch.testing.assert_close(out.cpu(),expected,rtol=.03,atol=.006)
        for graph in (False,True): measure(f'fp16_mm_{m}x{k}x{n}',fn,flops=2*m*k*n,graph=graph)
    x=torch.ones(1024).npu();out=torch.empty_like(x)
    for graph in (False,True):measure('small_add_1024',lambda:torch.add(x,1,out=out),graph=graph)
