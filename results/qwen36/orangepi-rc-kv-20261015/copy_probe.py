"""RC独立H2D/D2H探针；不加载模型或C8适配器。"""
import json
import time
import torch
import torch_npu

for size in (1024, 4 * 1024**2, 16 * 1024**2, 64 * 1024**2):
    source = torch.arange(size, dtype=torch.int32).to(torch.uint8)
    start = time.perf_counter()
    device = source.to('npu:0', non_blocking=True)
    torch.npu.synchronize()
    h2d = time.perf_counter() - start
    start = time.perf_counter()
    back = device.cpu()
    torch.npu.synchronize()
    d2h = time.perf_counter() - start
    passed = torch.equal(source, back)
    print(json.dumps({'bytes': size, 'h2d_s': h2d, 'd2h_s': d2h, 'exact_match': passed}), flush=True)
    assert passed
print('COPY_PROBE_PASS', flush=True)
