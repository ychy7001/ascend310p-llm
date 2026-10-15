"""独立RC元数据传输探针；不加载权重，逐步打印便于识别阻塞点。"""
import json
import torch
import torch_npu

torch.npu.set_device(0)
for dtype in (torch.int32, torch.int64):
    cpu = torch.tensor([[11, 29]], dtype=dtype)
    device = cpu.to('npu', non_blocking=False)
    for label, extra_stream, nonblocking in [('blocking-default', False, False),
                                              ('async-default', False, True),
                                              ('async-copy-stream', True, True)]:
        host = torch.empty(cpu.shape, dtype=dtype, pin_memory=True)
        default = torch.npu.current_stream()
        stream = torch.npu.Stream() if extra_stream else default
        event = torch.npu.Event()
        print(json.dumps({'stage':'start','dtype':str(dtype),'mode':label}), flush=True)
        with torch.npu.stream(stream):
            if extra_stream:
                stream.wait_stream(default)
            host.copy_(device, non_blocking=nonblocking)
            event.record()
        event.synchronize()
        assert torch.equal(host, cpu), (host, cpu)
        print(json.dumps({'stage':'passed','dtype':str(dtype),'mode':label,'values':host.tolist()}), flush=True)
print('ALL_COPY_CASES_PASSED', flush=True)
