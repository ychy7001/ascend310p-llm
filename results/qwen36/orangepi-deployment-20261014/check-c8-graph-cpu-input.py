"""旧 AI CPU 随机生成不兼容：测试数据在 CPU 生成，再验证真实 NPU Graph。"""
import os
import runpy
import sys
import torch
import torch_npu
randn, randn_like = torch.randn, torch.randn_like


def cpu_randn(*args, **kwargs):
    device = kwargs.pop('device', None)
    out = randn(*args, **kwargs)
    return out.to(device) if device is not None else out


def cpu_randn_like(tensor, **kwargs):
    device = kwargs.pop('device', tensor.device)
    return randn_like(tensor.cpu(), **kwargs).to(device)


torch.randn = cpu_randn
torch.randn_like = cpu_randn_like
sys.argv = ['check_c8_graph_read_310.py', '--library',
            os.environ.get('C8_LIBRARY', '/work/artifacts/c8-transdata-build/output/libqwen36_c8.so'), '--device-length']
runpy.run_path('/work/repo/experimental/check_c8_graph_read_310.py', run_name='__main__')
