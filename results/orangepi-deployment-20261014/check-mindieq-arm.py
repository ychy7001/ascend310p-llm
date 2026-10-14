"""当前旧 AI CPU 使用已编译算子；复用整数参考/动态边界 Graph 门禁。"""
import runpy
import torch
import torch_npu

torch.npu.set_compile_mode(jit_compile=False)
runpy.run_path('/work/repo/tools/bench_mindieq_grouped_310.py', run_name='__main__')
