"""当前旧 AI CPU 使用已编译算子；复用整数参考/动态边界 Graph 门禁。"""
import runpy
import sys
import torch
import torch_npu

torch.npu.set_compile_mode(jit_compile=False)
sys.path.insert(0, '/work/repo/tools')
script = 'bench_mindieq_grouped_310.py'
if '--shared' in sys.argv:
    sys.argv.remove('--shared')
    script = 'check_mindieq_shared_gate_310.py'
runpy.run_path('/work/repo/tools/' + script, run_name='__main__')
