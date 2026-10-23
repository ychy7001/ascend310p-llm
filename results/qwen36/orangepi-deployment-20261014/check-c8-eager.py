import os
import runpy
import sys
import torch
import torch_npu
torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)
sys.argv = ['check_c8_cache_310.py', '--library', os.environ.get('C8_LIBRARY', '/work/artifacts/c8-transdata-build/output/libqwen36_c8.so')]
runpy.run_path('/work/repo/experimental/check_c8_cache_310.py', run_name='__main__')
