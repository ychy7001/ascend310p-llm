#!/usr/bin/env python3
"""固定官方 WY bridge 的私有 Torch 注册；不导入 cann_ops_transformer。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch
import torch_npu
from torch.utils.cpp_extension import load
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--adapter-root',type=Path,required=True)
p.add_argument('--build',type=Path,required=True)
a=p.parse_args();a.build.mkdir(exist_ok=True);torch.set_num_threads(2)
source=a.source_root/'experimental/attention/chunk_gated_delta_rule_compute_wy/torch_extension/csrc/chunk_gated_delta_rule_compute_wy.cpp'
text=source.read_text()
assert hashlib.sha256(source.read_bytes()).hexdigest()=='8c8f0b260596550d338945ea529d51058de92904ad27326c3609a980ef4693c7'
text=text[:text.index('PYBIND11_MODULE')]
text=text.replace('#include <torch/extension.h>','#include <torch/library.h>\n#include <torch/types.h>')
text=text.replace('#include "aclnn_common.h"','#include "aclnn_torch_adapter/op_api_common.h"\nthread_local char g_hashBuf[kHashBufSize];\nthread_local int g_hashOffset = 0;')
text=text.replace('ACLNN_CMD(', 'EXEC_NPU_CMD(')
text+='\n} // namespace op_api\nTORCH_LIBRARY(cann_wy921, m) { m.def("compute(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, int chunk_size) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"); }\nTORCH_LIBRARY_IMPL(cann_wy921, PrivateUse1, m) { m.impl("compute", &op_api::chunk_gated_delta_rule_compute_wy); }\n'
assert 'PYBIND11_MODULE' not in text and 'TORCH_LIBRARY(cann_ops_transformer' not in text
cpp=a.build/'cann_wy921.cpp';cpp.write_text(text)
npu_root=Path(torch_npu.__file__).parent
include=Path(os.environ['ASCEND_HOME_PATH'])/'include'
os.environ['MAX_JOBS']='2'
load(name='cann_wy921',sources=[str(cpp),str(a.adapter_root/'aclnn_torch_adapter/NPUBridge.cpp')],
 extra_include_paths=[str(a.adapter_root),str(a.adapter_root/'common/include'),str(include),str(npu_root/'include')],
 extra_cflags=['-O2','-fvisibility=hidden','-DASCEND_PLATFORM_310P'],
 extra_ldflags=[f'-L{npu_root/"lib"}','-ltorch_npu'],is_python_module=False,
 verbose=True,build_directory=str(a.build))
ops=[x for x in torch._C._dispatch_get_all_op_names() if 'cann_wy921' in x or ('cann_ops_transformer' in x and 'compute_wy' in x)]
assert ops==['cann_wy921::compute'],ops
(a.build/'provenance.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'generated_bridge_sha256':hashlib.sha256(cpp.read_bytes()).hexdigest(),'registered_ops':ops,
 'fallback':False,'max_jobs':2,'device_exposed':False},indent=2)+'\n')
