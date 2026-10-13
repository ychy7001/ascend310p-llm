# SPDX-License-Identifier: Apache-2.0
"""Extract the pinned PR's WY binding into an isolated namespace, preserving license."""
import argparse
from pathlib import Path
import torch_npu
from torch.utils.cpp_extension import load
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root', type=Path, required=True)
p.add_argument('--build', type=Path, required=True)
a = p.parse_args()
a.build.mkdir(parents=True, exist_ok=True)
text = (a.source_root / 'csrc/torch_binding.cpp').read_text()
start = text.index('std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_compute_wy(')
end = text.index('\nstd::vector<int64_t> get_npu_storage_shape', start)
source = a.build / 'wy.cpp'
source.write_text(text[:text.index('#include')] + '\n// Extracted for standalone testing; namespace modified.\n'
    '#include <torch/library.h>\n#include <torch/types.h>\n#include "aclnn_torch_adapter/op_api_common.h"\n'
    'thread_local char g_hashBuf[kHashBufSize];\nthread_local int g_hashOffset = 0;\n'
    + text[start:end] + '\nTORCH_LIBRARY(pr17640_wy, m) { m.def("compute(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, int? chunk_size=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)"); }\n'
    'TORCH_LIBRARY_IMPL(pr17640_wy, PrivateUse1, m) { m.impl("compute", &chunk_gated_delta_rule_compute_wy); }\n')
npu_root = Path(torch_npu.__file__).parent
load(name='pr17640_wy', sources=[str(source), str(a.source_root/'csrc/aclnn_torch_adapter/NPUBridge.cpp')],
     extra_include_paths=[str(a.source_root/'csrc'),str(a.source_root/'csrc/common/include'),str(npu_root/'include'),'/usr/local/Ascend/ascend-toolkit/latest/include'],
     extra_cflags=['-O2','-fvisibility=hidden'], extra_ldflags=[f'-L{npu_root/"lib"}','-ltorch_npu'],
     is_python_module=False, verbose=True, build_directory=str(a.build))
