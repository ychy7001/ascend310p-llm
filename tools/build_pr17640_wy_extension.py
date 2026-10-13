# SPDX-License-Identifier: Apache-2.0
"""Extract the pinned PR's WY binding into an isolated namespace, preserving license."""
import argparse
from pathlib import Path
import torch_npu
from torch.utils.cpp_extension import load
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root', type=Path, required=True)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--full-gdn', action='store_true')
a = p.parse_args()
a.build.mkdir(parents=True, exist_ok=True)
text = (a.source_root / 'csrc/torch_binding.cpp').read_text()
start = text.index('std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_compute_wy(')
end = text.index('\nstd::vector<int64_t> get_npu_storage_shape', start)
name = 'pr17640_gdn' if a.full_gdn else 'pr17640_wy'
functions = text[start:end]
definitions = ['compute(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, int? chunk_size=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor)']
implementations = [('compute', 'chunk_gated_delta_rule_compute_wy')]
if a.full_gdn:
    first = text.index('std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_fwd_h_vllm(')
    last = text.index('\nat::Tensor npu_sparse_attention_score_prefill(', first)
    functions += text[first:last]
    for op in ('chunk_gated_delta_rule_fwd_h_vllm', 'chunk_fwd_o_vllm'):
        # Reuse exact schemas from pinned Torch registrations.
        import re
        schema = re.search(r'"(' + op + r'\(Tensor[^"\n]+)"', text).group(1)
        definitions.append(schema)
        implementations.append((op, op))
registration = '\nTORCH_LIBRARY(' + name + ', m) {\n'
registration += ''.join('m.def("' + schema + '");\n' for schema in definitions) + '}\n'
registration += 'TORCH_LIBRARY_IMPL(' + name + ', PrivateUse1, m) {\n'
registration += ''.join('m.impl("' + op + '", &' + fn + ');\n' for op, fn in implementations) + '}\n'
source = a.build / 'wy.cpp'
source.write_text(text[:text.index('#include')] + '\n// Extracted for standalone testing; namespace modified.\n'
    '#include <torch/library.h>\n#include <torch/types.h>\n#include "aclnn_torch_adapter/op_api_common.h"\n'
    'thread_local char g_hashBuf[kHashBufSize];\nthread_local int g_hashOffset = 0;\n'
    + functions + registration)
npu_root = Path(torch_npu.__file__).parent
load(name=name, sources=[str(source), str(a.source_root/'csrc/aclnn_torch_adapter/NPUBridge.cpp')],
     extra_include_paths=[str(a.source_root/'csrc'),str(a.source_root/'csrc/common/include'),str(npu_root/'include'),'/usr/local/Ascend/ascend-toolkit/latest/include'],
     extra_cflags=['-O2','-fvisibility=hidden','-DASCEND_PLATFORM_310P'], extra_ldflags=[f'-L{npu_root/"lib"}','-ltorch_npu'],
     is_python_module=False, verbose=True, build_directory=str(a.build))
