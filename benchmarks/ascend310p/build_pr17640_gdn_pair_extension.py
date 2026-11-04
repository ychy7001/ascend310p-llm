# SPDX-License-Identifier: Apache-2.0
"""固定 PR17640 的 FwdH/FwdO 成对绑定；保持上游 NZ 状态契约。"""
import argparse
import os
import re
from pathlib import Path
import torch_npu
from torch.utils.cpp_extension import load

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root', type=Path, required=True)
p.add_argument('--build', type=Path, required=True)
a = p.parse_args()
a.build.mkdir(parents=True, exist_ok=False)
text = (a.source_root/'csrc/torch_binding.cpp').read_text()
start = text.index('std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_gated_delta_rule_fwd_h_vllm(')
end = text.index('\nat::Tensor npu_sparse_attention_score_prefill(', start)
functions = text[start:end]
registrations = []
for original, name in [('chunk_gated_delta_rule_fwd_h_vllm', 'fwd_h'), ('chunk_fwd_o_vllm', 'fwd_o')]:
    schema = re.search(r'"('+original+r'\(Tensor[^"\n]+)"', text).group(1)
    registrations.append((schema.replace(original+'(', name+'(', 1), original, name))
source = a.build/'pr17640_gdn_pair.cpp'
source.write_text(text[:text.index('#include')] + '\n'
    '#include <torch/library.h>\n#include <torch/types.h>\n'
    '#include "aclnn_torch_adapter/op_api_common.h"\n'
    'thread_local char g_hashBuf[kHashBufSize];\nthread_local int g_hashOffset = 0;\n' + functions +
    '\nTORCH_LIBRARY(pr17640_gdn_pair, m) {\n' +
    ''.join('m.def("'+schema+'");\n' for schema, _, _ in registrations) + '}\n' +
    'TORCH_LIBRARY_IMPL(pr17640_gdn_pair, PrivateUse1, m) {\n' +
    ''.join('m.impl("'+name+'", &'+original+');\n' for _, original, name in registrations) + '}\n')
npu = Path(torch_npu.__file__).parent
cann_include = Path(os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest')) / 'include'
if not cann_include.is_dir():
    raise RuntimeError('CANN include directory missing: ' + str(cann_include))
load(name='pr17640_gdn_pair', sources=[str(source), str(a.source_root/'csrc/aclnn_torch_adapter/NPUBridge.cpp')],
     extra_include_paths=[str(a.source_root/'csrc'), str(a.source_root/'csrc/common/include'), str(npu/'include'), str(cann_include)],
     extra_cflags=['-O2', '-fvisibility=hidden', '-DASCEND_PLATFORM_310P'],
     extra_ldflags=[f'-L{npu/"lib"}', '-ltorch_npu'], is_python_module=False,
     verbose=True, build_directory=str(a.build))
