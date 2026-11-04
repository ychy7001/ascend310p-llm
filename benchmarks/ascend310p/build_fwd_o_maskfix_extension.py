# SPDX-License-Identifier: Apache-2.0
"""复用固定PR的FwdO绑定契约，独立namespace调用改名后的生产mask实验算子。"""
import argparse
import re
from pathlib import Path
import torch_npu
from torch.utils.cpp_extension import load
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--build',type=Path,required=True)
a=p.parse_args();a.build.mkdir(parents=True,exist_ok=False)
text=(a.source_root/'csrc/torch_binding.cpp').read_text()
start=text.index('at::Tensor chunk_fwd_o_vllm(')
end=text.index('\nat::Tensor npu_sparse_attention_score_prefill(',start)
function=text[start:end].replace('chunk_fwd_o_vllm','maskfix_run').replace('aclnnChunkFwdOVllm','aclnnChunkFwdOMaskFix')
schema=re.search(r'"(chunk_fwd_o_vllm\(Tensor[^"\n]+)"',text).group(1).replace('chunk_fwd_o_vllm(', 'run(',1)
source=a.build/'maskfix.cpp'
source.write_text(text[:text.index('#include')]+'\n// Isolated mask experiment; binding contract reused from GruntovDima PR17640.\n'
 '#include <torch/library.h>\n#include <torch/types.h>\n#include "aclnn_torch_adapter/op_api_common.h"\n'
 'thread_local char g_hashBuf[kHashBufSize];\nthread_local int g_hashOffset = 0;\n'+function+
 '\nTORCH_LIBRARY(maskfix_fwd_o, m) {m.def("'+schema+'");}\n'
 'TORCH_LIBRARY_IMPL(maskfix_fwd_o, PrivateUse1, m) {m.impl("run", &maskfix_run);}\n')
npu=Path(torch_npu.__file__).parent
load(name='maskfix_fwd_o',sources=[str(source),str(a.source_root/'csrc/aclnn_torch_adapter/NPUBridge.cpp')],
 extra_include_paths=[str(a.source_root/'csrc'),str(a.source_root/'csrc/common/include'),str(npu/'include'),'/usr/local/Ascend/ascend-toolkit/latest/include'],
 extra_cflags=['-O2','-fvisibility=hidden','-DASCEND_PLATFORM_310P'],
 extra_ldflags=[f'-L{npu/"lib"}','-ltorch_npu'],is_python_module=False,verbose=True,build_directory=str(a.build))
