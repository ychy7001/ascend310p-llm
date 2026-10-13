# SPDX-License-Identifier: Apache-2.0
"""外部固定绑定的隔离补丁：通过torch-npu任务队列提交ACL调用，保持生产者顺序。
不复制外部源码，生成到新目录；原始文件不改写。仅用于实验。
"""
import argparse
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
s=a.source.read_text()
s=s.replace('#include "torch_npu/csrc/core/npu/NPUStream.h"','#include "torch_npu/csrc/core/npu/NPUStream.h"\n#include "torch_npu/csrc/framework/OpCommand.h"')
old='at::Tensor qmm_v5_w4a8_npu(const at::Tensor &x1, const at::Tensor &x1_scale,\n                           const at::Tensor &x2_packed, const at::Tensor &x2_scale)'
assert old in s
s=s.replace(old,old.replace('qmm_v5_w4a8_npu','qmm_v5_w4a8_raw').replace('const at::Tensor &x2_scale)','const at::Tensor &x2_scale, const at::Tensor &out)'))
allocation='    auto out = at::empty({m, n}, x1.options().dtype(c10::kHalf));'
assert allocation in s;s=s.replace(allocation,'    // Output allocated by queue-aware wrapper.')
end=s.index('\n}',s.index('at::Tensor qmm_v5_w4a8_raw'))+2
wrapper='''

// Experimental modification: preserve torch-npu producer/consumer queue order.
at::Tensor qmm_v5_w4a8_npu(const at::Tensor &x1, const at::Tensor &x1_scale,
                           const at::Tensor &x2_packed, const at::Tensor &x2_scale)
{
    auto out = at::empty({x1.size(0), x2_scale.size(0)}, x1.options().dtype(c10::kHalf));
    auto handler = [x1, x1_scale, x2_packed, x2_scale, out]() -> int {
        qmm_v5_w4a8_raw(x1, x1_scale, x2_packed, x2_scale, out);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("MindIEQQmmQueued", handler);
    return out;
}
'''
s=s[:end]+wrapper+s[end:]
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:f.write(s)
