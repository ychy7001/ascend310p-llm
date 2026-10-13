# SPDX-License-Identifier: Apache-2.0
"""外部固定绑定的隔离补丁：通过torch-npu任务队列提交ACL调用，保持生产者顺序。
不复制外部源码，生成到新目录；原始文件不改写。仅用于实验。
"""
import argparse
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--caller-workspace',action='store_true',help='隔离候选：调用线程准备ACL句柄/工作区，任务线程仅提交计算')
p.add_argument('--record-stream',action='store_true',help='记录ACL输入/输出/工作空间使用流；隔离生命周期候选')
p.add_argument('--record-stream-scope',choices=['all','inputs','workspace','none'],default='all',help='生命周期消融实验，须配合--record-stream')
a=p.parse_args()
if not a.record_stream and a.record_stream_scope!='all':
    p.error('--record-stream-scope requires --record-stream')
s=a.source.read_text()
s=s.replace('#include "torch_npu/csrc/core/npu/NPUStream.h"','#include "torch_npu/csrc/core/npu/NPUStream.h"\n#include "torch_npu/csrc/framework/OpCommand.h"')
if a.caller_workspace:
    if a.record_stream:
        p.error('--caller-workspace is a separate candidate; omit --record-stream')
    s=s.replace('#include "torch_npu/csrc/framework/OpCommand.h"',
                '#include "torch_npu/csrc/framework/OpCommand.h"\n#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"')
    start=s.index('    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);')
    end=s.index('    return out;', start)
    s=s[:start]+'''    // Experimental modification: allocate and record on the caller thread.
    // The queue worker performs no Torch allocation or Tensor destruction.
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto &tensor : {x1, x1_scale, x2_packed, x2_scale, out, wsTensor}) {
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
    auto aclStream = stream.stream(false);
    auto handler = [wsPtr, wsSize, executor, aclStream,
                    x1T, x2T, x1sT, x2sT, outT]() -> int {
        auto ret = aclnnQuantMatmulV5(wsPtr, wsSize, executor, aclStream);
        aclDestroyTensor(x1T); aclDestroyTensor(x2T); aclDestroyTensor(x1sT);
        aclDestroyTensor(x2sT); aclDestroyTensor(outT);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclnnQuantMatmulV5 failed: ", ret);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("MindIEQQmmCallerWorkspace", handler);
'''+s[end:]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(s)
    raise SystemExit(0)
old='at::Tensor qmm_v5_w4a8_npu(const at::Tensor &x1, const at::Tensor &x1_scale,\n                           const at::Tensor &x2_packed, const at::Tensor &x2_scale)'
assert old in s
s=s.replace(old,old.replace('qmm_v5_w4a8_npu','qmm_v5_w4a8_raw').replace('const at::Tensor &x2_scale)','const at::Tensor &x2_scale, const at::Tensor &out)'))
allocation='    auto out = at::empty({m, n}, x1.options().dtype(c10::kHalf));'
assert allocation in s;s=s.replace(allocation,'    // Output allocated by queue-aware wrapper.')
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
if a.record_stream:
    s=s.replace('#include "torch_npu/csrc/framework/OpCommand.h"',
                '#include "torch_npu/csrc/framework/OpCommand.h"\n#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"')
    s=s.replace('const at::Tensor &out)\n{', 'const at::Tensor &out, c10_npu::NPUStream stream)\n{', 1)
    s=s.replace('    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);',
                '    c10_npu::NPUCachingAllocator::recordStream(wsTensor.storage().data_ptr(), stream);\n'
                '    auto aclStream = stream.stream(false);', 1)
    wrapper=wrapper.replace('    auto handler = [x1, x1_scale, x2_packed, x2_scale, out]() -> int {',
        '    auto stream = c10_npu::getCurrentNPUStream();\n'
        '    for (const auto &tensor : {x1, x1_scale, x2_packed, x2_scale, out}) {\n'
        '        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);\n'
        '    }\n'
        '    auto handler = [x1, x1_scale, x2_packed, x2_scale, out, stream]() -> int {')
    wrapper=wrapper.replace('x2_scale, out);', 'x2_scale, out, stream);')
    if a.record_stream_scope in ('inputs','none'):
        s=s.replace('    c10_npu::NPUCachingAllocator::recordStream(wsTensor.storage().data_ptr(), stream);\n','')
    if a.record_stream_scope in ('workspace','none'):
        wrapper=wrapper.replace('    for (const auto &tensor : {x1, x1_scale, x2_packed, x2_scale, out}) {\n'
                                '        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);\n'
                                '    }\n','')
    wrapper=wrapper.replace('MindIEQQmmQueued', 'MindIEQQmmRecorded')
end=s.index('\n}',s.index('at::Tensor qmm_v5_w4a8_raw'))+2
s=s[:end]+wrapper+s[end:]
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:f.write(s)
