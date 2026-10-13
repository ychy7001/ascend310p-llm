# SPDX-License-Identifier: Apache-2.0
"""隔离生成W4绑定生命周期候选；不改原始公共适配头或现有服务扩展。"""
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--adapter-source', type=Path, required=True)
parser.add_argument('--extension-source', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
args = parser.parse_args()
source = args.adapter_source.read_text()
for header in ('NPUBridge.h', 'NPUStorageImpl.h'):
    source = source.replace(f'#include "{header}"', f'#include "aclnn_torch_adapter/{header}"')
source = '#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"\n' + source
needle = '    uint64_t workspace_size = 0;'
assert source.count(needle) == 1
# 本项目4个算子的参数均为Tensor；不用于含bool/int属性的其他ACL算子。
source = source.replace(needle, '''    auto qwen_keep = std::make_tuple(__VA_ARGS__);                         \\
    auto qwen_stream = c10_npu::getCurrentNPUStream();                        \\
    std::apply([&](const auto&... tensors) {                                  \\
      (c10_npu::NPUCachingAllocator::recordStream(                             \\
          tensors.storage().data_ptr(), qwen_stream), ...);                    \\
    }, qwen_keep);                                                            \\
    at::Tensor workspace_tensor;                                              \\
''' + needle)
needle = '      auto workspace_tensor ='
assert source.count(needle) == 1
source = source.replace(needle, '      workspace_tensor =')
needle = '      workspace_addr = const_cast<void *>(workspace_tensor.storage().data());'
assert source.count(needle) == 1
source = source.replace(needle, needle + ''' \\
      c10_npu::NPUCachingAllocator::recordStream(                              \\
          workspace_tensor.storage().data_ptr(), qwen_stream);''')
needle = '                     acl_stream, executor]() -> int {'
assert source.count(needle) == 1
source = source.replace(needle, '                     acl_stream, executor, qwen_keep, workspace_tensor]() -> int {')
args.output_dir.mkdir(parents=True, exist_ok=False)
adapter = args.output_dir / 'op_api_common.h'
adapter.write_text(source)
extension = args.extension_source.read_text().replace(
    '#include "aclnn_torch_adapter/op_api_common.h"',
    '#include "op_api_common.h"',
)
(args.output_dir / 'qwen36_int4_custom.cpp').write_text(extension)
