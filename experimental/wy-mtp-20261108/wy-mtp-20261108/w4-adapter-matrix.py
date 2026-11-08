# SPDX-License-Identifier: Apache-2.0
"""可选独立 PR GDN 接入；外部固定源码与 OPP 由实验启动器提供。"""
import importlib
import os
from pathlib import Path
import torch

torch.ops.load_library(os.environ['QWEN36_GDN_LIBRARY'])
module=importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
source_path=Path(os.environ['QWEN36_GDN_SOURCE'])
source=source_path.read_text().replace('torch.ops._C_ascend.', 'torch.ops.pr17640_gdn.').replace('"_C_ascend"','"pr17640_gdn"').replace('chunk_gated_delta_rule_compute_wy','compute')
namespace={'__name__':'qwen36_pr17640_gdn'}
exec(compile(source,str(source_path),'exec'),namespace)
# Experiment-only selection of the frozen wrapper's existing WY functions.
mode=os.environ.get('WY_MATRIX_MODE','native')
if mode not in ('native','torch'):
    raise RuntimeError('WY_MATRIX_MODE must be native or torch')
original_wy=namespace['_compute_kernel_inputs_from_npu_wy']
torch_wy=namespace['_compute_kernel_inputs_from_torch_wy']
seen=set()
def measured_wy(q,k,v,g,beta,chunk_size):
    eligible=namespace['_can_use_npu_compute_wy'](q,k,v,g,beta,chunk_size)
    key=(mode,eligible)
    if key not in seen:
        print('[WY matrix] mode=%s native_eligible=%s q_shape=%s dtype=%s' % (mode,eligible,tuple(q.shape),q.dtype),flush=True)
        seen.add(key)
    if mode=='native':
        if not eligible:
            raise RuntimeError('WY matrix native arm would fall back to Torch')
        return original_wy(q,k,v,g,beta,chunk_size)
    return torch_wy(q,k,v,g,beta,chunk_size)
namespace['_compute_kernel_inputs_from_npu_wy']=measured_wy
replacement=namespace['chunk_gated_delta_rule_310']
module.chunk_gated_delta_rule_310=replacement
# The serving image imports this consumer before the optional bootstrap hook.
consumer=importlib.import_module('vllm_ascend._310p.ops.fla.gdn_310')
consumer.chunk_gated_delta_rule_310=replacement
package=importlib.import_module('vllm_ascend._310p.ops.fla')
package.chunk_gated_delta_rule_310=replacement
assert consumer.chunk_gated_delta_rule_310.__code__.co_filename == str(source_path)
print('[Qwen36 GDN] fixed PR WY/FwdH/FwdO enabled via independent namespace',flush=True)
