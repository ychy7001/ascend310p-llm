#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""无模型权重的单 chunk/head WY 复现；退出1表示冻结逐点容差超标。"""
import argparse,json,os
from pathlib import Path
import torch
import torch_npu
from wy_snapshot_reference import metrics,reference
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True);p.add_argument('--library',type=Path,required=True)
p.add_argument('--vendor',type=Path,required=True);p.add_argument('--device',type=int,choices=(0,1),required=True)
p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
os.environ['ASCEND_CUSTOM_OPP_PATH']=str(a.vendor)+':'+os.environ.get('ASCEND_CUSTOM_OPP_PATH','')
torch.npu.set_device(a.device);torch_npu.npu.set_compile_mode(jit_compile=False)
from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import _compute_kernel_inputs_from_torch_wy
saved=torch.load(a.input,weights_only=True);inputs=saved['inputs'];expected=reference(inputs)
torch.ops.load_library(str(a.library));device=[x.npu() for x in inputs]
actual=[x.cpu() for x in torch.ops.pr17640_wy.compute(*device,64)]
baseline=[x.cpu() for x in _compute_kernel_inputs_from_torch_wy(*device,64)]
repeat=[x.cpu() for x in torch.ops.pr17640_wy.compute(*device,64)]
result={'scope':'WY only; no weights, no tokenizer, no state, no varlen, no TP communication',
        'shapes':[list(x.shape) for x in inputs],
        'threshold':{'atol':.0005,'rtol':.04,'max_relative_rmse':.02,'q_k_exact':True},
        'native_vs_cpu':metrics(actual,expected),'native_vs_torch':metrics(actual,baseline),
        'torch_vs_cpu':metrics(baseline,expected),'native_repeat_equal':[torch.equal(x,y) for x,y in zip(actual,repeat)]}
result['passed']=all(x['passed'] for key in ('native_vs_cpu','native_vs_torch') for x in result[key])
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
raise SystemExit(0 if result['passed'] else 1)
