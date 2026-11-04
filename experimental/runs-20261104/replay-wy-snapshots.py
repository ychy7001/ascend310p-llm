#!/usr/bin/env python3
"""离线重放真实 WY 输入；无权重，原生路径禁止回退。CPU oracle 在本机另算。"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import torch
import torch_npu
from native_wy import supported

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pair',type=Path,required=True)
p.add_argument('--wy',type=Path,required=True)
p.add_argument('--captures',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--candidate',choices=('native','cann'),required=True)
p.add_argument('--cann',type=Path)
a=p.parse_args()
torch.set_num_threads(1)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
vendors=[a.pair/'opp/vendors/hyperqwen_gdn_pair921_transformer']
if a.candidate=='native':vendors.append(a.wy/'opp/vendors/hyperqwen_wy921_transformer')
else:
 assert a.cann is not None
 vendors.insert(0,a.cann/'opp/vendors/wy921_cann_transformer')
os.environ['ASCEND_CUSTOM_OPP_PATH']=':'.join(map(str,vendors))+':'+os.environ.get('ASCEND_CUSTOM_OPP_PATH','')
module=importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()=='4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f'
a.out.mkdir(exist_ok=False)
for device in (0,1):
 torch.npu.set_device(device)
 torch_npu.npu.set_compile_mode(jit_compile=False)
 if a.candidate=='native':torch.ops.load_library(str(a.wy/'extension/pr17640_wy.so'))
 if a.candidate=='cann':torch.ops.load_library(str(a.cann/'extension/cann_wy921.so'))
 for path in sorted(a.captures.glob(f'rank{device}-call*.pt')):
  saved=torch.load(path,weights_only=True);inputs=[x.npu() for x in saved['inputs']]
  assert supported(*inputs,64),'No fallback permitted'
  baseline=module._compute_kernel_inputs_from_torch_wy(*inputs,64)
  op=torch.ops.pr17640_wy.compute if a.candidate=='native' else torch.ops.cann_wy921.compute
  actual=op(*(x.contiguous() for x in inputs),64)
  cpu_actual=[x.cpu() for x in actual]
  repeated=op(*(x.contiguous() for x in inputs),64)
  row={'source':path.name,'metadata':saved['metadata'],
       'torch_equal_capture':[torch.equal(x.cpu(),y) for x,y in zip(baseline,saved['torch_outputs'])],
       'candidate_repeat_equal':[torch.equal(x.cpu(),y) for x,y in zip(repeated,cpu_actual)],
       'loaded_apis':sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcust_opapi.so' in line})}
  torch.save({'metadata':saved['metadata'],'torch_outputs':[x.cpu() for x in baseline],a.candidate+'_outputs':cpu_actual},a.out/path.name)
  with (a.out/'replay.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
  print(json.dumps(row),flush=True)
