#!/usr/bin/env python3
"""同进程 Torch/修复WY，真实快照及冻结触发集，禁止回退。"""
import argparse,hashlib,importlib,json,os
from pathlib import Path
import torch,torch_npu
p=argparse.ArgumentParser()
for name in ('pair','wy','captures','triggers','out'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1)
for root in (a.pair,a.wy):
 for name,expected in json.loads((root/'operator-artifacts.json').read_text()).items():
  assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
vendors=[a.wy/'opp/vendors/hyperqwen_postbeta_neumaier_transformer',a.pair/'opp/vendors/hyperqwen_gdn_pair921_transformer']
os.environ['ASCEND_CUSTOM_OPP_PATH']=':'.join(map(str,vendors))+':'+os.environ.get('ASCEND_CUSTOM_OPP_PATH','')
m=importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
assert hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()=='4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f'
torch.ops.load_library(str(a.wy/'extension/pr17640_wy.so'));a.out.mkdir(exist_ok=False)
for rank in (0,1):
 torch.npu.set_device(rank);torch_npu.npu.set_compile_mode(jit_compile=False)
 for label,source in [('snapshots',a.captures),('triggers',a.triggers)]:
  dest=a.out/label;dest.mkdir(exist_ok=True)
  for path in sorted(source.glob(f'rank{rank}-call*.pt')):
   saved=torch.load(path,weights_only=True);inputs=[x.npu() for x in saved['inputs']]
   actual=[x.cpu() for x in torch.ops.pr17640_wy.compute(*inputs,64)]
   baseline=[x.cpu() for x in m._compute_kernel_inputs_from_torch_wy(*inputs,64)]
   repeats=[]
   for _ in range(3):
    repeat=torch.ops.pr17640_wy.compute(*inputs,64)
    repeats.append([torch.equal(x.cpu(),y) for x,y in zip(repeat,actual)])
   row={'source':path.name,'set':label,'metadata':saved['metadata'],'repeats':repeats,'repeat_pass':all(all(r) for r in repeats),'torch_equal_capture':[torch.equal(x,y) for x,y in zip(baseline,saved.get('torch_outputs',[]))]}
   torch.save({'metadata':saved['metadata'],'native_outputs':actual,'torch_outputs':baseline},dest/path.name)
   with (a.out/'replay.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   print(json.dumps(row),flush=True)
loaded=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcust_opapi.so' in line})
assert str((vendors[0]/'op_api/lib/libcust_opapi.so').resolve()) in loaded,loaded
(a.out/'loaded-apis.json').write_text(json.dumps(loaded))
