#!/usr/bin/env python3
"""真实快照上连发3次WY，在读回前保留全部输出；检查异步运行重复性。"""
import argparse,importlib,json,os
from pathlib import Path
import torch,torch_npu
p=argparse.ArgumentParser()
for name in ('pair','wy','captures','triggers','out'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
vendors=[a.wy/'opp/vendors/hyperqwen_gram_postbeta128_transformer',a.pair/'opp/vendors/hyperqwen_gdn_pair921_transformer']
os.environ['ASCEND_CUSTOM_OPP_PATH']=':'.join(map(str,vendors))+':'+os.environ.get('ASCEND_CUSTOM_OPP_PATH','')
torch.ops.load_library(str(a.wy/'extension/pr17640_wy.so'));rows=[]
for rank in (0,1):
 torch.npu.set_device(rank);torch_npu.npu.set_compile_mode(jit_compile=False)
 for path in [*sorted(a.captures.glob(f'rank{rank}-call*.pt')),*sorted(a.triggers.glob(f'rank{rank}-call*.pt'))]:
  saved=torch.load(path,weights_only=True);inputs=[x.npu() for x in saved['inputs']]
  anchor=[x.cpu() for x in torch.ops.pr17640_wy.compute(*inputs,64)]
  pending=[torch.ops.pr17640_wy.compute(*inputs,64) for _ in range(3)]
  equal=[[torch.equal(x.cpu().view(torch.uint8),y.view(torch.uint8)) for x,y in zip(result,anchor)] for result in pending]
  row={'source':path.name,'set':'triggers' if path.parent.name=='triggers' else 'snapshots','passed':all(all(r) for r in equal),'repeat_equal':equal}
  rows.append(row);print(json.dumps(row),flush=True)
result={'passed':len(rows)==132 and all(x['passed'] for x in rows),'cases':len(rows),'passed_cases':sum(x['passed'] for x in rows),'rows':rows,'method':'byte equality, 3 queued calls before CPU readback; synchronized anchor from same input'}
a.out.write_text(json.dumps(result,indent=2)+'\n');assert result['passed'],'Queued WY repeat mismatch'
