#!/usr/bin/env python3
"""真实 WY 快照对应的完整 GDN 输出/state 对照；沿用配套 H/O 和原容差。"""
import argparse,hashlib,importlib,json,os
from pathlib import Path
import torch,torch_npu
from native_wy import supported
p=argparse.ArgumentParser(description=__doc__)
for name in ('pair','wy','captures','out'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--candidate',choices=('native','cann'),required=True);a=p.parse_args();torch.set_num_threads(1)
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
wyvendor='hyperqwen_wy921_transformer' if a.candidate=='native' else 'wy921_cann_transformer'
vendors=[a.wy/'opp/vendors'/wyvendor,a.pair/'opp/vendors/hyperqwen_gdn_pair921_transformer']
os.environ['ASCEND_CUSTOM_OPP_PATH']=':'.join(map(str,vendors))+':'+os.environ.get('ASCEND_CUSTOM_OPP_PATH','')
m=importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
assert hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()=='4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f'
original=m._compute_kernel_inputs_from_torch_wy
torch.ops.load_library(str(a.pair/'extension/pr17640_gdn_pair.so'))
torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h_vllm=torch.ops.pr17640_gdn_pair.fwd_h
torch.ops._C_ascend.chunk_fwd_o_vllm=torch.ops.pr17640_gdn_pair.fwd_o
name='pr17640_wy' if a.candidate=='native' else 'cann_wy921'
torch.ops.load_library(str(a.wy/'extension'/f'{name}.so'))
operator=getattr(torch.ops,name).compute
calls=0
def helper(q,k,v,g,beta,chunk_size):
 global calls
 assert supported(q,k,v,g,beta,chunk_size),'No fallback'
 calls+=1
 return operator(*(x.contiguous() for x in (q,k,v,g,beta)),chunk_size)
a.out.mkdir(exist_ok=False)
for rank in (0,1):
 torch.npu.set_device(rank);torch_npu.npu.set_compile_mode(jit_compile=False)
 for path in sorted(a.captures.glob(f'rank{rank}-call*.pt')):
  saved=torch.load(path,weights_only=True);meta=saved['metadata'];cu=meta['cu_seqlens'];assert cu is not None and len(cu)==2 and cu[0]==0
  length=cu[-1];inputs=[x[:,:length].contiguous().npu() for x in saved['inputs']]
  kw={'initial_state':saved['initial_state'].npu(),'output_final_state':True,'head_first':False,
      'cu_seqlens':torch.tensor(cu,dtype=torch.int64,device='npu'),'use_qk_l2norm_in_kernel':False}
  output={}
  for mode,fn in [('torch',original),(a.candidate,helper)]:
   m._compute_kernel_inputs_from_torch_wy=fn
   result=m.chunk_gated_delta_rule_310(*inputs,**kw)
   output[mode+'_outputs']=[x.cpu() for x in result]
  torch.save({'metadata':meta,**output},a.out/path.name)
  print(json.dumps({'source':path.name,'candidate':a.candidate,'cumulative_calls':calls}),flush=True)
assert calls==120,calls
