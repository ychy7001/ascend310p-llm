#!/usr/bin/env python3
"""固定9.2.1配套H/O，只切WY；两路径分别对CPU递推并检查重复性。"""
import argparse,hashlib,importlib,json,os
from pathlib import Path
import torch,torch_npu
from check_native_wy import compare,recurrent_reference
from native_wy import supported
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pair',type=Path,required=True);p.add_argument('--wy',type=Path,required=True);p.add_argument('--device',type=int,choices=(0,1),required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2)
vendors=[str(a.wy/'opp/vendors/hyperqwen_postbeta_refined_transformer'),str(a.pair/'opp/vendors/hyperqwen_gdn_pair921_transformer')]
os.environ['ASCEND_CUSTOM_OPP_PATH']=':'.join(vendors+[x for x in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':') if x and x not in vendors])
for root in (a.pair,a.wy):
 for name,expected in json.loads((root/'operator-artifacts.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
# Bootstrap prepends its builtin vendor; restore isolated vendors before ACL init and bridge dlopen.
os.environ['ASCEND_CUSTOM_OPP_PATH']=':'.join(vendors+[x for x in os.environ.get('ASCEND_CUSTOM_OPP_PATH','').split(':') if x and x not in vendors])
torch.npu.set_device(a.device);torch_npu.npu.set_compile_mode(jit_compile=False)
m=importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule');sha=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest();assert sha=='4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f'
# Both vendor paths are set before either bridge snapshots the search order.
torch.ops.load_library(str(a.pair/'extension/pr17640_gdn_pair.so'));torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h_vllm=torch.ops.pr17640_gdn_pair.fwd_h;torch.ops._C_ascend.chunk_fwd_o_vllm=torch.ops.pr17640_gdn_pair.fwd_o
torch.ops.load_library(str(a.wy/'extension/pr17640_wy.so'));original=m._compute_kernel_inputs_from_torch_wy;calls=[]
def native(q,k,v,g,beta,chunk_size):
 ok=supported(q,k,v,g,beta,chunk_size);calls.append(ok)
 if ok:return torch.ops.pr17640_wy.compute(q.contiguous(),k.contiguous(),v.contiguous(),g.contiguous(),beta.contiguous(),chunk_size)
 raise RuntimeError('Native WY unsupported inputs; fallback prohibited')
cases=[(t,8,16,1.,None) for t in (1,63,64,65,111,127,128,129,512,1025)]+[(128,16,32,1.,None),(177,8,16,1.,[0,1,65,177]),(128,8,24,1.,None),(512,8,16,8.,None),(64,8,16,.01,None)]
rows=[]
with a.out.open('x') as f:
 f.write(json.dumps({'kind':'config','candidate':'native','namespace':'pr17640_wy','fallback':False,'chunk_sha256':sha,'device':a.device,'scope':'Same9.2.1 chunk, pairedH/O, synthetic inputs; only WY differs. No model or performance acceptance.','atol':.0002,'rtol':.04,'max_relative_rmse':.02})+'\n')
 for idx,(t,hk,hv,decay,cu) in enumerate(cases):
  torch.manual_seed(310+idx);q=torch.nn.functional.normalize(torch.randn(1,t,hk,128),dim=-1).half();k=torch.nn.functional.normalize(torch.randn_like(q).float(),dim=-1).half();v=(torch.randn(1,t,hv,128)*.1).half();g=-torch.rand(1,t,hv)*decay;beta=(.05+.9*torch.rand(1,t,hv)).half();cu=cu or [0,t];state=(torch.randn(len(cu)-1,hv,128,128)*.01).half();expected=recurrent_reference(q,k,v,g,beta,state,cu);device=[x.npu() for x in (q,k,v,g,beta)];kw=dict(initial_state=state.npu(),output_final_state=True,head_first=False,use_qk_l2norm_in_kernel=False)
  if len(cu)>2:kw['cu_seqlens']=torch.tensor(cu,dtype=torch.int64,device='npu')
  row={'kind':'case','case':idx,'tokens':t,'hk':hk,'hv':hv,'decay':decay,'cu':cu,'status':'PASS'};outputs={}
  for mode,helper in [('torch',original),('native',native)]:
   m._compute_kernel_inputs_from_torch_wy=helper
   try:
    actual=m.chunk_gated_delta_rule_310(*device,**kw);row[mode+'_vs_cpu']=compare(actual,expected);outputs[mode]=actual
    for _ in range(3):compare(m.chunk_gated_delta_rule_310(*device,**kw),actual,atol=0,rtol=0)
    row[mode+'_repeats']=3
   except AssertionError as exc:
    if mode=='native':row.update(status='FAIL')
    row[mode+'_error']=str(exc)
  if 'torch' in outputs and 'native' in outputs:
   try:row['native_vs_torch']=compare(outputs['native'],outputs['torch'])
   except AssertionError as exc:row['paired_error']=str(exc)
  if row['status']=='FAIL':torch.save({'inputs':(q,k,v,g,beta,state),'cu':cu,'expected':expected,'outputs':{k:tuple(z.cpu() for z in out) for k,out in outputs.items()}},a.out.with_suffix(f'.case{idx}.pt'))
  rows.append(row);f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
 assert calls and all(calls),'Native route not exercised or unexpected fallback'
 loaded=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcust_opapi.so' in line})
 for vendor in vendors:assert str((Path(vendor)/'op_api/lib/libcust_opapi.so').resolve()) in loaded,loaded
 summary={'kind':'summary','candidate':'native','namespace':'pr17640_wy','cases':len(rows),'passed':sum(x['status']=='PASS' for x in rows),'native_calls':len(calls),'loaded_apis':loaded};f.write(json.dumps(summary)+'\n');print(json.dumps(summary),flush=True)
if summary['passed']!=summary['cases']:raise SystemExit(1)
