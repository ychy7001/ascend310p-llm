"""Replay all 480 B same-input WY captures; save outputs for independent CPU G0."""
import argparse,hashlib,importlib,json,os,time
from pathlib import Path
import torch,torch_npu
p=argparse.ArgumentParser()
for key in ('pair','wy','captures','selection','out'):p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1)
for root in (a.pair,a.wy):
 for name,expected in json.loads((root/'operator-artifacts.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
vendors=[a.wy/'opp/vendors/hyperqwen_gram_postbeta128_transformer',a.pair/'opp/vendors/hyperqwen_gdn_pair921_transformer']
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
os.environ['ASCEND_CUSTOM_OPP_PATH']=':'.join(map(str,vendors))+':'+os.environ.get('ASCEND_CUSTOM_OPP_PATH','')
module=importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
assert hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()=='4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f'
torch.ops.load_library(str(a.wy/'extension/pr17640_wy.so'))
rows=json.loads(a.selection.read_text());assert len(rows)==480;a.out.mkdir(exist_ok=False)
for i,row in enumerate(rows):
 torch.npu.set_device(row['rank']);torch_npu.npu.set_compile_mode(jit_compile=False)
 source=a.captures/row['file'];assert hashlib.sha256(source.read_bytes()).hexdigest()==row['sha256']
 saved=torch.load(source,map_location='cpu',weights_only=False);inputs=[x.npu() for x in saved['inputs']]
 actual=[x.cpu() for x in torch.ops.pr17640_wy.compute(*inputs,saved['chunk_size'])]
 baseline=[x.cpu() for x in module._compute_kernel_inputs_from_torch_wy(*inputs,saved['chunk_size'])]
 serial=[]
 for _ in range(3):serial.append([torch.equal(x.cpu().view(torch.uint8),y.view(torch.uint8)) for x,y in zip(torch.ops.pr17640_wy.compute(*inputs,saved['chunk_size']),actual)])
 pending=[torch.ops.pr17640_wy.compute(*inputs,saved['chunk_size']) for _ in range(3)]
 queued=[[torch.equal(x.cpu().view(torch.uint8),y.view(torch.uint8)) for x,y in zip(v,actual)] for v in pending]
 record={'index':i,'source':row['file'],'source_sha256':row['sha256'],'rank':row['rank'],'case':row['case'],'layer':row['layer'],
  'serial_repeats':serial,'queued_repeats':queued,'repeat_pass':all(all(x) for x in serial+queued)}
 torch.save({'native_outputs':actual,'torch_outputs':baseline},a.out/(f'{i:04d}.pt'))
 with (a.out/'replay.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
 print(json.dumps(record),flush=True)
 assert record['repeat_pass'],'Serial/queued bitwise repeat failed'
loaded=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcust_opapi.so' in line})
assert str((vendors[0]/'op_api/lib/libcust_opapi.so').resolve()) in loaded,loaded
(a.out/'loaded-apis.json').write_text(json.dumps(loaded,indent=2))
