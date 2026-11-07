"""One captured input: pinned Torch WY versus isolated blocked UT candidate."""
import ast, hashlib, json, time
from pathlib import Path
import torch, torch_npu
base=Path('/runtime'); out=base/'torch-ut-graph-identity-20261107/data';out.mkdir(exist_ok=False)
source=base/'runtime-wy-pinned.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f'
from torch_ut_graph_candidate import load_pair
baseline,candidate,graph_ut=load_pair(source)
ns={'candidate':candidate}
torch.set_num_threads(1)
items=[]
for label,folder in [('snapshots',base/'wy921-localize-v1/capture/captures'),('triggers',base/'wy4-candidate-v3/triggers')]:
 for path in sorted(folder.glob('rank*-call*.pt')):
  items.append((label,path,int(path.name[4]),None))
assert len(items)==132
selection=json.loads((base/'wy-guarded-20261106/boundary-selection.json').read_text());assert len(selection)==480
items += [('boundaries',base/'wy-boundary-20261106/capture'/r['file'],r['rank'],r['sha256']) for r in selection]
freeze={'baseline_sha':hashlib.sha256(source.read_bytes()).hexdigest(),'candidate_sha':hashlib.sha256((base/'torch_ut_graph_candidate.py').read_bytes()).hexdigest(),'script_sha':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'inputs':[{'set':label,'file':str(path.relative_to(base)),'sha':hashlib.sha256(path.read_bytes()).hexdigest()} for label,path,_,_ in items], 'scope':'612 fixed-input graph UT byte-identity/repeat check against pinned Torch baseline; not an absolute CPU G0 or model/performance acceptance'}
(out/'freeze.json').write_text(json.dumps(freeze,indent=2))
for i,(label,path,rank,sha) in enumerate(items):
 if sha: assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
 torch.npu.set_device(rank);torch_npu.npu.set_compile_mode(jit_compile=False)
 saved=torch.load(path,map_location='cpu',weights_only=False);args=[x.npu() for x in saved['inputs']]
 chunk=saved.get('chunk_size',64)
 original=[x.cpu() for x in baseline(*args,chunk)]
 actual=[x.cpu() for x in ns['candidate'](*args,chunk)]
 serial=[[torch.equal(x.cpu().view(torch.uint8),y.view(torch.uint8)) for x,y in zip(ns['candidate'](*args,chunk),actual)] for _ in range(3)]
 pending=[ns['candidate'](*args,chunk) for _ in range(3)]
 queued=[[torch.equal(x.cpu().view(torch.uint8),y.view(torch.uint8)) for x,y in zip(v,actual)] for v in pending]
 exact=[torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(original,actual)]
 record={'identity':exact,'all_outputs_exact':all(exact),'index':i,'set':label,'file':str(path.relative_to(base)),'source_sha':freeze['inputs'][i]['sha'],'rank':rank,'q_shape':list(args[0].shape),'serial':serial,'queued':queued,'repeat_pass':all(all(v) for v in serial+queued),'qkg_exact':all(torch.equal(original[j],actual[j]) for j in (0,1,4))}
 if not all(exact): torch.save({'candidate_outputs':actual,'torch_outputs':original},out/f'{i:04d}-mismatch.pt')
 with (out/'rows.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
 if i%30==0:print(i+1,'/',len(items),flush=True)
 assert record['repeat_pass'] and record['all_outputs_exact'],record
(out/'device-summary.json').write_text(json.dumps({'cases':len(items),'repeat_passed':True,'qkg_exact':True,'all_outputs_exact':True,'status':'baseline_identity_pass_not_absolute_G0','captures':graph_ut.captures,'cache_entries':len(graph_ut.cache)},indent=2))
