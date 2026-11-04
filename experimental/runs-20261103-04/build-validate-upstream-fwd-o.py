#!/usr/bin/env python3
"""Dense基线退出后，构建固定PR17640的FwdO原实现；只验证算子，不启动模型。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args();root=a.root.resolve();out=root/'upstream-fwd-o';out.mkdir(exist_ok=False)
steps=[];paused=False;finished=False;run_id=str(time.time_ns())
def state(value,**kw):
 f=out/'status.tmp';f.write_text(json.dumps(dict(state=value,steps=steps,**kw),indent=2)+'\n');f.replace(out/'status.json')
def run(label,cmd):
 state('running',current=label);begin=time.monotonic()
 with (out/(label+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 steps.append(dict(label=label,returncode=r.returncode,elapsed_s=time.monotonic()-begin))
 if r.returncode:raise RuntimeError(label+' failed')
try:
 state('waiting_quant_isolation_cleanup')
 while True:
  f=root/'dense-quant-isolation/status.json';j=json.loads(f.read_text()) if f.exists() else {}
  if j.get('state')=='complete':break
  if j.get('state')=='failed':raise RuntimeError('Quant isolation incomplete; avoid overlapping recovery')
  time.sleep(5)
 native=root/'native-gdn';source=out/'source'
 def ignore(path,names):
  excluded={'.git','__pycache__'}
  if Path(path)==native/'source/csrc':excluded.update({'build','build_out','build_cache','output'})
  return set(names)&excluded
 shutil.copytree(native/'source',source,ignore=ignore,symlinks=True)
 tree=json.loads((native/'tree.json').read_text());verified=[]
 for n in tree['tree']:
  if n['type']!='blob' or not n['path'].startswith(('csrc/moe/chunk_fwd_o_vllm/','csrc/moe/common/','csrc/common/')):continue
  data=(source/n['path']).read_bytes()
  if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=n['sha']:raise RuntimeError('Source mismatch: '+n['path'])
  verified.append(n['path'])
 cache=root/'fwd-o-maskfix/source/csrc/third_party/catlass';dest=source/'csrc/third_party/catlass'
 if dest.exists():shutil.rmtree(dest)
 shutil.copytree(cache,dest,symlinks=True)
 (out/'source-provenance.json').write_text(json.dumps(dict(commit='18163f6b1d208ab3b4fb06e9752225611c37f40e',upstream_author='GruntovDima',verified_git_blobs=verified,algorithm_changes=[],scope='Existing PR FwdO only; original production FwdH retained; independent torch namespace; no model launch'),indent=2)+'\n')
 cmd=json.loads((root/'fwd-o-maskfix/build-command.json').read_text())
 cmd[cmd.index('--name')+1]='qwen-upstream-fwd-o-build-'+run_id
 cmd[cmd.index(str(root/'fwd-o-maskfix')+':/work:rw')]=str(out)+':/work:rw'
 cmd[-1]='''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /work/source/csrc
bash build.sh --pkg --ops=chunk_fwd_o_vllm --soc=ascend310p --vendor_name=hyperqwen_upstream_o -j8
python3 /project/build_pr17640_fwd_o_extension.py --source-root /work/source --build /work/extension
'''
 (out/'build-command.json').write_text(json.dumps(cmd,indent=2)+'\n');run('build',cmd)
 packages=list((source/'csrc/build_out').glob('*.run'))
 if len(packages)!=1:raise RuntimeError('Ambiguous package')
 cmd[cmd.index('--name')+1]='qwen-upstream-fwd-o-install-'+run_id
 cmd[-1]='bash /work/'+str(packages[0].relative_to(out))+' --quiet --install-path=/work/opp'
 run('install',cmd)
 lib=out/'opp/vendors/hyperqwen_upstream_o_transformer/op_api/lib/libcust_opapi.so'
 symbols=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True)
 if not re.search(r'\baclnnChunkFwdOVllm$',symbols,re.M) or re.search(r'\baclnnChunkFwdO$|\baclnnChunkGatedDeltaRule',symbols,re.M):raise RuntimeError('Unexpected APIs')
 (out/'package-isolation.json').write_text(json.dumps(dict(symbols=symbols,sha256=hashlib.sha256(lib.read_bytes()).hexdigest()),indent=2)+'\n')
 paused=True;run('pause-a2-temporary-unit',['sudo','-n','systemctl','stop','llm-backend-primary.service'])
 idle=0
 for _ in range(90):
  info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
  idle=idle+1 if 'No running processes found' in info.stdout else 0
  if idle>=2:break
  time.sleep(2)
 else:raise RuntimeError('NPU not idle')
 cmd=json.loads((native/'validation-command.json').read_text())
 cmd[cmd.index('--name')+1]='qwen-upstream-fwd-o-validate-'+run_id
 image=json.loads((root/'production-manifest.json').read_text())['image'];idx=cmd.index(image)
 cmd[idx:idx]=['-v',str(out)+':/upstream-o:rw']
 cmd[-1]='''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="/upstream-o/opp/vendors/hyperqwen_upstream_o_transformer:/work/opp/vendors/hyperqwen_wy_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"
cd /work/tools
python3 probe_fwd_o_frozen.py --work /work --out-dir /upstream-o/frozen-validation --fwd-o-library /upstream-o/extension/pr17640_fwd_o.so --fwd-o-namespace pr17640_fwd_o --candidate-label upstream_fwd_o --repeats 48
python3 - <<'GATE'
import json
from pathlib import Path
s=json.loads(Path('/upstream-o/frozen-validation/summary.json').read_text())
assert s['upstream_fwd_o']==dict(tested=48,within_tolerance=48,output_equal_first=48),s
GATE
python3 probe_gdn_intermediates.py --work /work --out-dir /upstream-o/chain-validation --fwd-o-library /upstream-o/extension/pr17640_fwd_o.so --fwd-o-namespace pr17640_fwd_o
python3 - <<'GATE'
import json,sys
from pathlib import Path
import torch
from vllm_ascend.utils import enable_custom_op
assert enable_custom_op()
s=json.loads(Path('/upstream-o/chain-validation/summary.json').read_text())
assert all(x['tested']==24 and x['outside_tolerance_runs']==0 and not any(x['interface_variation'].values()) for x in s.values()),s
rows=list(map(json.loads,Path('/upstream-o/chain-validation/intermediates.jsonl').read_text().splitlines()))
assert all(all(x['interfaces_equal_first']['fwd_o_raw']) and x['raw_matches_final'] for x in rows)
torch.ops.load_library('/upstream-o/extension/pr17640_fwd_o.so')
torch.ops._C_ascend.chunk_fwd_o=torch.ops.pr17640_fwd_o.run
sys.argv=['check_native_wy.py','--library','/work/wy-extension/pr17640_wy.so','--out','/upstream-o/full-gdn-gate.jsonl']
from check_native_wy import main
main()
rows=list(map(json.loads,Path('/upstream-o/full-gdn-gate.jsonl').read_text().splitlines()))
assert len(rows)==10 and all(x['status']=='PASS' and not x.get('baseline_reference_error') for x in rows),rows
GATE
'''
 (out/'validation-command.json').write_text(json.dumps(cmd,indent=2)+'\n');run('validate',cmd);finished=True
except Exception as error:
 state('failed',error=str(error));raise
finally:
 if paused:
  with (out/'restore-a2-service.log').open('x') as log:r=subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:finished=False;state('failed',error='Could not restore a2 temporary unit')
 if finished:state('passed_operator_validation',scope='Existing PR FwdO, original FwdH; operator-level only; no model/performance acceptance')
