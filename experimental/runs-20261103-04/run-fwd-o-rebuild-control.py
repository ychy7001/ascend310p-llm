#!/usr/bin/env python3
"""对照：保留独立API改名，但恢复全部生产kernel源码，增量重编译并检查数值。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args();root=a.root.resolve();out=root/'fwd-o-rebuild-control'
if out.exists():raise RuntimeError('Control output already exists')
steps=[];paused=False;finished=False;run_id=str(time.time_ns())

def state(value,**kw):
 f=out/'status.tmp';f.write_text(json.dumps(dict(state=value,steps=steps,**kw),indent=2)+'\n');f.replace(out/'status.json')
def run(label,cmd):
 state('running',current=label);begin=time.monotonic()
 with (out/(label+'.log')).open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 steps.append(dict(label=label,returncode=r.returncode,elapsed_s=time.monotonic()-begin))
 if r.returncode:raise RuntimeError(label+' failed')
# Exact copied build paths remain /work/... inside the independent container.
excluded={'status.json','validate.log','frozen-validation','chain-validation','full-gdn-gate.jsonl','opp'}
def exclude_results(path,names):
 return {name for name in names if name in excluded or name.endswith('.log')} if Path(path)==root/'fwd-o-maskfix' else set()
shutil.copytree(root/'fwd-o-maskfix',out,symlinks=True,ignore=exclude_results)

try:
 state('preparing')
 rel='op_kernel/arch20/gemm/kernel/gdn_fwd_o_kernel.hpp'
 original=root/'production-kernel-source-audit/csrc__moe__chunk_fwd_o'/rel
 data=original.read_text().replace('ChunkFwdO','ChunkFwdOMaskFix').replace('CHUNK_FWD_O','CHUNK_FWD_O_MASK_FIX').replace('chunk_fwd_o','chunk_fwd_o_mask_fix')
 target=out/'source/csrc/moe/chunk_fwd_o_mask_fix'/rel;target.write_text(data)
 (out/'rebuild-control-definition.json').write_text(json.dumps(dict(scope='No mask/math/synchronization changes; isolated API rename only',
  original_source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),controlled_header_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
  framework='Same pinned PR build framework and extension as failed variants; incremental build; old results are explicitly excluded'),indent=2)+'\n')
 # Force the custom kernel build to reevaluate the changed header; preserve host deps.
 (out/'source/csrc/moe/chunk_fwd_o_mask_fix/op_kernel/chunk_fwd_o_mask_fix.cpp').touch()
 cmd=json.loads((out/'build-command.json').read_text())
 cmd[cmd.index('--name')+1]='qwen36-fwd-o-rebuild-control-20261103-'+run_id
 oldmount=str(root/'fwd-o-maskfix')+':/work:rw'
 cmd[cmd.index(oldmount)]=str(out)+':/work:rw'
 cmd[-1]='''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /work/source/csrc/build
cmake --build . --target package -j4
'''
 (out/'rebuild-command.json').write_text(json.dumps(cmd,indent=2)+'\n');run('rebuild',cmd)
 packages=[x for x in (out/'source/csrc').rglob('*.run') if x.is_file()]
 if not packages or len({hashlib.sha256(x.read_bytes()).hexdigest() for x in packages})!=1:raise RuntimeError('Ambiguous rebuilt packages')
 cmd[cmd.index('--name')+1]='qwen36-fwd-o-rebuild-install-20261103-'+run_id
 cmd[-1]='bash /work/'+str(packages[0].relative_to(out))+' --quiet --install-path=/work/opp'
 run('install',cmd)
 bins={str(f.relative_to(out/'opp')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (out/'opp').rglob('*.o')}
 oldbins={str(f.relative_to(root/'fwd-o-maskfix/opp')):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'fwd-o-maskfix/opp').rglob('*.o')}
 (out/'binary-comparison.json').write_text(json.dumps(dict(rebuilt=bins,previous_fence=oldbins),indent=2)+'\n')
 if not bins or bins==oldbins:raise RuntimeError('Kernel binary did not change; investigate build dependencies before numerical claims')
 paused=True
 run('pause-a2-temporary-unit',['sudo','-n','systemctl','stop','llm-backend-primary.service'])
 idle=0
 for _ in range(90):
  r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
  idle=idle+1 if 'No running processes found' in r.stdout else 0
  if idle>=2:break
  time.sleep(2)
 else:raise RuntimeError('NPU not idle')
 cmd=json.loads((root/'fwd-o-maskfix/validation-command.json').read_text())
 cmd[cmd.index('--name')+1]='qwen36-fwd-o-rebuild-validate-20261103-'+run_id
 oldmount=str(root/'fwd-o-maskfix')+':/maskfix:rw';cmd[cmd.index(oldmount)]=str(out)+':/maskfix:rw'
 cmd[-1]='''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="/maskfix/opp/vendors/hyperqwen_maskfix_transformer:/work/opp/vendors/hyperqwen_wy_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"
cd /work/tools
python3 probe_fwd_o_frozen.py --work /work --out-dir /maskfix/frozen-control --maskfix-library /maskfix/extension/maskfix_fwd_o.so --repeats 48
python3 probe_gdn_intermediates.py --work /work --out-dir /maskfix/chain-control --maskfix-library /maskfix/extension/maskfix_fwd_o.so
'''
 (out/'control-validation-command.json').write_text(json.dumps(cmd,indent=2)+'\n');run('control-validate',cmd)
 finished=True
except Exception as e:
 state('failed',error=str(e));raise
finally:
 if paused:
  with (out/'control-restore-a2-service.log').open('x') as f:r=subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],stdout=f,stderr=subprocess.STDOUT)
  if r.returncode:finished=False;state('failed',error='Could not restore a2 temporary unit')
 if finished:state('complete',scope='Unmodified-kernel rebuild control; diagnostic only. Operator namespace in raw probe remains maskfix for ABI reuse, but no fix is present.')
