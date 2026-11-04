#!/usr/bin/env python3
"""固定输入FwdO复现后构建最小mask试验包，并独占验证；从不覆盖生产OPP。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--attempt',default='v4')
p.add_argument('--dependency-cache',type=Path)
a=p.parse_args();root=a.root.resolve();native=root/'native-gdn'
out=root/'fwd-o-maskfix';out.mkdir(exist_ok=False)
steps=[];paused=False;finished=False

def state(value,**kw):
 f=out/'status.tmp';f.write_text(json.dumps(dict(state=value,steps=steps,**kw),indent=2)+'\n');f.replace(out/'status.json')
def run(label,cmd):
 state('running',current=label);begin=time.monotonic()
 with (out/(label+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 steps.append(dict(label=label,returncode=r.returncode,elapsed_s=time.monotonic()-begin))
 if r.returncode:raise RuntimeError(label+' failed')
def idle():
 count=0
 for _ in range(90):
  r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
  count=count+1 if 'No running processes found' in r.stdout else 0
  if count>=2:return
  time.sleep(2)
 raise RuntimeError('NPU cleanup incomplete')
try:
 state('waiting_fwd_o_chain_isolation')
 while True:
  j=json.loads((root/'gdn-intermediate-isolation-raw/status.json').read_text())
  if j['state'] in ('failed','not_started'):state('not_started',reason='isolation did not complete');raise SystemExit(0)
  if j['state']=='complete':break
  time.sleep(5)
 diagnostic=json.loads((native/'intermediate-isolation-raw/summary.json').read_text())['torch_wy']
 if not diagnostic['outside_tolerance_runs'] or diagnostic['raw_output_differs_final_cpu'] or any(diagnostic['interface_variation'].values()):
  state('not_started',reason='Did not isolate raw FwdO output deviation with stable inputs');raise SystemExit(0)
 source=out/'source'
 state('preparing_source')
 def ignore_source(path,names):
  ignored={'.git','__pycache__'}
  if Path(path)==native/'source/csrc':ignored.update({'build','build_out','build_cache','output'})
  return set(names)&ignored
 shutil.copytree(native/'source',source,ignore=ignore_source)
 if a.dependency_cache:shutil.copytree(a.dependency_cache,source/'csrc/build_cache')
 overlay=out/'production-overlay';overlay.mkdir()
 with tarfile.open(root/'production-fwd-o-maskfix-inputs.tar.gz') as tf:
  for member in tf.getmembers():
   if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:raise RuntimeError('Unsafe overlay member')
  tf.extractall(overlay,filter='data')
 tree=json.loads((overlay/'git-tree.json').read_text());verified=[]
 for node in tree['tree']:
  if node['type']!='blob' or not node['path'].startswith(('csrc/moe/chunk_fwd_o/','csrc/moe/common/')):continue
  data=(overlay/node['path']).read_bytes()
  if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=node['sha']:raise RuntimeError('Overlay Git blob mismatch')
  dest=source/node['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);verified.append(node['path'])
 run('apply-mask-patch',['git','-C',str(source),'apply',str(root/'production-fwd-o-mask-fences.unvalidated.patch')])
 old=source/'csrc/moe/chunk_fwd_o';renamed=source/'csrc/moe/chunk_fwd_o_mask_fix'
 old.rename(renamed)
 for path in sorted(renamed.rglob('*'),key=lambda p:len(p.parts),reverse=True):
  if not path.is_file():continue
  text=path.read_text().replace('ChunkFwdO','ChunkFwdOMaskFix').replace('CHUNK_FWD_O','CHUNK_FWD_O_MASK_FIX').replace('chunk_fwd_o','chunk_fwd_o_mask_fix')
  path.write_text(text)
  newname=path.name.replace('chunk_fwd_o','chunk_fwd_o_mask_fix')
  if newname!=path.name:path.rename(path.with_name(newname))
 cache=Path('/path/to/lab-runtime/mindieq-r3-20261013/pr17640/csrc/third_party/catlass')
 head=subprocess.check_output(['git','-C',str(cache),'rev-parse','HEAD'],text=True).strip()
 if head!='41bf90da655bba3c66d0acd7e00abe33960ecfd6':raise RuntimeError('Catlass source changed')
 catlass=source/'csrc/third_party/catlass'
 if catlass.exists():shutil.rmtree(catlass)
 shutil.copytree(cache,catlass,ignore=shutil.ignore_patterns('.git','build'))
 (out/'source-provenance.json').write_text(json.dumps(dict(build_framework_commit='18163f6b1d208ab3b4fb06e9752225611c37f40e',
  production_operator_commit=tree['sha'],verified_production_blobs=verified,catlass_commit=head,
  changes=['V/S fences around original scalar tail mask updates','isolated operator/API rename'],
  credit='Independent minimal fence experiment on production FwdO; original kernel authorship retained. Prior vector rewrite failed and is preserved.'),indent=2)+'\n')
 manifest=json.loads((root/'production-manifest.json').read_text());image=manifest['image']
 cmd=['podman','run','--name','qwen36-fwd-o-maskfix-build-20261103-'+a.attempt,'--network','none','--entrypoint','/bin/bash',
  '--group-add','keep-groups','-e','OMP_NUM_THREADS=1','-e','MAX_JOBS=4','-v',f'{out}:/work:rw','-v',f'{root}:/project:ro']
 for device in ['davinci0','davinci1','davinci_manager','devmm_svm','hisi_hdc']:cmd+=['--device','/dev/'+device]
 for mount in ['/usr/local/Ascend/driver/lib64','/etc/ascend_install.info','/usr/local/Ascend/driver/version.info','/usr/local/dcmi']:cmd+=['-v',f'{mount}:{mount}:ro']
 shell='''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /work/source/csrc
bash build.sh --pkg --ops=chunk_fwd_o_mask_fix --soc=ascend310p --vendor_name=hyperqwen_maskfix -j8
python3 /project/build_fwd_o_maskfix_extension.py --source-root /work/source --build /work/extension
'''
 cmd += [image,'--noprofile','--norc','-c',shell]
 (out/'build-command.json').write_text(json.dumps(cmd,indent=2)+'\n');run('build',cmd)
 packages=[x for x in (source/'csrc').rglob('*.run') if x.is_file()]
 if not packages or len({hashlib.sha256(x.read_bytes()).hexdigest() for x in packages})!=1:raise RuntimeError('Missing or ambiguous package')
 package=packages[0].relative_to(out)
 install=cmd[:-4]+['--noprofile','--norc','-c',f'bash /work/{package} --quiet --install-path=/work/opp']
 install[install.index('--name')+1]='qwen36-fwd-o-maskfix-install-20261103-'+a.attempt
 run('install-isolated-package',install)
 lib=out/'opp/vendors/hyperqwen_maskfix_transformer/op_api/lib/libcust_opapi.so'
 symbols=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True)
 if not re.search(r'\baclnnChunkFwdOMaskFix$',symbols,re.M) or re.search(r'\baclnnChunkFwdO$|aclnnChunkGatedDeltaRule',symbols,re.M):raise RuntimeError('Unexpected package API exports')
 (out/'package-isolation.json').write_text(json.dumps(dict(symbols=symbols,sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),
  scope='Only renamed FwdO API; production OPP and FwdH untouched'),indent=2)+'\n')
 paused=True
 run('pause-a2-temporary-unit',['sudo','-n','systemctl','stop','llm-backend-primary.service']);idle()
 cmd=json.loads((native/'validation-command.json').read_text())
 cmd[cmd.index('--name')+1]='qwen36-fwd-o-maskfix-validate-20261103-'+a.attempt
 index=cmd.index(image);cmd[index:index]=['-v',f'{out}:/maskfix:rw']
 cmd[-1]='''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="/maskfix/opp/vendors/hyperqwen_maskfix_transformer:/work/opp/vendors/hyperqwen_wy_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"
cd /work/tools
python3 probe_fwd_o_frozen.py --work /work --out-dir /maskfix/frozen-validation --maskfix-library /maskfix/extension/maskfix_fwd_o.so --repeats 48
python3 - <<'GATE'
import json
from pathlib import Path
summary=json.loads(Path('/maskfix/frozen-validation/summary.json').read_text())
assert summary['maskfix']==dict(tested=48,within_tolerance=48,output_equal_first=48),summary
GATE
python3 check_maskfix_gdn.py --library /work/wy-extension/pr17640_wy.so --out /maskfix/full-gdn-gate.jsonl
'''
 (out/'validation-command.json').write_text(json.dumps(cmd,indent=2)+'\n');run('validate',cmd)
 frozen=json.loads((out/'frozen-validation/summary.json').read_text())
 if frozen['maskfix']!={'tested':48,'within_tolerance':48,'output_equal_first':48}:raise RuntimeError('Fixed-input maskfix did not pass')
 rows=list(map(json.loads,(out/'full-gdn-gate.jsonl').read_text().splitlines()))
 if len(rows)!=10 or any(r['status']!='PASS' or r.get('baseline_reference_error') for r in rows):raise RuntimeError('Full GDN/reference gate failed')
 finished=True
except Exception as e:
 state('failed',error=str(e));raise
finally:
 if paused:
  with (out/'restore-a2-service.log').open('x') as log:r=subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:finished=False;state('failed',error='Could not restore a2 temporary unit')
 if finished:state('passed_operator_validation',scope='Fixed-input FwdO plus full synthetic GDN gate; no model performance or business acceptance yet')
