#!/usr/bin/env python3
"""固定PR17640源码在9.2.1重编译WY；不暴露NPU，不暂停原服务。"""
import argparse,hashlib,json,shutil,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();out=r/'native-wy921-v1';out.mkdir(exist_ok=False)
image='d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a'
def state(value,**kw):(out/'status.json').write_text(json.dumps(dict(state=value,**kw),indent=2)+'\n')
try:
 state('preparing');src=r/'native-gdn/source'
 def ignore(path,names):
  skip={'.git','__pycache__'}
  if Path(path)==src/'csrc':skip|={'build','build_out','build_cache','output'}
  return set(names)&skip
 shutil.copytree(src,out/'source',symlinks=True,ignore=ignore)
 tree=json.loads((r/'native-gdn/tree.json').read_text());verified=[]
 for x in tree['tree']:
  name=x['path']
  if x['type']!='blob' or not (name.startswith(('csrc/moe/chunk_gated_delta_rule_compute_wy/','csrc/common/','csrc/aclnn_torch_adapter/')) or name in ('csrc/torch_binding.cpp','csrc/build.sh')):continue
  b=(out/'source'/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==x['sha'],name;verified.append(name)
 assert any('compute_wy' in x for x in verified)
 tools=out/'tools';tools.mkdir();shutil.copy2(r/'build_pr17640_wy_extension.py',tools)
 template=json.loads((r/'upstream-fwd-o/build-command.json').read_text());cmd=[];i=0
 while i<len(template):
  if template[i]=='--device':i+=2;continue
  cmd.append(template[i]);i+=1
 cmd[cmd.index('--name')+1]='native-wy921-build-'+str(time.time_ns())
 cmd[cmd.index(str(r/'upstream-fwd-o')+':/work:rw')]=str(out)+':/work:rw'
 old=json.loads((r/'production-manifest.json').read_text())['image'];cmd[cmd.index(old)]=image
 cmd[-1]='''set -eo pipefail
source /usr/local/Ascend/cann/set_env.sh
cd /work/source/csrc
bash build.sh --pkg --ops=chunk_gated_delta_rule_compute_wy --soc=ascend310p --vendor_name=hyperqwen_wy921 -j4
python3 /work/tools/build_pr17640_wy_extension.py --source-root /work/source --build /work/extension
'''
 (out/'source-provenance.json').write_text(json.dumps(dict(commit='18163f6b1d208ab3b4fb06e9752225611c37f40e',image=image,verified_git_blobs=verified,scope='Rebuild unchanged upstream WY; no NPU devices; source cache dependencies reused'),indent=2))
 (out/'build-command.json').write_text(json.dumps(cmd,indent=2));state('building')
 with (out/'build.log').open('x') as f:res=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 assert res.returncode==0,f'Build exit {res.returncode}'
 packages=list((out/'source/csrc/build_out').glob('*.run'));assert len(packages)==1
 cmd[cmd.index('--name')+1]='native-wy921-install-'+str(time.time_ns());cmd[-1]='bash /work/'+str(packages[0].relative_to(out))+' --quiet --install-path=/work/opp'
 with (out/'install.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=120)
 api=out/'opp/vendors/hyperqwen_wy921_transformer/op_api/lib/libcust_opapi.so';symbols=subprocess.check_output(['nm','-D','--defined-only',str(api)],text=True)
 assert 'aclnnChunkGatedDeltaRuleComputeWy' in symbols and not any(s in symbols for s in ('FwdH','FwdO'))
 artifacts={str(x.relative_to(out)):hashlib.sha256(x.read_bytes()).hexdigest() for directory in [out/'extension',out/'opp'] for x in directory.rglob('*') if x.is_file()}
 (out/'operator-artifacts.json').write_text(json.dumps(artifacts,indent=2));state('built_requires_numerical_gate',package=str(packages[0].relative_to(out)),artifacts=len(artifacts),symbols=symbols)
except Exception as e:state('failed',error=str(e));raise
