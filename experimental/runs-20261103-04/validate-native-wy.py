#!/usr/bin/env python3
"""等待 WY 构建，隔离安装、检查符号，再执行生产栈完整数值门禁。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--work',type=Path,required=True)
p.add_argument('--container',default='qwen36-native-wy-validate-20261103')
a=p.parse_args()
root=a.work.resolve()
status=root/'validation-status.json'
def report(stage,**extra):
    status.write_text(json.dumps(dict(stage=stage,**extra),indent=2)+'\n')
try:
    report('waiting_build')
    for _ in range(720):
        j=json.loads((root/'build-status.json').read_text())
        if j['stage']=='failed': raise RuntimeError('Build failed; inspect build-status.json')
        if j['stage']=='built': break
        time.sleep(5)
    else: raise TimeoutError('build')
    packages=[root/x for x in j['packages'] if (root/x).is_file()]
    if not packages: raise RuntimeError('No build package')
    if len({hashlib.sha256(x.read_bytes()).hexdigest() for x in packages})!=1:
        raise RuntimeError('Multiple different packages; inspect before selecting')
    package=packages[0].relative_to(root)
    image=json.loads((root/'tools/production-manifest.json').read_text())['image']
    cmd=['podman','run','--name',a.container,'--network','none',
         '--entrypoint','/bin/bash','--group-add','keep-groups','-e','OMP_NUM_THREADS=1',
         '-v',f'{root}:/work:rw']
    for key,value in json.loads((root/'tools/production-manifest.json').read_text())['env'].items():
        cmd += ['-e',f'{key}={value}']
    for device in ('davinci0','davinci1','davinci_manager','devmm_svm','hisi_hdc'):
        cmd+=['--device','/dev/'+device]
    for mount in ('/usr/local/Ascend/driver/lib64','/etc/ascend_install.info',
                  '/usr/local/Ascend/driver/version.info','/usr/local/dcmi'):
        cmd+=['-v',f'{mount}:{mount}:ro']
    shell='''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash /work/PACKAGE --quiet --install-path=/work/opp
export ASCEND_CUSTOM_OPP_PATH="/work/opp/vendors/hyperqwen_wy_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"
cd /work/tools
python3 - <<'CHECK'
import hashlib,json,pathlib,subprocess
p=pathlib.Path('/work/opp/vendors/hyperqwen_wy_transformer')
lib=p/'op_api/lib/libcust_opapi.so'
symbols=subprocess.run(['nm','-D','--defined-only',str(lib)],capture_output=True,text=True,check=True).stdout
if 'aclnnChunkGatedDeltaRuleComputeWy' not in symbols or any(x in symbols for x in ('FwdH','FwdO')):
    raise RuntimeError('Unexpected OPP API exports')
files=[str(f.relative_to(p)) for f in p.rglob('*') if f.is_file()]
if any('fwd_h' in f or 'fwd_o' in f for f in files):raise RuntimeError('Unexpected non-WY OPP files')
pathlib.Path('/work/package-isolation.json').write_text(json.dumps(dict(symbols=symbols,files=files,
    library_sha256=hashlib.sha256(lib.read_bytes()).hexdigest(),scope='WY-only package; no FwdH/FwdO exports or kernels'),indent=2)+'\\n')
CHECK
python3 check_native_wy.py --library /work/wy-extension/pr17640_wy.so --out /work/numerical-gate.jsonl
'''.replace('PACKAGE',str(package))
    cmd += [image,'--noprofile','--norc','-c',shell]
    (root/'validation-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    report('validating')
    with (root/'validation.log').open('x') as log:
        r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f'validation exited {r.returncode}; inspect validation.log')
    rows=[json.loads(x) for x in (root/'numerical-gate.jsonl').read_text().splitlines()]
    if len(rows)!=10 or any(x['status']!='PASS' for x in rows):raise RuntimeError('Incomplete numerical gate')
    report('passed',cases=len(rows))
except Exception as e:
    report('failed',error=str(e))
    raise
