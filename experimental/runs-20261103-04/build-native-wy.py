#!/usr/bin/env python3
"""在 a2 上验证固定 PR archive，使用生产镜像只构建 WY OPP/绑定。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--work', type=Path, required=True)
p.add_argument('--tools', type=Path, required=True)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--download-pid', type=int, required=True)
a=p.parse_args()
root=a.work.resolve()
status=root/'build-status.json'
def report(stage,**kw):
    status.write_text(json.dumps(dict(stage=stage,**kw),indent=2)+'\n')
report('waiting_source')
try:
    # Curl is a session child; disappearance alone is insufficient, gzip/tree checks follow.
    deadline=time.monotonic()+1900
    while Path(f'/proc/{a.download_pid}').exists():
        stat=Path(f'/proc/{a.download_pid}/stat').read_text().split(') ',1)[1].split()[0]
        if stat=='Z': break
        if time.monotonic()>deadline: raise TimeoutError('source download')
        time.sleep(5)
    source=root/'source'
    source.mkdir(exist_ok=False)
    archive=root/'source.tar.gz'
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            path=Path(member.name)
            if path.is_absolute() or '..' in path.parts:
                raise RuntimeError('unsafe archive path')
    subprocess.run(['tar','-xzf',str(archive),'-C',str(source),'--strip-components=1'],check=True)
    tree=json.loads((root/'tree.json').read_text())
    verified=0
    for node in tree['tree']:
        if node['type']!='blob': continue
        f=source/node['path']
        data=f.readlink().as_posix().encode() if f.is_symlink() else f.read_bytes()
        actual=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual!=node['sha']: raise RuntimeError('Git blob mismatch: '+node['path'])
        verified+=1
    (root/'source-verification.json').write_text(json.dumps(dict(
        commit='18163f6b1d208ab3b4fb06e9752225611c37f40e', verified_git_blobs=verified,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        dependency_cache_source='/path/to/lab-runtime/mindieq-r3-20261013/pr17640/csrc/third_party',
        scope='PR source tree verified before build; submodules not included in archive'),indent=2)+'\n')
    cache=Path('/path/to/lab-runtime/mindieq-r3-20261013/pr17640/csrc/third_party')
    for name in ('pkg','json','abseil-cpp','makeself','ascend_protobuf'):
        shutil.copytree(cache/name,source/'csrc/third_party'/name,dirs_exist_ok=True)
    image=json.loads(a.manifest.read_text())['image']
    cmd=['podman','run','--name','qwen36-native-wy-build-20261103','--network','host',
         '--entrypoint','/bin/bash','--group-add','keep-groups','-e','OMP_NUM_THREADS=1',
         '-e','MAX_JOBS=4','-v',f'{root}:/work:rw','-v',f'{a.tools.resolve()}:/project:ro']
    for device in ('davinci0','davinci1','davinci_manager','devmm_svm','hisi_hdc'):
        cmd+=['--device','/dev/'+device]
    for mount in ('/usr/local/Ascend/driver/lib64','/etc/ascend_install.info',
                  '/usr/local/Ascend/driver/version.info','/usr/local/dcmi'):
        cmd+=['-v',f'{mount}:{mount}:ro']
    shell='''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /work/source/csrc
bash build.sh --pkg --ops=chunk_gated_delta_rule_compute_wy --soc=ascend310p --vendor_name=hyperqwen_wy -j8
python3 /project/build_pr17640_wy_extension.py --source-root /work/source --build /work/wy-extension
'''
    cmd += [image,'--noprofile','--norc','-c',shell]
    (root/'build-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    report('building')
    with (root/'build.log').open('x') as log:
        r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f'build exited {r.returncode}; inspect build.log')
    report('built',packages=[str(x.relative_to(root)) for x in (source/'csrc').rglob('*.run') if x.is_file()],
           libraries=[str(x.relative_to(root)) for x in (root/'wy-extension').glob('*.so')])
except Exception as e:
    report('failed',error=str(e))
    raise
