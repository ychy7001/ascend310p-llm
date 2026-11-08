#!/usr/bin/env python3
"""Frozen W4 image, existing WY paths, MTP0/2; a2 diagnostic only."""
import fcntl
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import run_wy_mtp_matrix_20261108 as common

ROOT=common.ROOT
IMAGE='registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:8e17ec9ae646b1e418cf119bbb347238c4505b8b0037782a1345914824ec8d35'
ASSETS=ROOT/'wy-mtp-20261108'
OUT=ROOT/'w4-wy-mtp-matrix-20261108-v1'
common.pdf.MODEL='qwen36-w4a8-c8'

def launch(mode,mtp):
    directory=OUT/f'w4-{mode}-mtp{mtp}'
    directory.mkdir()
    (directory/'cache').mkdir()
    env={'MODEL_DIR':'/model','PORT':'18126','HOST':'127.0.0.1','SERVED_MODEL_NAME':common.pdf.MODEL,'QWEN36_PROFILE':'long-prefill','QWEN36_C8_ENABLED':'0','NUM_SPECULATIVE_TOKENS':str(mtp or 1),'WY_MATRIX_MTP':str(mtp),'WY_MATRIX_MODE':mode,'MAX_MODEL_LEN':'114688','MAX_BATCHED_TOKENS':'2048','MEMORY_UTILIZATION':'0.95','MAX_NUM_SEQS':'1','ENABLE_PREFIX_CACHING':'0','ASYNC_SCHEDULING':'1','LANGUAGE_MODEL_ONLY':'0','QWEN36_SYNC_METADATA':'0','OMP_NUM_THREADS':'1','ASCEND_RT_VISIBLE_DEVICES':'0','ASCEND_VISIBLE_DEVICES':'0'}
    name=f'w4-wy-{mode}-mtp{mtp}-{time.time_ns()}'
    cmd=['podman','run','-d','--pull=never','--name',name,'--network','host','--group-add','keep-groups','--ipc','shareable','--shm-size','32g']
    for key,value in env.items(): cmd+=['--env',key+'='+value]
    for device in ('davinci0','davinci_manager','devmm_svm','hisi_hdc'): cmd+=['--device','/dev/'+device]
    mounts=[(Path('/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012/model'),'/model','ro'),(directory,'/experiment','rw'),(directory/'cache','/root/.cache','rw'),(ASSETS/'w4-adapter-matrix.py','/c8adapter/pr17640_gdn_adapter.py','ro'),(ASSETS/'w4-start-matrix.sh','/opt/qwen36/start.sh','ro')]
    for path in ('/usr/local/bin/npu-smi','/etc/ascend_install.info','/usr/local/Ascend/driver/version.info','/usr/local/Ascend/driver/lib64','/usr/local/dcmi'): mounts.append((path,path,'ro'))
    for source,target,access in mounts: cmd+=['-v',f'{source}:{target}:{access}']
    cmd+=['--entrypoint','/bin/bash',IMAGE,'/opt/qwen36/start.sh']
    common.write(directory/'launch-command.json',cmd)
    common.write(directory/'env.json',env)
    result=common.run(cmd,capture_output=True,text=True)
    (directory/'container-id.txt').write_text(result.stdout)
    return name,directory

def main():
    OUT.mkdir(exist_ok=False)
    begin=time.monotonic()
    states=[]
    with open('/tmp/a310-npu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=common.pdf.snapshot()
        common.write(OUT/'window-start.json',before)
        assert before['rootless']['stdout']==before['rootful']['stdout']=='', 'active containers'
        assert before['primary']['stdout']==before['image']['stdout']=='inactive', 'active service'
        assert 'No running processes' in before['npu']['stdout'], 'NPU busy'
        common.write(OUT/'freeze.json',{'image':IMAGE,'image_inspect':json.loads(subprocess.check_output(['podman','image','inspect',IMAGE],text=True)),'sources':{str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'run_wy_mtp_matrix_20261108.py',*ASSETS.glob('*')] if p.is_file()},'scope':'Existing W4 independent-namespace GDN. Adapter selects existing Torch/native WY functions; kernels and model weights unchanged. MTP0 omits speculative-config; other profile settings retained. Not the TP2 binding.'})
        error=None
        try:
            for index,(mode,mtp) in enumerate((('native',2),('torch',2),('native',0),('torch',0))):
                if index: time.sleep(60)
                common.write(OUT/'status.json',{'state':'launching','mode':mode,'mtp':mtp,'unix_ns':time.time_ns()})
                name,directory=launch(mode,mtp)
                try:
                    common.pdf.wait_health(name)
                    common.bench(directory,mtp,OUT)
                finally:
                    state=common.stop(name,directory)
                    states.append({'mode':mode,'mtp':mtp,'state':state})
                    common.write(OUT/'container-states.json',states)
        except BaseException:
            error=traceback.format_exc()
            (OUT/'error.txt').write_text(error)
        after=common.pdf.snapshot()
        clean=after['rootless']['stdout']==after['rootful']['stdout']=='' and 'No running processes' in after['npu']['stdout']
        common.write(OUT/'window-closed.json',{'elapsed_s':time.monotonic()-begin,'cleanup_passed':clean,'snapshot':after,'error':error,'states':states})
        common.write(OUT/'status.json',{'state':'failed' if error or not clean else 'complete','error':error,'cleanup_passed':clean,'elapsed_s':time.monotonic()-begin})
        if error or not clean: raise RuntimeError(error or 'cleanup failed')

if __name__=='__main__': main()
