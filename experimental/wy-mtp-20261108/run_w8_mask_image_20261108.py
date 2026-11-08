#!/usr/bin/env python3
"""a2: validate the packaged default entrypoint after the mask diagnostic passes."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path('/home/ychy/ascend-lab-runtime/hyperqwen-20261103')
sys.path.insert(0,str(ROOT))
import run_w8_mask_fix_20261108 as common
pdf=common.pdf
previous=ROOT/'w8-mask-fix-20261108-v1'
for _ in range(180):
    status=json.loads((previous/'status.json').read_text())
    if status['state']=='failed': raise RuntimeError('previous diagnostic failed; image validation canceled')
    if status['state']=='complete': break
    time.sleep(10)
else: raise TimeoutError('previous diagnostic did not complete within 30 minutes')
time.sleep(max(0,60-(time.time()-(previous/'window-closed.json').stat().st_mtime)))
out=ROOT/'w8-mask-image-20261108-v1'
out.mkdir(exist_ok=False)
(out/'cache').mkdir()
name='wy-mtp-packaged-'+str(time.time_ns())
image='localhost/qwen36-w8-wy-mtp-maskfix:20261108'
begin=time.monotonic()
with open('/tmp/a310-npu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=pdf.snapshot()
    assert before['rootless']['stdout']==before['rootful']['stdout']=='' and 'No running processes' in before['npu']['stdout']
    common.write(out/'window-start.json',before)
    inspect=json.loads(subprocess.check_output(['podman','image','inspect',image],text=True))[0]
    assert inspect['Config']['Entrypoint']==['bash','/opt/qwen36/start-wy-mtp.sh']
    common.write(out/'image.json',inspect)
    cmd=['podman','run','-d','--pull=never','--name',name,'--network','host','--group-add','keep-groups','--ipc','shareable','--shm-size','32g','-e','HOST=127.0.0.1','-e','PORT=18126']
    for device in ('davinci0','davinci1','davinci_manager','devmm_svm','hisi_hdc'):
        cmd+=['--device','/dev/'+device]
    for path in ('/usr/local/bin/npu-smi','/etc/ascend_install.info','/usr/local/Ascend/driver/version.info','/usr/local/Ascend/driver/lib64','/usr/local/dcmi'):
        cmd+=['-v',path+':'+path+':ro']
    cmd+=['-v',str(ROOT/'production-w8a8/model')+':/model:ro','-v',str(out/'cache')+':/root/.cache:rw',image]
    common.write(out/'launch-command.json',cmd)
    error=None
    state=None
    launched=False
    try:
        common.run(cmd,capture_output=True,text=True)
        launched=True
        common.write(out/'status.json',{'state':'launching'})
        common.wait_health(name)
        pdf.MODEL='qwen36-w8a8-wy-mtp'
        common.bench(out,2,out,smoke_only=True)
        expected={'numbers':list(range(1,21)),'status':'ok'}
        prompt='Return exactly this JSON object, no other text: '+json.dumps(expected)
        response=common.post('/v1/chat/completions',{'model':pdf.MODEL,'messages':[{'role':'user','content':prompt}],'temperature':0,'seed':0,'max_tokens':192,'chat_template_kwargs':{'enable_thinking':False}})
        common.write(out/'multi-step-json.json',response)
        assert json.loads(response['choices'][0]['message']['content'])==expected
        manifest=json.loads((ROOT/'a310-ocr-pdf-smoke-20261107/manifest.json').read_text())
        case=next(case for case in manifest['cases'] if case['id']=='pdf-02')
        row=pdf.request(case,ROOT/'a310-ocr-pdf-smoke-20261107',manifest['prompt'],'packaged-mtp2',0)
        common.write(out/'pdf-02.json',row)
        assert row['valid']
    except BaseException:
        error=traceback.format_exc()
    finally:
        if launched: state=common.stop(name,out)
        after=pdf.snapshot()
        clean=after['rootless']['stdout']==after['rootful']['stdout']=='' and 'No running processes' in after['npu']['stdout']
        common.write(out/'window-closed.json',{'elapsed_s':time.monotonic()-begin,'cleanup_passed':clean,'error':error,'container_state':state,'snapshot':after})
        common.write(out/'status.json',{'state':'failed' if error or not clean else 'complete','elapsed_s':time.monotonic()-begin,'cleanup_passed':clean,'error':error})
    if error or not clean: raise RuntimeError(error or 'cleanup failed')
