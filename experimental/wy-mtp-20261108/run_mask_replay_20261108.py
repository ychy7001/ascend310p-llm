#!/usr/bin/env python3
"""a2 single-chip mask graph replay audit with independent cleanup."""
import sys,pathlib,subprocess,fcntl,time,json
r=pathlib.Path('/home/ychy/ascend-lab-runtime/hyperqwen-20261103')
sys.path.insert(0,str(r))
import run_pr17640_pdf_ab_20261107 as pdf
out=r/'mtp-graph-fix-20261108/mask-replay-v2'
out.mkdir(exist_ok=False)
name='w8-splitfuse-mask-replay-20261108-v2'
begin=time.monotonic()
with open('/tmp/a310-npu.lock','a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=pdf.snapshot()
    assert before['rootless']['stdout']==before['rootful']['stdout']=='' and 'No running processes' in before['npu']['stdout']
    cmd=['podman','run','--name',name,'--pull=never','--group-add','keep-groups','--network','none','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0','-e','PYTHONPATH=/experiment','-e','OMP_NUM_THREADS=1']
    for dev in ('davinci0','davinci_manager','devmm_svm','hisi_hdc'):
        cmd+=['--device','/dev/'+dev]
    for path in ('/usr/local/Ascend/driver/lib64','/usr/local/Ascend/driver/version.info','/usr/local/dcmi','/etc/ascend_install.info'):
        cmd+=['-v',path+':'+path+':ro']
    cmd+=['-v',str(r/'mtp-graph-fix-20261108/splitfuse-fix')+':/experiment:ro','--entrypoint','bash','localhost/qwen36-pr17640-wy-locked:20261108','-c','source /usr/local/Ascend/ascend-toolkit/set_env.sh; exec python3 /experiment/check_mask_replay.py']
    code=-1
    try:
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
        (out/'stdout.txt').write_text(result.stdout)
        (out/'stderr.txt').write_text(result.stderr)
        code=result.returncode
    finally:
        subprocess.run(['podman','rm','-f',name],capture_output=True)
        after=pdf.snapshot()
        clean=after['rootless']['stdout']==after['rootful']['stdout']=='' and 'No running processes' in after['npu']['stdout']
        (out/'window.json').write_text(json.dumps({'exit_code':code,'elapsed_s':time.monotonic()-begin,'cleanup_passed':clean,'before':before,'after':after,'command':cmd},indent=2))
    print('exit',code,'clean',clean)
    print(result.stdout[-2500:])
    print(result.stderr[-1500:])
    if code or not clean: sys.exit(1)
