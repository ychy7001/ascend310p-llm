import sys, pathlib, json, subprocess, fcntl, time, hashlib
ROOT=pathlib.Path('/home/ychy/ascend-lab-runtime/hyperqwen-20261103');sys.path.insert(0,str(ROOT))
import run_w8_mask_fix_20261108 as common
out=ROOT/'graph-localize/pr18730-original-repro';out.mkdir()
start=time.monotonic();name='pr18730-original-repro'
with open('/tmp/a310-npu.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 before=common.pdf.snapshot();common.write(out/'window-start.json',before)
 assert before['rootless']['stdout']==before['rootful']['stdout']=='' and 'No running processes' in before['npu']['stdout']
 cmd=['podman','run','--name',name,'--pull=never','--network=none','--group-add','keep-groups','--shm-size','2g','--entrypoint','bash','-e','TORCH_DEVICE_BACKEND_AUTOLOAD=0']
 for d in ['davinci0','davinci_manager','devmm_svm','hisi_hdc']:cmd+=['--device','/dev/'+d]
 for p in ['/usr/local/Ascend/driver/lib64','/usr/local/Ascend/driver/version.info','/usr/local/dcmi','/etc/ascend_install.info']:cmd+=['-v',p+':'+p+':ro']
 work=ROOT/'graph-localize/pr18730-audit';cmd+=['-v',str(work)+':/audit:ro','localhost/qwen36-w8-wy-mtp-production:20261108','-c','source /usr/local/Ascend/ascend-toolkit/set_env.sh; export ASCEND_CUSTOM_OPP_PATH=/pr17640-opp/vendors/custom_transformer; python3 /audit/repro_gdn_17856.py']
 common.write(out/'command.json',cmd)
 code=None;error=None
 try:
  with (out/'repro.log').open('w') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=180).returncode
 except Exception as e:error=str(e)
 finally:
  subprocess.run(['podman','rm','-f',name],capture_output=True,timeout=45)
  after=common.pdf.snapshot();clean=after['rootless']['stdout']==after['rootful']['stdout']=='' and 'No running processes' in after['npu']['stdout']
  common.write(out/'status.json',{'exit_code':code,'error':error,'elapsed_s':time.monotonic()-start,'cleanup_passed':clean,'script_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [work/'repro_gdn_17856.py',work/'check_gdn_state_table_310.py']},'snapshot':after})
 print((out/'status.json').read_text());print((out/'repro.log').read_text()[-5000:])
