#!/usr/bin/env python3
"""One-shot local orchestration: wait for verified files, run gated service, smoke.
No fallback to eager dequant or partially downloaded checkpoints.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

work=Path('/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018')
model=Path('/home/ychy/models/Qwen3.8-Flash-Next-W4A16-G128-300i-d32978c')
(work/'serving-orchestration.pid').write_text(str(os.getpid()))

def status(phase, **extra):
    d={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'phase':phase,**extra}
    (work/'serving-status.json').write_text(json.dumps(d,indent=2))
    print(json.dumps(d),flush=True)

try:
    for _ in range(240):
        if (model/'download-verified.json').exists(): break
        # Detect a dead downloader and fail visibly instead of waiting forever.
        pid=int((model/'download.pid').read_text())
        os.kill(pid,0)
        status('waiting_for_verified_checkpoint',download_pid=pid)
        time.sleep(60)
    else: raise TimeoutError('Checkpoint download not complete after four hours')
    status('launching_gated_tp2_service')
    subprocess.run(['bash',str(work/'launch-text-tp2-on-a2.sh')],check=True)
    status('waiting_for_model_load')
    for _ in range(180):
        state=subprocess.check_output(['podman','inspect','--format','{{.State.Status}}','qwen38-tp2-text-4k'],text=True).strip()
        if state!='running': raise RuntimeError('Service container exited during loading')
        try:
            with urllib.request.urlopen('http://127.0.0.1:18038/health',timeout=5) as response:
                if response.status==200: break
        except OSError: pass
        time.sleep(20)
    else: raise TimeoutError('Service did not become healthy within one hour')
    status('running_real_checkpoint_generation_smoke')
    subprocess.run(['python3',str(work/'qwen38_serving_smoke.py'),'--output',str(work/'serving-smoke.json')],check=True)
    status('basic_generation_passed',port=18038)
except Exception as exc:
    status('failed',error=str(exc))
    raise
finally:
    if subprocess.run(['podman','container','exists','qwen38-tp2-text-4k']).returncode==0:
        with (work/'qwen38-serving.log').open('wb') as out:
            subprocess.run(['podman','logs','--timestamps','qwen38-tp2-text-4k'],stdout=out,stderr=subprocess.STDOUT)
    with (work/'npu-after-serving.txt').open('wb') as out:
        subprocess.run(['npu-smi','info'],stdout=out,stderr=subprocess.STDOUT)
