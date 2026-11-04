#!/usr/bin/env python3
"""成对H/O算子门禁通过后，生产栈35B TP2模型质量筛选；不自动验收。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--pair-dir',type=Path,required=True)
p.add_argument('--port',type=int,default=18112)
a=p.parse_args()
root=a.root.resolve();pair=a.pair_dir.resolve()
out=root/'production-gdn-pair-quality'
out.mkdir(exist_ok=False)
shutil.copy2(__file__,out/'supervisor-used.py')
steps=[];paused=False;launched=False;finished=False;primary_error=None
container='qwen36-gdn-pair-quality-'+str(time.time_ns())
base=f'http://127.0.0.1:{a.port}'
model='Qwen3.6-35B-A3B-w8a8'
common=['--base-url',base,'--model',model]
fixtures=Path('/path/to/lab-runtime/a1-310p-tune-20261101')

def state(value,**extra):
    tmp=out/'status.tmp';tmp.write_text(json.dumps(dict(state=value,steps=steps,**extra),indent=2)+'\n');tmp.replace(out/'status.json')

def run(label,cmd,timeout=1800):
    state('running',current=label)
    (out/(label+'-command.json')).write_text(json.dumps(cmd,indent=2)+'\n')
    begin=time.monotonic()
    with (out/(label+'.log')).open('x') as log:
        r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    steps.append(dict(label=label,returncode=r.returncode,elapsed_s=time.monotonic()-begin))
    if r.returncode:raise RuntimeError(label+' failed; inspect raw outputs')

def quality(label):
    run(label,['python3',str(root/'check_cache_request_reuse.py'),*common,'--out',str(out/(label+'.json'))],600)
    j=json.loads((out/(label+'.json')).read_text())
    if j['strict_matches']!=8 or not j['reuse']['identical_text'] or j['reuse']['canary_leaked']:
        raise RuntimeError(label+' strict short/ABA gate failed')

def terminate(signum,frame):
    raise RuntimeError('Supervisor interrupted: '+str(signum))
signal.signal(signal.SIGTERM,terminate)
try:
    status=json.loads((pair/'status.json').read_text())
    if status['state']!='passed_operator_validation' or not (pair/'restore-a2-service.log').exists():
        raise RuntimeError('Pair operator gate or previous cleanup incomplete')
    manifest=json.loads((root/'production-manifest.json').read_text())
    manifest['env']['EXPERIMENT_NATIVE_WY']='1'
    manifest['env']['PYTHONFAULTHANDLER']='1'
    manifest['experiment_scope']='Full upstream paired FwdH/FwdO + native WY quality screen on production image; not WY-only A/B'
    (out/'candidate-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'candidate-definition.json').write_text(json.dumps(dict(
        source_commit='18163f6b1d208ab3b4fb06e9752225611c37f40e',author='GruntovDima; existing PR17640 reuse',
        changed_components=['paired upstream FwdH/FwdO including NZ binding','native WY'],
        unchanged=['production image','W8A8 TP2','Graph[1,2]','MTP0','maxlen262144','batch4096','seqs4','FP16 SSM','prefix on'],
        stage='quality only; strict short, facts, original business shape, GSM64; semantic review required',
        pair_gate=status,manifest_sha256=hashlib.sha256((root/'production-manifest.json').read_bytes()).hexdigest()),indent=2)+'\n')
    paused=True
    subprocess.run(['sudo','-n','systemctl','stop','llm-backend-primary.service'],check=True)
    idle=0
    for _ in range(90):
        info=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
        idle=idle+1 if 'No running processes found' in info.stdout else 0
        if idle>=2:break
        time.sleep(2)
    else:raise RuntimeError('NPU not idle')
    launched=True
    run('launch',['python3',str(root/'launch-production-replica.py'),'--manifest',str(out/'candidate-manifest.json'),
        '--start-script',str(root/'start-gdn-pair.sh'),'--model-dir',str(root/'production-w8a8/model'),
        '--out',str(out),'--container',container,'--port',str(a.port),
        '--native-wy-dir',str(root/'native-gdn'),'--gdn-pair-dir',str(pair)],120)
    state('waiting_health',container=container)
    for _ in range(240):
        try:
            with urllib.request.urlopen(base+'/health',timeout=3) as r:
                if r.status==200:break
        except Exception:pass
        inspect=subprocess.run(['podman','inspect','--format','{{.State.Status}}',container],capture_output=True,text=True)
        if inspect.stdout.strip()=='exited':raise RuntimeError('Candidate exited during startup')
        time.sleep(5)
    else:raise TimeoutError('Health not ready in startup budget')
    quality('quality-before')
    run('business-facts',['python3',str(root/'check_business_facts.py'),*common,'--cases','8','--concurrencies','1','3','--out',str(out/'business-facts.jsonl')],900)
    run('business-quality',['python3',str(root/'check_business_quality.py'),'--url',base,'--model',model,
        '--fixtures-dir',str(fixtures),'--out',str(out/'business-quality.jsonl')],1200)
    # All original business responses are retained for owner/deadline semantic review.
    # Automatic completion/schema checks are never called a semantic pass.
    run('gsm64',['python3',str(root/'eval_gsm8k_http.py'),*common,
        '--dataset','/path/to/lab-runtime/qwen36-public-release-20261012/gsm8k-test.jsonl',
        '--limit','64','--seed','310','--max-tokens','512','--label','production-gdn-pair-quality',
        '--out',str(out/'gsm64.jsonl')],2400)
    quality('quality-after')
    finished=True
    state('quality_screen_complete_requires_semantic_review',scope='No performance or deployment acceptance; inspect owner/deadline mappings and paired GSM outcomes')
except Exception as error:
    primary_error=str(error);state('failed',error=primary_error);raise
finally:
    if launched:
        subprocess.run(['podman','stop','--time','30',container],capture_output=True,timeout=45)
        with (out/'server.log').open('w') as log:subprocess.run(['podman','logs',container],stdout=log,stderr=subprocess.STDOUT)
        with (out/'container-inspect.json').open('w') as log:subprocess.run(['podman','inspect',container],stdout=log,stderr=subprocess.STDOUT)
    if paused:
        with (out/'restore-a2-service.log').open('x') as log:
            subprocess.run(['sudo','-n','systemctl','reset-failed','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
            restored=subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
        if restored.returncode:state('failed',error='Temporary service restore failed',primary_error=primary_error)
