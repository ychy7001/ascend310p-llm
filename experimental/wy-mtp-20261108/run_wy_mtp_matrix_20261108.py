#!/usr/bin/env python3
"""a2 only: frozen-image WY/MTP integration and cold-prefill experiment."""
import argparse
import copy
import fcntl
import hashlib
import json
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path('/home/ychy/ascend-lab-runtime/hyperqwen-20261103')
BASE = 'http://127.0.0.1:18126'
IMAGE = 'localhost/qwen36-pr17640-wy-locked:20261108'
sys.path.insert(0, str(ROOT))
import run_pr17640_pdf_ab_20261107 as pdf

def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def run(cmd, **kwargs):
    return subprocess.run(cmd, check=True, timeout=kwargs.pop('timeout', 60), **kwargs)

def post(endpoint, payload, timeout=300):
    req = urllib.request.Request(BASE+endpoint, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)

def launch(out, mtp, eager=False):
    directory = out / f'w8-wy-mtp{mtp}'
    directory.mkdir()
    (directory/'cache').mkdir()
    manifest = copy.deepcopy(json.loads((ROOT/'production-manifest.json').read_text()))
    manifest['image'] = IMAGE
    command = manifest['command']
    command[command.index('--host')+1] = '127.0.0.1'
    command[command.index('--port')+1] = '18126'
    if eager:
        i=command.index('--compilation-config')
        del command[i:i+2]
        command += ['--enforce-eager']
    if mtp:
        command += ['--speculative-config', json.dumps({'method':'mtp','num_speculative_tokens':mtp})]
    manifest['env'].update({'TORCH_DEVICE_BACKEND_AUTOLOAD':'0','QWEN36_C8_ENABLED':'0','ASCEND_CUSTOM_OPP_PATH':'/pr17640-opp/vendors/custom_transformer'})
    write(directory/'manifest.json',manifest)
    mounts = [
        (ROOT/'production-w8a8/model','/models/Qwen3.6-35B-A3B-w8a8','ro'),
        (directory,'/experiment','rw'), (directory/'cache','/root/.cache','rw'),
        (ROOT/'wy-boundary-20261106/tools/start-production-replica.sh','/workspace/start_vllm_ascend.sh','ro'),
        (directory/'manifest.json','/workspace/backend-primary.manifest.json','ro')]
    for path in ('/usr/local/bin/npu-smi','/etc/ascend_install.info','/usr/local/Ascend/driver/version.info','/usr/local/Ascend/driver/lib64','/usr/local/dcmi'):
        mounts.append((path,path,'ro'))
    name = f'wy-mtp-{mtp}-{time.time_ns()}'
    cmd = ['podman','run','-d','--pull=never','--name',name,'--network','host','--group-add','keep-groups','--ipc','shareable','--shm-size','32g']
    for device in ('davinci0','davinci1','davinci_manager','devmm_svm','hisi_hdc'):
        cmd += ['--device','/dev/'+device]
    for source,target,mode in mounts:
        cmd += ['-v',f'{source}:{target}:{mode}']
    cmd += ['--entrypoint','/bin/bash',IMAGE,'/workspace/start_vllm_ascend.sh']
    write(directory/'launch-command.json',cmd)
    result = run(cmd,capture_output=True,text=True)
    (directory/'container-id.txt').write_text(result.stdout)
    return name,directory

def stop(name,directory):
    run(['podman','stop','--time','30',name],capture_output=True,timeout=60)
    with (directory/'server.log').open('w') as log:
        run(['podman','logs',name],stdout=log,stderr=subprocess.STDOUT)
    detail=json.loads(subprocess.check_output(['podman','inspect',name],text=True))
    write(directory/'container-inspect.json',detail)
    run(['podman','rm',name],capture_output=True)
    return detail[0]['State']

def bench(directory,mtp,out,smoke_only=False):
    smoke=[]
    for index,prompt in enumerate(('Reply with OK.', 'Compute 17 + 25. Reply with the number only.', 'Return JSON with exactly these fields: invoice_total as 57.80, date as 2026-11-08.')):
        before=pdf.metrics()
        start=time.monotonic()
        result=post('/v1/chat/completions',{'model':pdf.MODEL,'messages':[{'role':'user','content':prompt}], 'temperature':0,'seed':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False},'cache_salt':f'{out.name}-mtp{mtp}-smoke{index}'})
        after=pdf.metrics()
        smoke.append({'prompt':prompt,'elapsed_s':time.monotonic()-start,'response':result,'metrics_delta':{k:after[k]-before[k] for k in before}})
        write(directory/'smoke.json',smoke)
        assert result.get('choices') and result.get('usage'), 'invalid smoke response'
    if smoke_only: return
    for length in (4096,8192,16384,32768):
        write(out/'status.json',{'state':'prefill','mtp':mtp,'length':length,'unix_ns':time.time_ns()})
        cmd=['python3',str(ROOT/'wy-boundary-20261106/tools/bench_server_phases.py'),'--base-url',BASE,'--model',pdf.MODEL,'--prompt-tokens',str(length),'--output-tokens','1','--repeats','3','--cache-salt-prefix',f'{out.name}-mtp{mtp}-{length}','--out',str(directory/f'prefill-{length}.jsonl')]
        run(cmd,timeout=900)
    write(out/'status.json',{'state':'decode','mtp':mtp,'unix_ns':time.time_ns()})
    cmd=['python3',str(ROOT/'wy-boundary-20261106/tools/bench_server_phases.py'),'--base-url',BASE,'--model',pdf.MODEL,'--prompt-tokens','4096','--output-tokens','128','--repeats','3','--cache-salt-prefix',f'{out.name}-mtp{mtp}-decode','--out',str(directory/'decode-4096-128.jsonl')]
    if mtp: cmd += ['--spec-metrics']
    run(cmd,timeout=1200)
    inputs=ROOT/'a310-ocr-pdf-smoke-20261107'
    manifest=json.loads((inputs/'manifest.json').read_text())
    for case in manifest['cases']:
        if case['id'] not in ('pdf-01','pdf-02','pdf-03','pdf-04'): continue
        for repeat in range(2):
            write(out/'status.json',{'state':'pdf','mtp':mtp,'case':case['id'],'repeat':repeat,'unix_ns':time.time_ns()})
            row=pdf.request(case,inputs,manifest['prompt'],f'mtp{mtp}',repeat)
            row['text_sha256']=hashlib.sha256(row['text'].encode()).hexdigest()
            write(directory/f'{case["id"]}-{repeat}.json',row)
            assert row['valid'], f'invalid PDF {case["id"]}'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--arms',default='1,0,2')
    parser.add_argument('--eager',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(exist_ok=False)
    begin=time.monotonic()
    states=[]
    with open('/tmp/a310-npu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        snapshot=pdf.snapshot()
        write(args.out/'window-start.json',snapshot)
        assert snapshot['rootless']['stdout']==snapshot['rootful']['stdout']=='', 'device host has active containers'
        assert snapshot['primary']['stdout']==snapshot['image']['stdout']=='inactive', 'device service active'
        assert 'No running processes' in snapshot['npu']['stdout'], 'NPU busy'
        write(args.out/'freeze.json',{'image':IMAGE,'image_inspect':json.loads(subprocess.check_output(['podman','image','inspect',IMAGE],text=True)),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'arms':args.arms,'eager':args.eager,'smoke_only':args.smoke_only,'scope':'WY/MTP integration diagnostic; no production adoption; no frozen kernel/protocol/worker changes'})
        error=None
        try:
            for index,mtp in enumerate(map(int,args.arms.split(','))):
                if index: time.sleep(60)
                write(args.out/'status.json',{'state':'launching','mtp':mtp,'unix_ns':time.time_ns()})
                name,directory=launch(args.out,mtp,args.eager)
                try:
                    pdf.wait_health(name)
                    write(args.out/'status.json',{'state':'smoke','mtp':mtp,'unix_ns':time.time_ns()})
                    bench(directory,mtp,args.out,args.smoke_only)
                finally:
                    state=stop(name,directory)
                    states.append({'mtp':mtp,'container_state':state})
                    write(args.out/'container-states.json',states)
        except BaseException:
            error=traceback.format_exc()
            (args.out/'error.txt').write_text(error)
        after=pdf.snapshot()
        clean=after['rootless']['stdout']==after['rootful']['stdout']=='' and 'No running processes' in after['npu']['stdout']
        write(args.out/'window-closed.json',{'elapsed_s':time.monotonic()-begin,'cleanup_passed':clean,'snapshot':after,'error':error,'states':states})
        write(args.out/'status.json',{'state':'failed' if error or not clean else 'complete','error':error,'cleanup_passed':clean,'elapsed_s':time.monotonic()-begin})
        if error or not clean: raise RuntimeError(error or 'cleanup failed')

if __name__=='__main__': main()
