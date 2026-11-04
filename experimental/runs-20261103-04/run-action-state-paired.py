#!/usr/bin/env python3
"""a2原生产栈与9.2.1配套H/O的同输入字段/长上下文回归；不跑性能。"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import urllib.request

IMAGE = 'd103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--attempt', default='v1')
a = p.parse_args()
if not re.fullmatch('[a-zA-Z0-9-]+', a.attempt): p.error('invalid attempt')
root = a.root.resolve(); out = root/('action-state-paired-'+a.attempt); out.mkdir(exist_ok=False)
shutil.copy2(__file__, out/'supervisor-used.py')
steps = []; paused = False; current_container = None; failure = None
base = 'http://127.0.0.1:18116'; model = 'Qwen3.6-35B-A3B-w8a8'


def state(status, **kw):
    tmp = out/'status.tmp'; tmp.write_text(json.dumps(dict(state=status, steps=steps, **kw), indent=2)+'\n'); tmp.replace(out/'status.json')


def http(path, payload=None, url=base):
    req = urllib.request.Request(url+path, data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900 if payload else 5) as r:
        return json.load(r) if payload is not None else r.status


def run(label, cmd, timeout=900):
    state('running', current=label, container=current_container)
    (out/(label+'-command.json')).write_text(json.dumps(cmd, indent=2)+'\n')
    begin = time.monotonic()
    with (out/(label+'.log')).open('x') as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    steps.append(dict(label=label, returncode=r.returncode, elapsed_s=time.monotonic()-begin))
    if r.returncode: raise RuntimeError(label+' execution failed; raw output retained')


def idle(label):
    for _ in range(90):
        text = subprocess.check_output(['npu-smi', 'info'], text=True)
        if 'No running processes found' in text and not re.search(r'\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|', text):
            (out/(label+'-idle.txt')).write_text(text); return
        time.sleep(2)
    raise RuntimeError('NPU devices did not become idle')


def health(container):
    for _ in range(240):
        try:
            if http('/health') == 200: return
        except Exception: pass
        status = subprocess.run(['podman','inspect','--format','{{.State.Status}}',container], capture_output=True, text=True)
        if status.stdout.strip() == 'exited': raise RuntimeError('Model exited during startup')
        time.sleep(5)
    raise TimeoutError('Model health startup budget')


def quality(label):
    run(label, ['python3',str(out/'tools/check_cache_request_reuse.py'),'--base-url',base,'--model',model,'--out',str(out/(label+'.json'))])
    result = json.loads((out/(label+'.json')).read_text())
    if result['strict_matches'] != 8 or not result['reuse']['identical_text'] or result['reuse']['canary_leaked']:
        raise RuntimeError(label+' short/ABA integrity gate failed')


def freeform(stage):
    sources = [json.loads(x) for x in (root/'production-w8a8/business-quality.jsonl').read_text().splitlines()]
    selected = [r for r in sources if r['kind']=='meeting' and r['id'] in (4,5,18)]
    assert len(selected)==3
    def one(source, c, repeat):
        payload = copy.deepcopy(source['request'])
        payload.update(seed=0, cache_salt=f'action-state-freeform-c{c}-r{repeat}-id{source["id"]}')
        row = dict(kind='freeform', id=source['id'], concurrency_limit=c, repeat=repeat, request=payload,
                   request_sha256=hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                   source_request_sha256=hashlib.sha256(json.dumps(source['request'],ensure_ascii=False,sort_keys=True).encode()).hexdigest(),
                   requires_manual_review=True, success=False)
        try:
            row['response'] = http('/v1/chat/completions',payload)
            row['success'] = row['response']['choices'][0]['finish_reason']=='stop'
        except Exception as error: row.update(error_type=type(error).__name__, error=str(error))
        return row
    with (out/(stage+'-freeform.jsonl')).open('x') as log:
        for c, repeat in ((1,0),(3,0),(3,1)):
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                for row in pool.map(lambda src: one(src,c,repeat),selected):
                    log.write(json.dumps(row,ensure_ascii=False)+'\n'); log.flush()
    # Completion/format alone is not a semantic gate; human review remains separate.


def stop(stage):
    global current_container
    if not current_container: return
    name = current_container
    subprocess.run(['podman','stop','--time','30',name],capture_output=True,timeout=45)
    with (out/(stage+'-server.log')).open('w') as log:
        subprocess.run(['podman','logs',name],stdout=log,stderr=subprocess.STDOUT,timeout=30)
    detail = json.loads(subprocess.check_output(['podman','inspect',name],text=True))
    (out/(stage+'-container-inspect.json')).write_text(json.dumps(detail,indent=2)+'\n')
    if detail[0]['State']['Status'] != 'exited': raise RuntimeError('Experiment container cleanup failed')
    current_container = None


def interrupted(signum, frame): raise RuntimeError(f'Interrupted: {signum}')
signal.signal(signal.SIGTERM, interrupted)
try:
    state('preparing')
    pair = root/'upstream-gdn-pair921-v1'
    assert json.loads((pair/'status.json').read_text())['state']=='passed_operator_validation'
    assert json.loads((root/'cann921-upgrade/a3b-quality-pair921-v2/adjudication.json').read_text())['status']=='not_accepted_business_semantics'
    assert http('/health',url='http://127.0.0.1:18081')==200
    tools = out/'tools'; tools.mkdir()
    for name in ('check_action_state_regression.py','check_cache_request_reuse.py','launch-production-replica.py',
                 'start-production-replica.sh','start-cann921-gdn-pair.sh'):
        shutil.copy2(root/name,tools/name)
    shutil.copy2(root/'action-state-fixtures-v1.json',out/'fixtures.json')
    freeze = {str(x.relative_to(out)):hashlib.sha256(x.read_bytes()).hexdigest()
              for x in [out/'fixtures.json',*tools.iterdir()]}
    freeze['paired_worker_sha256'] = hashlib.sha256((pair/'tools/cann921_gdn_pair_worker.py').read_bytes()).hexdigest()
    (out/'tool-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    paused = True
    subprocess.run(['sudo','-n','systemctl','stop','llm-backend-primary.service'],check=True)
    for stage in ('A-production','B-pair921'):
        idle(stage)
        directory=out/stage; directory.mkdir()
        manifest=json.loads((root/'production-manifest.json').read_text())
        extra=[]; script='start-production-replica.sh'
        if stage.startswith('B'):
            manifest['image']=IMAGE; manifest['experiment_kind']='runtime_gdn_pair_correctness'
            manifest['experiment_scope']='Whole official9.2.1 runtime + PR17640 paired H/O vs original production; original Torch WY; semantic diagnostics, not single-component/performance A/B'
            manifest['env'].pop('ASCEND_CUSTOM_PATH',None)
            script='start-cann921-gdn-pair.sh'
            extra=['--compatibility-image-id',IMAGE,'--cann921-gdn-pair-dir',str(pair)]
        (directory/'candidate-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        current_container='action-state-'+stage.lower()+'-'+str(time.time_ns())
        try:
            run(stage+'-launch',['python3',str(tools/'launch-production-replica.py'),'--manifest',str(directory/'candidate-manifest.json'),
                '--start-script',str(tools/script),'--model-dir',str(root/'production-w8a8/model'),'--out',str(directory),
                '--container',current_container,'--port','18116',*extra],120)
            state('waiting_health',current=stage,container=current_container);health(current_container)
            quality(stage+'-quality-before')
            run(stage+'-fields',['python3',str(tools/'check_action_state_regression.py'),'--base-url',base,
                '--fixtures',str(out/'fixtures.json'),'--out',str(out/(stage+'-fields.jsonl'))],1800)
            state('running',current=stage+'-freeform',container=current_container);freeform(stage)
            quality(stage+'-quality-after')
        finally: stop(stage)
    state('paired_diagnostics_complete_requires_adjudication',scope='Fields/long context/freeform only; no GSM/performance/production acceptance')
except Exception as error:
    failure=str(error);state('failed',error=failure)
    raise
finally:
    try:
        if current_container: stop('emergency')
    finally:
        if paused:
            with (out/'restore-a2-service.log').open('x') as log:
                subprocess.run(['sudo','-n','systemctl','reset-failed','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
                r=subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:
                state('failed',error='Original service restore failed',primary_error=failure)
