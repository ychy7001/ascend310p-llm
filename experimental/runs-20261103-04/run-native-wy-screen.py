#!/usr/bin/env python3
"""WY 数值门禁通过后运行 W8 TP2 单变量筛选，收益足够则执行业务/GSM。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import urllib.request

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--native-dir',type=Path,required=True)
p.add_argument('--port',type=int,default=18108)
p.add_argument('--restore-a2-service',action='store_true',help='restore the user-authorized temporary a2 unit after experiment')
a=p.parse_args()
root=a.root.resolve()
native=a.native_dir.resolve()
output=root/'production-native-wy'
output.mkdir(exist_ok=False)
baseline=root/'production-w8a8'
container='qwen36-production-native-wy-20261103'
base=f'http://127.0.0.1:{a.port}'
model='Qwen3.6-35B-A3B-w8a8'
common=['--base-url',base,'--model',model]
steps=[]
launched=False
outcome=None
finished=False
fixtures=Path('/path/to/lab-runtime/a1-310p-tune-20261101')
def state(value,**extra):
    tmp=output/'status.tmp'
    tmp.write_text(json.dumps(dict(state=value,steps=steps,**extra),indent=2)+'\n')
    tmp.replace(output/'status.json')
def run(label,cmd):
    state('running',current=label)
    start=time.monotonic()
    with (output/(label+'.log')).open('x') as log:
        r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    steps.append(dict(label=label,returncode=r.returncode,elapsed_s=time.monotonic()-start))
    if r.returncode:raise RuntimeError(f'{label} exited {r.returncode}')
def quality(label):
    run(label,['python3',str(root/'check_cache_request_reuse.py'),*common,'--out',str(output/(label+'.json'))])
    j=json.loads((output/(label+'.json')).read_text())
    if j['strict_matches']!=8 or not j['reuse']['identical_text'] or j['reuse']['canary_leaked']:
        raise RuntimeError(label+' short quality gate failed')
def records(path,kind):
    return [r for r in map(json.loads,path.read_text().splitlines()) if r['kind']==kind]
try:
    state('waiting_numerical_gate')
    for _ in range(840):
        f=native/'validation-status.json'
        if f.exists():
            j=json.loads(f.read_text())
            if j['stage']=='failed':raise RuntimeError('Native operator validation failed; model not launched')
            if j['stage']=='passed':break
        time.sleep(5)
    else:raise TimeoutError('numerical gate')
    # Verify exclusive test devices after validator has exited.
    state('waiting_devices_idle')
    idle=0
    for _ in range(60):
        smi=subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
        idle=idle+1 if 'No running processes found' in smi.stdout else 0
        if idle>=2:break
        time.sleep(2)
    else:raise RuntimeError('NPU busy after validation; model not launched')
    for f in ('source-verification.json','package-isolation.json','numerical-gate.jsonl','validation-status.json'):
        shutil.copy2(native/f,output/f)
    manifest=json.loads((root/'production-manifest.json').read_text())
    manifest['env']['PYTHONFAULTHANDLER']='1'
    manifest_path=output/'candidate-manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'candidate-definition.json').write_text(json.dumps(dict(
        primary_variable='replace torch WY helper with PR17640 ComputeWy only',
        source_commit='18163f6b1d208ab3b4fb06e9752225611c37f40e',
        upstream_author='GruntovDima; reuse existing PR17640, not original kernel authorship',
        production_manifest_sha256=hashlib.sha256((root/'production-manifest.json').read_bytes()).hexdigest(),
        unchanged_components=['production FwdH/FwdO','Graph[1,2]','MTP0','TP2 W8A8','full context','prefix cache on'],
        diagnostic_python_faulthandler=True,
        selection='median Prefill gain >=5% at both 4K and 32K; all C1 Decode ratios >=0.95; then business/GSM gates'),indent=2)+'\n')
    # Mark launch attempted so a partial launcher failure is also cleaned up.
    launched=True
    run('launch',['python3',str(root/'launch-production-replica.py'),
        '--manifest',str(manifest_path),'--start-script',str(root/'start-native-wy.sh'),
        '--model-dir',str(baseline/'model'),'--out',str(output),'--container',container,
        '--port',str(a.port),'--native-wy-dir',str(native),'--profiler'])
    state('waiting_health')
    for _ in range(240):
        try:
            with urllib.request.urlopen(base+'/health',timeout=3) as r:
                if r.status==200:break
        except Exception:pass
        j=subprocess.run(['podman','inspect','--format','{{.State.Status}}',container],capture_output=True,text=True)
        if j.stdout.strip()=='exited':raise RuntimeError('Candidate exited during startup')
        time.sleep(5)
    else:raise TimeoutError('candidate health')
    quality('quality-before')
    comparison={}
    for length in (1024,4096,16384,32768):
        label=f'c1-p{length}-o256'
        run(label,['python3',str(root/'bench_server_phases.py'),*common,
            '--prompt-tokens',str(length),'--output-tokens','256','--repeats','3','--seed','0',
            '--cache-salt-prefix',f'native-wy-{length}',
            '--out',str(output/(label+'.jsonl'))])
        old=records(baseline/(label+'.jsonl'),'request')
        new=records(output/(label+'.jsonl'),'request')
        if len(old)!=3 or len(new)!=3 or not all(r['valid'] for r in old+new):
            raise RuntimeError('Incomplete or invalid phase benchmark')
        if [r['prompt_ids_sha256'] for r in old]!=[r['prompt_ids_sha256'] for r in new]:
            raise RuntimeError('Candidate prompt tokens differ from frozen baseline')
        comparison[str(length)]={}
        for key in ('prefill_tok_s','decode_tok_s'):
            before=statistics.median(r[key] for r in old)
            after=statistics.median(r[key] for r in new)
            comparison[str(length)][key]=dict(baseline=before,candidate=after,ratio=after/before)
    (output/'phase-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    run('business-facts',['python3',str(root/'check_business_facts.py'),*common,
        '--cases','8','--concurrencies','1','3','--out',str(output/'business-facts.jsonl')])
    quality('quality-after-phases')
    selected=(all(comparison[str(n)]['prefill_tok_s']['ratio']>=1.05 for n in (4096,32768)) and
              all(comparison[str(n)]['decode_tok_s']['ratio']>=.95 for n in (1024,4096,16384,32768)))
    if not selected:
        outcome=dict(decision='not_selected',reason='Prefill gain insufficient or C1 Decode regressed',comparison=comparison)
    else:
        for c in (1,3,4):
            label=f'c{c}-p1024-o256-stream'
            run(label,['python3',str(root/'bench_http_stream.py'),*common,'--prompt-tokens','1024',
                '--output-tokens','256','--repeats','3','--seed','0','--concurrency',str(c),
                '--out',str(output/(label+'.jsonl'))])
        run('business-quality',['python3',str(root/'check_business_quality.py'),'--url',base,
            '--model',model,'--fixtures-dir',str(fixtures),'--out',str(output/'business-quality.jsonl')])
        shutil.copy2(baseline/'replay-prompts.json',output/'replay-prompts.json')
        run('business-replay-long',['python3',str(root/'replay_stage_workload.py'),
            '--workload',str(fixtures/'a1-310p-tune.workload.json'),
            '--recording','chinese_company_annual_meeting_planning','--urls',base,'--model',model,
            '--prompts',str(output/'replay-prompts.json'),'--out',str(output/'business-replay-long.json')])
        run('gsm64',['python3',str(root/'eval_gsm8k_http.py'),*common,
            '--dataset','/path/to/lab-runtime/qwen36-public-release-20261012/gsm8k-test.jsonl',
            '--limit','64','--seed','310','--max-tokens','512','--label','production-native-wy',
            '--out',str(output/'gsm64.jsonl')])
        old=json.loads((baseline/'gsm64.jsonl').read_text().splitlines()[-1])
        new=json.loads((output/'gsm64.jsonl').read_text().splitlines()[-1])
        acceptable=new['correct']>=old['correct'] and new['truncated']<=old['truncated']
        outcome=dict(decision='promising_requires_repeat_and_semantic_review' if acceptable else 'rejected_quality',
            comparison=comparison,baseline_gsm=old,candidate_gsm=new)
    quality('quality-after')
    state('finishing',**outcome)
    finished=True
except Exception as e:
    state('failed',error=str(e))
    raise
finally:
    if launched:
        subprocess.run(['podman','stop','--time','90',container],capture_output=True,text=True)
        with (output/'server.log').open('w') as f:
            subprocess.run(['podman','logs',container],stdout=f,stderr=subprocess.STDOUT)
        subprocess.run(['podman','inspect',container],stdout=(output/'container-inspect.json').open('w'))
        check=subprocess.run(['podman','inspect','--format','{{.State.Status}}',container],capture_output=True,text=True)
        if finished and check.stdout.strip()!='exited':
            finished=False
            state('failed',error='Candidate cleanup incomplete',outcome=outcome)
    if a.restore_a2_service:
        # This flag is only used after the explicit a2 temporary-service authorization.
        with (output/'restore-a2-service.log').open('x') as log:
            restored=subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],
                                    stdout=log,stderr=subprocess.STDOUT)
        if restored.returncode:
            finished=False
            state('failed',error='Could not restore original a2 temporary service',outcome=outcome)
    if finished:state('complete',**outcome)
