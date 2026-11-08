#!/usr/bin/env python3
"""Run one isolated a2 production-profile qualification window; archive failures."""
import argparse, concurrent.futures, fcntl, hashlib, json, os, subprocess, sys, time, traceback, urllib.error
from pathlib import Path
ROOT=Path('/home/ychy/ascend-lab-runtime/hyperqwen-20261103')
sys.path[:0]=[str(ROOT),str(ROOT/'wy-boundary-20261106/tools')]
import run_w8_mask_fix_20261108 as common
import bench_http_stream as http
import bench_server_phases as phases
pdf=common.pdf
IMAGE='localhost/qwen36-w8-wy-mtp-maskfix:20261108'
MODEL='qwen36-w8a8-wy-mtp'
BASE=common.BASE

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--batch',type=int,default=4096)
    p.add_argument('--mtp',type=int,default=2)
    p.add_argument('--mode',choices=['qualify','tune'],default='qualify')
    p.add_argument('--prefix',type=int,default=0)
    a=p.parse_args(); out=a.out; out.mkdir(); (out/'cache').mkdir()
    begin=time.monotonic(); name='wy-production-'+str(time.time_ns()); started=False; state=None; error=None
    def status(phase,**values): common.write(out/'status.json',{'state':phase,'elapsed_s':time.monotonic()-begin,**values})
    def command(args,filename,timeout=1200):
        common.write(out/(filename+'.command.json'),args)
        with (out/(filename+'.log')).open('w') as log:
            common.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    def bench(length,output=1,repeats=3):
        key=f'p{length}-o{output}'; status(key)
        args=['python3',str(ROOT/'wy-boundary-20261106/tools/bench_server_phases.py'),'--base-url',BASE,'--model',MODEL,'--prompt-tokens',str(length),'--output-tokens',str(output),'--repeats',str(repeats),'--cache-salt-prefix',out.name+'-'+key,'--out',str(out/(key+'.jsonl'))]
        if a.mtp and output>1: args+=['--spec-metrics']
        command(args,key,timeout=2400)
    with open('/tmp/a310-npu.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=pdf.snapshot(); common.write(out/'window-start.json',before)
        assert before['rootless']['stdout']==before['rootful']['stdout']=='' and 'No running processes' in before['npu']['stdout'], 'device busy'
        cmd=['podman','run','-d','--pull=never','--name',name,'--network','host','--group-add','keep-groups','--ipc','shareable','--shm-size','32g']
        env={'HOST':'127.0.0.1','PORT':'18126','MAX_MODEL_LEN':'262144','MAX_NUM_SEQS':'4','MAX_BATCHED_TOKENS':str(a.batch),'MEMORY_UTILIZATION':'0.90','NUM_SPECULATIVE_TOKENS':str(a.mtp),'ENABLE_PREFIX_CACHING':str(a.prefix),'SERVED_MODEL_NAME':MODEL}
        for k,v in env.items(): cmd+=['-e',k+'='+v]
        for d in ('davinci0','davinci1','davinci_manager','devmm_svm','hisi_hdc'): cmd+=['--device','/dev/'+d]
        for path in ('/usr/local/bin/npu-smi','/etc/ascend_install.info','/usr/local/Ascend/driver/version.info','/usr/local/Ascend/driver/lib64','/usr/local/dcmi'): cmd+=['-v',path+':'+path+':ro']
        entry=ROOT/'wy-mtp-production-20261108/start-wy-mtp.sh'
        cmd+=['-v',str(ROOT/'production-w8a8/model')+':/model:ro','-v',str(out/'cache')+':/root/.cache:rw','-v',str(entry)+':/opt/qwen36/start-wy-mtp.sh:ro',IMAGE]
        common.write(out/'launch-command.json',cmd)
        common.write(out/'freeze.json',{'env':env,'entry_sha256':hashlib.sha256(entry.read_bytes()).hexdigest(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'image':json.loads(subprocess.check_output(['podman','image','inspect',IMAGE],text=True))[0]})
        try:
            common.run(cmd,capture_output=True,text=True); started=True; status('launching')
            common.wait_health(name); status('smoke')
            pdf.MODEL=MODEL; common.bench(out,a.mtp,out,smoke_only=True)
            if a.mode=='qualify':
                # Exercise multiple speculative verification layouts before the long run.
                expected={'numbers':list(range(1,21)),'status':'ok'}
                def exact(i):
                    r=common.post('/v1/chat/completions',{'model':MODEL,'messages':[{'role':'user','content':'Return exactly this JSON object, no other text: '+json.dumps(expected)}],'temperature':0,'max_tokens':192,'chat_template_kwargs':{'enable_thinking':False},'cache_salt':out.name+'-json'+str(i)})
                    assert json.loads(r['choices'][0]['message']['content'])==expected, r
                    return r
                status('concurrent-json')
                for count in (2,3,4):
                    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool: rows=list(pool.map(exact,range(count)))
                    common.write(out/f'concurrent-json-{count}.json',rows)
            for length in ((4096,8192,16384,32768) if a.mode=='qualify' else (4096,32768)): bench(length)
            bench(4096,128)
            if a.mode=='qualify':
                status('concurrent-decode')
                command(['python3',str(ROOT/'wy-boundary-20261106/tools/bench_http_stream.py'),'--base-url',BASE,'--model',MODEL,'--prompt-tokens','4096','--output-tokens','128','--concurrency','4','--repeats','3','--cache-salt-prefix',out.name+'-c4','--out',str(out/'c4-p4096-o128.jsonl')],'concurrency',1200)
                # No warmup at the capacity boundary: one exact context-budget request.
                with http.post(BASE,'/tokenize',{'model':MODEL,'prompt':'Cold full context capacity verification. Numerical precision and efficient prefill. ','add_special_tokens':False}) as response: unit=json.load(response)['tokens']
                for length in (131072,262016):
                    status('long-context',prompt_tokens=length)
                    tokens=(unit*((length+len(unit)-1)//len(unit)))[:length]
                    before=phases.snapshot(BASE,bool(a.mtp)); cache_before=phases.prefix_snapshot(BASE)
                    result=http.request(BASE,MODEL,tokens,128,cache_salt=out.name+'-full'+str(length))
                    after=phases.snapshot(BASE,bool(a.mtp)); cache_after=phases.prefix_snapshot(BASE)
                    common.write(out/f'full-{length}.json',{'result':result,'metrics_before':before,'metrics_after':after,'cache_before':cache_before,'cache_after':cache_after,'prompt_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest()})
                    assert cache_after['vllm:prefix_cache_hits_total']==cache_before['vllm:prefix_cache_hits_total']
                status('boundary-rejection')
                try:
                    with http.post(BASE,'/v1/completions',{'model':MODEL,'prompt':[unit[0]]*262144,'max_tokens':1}) as response:
                        raise AssertionError('over-limit request unexpectedly accepted')
                except urllib.error.HTTPError as e:
                    body=e.read().decode(); common.write(out/'boundary-rejection.json',{'code':e.code,'body':body}); assert e.code==400
                status('pdf')
                os.environ['PDF_AB_OUT']=str(out)
                inputs=ROOT/'a310-ocr-pdf-smoke-20261107'; manifest=json.loads((inputs/'manifest.json').read_text())
                for case in manifest['cases']:
                    if case['id'] in ('pdf-01','pdf-02','pdf-03','pdf-04'):
                        row=pdf.request(case,inputs,manifest['prompt'],'production-profile',0); common.write(out/(case['id']+'.json'),row); assert row['valid']
                # Confirm long context and PDF did not poison the subsequent decode state.
                status('post-long-json'); common.write(out/'post-long-json.json',exact(99))
            common.write(out/'resource-final.json',pdf.snapshot())
        except BaseException: error=traceback.format_exc()
        finally:
            if started: state=common.stop(name,out)
            after=pdf.snapshot(); clean=after['rootless']['stdout']==after['rootful']['stdout']=='' and 'No running processes' in after['npu']['stdout']
            common.write(out/'window-closed.json',{'elapsed_s':time.monotonic()-begin,'cleanup_passed':clean,'error':error,'container_state':state,'snapshot':after})
            status('failed' if error or not clean else 'complete',error=error,cleanup_passed=clean)
        if error or not clean: raise RuntimeError(error or 'cleanup failed')
if __name__=='__main__': main()
