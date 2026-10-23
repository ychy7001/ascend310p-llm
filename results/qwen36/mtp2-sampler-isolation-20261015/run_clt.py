"""独占设备执行sampler探针和MTP2候选；异常自动恢复原容器，不修改原源码。"""
import argparse,json,subprocess,time
from pathlib import Path

ROOT=Path('/home/clt/qwen36-mtp2-sampler-20261015')
BASE='qwen36-c16-112k-20261015'
PROBE='qwen36-mtp2-bounded-probe-20261015'
CANDIDATE='qwen36-mtp2-bounded-20261015'
TOOLS=Path('/home/clt/qwen36-unified-2048-20261015/tools')

def run(*args,**kw):
    return subprocess.run(args,check=True,text=True,**kw)

def clone_args(name,meta,overrides=None,probe=False):
    args=['docker','run','--name',name,'--network','host']
    if meta['HostConfig']['Privileged']:args+=['--privileged']
    for device in meta['HostConfig'].get('Devices') or []:
        args+=['--device',f"{device['PathOnHost']}:{device['PathInContainer']}:{device['CgroupPermissions']}"]
    for mount in meta['HostConfig'].get('Binds') or []:args+=['-v',mount]
    args+=['-v',f'{ROOT}:/experiment:ro']
    env=dict(e.split('=',1) for e in meta['Config']['Env'] if '=' in e)
    env.update(overrides or {})
    for k,v in env.items():args+=['-e',f'{k}={v}']
    if probe:
        prelude=meta['Config']['Cmd'][2].split('    python3 -c')[0]
        args+=['--entrypoint','bash',meta['Config']['Image'],'-lc',prelude+'\npython3 /experiment/check_sampler.py --device npu\n']
    else:
        args+=['-d',meta['Config']['Image'],*meta['Config']['Cmd']]
    return args

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-only-after-probe',action='store_true',help='要求已记录40项数值通过；不将探针退出失败改记为成功')
    parser.add_argument('--eager',action='store_true',help='仅关Graph，其余保持固定sampler候选')
    args=parser.parse_args()
    candidate=CANDIDATE+('-eager' if args.eager else '')
    out=ROOT/('eager' if args.eager else 'graph')
    out.mkdir(exist_ok=True)
    meta=json.loads(run('docker','inspect',BASE,capture_output=True).stdout)[0]
    (out/'baseline-inspect.json').write_text(json.dumps(meta,indent=2))
    for name in ((candidate,) if args.model_only_after_probe else (PROBE,candidate)):
        if subprocess.run(['docker','inspect',name],capture_output=True).returncode==0:
            raise RuntimeError('experiment container already exists: '+name)
    result={'passed':False,'stage':'probe','baseline':BASE,'candidate':candidate,'graph':not args.eager}
    try:
        run('docker','stop','-t','40',BASE)
        if args.model_only_after_probe:
            text=(ROOT/'npu-parity.txt').read_text()
            assert '"passed": true, "cases": 40' in text
            result['probe_scope']='40 numerical cases passed; prior process cleanup timed out, recorded separately'
        else:
            with (ROOT/'npu-parity.txt').open('w') as log:
                run(*clone_args(PROBE,meta,probe=True),stdout=log,stderr=subprocess.STDOUT,timeout=180)
        result['stage']='model-short'
        overrides={'PYTHONPATH':'/experiment/candidate_bootstrap:/experiment:/c8bootstrap','NUM_SPECULATIVE_TOKENS':'2',
                   'GRAPH_CAPTURE_SIZES':'1,2,3','QWEN36_C8_ALLOW_MTP2':'1'}
        if args.eager:
            overrides['ENFORCE_EAGER']='1'
            overrides['QWEN36_C8_ALLOW_GRAPH']='0'
            overrides['QWEN36_C8_FORCE_UPDATABLE_GRAPH']='0'
        run(*clone_args(candidate,meta,overrides))
        with (out/'model-short.txt').open('w') as log:
            run('python3',str(TOOLS/'validate.py'),'--base-url','http://127.0.0.1:18016',
                '--model','qwen36-w4a8-c8','--out',str(out/'short'),'--smoke-only',
                stdout=log,stderr=subprocess.STDOUT,timeout=1500)
        result['stage']='model-32k'
        with (out/'model-32k.txt').open('w') as log:
            run('python3',str(TOOLS/'bench_server_phases.py'),'--base-url','http://127.0.0.1:18016',
                '--model','qwen36-w4a8-c8','--out',str(out/'p32768-o128.jsonl'),
                '--prompt-tokens','32768','--output-tokens','128','--repeats','2',
                '--unique-prompts','--spec-metrics','--cache-salt-prefix','bounded-greedy',
                stdout=log,stderr=subprocess.STDOUT,timeout=600)
        result.update(passed=True,stage='short-and-32k-passed')
    except Exception as error:
        result['error']=repr(error)
    finally:
        for name in (PROBE,candidate):
            subprocess.run(['docker','stop','-t','20',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            p=subprocess.run(['docker','logs',name],capture_output=True,text=True)
            (out/f'{name}-service.txt').write_text(p.stdout+p.stderr)
        run('docker','start',BASE)
        result['baseline_restart_requested']=True
        (out/'outcome.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)

if __name__=='__main__':
    main()
