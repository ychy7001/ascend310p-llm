import json,subprocess,time,urllib.request
from pathlib import Path
p=Path('/home/clt/qwen36-kv-comparison-20261015')
log=p/'comparison-progress.jsonl'
def emit(row):
    with log.open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def wait_health(name):
    for i in range(150):
        info=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
        if not info['State']['Running'] or info['RestartCount']>0:
            raise RuntimeError('container failed or restarted: '+name)
        try:
            with urllib.request.urlopen('http://127.0.0.1:18016/health',timeout=3) as r:
                if r.status==200:return
        except Exception:pass
        if i%12==0:emit({'stage':'waiting','container':name,'attempt':i})
        time.sleep(5)
    raise RuntimeError('health timeout '+name)
try:
    wait_health('qwen36-c16-112k-20261015')
    emit({'stage':'c16_healthy'})
    subprocess.run(['python3','/home/clt/qwen36-optimized-20261015/repo/tools/bench_generation_sanity.py','--base-url','http://127.0.0.1:18016','--model','qwen36-w4a8-c8','--out',str(p/'c16/logs/sanity.json'),'--max-tokens','64'],check=True)
    subprocess.run(['python3','/home/clt/qwen36-optimized-20261015/validate_deployment.py','--deployment',str(p/'c16')],check=True)
    emit({'stage':'c16_complete','pass':True})
except Exception as e:
    emit({'stage':'c16_failed','error':str(e)})
finally:
    logs=subprocess.run(['docker','logs','qwen36-c16-112k-20261015'],capture_output=True,text=True)
    (p/'c16/logs/container.log').write_text(logs.stdout+logs.stderr)
    subprocess.run(['docker','stop','-t','30','qwen36-c16-112k-20261015'],check=False)
    subprocess.run(['docker','start','qwen36-optimized-112k-20261015'],check=True)
    wait_health('qwen36-optimized-112k-20261015')
    emit({'stage':'c8_restored_healthy'})
    subprocess.run(['python3','/home/clt/qwen36-optimized-20261015/repo/tools/bench_server_phases.py','--base-url','http://127.0.0.1:18016','--model','qwen36-w4a8-c8','--out',str(p/'c8-restored-8192.jsonl'),'--prompt-tokens','8192','--output-tokens','128','--repeats','3','--unique-prompts','--spec-metrics'],check=True)
    emit({'stage':'comparison_complete'})
