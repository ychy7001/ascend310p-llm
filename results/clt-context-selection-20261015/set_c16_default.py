import json,subprocess,time,urllib.request,shutil
from pathlib import Path
root=Path('/home/clt/qwen36-kv-comparison-20261015')
deploy=Path('/home/clt/qwen36-optimized-20261015')
log=root/'default-switch-progress.jsonl'
def emit(data):
    with log.open('a') as f:f.write(json.dumps(data)+'\n')
    print(json.dumps(data),flush=True)
def health(name):
    for i in range(120):
        info=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
        if not info['State']['Running'] or info['RestartCount']:
            raise RuntimeError('container stopped or restarted')
        try:
            with urllib.request.urlopen('http://127.0.0.1:18016/health',timeout=3) as response:
                if response.status==200:return
        except Exception:pass
        if i%12==0:emit({'stage':'waiting','attempt':i})
        time.sleep(5)
    raise RuntimeError('health timeout')
try:
    subprocess.run(['docker','stop','qwen36-optimized-112k-20261015'],check=True)
    subprocess.run(['docker','start','qwen36-c16-112k-20261015'],check=True)
    health('qwen36-c16-112k-20261015')
    emit({'stage':'c16_healthy'})
    subprocess.run(['python3',str(deploy/'repo/tools/bench_generation_sanity.py'),'--base-url','http://127.0.0.1:18016','--model','qwen36-w4a8-c8','--out',str(root/'c16-default-sanity.json'),'--max-tokens','64'],check=True)
    subprocess.run(['python3',str(deploy/'repo/tools/bench_server_phases.py'),'--base-url','http://127.0.0.1:18016','--model','qwen36-w4a8-c8','--out',str(root/'c16-default-8192.jsonl'),'--prompt-tokens','8192','--output-tokens','128','--repeats','3','--unique-prompts','--spec-metrics'],check=True)
    for filename in ['candidate.env','start-candidate.sh']:
        backup=deploy/(filename+'.before-c16-default')
        if not backup.exists():shutil.copy2(deploy/filename,backup)
    shutil.copy2(root/'c16.env',deploy/'candidate.env')
    (deploy/'start-candidate.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
cd /home/clt/qwen36-optimized-20261015
source ./candidate.env
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    exec docker start "$CONTAINER_NAME"
fi
exec bash repo/experimental/run-c8-service.sh
''')
    (deploy/'start-candidate.sh').chmod(0o755)
    (deploy/'README.md').write_text((deploy/'README.md').read_text()+'\n2026-10-15 默认更新：C16/W4A8/MTP1/Decode Graph，112K，18016，容器qwen36-c16-112k-20261015。API兼容模型名qwen36-w4a8-c8保留，不代表KV类型。C8容器停止并保留，回滚先stop C16再start qwen36-optimized-112k-20261015。\n')
    emit({'stage':'default_c16_complete','container':'qwen36-c16-112k-20261015','port':18016})
except Exception as error:
    emit({'stage':'failed','error':str(error)})
    subprocess.run(['docker','stop','qwen36-c16-112k-20261015'],check=False)
    subprocess.run(['docker','start','qwen36-optimized-112k-20261015'],check=True)
    health('qwen36-optimized-112k-20261015')
    emit({'stage':'rollback_c8_healthy'})
    raise
