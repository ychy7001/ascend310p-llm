import datetime,json,re,subprocess,time
from pathlib import Path
root=Path('/home/clt/qwen36-kv-comparison-20261015')
while True:
    try:
        done=(root/'comparison-progress.jsonl').exists() and 'comparison_complete' in (root/'comparison-progress.jsonl').read_text()
        if done:break
        r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15)
        m=re.search(r'(\d+)\s*/\s*44215',r.stdout)
        containers=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
        row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'device_used_mb':int(m.group(1)) if m else None,'total_mb':44215,'active_model_containers':[n for n in containers if n.startswith('qwen36')],'returncode':r.returncode}
        with (root/'memory-samples.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        time.sleep(10)
    except Exception as e:
        print(str(e),flush=True);break
