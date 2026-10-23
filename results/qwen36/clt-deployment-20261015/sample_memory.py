import datetime,json,re,subprocess,time
from pathlib import Path
out=Path('/home/clt/qwen36-optimized-20261015/logs/memory-during-validation.jsonl')
while Path('/proc/16410/cmdline').exists() and b'validate_deployment' in Path('/proc/16410/cmdline').read_bytes():
    r=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15)
    m=re.search(r'(\d+)\s*/\s*44215',r.stdout)
    row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'npu_smi_returncode':r.returncode,'device_used_mb':int(m.group(1)) if m else None,'device_total_mb':44215}
    with out.open('a') as f:f.write(json.dumps(row)+'\n')
    time.sleep(10)
