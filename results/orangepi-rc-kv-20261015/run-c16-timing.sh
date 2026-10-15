#!/usr/bin/env bash
set -euo pipefail
python3 -u - <<'PY'
import json,time,subprocess,urllib.request
from pathlib import Path
p=Path('results/orangepi-rc-kv-20261015')
end=time.monotonic()+1000
while time.monotonic()<end:
 try:
  with urllib.request.urlopen('http://orangepi:18024/health',timeout=3) as r:
   if r.status==200:break
 except OSError:time.sleep(5)
else:raise RuntimeError('C16 blocking-MM-H2D candidate health timeout')
print('C16 retry healthy',flush=True)
p.joinpath('c16-env.json').write_text(subprocess.check_output(['ssh','orangepi','docker inspect qwen36-orangepi --format "{{json .Config.Env}}"'],text=True))
subprocess.run(['python3','tools/check_cache_request_reuse.py','--base-url','http://orangepi:18024','--model','qwen36-w4a8','--out',str(p/'c16-quality.json')],check=True,timeout=600)
q=json.loads(p.joinpath('c16-quality.json').read_text());assert q['strict_matches']==8 and q['reuse']['identical_text'] and not q['reuse']['canary_leaked']
for n,repeats in ((8192,3),(32768,3),(112000,1)):
 subprocess.run(['python3','tools/bench_server_phases.py','--base-url','http://orangepi:18024','--model','qwen36-w4a8','--out',str(p/f'c16-p{n}-o128.jsonl'),'--prompt-tokens',str(n),'--output-tokens','128','--repeats',str(repeats),'--unique-prompts','--spec-metrics','--cache-salt-prefix',f'rc-kv-c16-{n}'],check=True,timeout=1500)
print('C16 fixed-output comparison complete',flush=True)
PY
