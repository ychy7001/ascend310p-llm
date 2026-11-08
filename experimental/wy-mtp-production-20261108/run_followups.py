#!/usr/bin/env python3
"""Continue only after a clean qualification, with required device cooldowns."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/ychy/ascend-lab-runtime/hyperqwen-20261103')
work=ROOT/'wy-mtp-production-20261108'
previous=ROOT/'w8-production-full-v5'
deadline=time.monotonic()+3600
while True:
    if not (previous/'status.json').exists():
        if time.monotonic()>deadline: raise SystemExit('no qualification status')
        time.sleep(5); continue
    status=json.loads((previous/'status.json').read_text())
    if status['state']=='failed': raise SystemExit('qualification failed; follow-ups canceled')
    if status['state']=='complete': break
    if time.monotonic()>deadline: raise SystemExit('qualification wait budget exceeded; no device action')
    time.sleep(20)
for window,batch,mtp in [('w8-production-b2048-v1',2048,2),('w8-production-mtp0-v3',4096,0)]:
    time.sleep(max(0,60-(time.time()-(previous/'window-closed.json').stat().st_mtime)))
    out=ROOT/window
    with (work/(window+'.log')).open('w') as log:
        subprocess.run(['python3',str(work/'run_profile.py'),'--out',str(out),'--batch',str(batch),'--mtp',str(mtp),'--mode','tune','--eager','1'],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    previous=out
