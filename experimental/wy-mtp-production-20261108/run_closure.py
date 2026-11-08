#!/usr/bin/env python3
"""After full-context qualification, run the matched control and final artifact."""
import json,subprocess,time
from pathlib import Path
ROOT=Path('/home/ychy/ascend-lab-runtime/hyperqwen-20261103');work=ROOT/'wy-mtp-production-20261108'
previous=ROOT/'w8-production-full-v6';deadline=time.monotonic()+3600
while True:
    if time.monotonic()>deadline: raise SystemExit('qualification wait budget exceeded')
    if not (previous/'status.json').exists():time.sleep(5);continue
    state=json.loads((previous/'status.json').read_text())['state']
    if state=='failed': raise SystemExit('qualification failed; no further device work')
    if state=='complete':break
    time.sleep(20)
for window,mode,mtp in [('w8-production-mtp0-final','tune',0),('w8-production-packaged-final','packaged',2)]:
    time.sleep(max(0,60-(time.time()-(previous/'window-closed.json').stat().st_mtime)))
    with (work/(window+'.log')).open('w') as log:
        subprocess.run(['python3',str(work/'run_profile.py'),'--out',str(ROOT/window),'--batch','2048','--mtp',str(mtp),'--mode',mode,'--eager','1','--served-model-name','Qwen3.6-35B-A3B-w8a8'],stdout=log,stderr=subprocess.STDOUT,check=True)
    previous=ROOT/window
(work/'closure-complete.json').write_text(json.dumps({'complete':True,'final_window':str(previous)})+'\n')
