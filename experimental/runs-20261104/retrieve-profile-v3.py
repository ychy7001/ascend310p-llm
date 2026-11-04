#!/usr/bin/env python3
"""Sequential, bandwidth-limited read-only retrieval. Never executes remote analysis."""
import json
from pathlib import Path
import subprocess
import time
ROOT=Path('results/decode-attribution-20261104/v3/raw')
REMOTE='/home/ychy/ascend-lab-runtime/decode-2-20261104/experiment/profiles'
names=sorted({line.split('\t',1)[1].split('/')[-1] for line in (ROOT/'remote-profile-inventory.txt').read_text().splitlines() if '\t' in line},key=lambda n:(n.split('_20261104')[1],n))
records=[]
for name in names:
    start=time.time();target=ROOT/'profiles'/name;target.mkdir(parents=True,exist_ok=True)
    print('START',name,flush=True)
    with (ROOT/(name+'.rsync.log')).open('w') as log:
        result=subprocess.run(['rsync','-a','--bwlimit=20000','--partial','--info=stats2','-e','ssh -o BatchMode=yes',f'ychy-a2:{REMOTE}/{name}/',str(target)+'/'],stdout=log,stderr=subprocess.STDOUT)
    record={'directory':name,'returncode':result.returncode,'seconds':time.time()-start};records.append(record)
    (ROOT/'retrieval.json').write_text(json.dumps(records,indent=2)+'\n')
    print('DONE',json.dumps(record),flush=True)
    if result.returncode:raise SystemExit(result.returncode)
