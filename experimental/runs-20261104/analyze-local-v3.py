#!/usr/bin/env python3
"""Single local CPU parser with an 8 GiB address-space and one-hour CPU cap."""
from pathlib import Path
import json, resource, subprocess, sys, time
ROOT=Path('results/decode-attribution-20261104/v3')
profiles=ROOT/'raw/profiles'
records=[]
def limits():
 resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
 resource.setrlimit(resource.RLIMIT_CPU,(3600,3600))
for case,stamp in [('b1-long','093151'),('b2-long','093306')]:
 for rank in (0,1):
  source=next(profiles.glob(f'*rank{rank}_*20261104{stamp}*'))/'ASCEND_PROFILER_OUTPUT'
  out=ROOT/'profiles'/f'{case}-rank{rank}.json'
  if out.exists():continue
  cmd=[sys.executable,'tools/analyze_decode_profile.py','--csv',str(source/'kernel_details.csv'),'--trace',str(source/'trace_view.json'),'--ledger','results/decode-attribution-20261104/v2/module-shape-ledger.json','--rank',str(rank),'--out',str(out)]
  start=time.monotonic()
  with (ROOT/'raw'/f'analyze-{case}-rank{rank}.log').open('w') as log:
   r=subprocess.run(['/usr/bin/time','-v',*cmd],stdout=log,stderr=log,preexec_fn=limits,timeout=3700)
  records.append({'case':case,'rank':rank,'returncode':r.returncode,'seconds':time.monotonic()-start})
  (ROOT/'raw/local-analysis-runs.json').write_text(json.dumps(records,indent=2)+'\n')
  print(records[-1],flush=True)
  if r.returncode:raise SystemExit(r.returncode)
