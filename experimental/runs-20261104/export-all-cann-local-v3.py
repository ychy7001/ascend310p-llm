#!/usr/bin/env python3
"""One matched CANN export subprocess at a time, only on copied local captures."""
import subprocess,sys
from pathlib import Path
ROOT=Path('results/decode-attribution-20261104/v3/raw')
for stamp,label in [('093327','b3'),('093352','b4')]:
 for rank in (0,1):
  source=next((ROOT/'profiles').glob(f'*rank{rank}_*20261104{stamp}*'));capture=next(source.glob('PROF*'))
  for kind in ['summary','timeline']:
   if kind=='summary' and (list(capture.glob('mindstudio_profiler_output/op_summary*.csv'))+list(capture.glob('device_*/summary/op_summary*.csv'))):continue
   if kind=='timeline' and list(capture.glob('device_*/timeline/task_time*.json')):continue
   with (ROOT/f'cann-export-{label}-rank{rank}-{kind}.log').open('w') as log:
    code=subprocess.run(['/usr/bin/time','-v',sys.executable,'experimental/runs-20261104/export-cann-local-v3.py','--capture',str(capture),'--kind',kind],stdout=log,stderr=log,timeout=1850).returncode
   print(label,rank,kind,code,flush=True)
   if code:raise SystemExit(code)
