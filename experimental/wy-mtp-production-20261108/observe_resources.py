#!/usr/bin/env python3
"""Read-only observed NPU snapshots; this is not allocator peak instrumentation."""
import json,subprocess,sys,time
from pathlib import Path
window=Path(sys.argv[1])
with (window/'resource-samples.jsonl').open('x') as output:
    while True:
        state=json.loads((window/'status.json').read_text())['state']
        snapshot=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=20)
        output.write(json.dumps({'unix_s':time.time(),'state':state,'exit_code':snapshot.returncode,'npu':snapshot.stdout})+'\n'); output.flush()
        if state in ('failed','complete'): break
        time.sleep(10)
