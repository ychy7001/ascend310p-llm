"""10秒一次npu-smi采样；可能漏掉瞬态峰值，不宣称完整峰值。"""
import json
from pathlib import Path
import subprocess
import sys
import time

with Path(sys.argv[1]).open('a') as output:
    for _ in range(180):
        started = time.time()
        result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
        output.write(json.dumps({'time': started, 'returncode': result.returncode,
                                 'stdout': result.stdout, 'stderr': result.stderr}) + '\n')
        output.flush()
        time.sleep(max(0, 10 - (time.time() - started)))
