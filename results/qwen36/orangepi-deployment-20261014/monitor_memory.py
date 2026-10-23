import json
import sys
import time
from pathlib import Path

with Path(sys.argv[1] if len(sys.argv)>1 else '/data/qwen36-deploy/memory.jsonl').open('a') as output:
    for _ in range(180):
        memory = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            memory[key] = int(value.split()[0])
        record = {'time': time.time(), **memory}
        output.write(json.dumps(record) + '\n')
        output.flush()
        time.sleep(10)
