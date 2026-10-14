import json
import time
from pathlib import Path

with Path('/data/qwen36-deploy/memory.jsonl').open('a') as output:
    for _ in range(180):
        memory = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            memory[key] = int(value.split()[0])
        record = {'time': time.time(), **memory}
        output.write(json.dumps(record) + '\n')
        output.flush()
        time.sleep(10)
