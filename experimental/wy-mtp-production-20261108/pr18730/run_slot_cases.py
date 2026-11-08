"""Use independent processes so one device exception cannot poison later checks."""
import json
from pathlib import Path
import subprocess
import sys

cases = [
    ['--length', '1'], ['--length', '2'], ['--length', '3'], ['--length', '4'],
    ['--length', '8'], ['--length', '16'],
    ['--length', '2', '--dtype', 'int64'],
    ['--length', '2', '--method', 'shift'],
    ['--length', '2', '--method', 'wide'],
    ['--length', '2', '--method', 'padded'],
    ['--length', '2', '--jit'],
]
rows = []
for index, args in enumerate(cases):
    cmd = [sys.executable, '/audit/repro_slot_mul.py', *args]
    with Path(f'/out/case-{index:02}.log').open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=50)
    rows.append({'args': args, 'exit_code': result.returncode})
    Path('/out/cases.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows[-1]), flush=True)
