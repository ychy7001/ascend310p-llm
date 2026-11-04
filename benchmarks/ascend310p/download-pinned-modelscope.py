#!/usr/bin/env python3
"""按冻结 ModelScope 文件清单逐文件下载并校验大小/SHA；不执行模型代码。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.parse

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', required=True)
p.add_argument('--file-list', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--wait-for-status', type=Path)
a = p.parse_args()
a.out.mkdir(exist_ok=False)
model = a.out / 'model'
model.mkdir()
steps = []

def state(value, **kw):
    f = a.out / 'status.tmp'
    f.write_text(json.dumps(dict(state=value, repo=a.repo, files=steps, **kw), indent=2) + '\n')
    f.replace(a.out / 'status.json')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(8 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()

try:
    listing = json.loads(a.file_list.read_text())
    if listing.get('Code') != 200 or not listing.get('Success'):
        raise ValueError('Invalid frozen source manifest')
    files = [f for f in listing['Data']['Files'] if f['Type'] == 'blob']
    for f in files:
        name = f['Path']
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Only flat checkpoint files supported')
        if len(f['Sha256']) != 64 or len(f['Revision']) != 40:
            raise ValueError('Missing pinned digest/revision')
    (a.out / 'source-file-list.json').write_bytes(a.file_list.read_bytes())
    state('waiting_benchmarks' if a.wait_for_status else 'preparing',
          manifest_sha256=digest(a.file_list), bytes=sum(f['Size'] for f in files))
    if a.wait_for_status:
        while True:
            j = json.loads(a.wait_for_status.read_text())
            if j['state'] in ('failed', 'not_started'):
                raise RuntimeError('Preceding experiment did not complete; download held')
            if j['state'] == 'complete':
                break
            time.sleep(5)
    for f in files:
        name = f['Path']
        target = model / name
        partial = model / (name + '.partial')
        query = urllib.parse.urlencode(dict(Revision=f['Revision'], FilePath=name))
        url = f'https://www.modelscope.cn/api/v1/models/{a.repo}/repo?{query}'
        state('downloading', current=name)
        begin = time.monotonic()
        with (a.out / 'download.log').open('a') as log:
            r = subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                '--retry', '5', '--retry-delay', '5', '--connect-timeout', '30',
                '--max-time', '3600', '--continue-at', '-', '--output', str(partial), url],
                stdout=log, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(f'Download failed: {name} (curl {r.returncode})')
        size = partial.stat().st_size
        actual = digest(partial)
        if size != f['Size'] or actual != f['Sha256']:
            raise RuntimeError(f'Source size/SHA mismatch: {name}; got {size}/{actual}')
        partial.replace(target)
        steps.append(dict(name=name, bytes=size, sha256=actual, revision=f['Revision'],
                          elapsed_s=time.monotonic()-begin))
        print(json.dumps(steps[-1]), flush=True)
    inventory = a.out / 'checkpoint-inventory.json'
    with (a.out / 'inventory.log').open('x') as log:
        r = subprocess.run(['python3', str(Path(__file__).with_name('inventory-checkpoint.py')),
            str(model), '--out', str(inventory)], stdout=log, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError('Checkpoint header/index inventory failed')
    state('complete', scope='All files verified against frozen remote metadata SHA256; safetensors headers/index validated. No model inference yet.')
except Exception as e:
    state('failed', error=str(e))
    raise
