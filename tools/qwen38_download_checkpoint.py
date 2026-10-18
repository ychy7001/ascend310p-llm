# SPDX-License-Identifier: Apache-2.0
"""Resume a pinned HF checkpoint from a mirror and verify every original hash.
Requires an HF model API ?blobs=true manifest. Never changes tensor metadata.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--endpoint', default='https://hf-mirror.com')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'download.pid').write_text(str(os.getpid()))
    revision = manifest['sha']
    files = manifest['siblings']
    print(json.dumps({'event':'start','repo':manifest['id'],'revision':revision,'files':len(files),'bytes':sum(f['size'] for f in files)}), flush=True)

    def check(path, item):
        h = hashlib.sha256() if 'lfs' in item else hashlib.sha1()
        if 'lfs' not in item: h.update(f"blob {item['size']}\0".encode())
        with path.open('rb') as f:
            while data := f.read(8*1024*1024): h.update(data)
        expected = item['lfs']['sha256'] if 'lfs' in item else item['blobId']
        if h.hexdigest() != expected: raise ValueError(f"Checksum mismatch: {item['rfilename']}")
        return h.hexdigest()

    def download(item):
        name = item['rfilename']
        if Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError(name)
        target = args.output/name; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != item['size']: raise ValueError(f'Existing wrong-size file: {name}')
            digest = check(target,item)
            return {'file':name,'bytes':item['size'],'hash':digest,'resumed_complete':True}
        part = target.with_name(target.name+'.part')
        url = f"{args.endpoint}/{manifest['id']}/resolve/{revision}/{urllib.parse.quote(name)}"
        for attempt in range(8):
            try:
                offset = part.stat().st_size if part.exists() else 0
                if offset > item['size']: raise ValueError(f'Oversize partial file: {name}')
                if offset < item['size']:
                    request = urllib.request.Request(url,headers={'User-Agent':'qwen38-checkpoint-download/1.0 (curl-compatible)', **({'Range':f'bytes={offset}-'} if offset else {})})
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    with opener.open(request,timeout=120) as response:
                        if offset and (response.status != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                            raise RuntimeError('Server did not honor resume range')
                        with part.open('ab' if offset else 'wb') as out:
                            while data := response.read(4*1024*1024):
                                offset += len(data)
                                if offset > item['size']: raise ValueError('Download exceeds manifest size')
                                out.write(data)
                if part.stat().st_size != item['size']: raise RuntimeError('Truncated response')
                digest = check(part,item)
                part.rename(target)
                record = {'event':'verified','file':name,'bytes':item['size'],'hash':digest}
                print(json.dumps(record),flush=True)
                return record
            except Exception as exc:
                print(json.dumps({'event':'retry','file':name,'attempt':attempt+1,'error':str(exc)}),flush=True)
                if attempt == 7: raise
                time.sleep(min(30,2**attempt))
    # Fetch configuration/tokenizer first; larger shards follow.
    files = sorted(files,key=lambda f:(f['rfilename'].endswith('.safetensors'),f['rfilename']))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(download, files))
    (args.output/'download-verified.json').write_text(json.dumps({'repo':manifest['id'],'revision':revision,'files':results},indent=2))
    print(json.dumps({'event':'complete','files':len(results)}),flush=True)


if __name__ == '__main__': main()
