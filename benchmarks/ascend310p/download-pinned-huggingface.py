#!/usr/bin/env python3
"""按冻结HF revision、LFS SHA256/Git blob下载GPU参考制品，不执行模型代码。"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import urllib.parse

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--proxy',required=True)
a=p.parse_args();a.out.mkdir(exist_ok=True);model=a.out/'model';model.mkdir(exist_ok=True)
listing=json.loads(a.manifest.read_text());revision=listing['revision'];repo=listing['repo'];files=listing['files'];steps=[];lock=threading.Lock()
assert len(revision)==40 and all(c in '0123456789abcdef' for c in revision)
assert all(Path(f['rfilename']).name==f['rfilename'] and f['rfilename'] not in ('.','..') for f in files)
source=a.out/'source-manifest.json'
if source.exists():assert source.read_bytes()==a.manifest.read_bytes()
else:source.write_bytes(a.manifest.read_bytes())
def state(value,**extra):
 with lock:
  tmp=a.out/'status.tmp';tmp.write_text(json.dumps({'state':value,'repo':repo,'revision':revision,'files':steps,**extra},indent=2)+'\n');tmp.replace(a.out/'status.json')
def validate(path,item):
 assert path.stat().st_size==item['size'],str(path)+' size mismatch'
 sha=hashlib.sha256();git=hashlib.sha1();git.update(f"blob {item['size']}\0".encode())
 with path.open('rb') as stream:
  while block:=stream.read(8*1024*1024):sha.update(block);git.update(block)
 if 'lfs' in item:assert sha.hexdigest()==item['lfs']['sha256'],str(path)+' LFS SHA mismatch'
 else:assert git.hexdigest()==item['blobId'],str(path)+' Git blob mismatch'
 return sha.hexdigest()
def download(item):
 name=item['rfilename'];target=model/name;part=model/(name+'.partial');begin=time.monotonic()
 if not target.exists():
  url=f'https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}?download=true'
  with (a.out/(name+'.download.log')).open('a') as log:
   r=subprocess.run(['curl','--proxy',a.proxy,'--fail','--location','--silent','--show-error','--retry','4','--retry-delay','3','--connect-timeout','25','--max-time','7200','--continue-at','-','--output',str(part),url],stdout=log,stderr=subprocess.STDOUT)
  if r.returncode:raise RuntimeError(f'{name}: curl {r.returncode}')
  digest=validate(part,item);part.replace(target)
 else:digest=validate(target,item)
 row={'name':name,'bytes':item['size'],'sha256':digest,'elapsed_s':time.monotonic()-begin}
 with lock:steps.append(row)
 state('downloading');print(json.dumps(row),flush=True)
try:
 state('preparing')
 metadata=[f for f in files if not f['rfilename'].endswith('.safetensors')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(download,metadata))
 index=json.loads((model/'model.safetensors.index.json').read_text());referenced=set(index['weight_map'].values());selected={f['rfilename'] for f in files}
 assert referenced<=selected,'Index references omitted file'
 state('downloading_weights',referenced_shards=sorted(referenced))
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(download,[f for f in files if f['rfilename'].endswith('.safetensors')]))
 state('complete',scope='Pinned bytes and LFS/Git digests verified; index file references complete; no model inference or precision equivalence claim')
except Exception as e:state('failed',error=str(e));raise
