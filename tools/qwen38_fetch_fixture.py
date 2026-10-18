"""Fetch pinned real tensors via checked HTTP ranges; never download a full model.
PLE files are explicitly SPARSE fixtures: only recorded rows contain real data.
"""
import argparse, concurrent.futures, hashlib, json, struct, urllib.request
from pathlib import Path

REPO = 'matteiuspi/Qwen3.8-Flash-Next-W4A16-G128-300i'
REV = 'd32978ccdde1729caec0034571b03a14779698e9'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--proxy', default='http://127.0.0.1:1081')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    wm = json.loads(a.index.read_text())['weight_map']; records = []
    def fetch(file, start, end):
        dest = a.output / 'ranges' / f'{file}.{start}-{end}'
        dest.parent.mkdir(exist_ok=True)
        if dest.exists(): return dest.read_bytes()
        url = f'https://huggingface.co/{REPO}/resolve/{REV}/{file}?range={start}-{end}'
        req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({'https': a.proxy}))
        with opener.open(req, timeout=90) as r:
            expected = f'bytes {start}-{end}/'
            if r.status != 206 or not r.headers.get('Content-Range', '').startswith(expected):
                raise RuntimeError(f'Range not honored: {r.status} {r.headers}')
            data = r.read(end-start+2)
        if len(data) != end-start+1: raise RuntimeError('Wrong byte count')
        dest.write_bytes(data)
        return data
    def header(file):
        first = fetch(file, 0, 7); n = struct.unpack('<Q', first)[0]
        if n > 8_000_000: raise ValueError('Unexpected header size')
        raw = fetch(file, 8, 7+n)
        return json.loads(raw), 8+n, first+raw
    eh, base, _ = header('experts-000.safetensors')
    selected = {k:v for k,v in eh.items() if any(f'.experts.{e}.' in k for e in (0,256))}
    def tensor(item):
        k,v = item; lo,hi = v['data_offsets']
        data = fetch('experts-000.safetensors', base+lo, base+hi-1)
        return k,v,data
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        tensors = list(pool.map(tensor, selected.items()))
    outhead = {}; payload = bytearray()
    for k,v,data in tensors:
        outhead[k] = dict(v, data_offsets=[len(payload),len(payload)+len(data)])
        payload.extend(data)
        records.append({'tensor':k, 'shape':v['shape'], 'dtype':v['dtype'], 'sha256':hashlib.sha256(data).hexdigest()})
    raw = json.dumps(outhead,separators=(',',':')).encode(); raw += b' '*((-len(raw))%8)
    (a.output/'experts.safetensors').write_bytes(struct.pack('<Q',len(raw))+raw+payload)
    plemap = {}; prows = []; shard_rows = None
    for shard in (0,127):
        key = f'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{shard}.weight'
        file = wm[key]; h,b,rawh = header(file); v = h[key]
        rows,dim = v['shape']; shard_rows = rows
        target = a.output / file
        with target.open('wb') as out:
            out.write(rawh); out.truncate(b+v['data_offsets'][1])
            for row in (0, 12345, rows-1):
                start=b+v['data_offsets'][0]+row*dim*2
                data=fetch(file,start,start+dim*2-1)
                out.seek(start);out.write(data)
                prows.append({'shard':shard,'row':row,'id':shard*rows+row,'hex':data.hex(),'sha256':hashlib.sha256(data).hexdigest()})
        plemap[key]=file
    (a.output/'model.safetensors.index.json').write_text(json.dumps({'weight_map':plemap}))
    manifest={'repo':REPO,'revision':REV,'experts':records,'ple_rows':prows,'ple_shard_rows':shard_rows,'ple_dim':dim,'warning':'PLE files are sparse fixtures; unrecorded rows are zero holes, NOT model weights','downloaded_range_bytes':sum(x.stat().st_size for x in (a.output/'ranges').iterdir())}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'expert_tensors':len(records),'real_ple_rows':len(prows),'downloaded_range_bytes':manifest['downloaded_range_bytes']}))
if __name__=='__main__': main()
