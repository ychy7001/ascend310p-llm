#!/usr/bin/env python3
"""Verify published immutable images via Registry V2; never print credentials."""
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path('/home/ychy/ascend-lab-runtime/hyperqwen-20261103/registry-delivery-20261108')
HOST='registry.cn-hangzhou.aliyuncs.com'
REPO='meetai/llm-service-vllm-ascend'
TOKEN=None
AUTH=json.loads((ROOT/'auth.json').read_text())['auths'][HOST]['auth']
ACCEPT='application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'

def fetch(path):
    global TOKEN
    headers={'Accept':ACCEPT}
    if TOKEN: headers['Authorization']='Bearer '+TOKEN
    try:
        return urllib.request.urlopen(urllib.request.Request('https://'+HOST+path,headers=headers),timeout=45)
    except urllib.error.HTTPError as e:
        if e.code!=401: raise
        challenge=dict(re.findall(r'(\w+)="([^"]*)"',e.headers['WWW-Authenticate']))
        url=challenge['realm']+'?'+urllib.parse.urlencode({'service':challenge.get('service',''),'scope':challenge.get('scope',f'repository:{REPO}:pull')})
        request=urllib.request.Request(url,headers={'Authorization':'Basic '+AUTH})
        with urllib.request.urlopen(request,timeout=45) as response: token=json.load(response)
        TOKEN=token.get('token') or token['access_token']
        return fetch(path)

records=[]
for variant,digest,expected_id in [
    ('tp1-w4a8','sha256:8e17ec9ae646b1e418cf119bbb347238c4505b8b0037782a1345914824ec8d35','b381d8e5ed45c11633a7bd07a50a0f5fcf8cf71bc797f1b543d589ab07dd759d'),
    ('tp2-w8a8-wy',(ROOT/'tp2-w8a8-wy.digest').read_text().strip(),'9df3c6f3e42084d77ef258853cf824a011b4d30a1b05acc72a834a691c073163')]:
    with fetch(f'/v2/{REPO}/manifests/{digest}') as response:
        raw=response.read()
        header_digest=response.headers.get('Docker-Content-Digest')
    assert 'sha256:'+hashlib.sha256(raw).hexdigest()==digest
    assert header_digest in (None,digest)
    manifest=json.loads(raw)
    assert manifest['config']['digest']=='sha256:'+expected_id
    (ROOT/(variant+'-remote-manifest.json')).write_bytes(raw)
    with fetch(f'/v2/{REPO}/blobs/sha256:{expected_id}') as response: config=response.read()
    assert hashlib.sha256(config).hexdigest()==expected_id
    data=json.loads(config)
    row={'variant':variant,'registry_image':f'{HOST}/{REPO}@{digest}','registry_digest':digest,'image_id':expected_id,'remote_config_digest':manifest['config']['digest'],'rootfs_diff_ids':data['rootfs']['diff_ids'],'registry_verified':True,'verified_at_utc':datetime.now(timezone.utc).isoformat(),'action':'reuse-existing' if variant=='tp1-w4a8' else 'pushed','mtp':variant=='tp1-w4a8'}
    if variant=='tp2-w8a8-wy': row['tag']=f'{HOST}/{REPO}:qwen36-tp2-w8a8-wy-20261108'
    records.append(row)
(ROOT/'registry-images.json').write_text(json.dumps({'schema_version':1,'images':records},indent=2)+'\n')
print(json.dumps([{k:v for k,v in r.items() if k!='rootfs_diff_ids'} for r in records],indent=2))
