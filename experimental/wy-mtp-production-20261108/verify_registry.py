#!/usr/bin/env python3
"""Verify an immutable registry manifest and config without exposing credentials."""
import argparse,hashlib,json,re,urllib.error,urllib.parse,urllib.request
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--auth-file',type=Path,required=True);p.add_argument('--digest-file',type=Path,required=True);p.add_argument('--image-id',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
host='registry.cn-hangzhou.aliyuncs.com';repo='meetai/llm-service-vllm-ascend';auth=json.loads(a.auth_file.read_text())['auths'][host]['auth'];token=None
def fetch(path):
 global token
 headers={'Accept':'application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json'}
 if token:headers['Authorization']='Bearer '+token
 try:return urllib.request.urlopen(urllib.request.Request('https://'+host+path,headers=headers),timeout=45)
 except urllib.error.HTTPError as error:
  if error.code!=401:raise
  challenge=dict(re.findall(r'(\w+)="([^"]*)"',error.headers['WWW-Authenticate']))
  url=challenge['realm']+'?'+urllib.parse.urlencode({'service':challenge.get('service',''),'scope':challenge.get('scope',f'repository:{repo}:pull')})
  with urllib.request.urlopen(urllib.request.Request(url,headers={'Authorization':'Basic '+auth}),timeout=45) as response:data=json.load(response)
  token=data.get('token') or data['access_token'];return fetch(path)
a.out.mkdir(exist_ok=True);digest=a.digest_file.read_text().strip();image_id=a.image_id.removeprefix('sha256:')
with fetch(f'/v2/{repo}/manifests/{digest}') as response:raw=response.read();header=response.headers.get('Docker-Content-Digest')
assert 'sha256:'+hashlib.sha256(raw).hexdigest()==digest and header in (None,digest)
manifest=json.loads(raw);assert manifest['config']['digest']=='sha256:'+image_id
with fetch(f'/v2/{repo}/blobs/sha256:{image_id}') as response:config=response.read()
assert hashlib.sha256(config).hexdigest()==image_id
data=json.loads(config);assert data['config']['Entrypoint']==['bash','/opt/qwen36/start-wy-mtp.sh']
(a.out/'remote-manifest.json').write_bytes(raw)
record={'registry_image':f'{host}/{repo}@{digest}','registry_digest':digest,'image_id':image_id,'verified_at_utc':datetime.now(timezone.utc).isoformat(),'registry_verified':True,'rootfs_diff_ids':data['rootfs']['diff_ids'],'entrypoint':data['config']['Entrypoint']}
(a.out/'registry-image.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='rootfs_diff_ids'}))
