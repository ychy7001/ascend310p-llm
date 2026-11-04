#!/usr/bin/env python3
"""只读检查镜像层的ACLNN导出、头文件、op信息和Python包，不启动容器。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--image',required=True)
a=p.parse_args()
detail=json.loads(subprocess.check_output(['podman','image','inspect',a.image],text=True))[0]
data=detail['GraphDriver']['Data']
layers=[data['UpperDir']]+data['LowerDir'].split(':')
needles=[b'aclnnChunkGatedDeltaRuleComputeWy',b'aclnnFusedGdnGating',b'ChunkGatedDeltaRuleComputeWy',b'FusedGdnGating',b'chunk_gated_delta_rule_compute_wy',b'fused_gdn_gating']
files=[];libraries=[];text_hits=[];package_paths=[];seen=set();checked_text=0
for index,layer in enumerate(layers):
 for directory,dirs,names in os.walk(layer):
  dirs[:]=[n for n in dirs if n not in ['.git','__pycache__']]
  for name in names:
   path=Path(directory)/name;rel=str(path.relative_to(layer))
   if rel in seen:continue
   seen.add(rel)
   if path.is_symlink() or not path.is_file():continue
   if not (rel.startswith('usr/local/Ascend/') or rel.startswith('usr/local/lib/python') or rel.startswith('usr/lib/python') or rel.startswith('vllm-workspace/')):continue
   if 'cann_ops_transformer' in rel:package_paths.append(rel)
   if any(n.decode().lower() in name.lower() for n in needles):files.append({'path':rel,'layer':index,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
   if 'opapi' in name and '.so' in name:
    result=subprocess.run(['nm','-D','--defined-only',str(path)],capture_output=True,text=True)
    matches=[line for line in result.stdout.splitlines() if any(n.decode() in line for n in needles)]
    libraries.append({'path':rel,'layer':index,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'nm_returncode':result.returncode,'matching_exports':matches})
   if path.suffix in ['.h','.json','.ini','.py'] and path.stat().st_size<20*1024*1024:
    content=path.read_bytes();checked_text+=1
    found=[n.decode() for n in needles if n in content]
    if found:text_hits.append({'path':rel,'layer':index,'matches':found,'sha256':hashlib.sha256(content).hexdigest()})
print(json.dumps({'image_id':detail['Id'],'image_architecture':detail['Architecture'],'image_repo_digests':detail.get('RepoDigests'),
 'method':'Read image layer files only, topmost path wins; no image mount or container/service operation. Deleted/opaque-path semantics are not resolved; flattened local-image scan is needed for authoritative positive presence.',
 'layer_count':len(layers),'text_files_checked':checked_text,'named_files':files,'opapi_libraries':libraries,'text_hits':text_hits,'cann_ops_transformer_paths':package_paths},indent=2))
