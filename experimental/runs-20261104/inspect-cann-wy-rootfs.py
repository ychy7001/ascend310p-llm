#!/usr/bin/env python3
"""本机CPU容器内扫描实际镜像rootfs，不导入torch_npu或初始化设备。"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sysconfig

needles=['aclnnChunkGatedDeltaRuleComputeWy','aclnnFusedGdnGating','ChunkGatedDeltaRuleComputeWy','FusedGdnGating','chunk_gated_delta_rule_compute_wy','fused_gdn_gating']
roots=[Path('/usr/local/Ascend'),Path('/vllm-workspace/vllm-ascend')]
roots.extend(Path(p) for p in set([sysconfig.get_path('purelib'),sysconfig.get_path('platlib')]))
seen=set();libs=[];names=[];text=[];count=0
for root in roots:
 for directory,dirs,files in os.walk(root):
  dirs[:]=[d for d in dirs if d not in ['.git','__pycache__']]
  for name in files:
   path=Path(directory)/name
   if path.is_symlink() or not path.is_file():continue
   key=str(path.resolve())
   if key in seen:continue
   seen.add(key)
   if any(n.lower() in name.lower() for n in needles):names.append(str(path))
   if 'opapi' in name and '.so' in name:
    result=subprocess.run(['nm','-D','--defined-only',str(path)],capture_output=True,text=True)
    libs.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'nm_returncode':result.returncode,
      'matching_exports':[line for line in result.stdout.splitlines() if any(n in line for n in needles)]})
   if path.suffix in ['.h','.json','.ini','.info','.py']:
    content=path.read_bytes();count+=1
    hits=[n for n in needles if n.encode() in content]
    if hits:text.append({'path':str(path),'matches':hits,'sha256':hashlib.sha256(content).hexdigest()})
packages={}
for name in ['cann_ops_transformer','torch_npu']:
 spec=importlib.util.find_spec(name)
 packages[name]={'present':spec is not None,'origin':spec.origin if spec else None}
runtime_sources={}
for name in ['cann_ops.py','chunk_gated_delta_rule.py','fused_gdn_gating.py','gdn_310.py']:
 path=Path('/vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/fla')/name
 runtime_sources[name]=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
print(json.dumps({'runtime_source_sha256':runtime_sources,'method':'Flattened exact-image rootfs, read symbols and text; no torch/torch_npu import, no device mount.',
 'text_files_checked':count,'named_files':names,'opapi_libraries':libs,'text_hits':text,'packages':packages,
 'device_paths':[str(p) for pattern in ['davinci*','devmm*','hisi*','nvidia*'] for p in Path('/dev').glob(pattern)]},indent=2))
