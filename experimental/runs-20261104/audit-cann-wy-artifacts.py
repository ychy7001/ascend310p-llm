#!/usr/bin/env python3
"""只提取并静态核验本轮CPU构建包，不安装或调用设备算子。"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 a=p.parse_args();records={'native_packages':{},'python_wheels':{}}
 names=['ChunkGatedDeltaRuleComputeWy','FusedGdnGating']
 ops=['chunk_gated_delta_rule_compute_wy','fused_gdn_gating']
 required={f'aclnn{name}{suffix}' for name in names for suffix in ['', 'GetWorkspaceSize']}
 for vendor in ['custom','wy921_cann']:
  package=a.repo/'build_out'/f'cann-ops-transformer-{vendor}_linux-x86_64.run'
  target=a.out/'artifact-audit'/vendor
  assert not target.exists()
  with (a.out/f'audit-extract-{vendor}.log').open('x') as log:
   subprocess.run(['bash',str(package),'--noexec','--extract='+str(target)],stdout=log,stderr=subprocess.STDOUT,check=True)
  base=target/'packages/vendors'/f'{vendor}_transformer'
  lib=base/'op_api/lib/libcust_opapi.so'
  output=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True)
  exports={line.split()[-1] for line in output.splitlines()}
  assert required<=exports
  headers=[base/'op_api/include/aclnnop'/f'aclnn_{op}.h' for op in ops]
  assert all(h.exists() for h in headers)
  info=base/'op_impl/ai_core/tbe/config/ascend310p/aic-ascend310p-ops-info.json'
  assert set(json.loads(info.read_text()))==set(names)
  kernels={}
  for op in ops:
   objects=list((base/'op_impl/ai_core/tbe/kernel/ascend310p'/op).glob('*.o'))
   assert len(objects)==1 and objects[0].read_bytes().startswith(b'\x7fELF')
   kernels[op]={'path':str(objects[0].relative_to(base)),'size_bytes':objects[0].stat().st_size,'sha256':sha(objects[0])}
  records['native_packages'][vendor]={'filename':package.name,'sha256':sha(package),'size_bytes':package.stat().st_size,
   'vendor_directory':f'{vendor}_transformer','library_sha256':sha(lib),'required_exports':sorted(required),
   'header_sha256':{h.name:sha(h) for h in headers},'opinfo_sha256':sha(info),'opinfo_operator_names':names,'kernels':kernels}
 for package in sorted((a.repo/'build_out').glob('*.whl')):
  with zipfile.ZipFile(package) as z:
   package_name=next(name.split('/')[0] for name in z.namelist() if name.endswith('/op_builder/builder.py'))
   found={}
   for op in ops:
    py=package_name+f'/ops/attention/{op}/{op}.py';cpp=package_name+f'/csrc/attention/{op}.cpp'
    content=z.read(py).decode();tree=ast.parse(content)
    assert any(isinstance(node,ast.FunctionDef) and node.name==op for node in tree.body)
    assert f'torch.ops.cann_ops_transformer.{op}' in content
    assert ('aclnn'+names[ops.index(op)]) in z.read(cpp).decode()
    found[op]={'python_sha256':hashlib.sha256(z.read(py)).hexdigest(),'cpp_sha256':hashlib.sha256(z.read(cpp)).hexdigest()}
   builder=z.read(package_name+'/op_builder/builder.py').decode()
   assert 'Library("cann_ops_transformer", "DEF")' in builder
   records['python_wheels'][package.name]={'sha256':sha(package),'size_bytes':package.stat().st_size,
    'python_package':package_name,'torch_namespace':'cann_ops_transformer','interfaces':found,
    'contains_JIT_cpp_sources':True,'CXX_bridge_loaded_or_device_called':False}
 records['scope']='Static artifact validation only: self-extract without installation, exported symbols, headers, 310P opinfo/kernel ELF, Python AST and packaged C++ entry points; no device initialization.'
 with (a.out/'artifact-audit.json').open('x') as stream:json.dump(records,stream,indent=2);stream.write('\n')
 print(json.dumps({'native_packages':list(records['native_packages']),'python_wheels':list(records['python_wheels']),'checks_passed':True},indent=2))


if __name__=='__main__':main()
