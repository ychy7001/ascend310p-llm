#!/usr/bin/env python3
"""Launch a pinned production replica with metadata observation only."""
import argparse,hashlib,json,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
root=a.root.resolve();out=root/'experiment';out.mkdir(exist_ok=False)
manifest=json.loads((root/'production-manifest.json').read_text())
image='registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:1a193ded0d75832321019542326740c7fc4509853556d032e773631466453b38'
assert manifest['image']==image
command=manifest['command'];command[command.index('--port')+1]='18106';command[command.index('--host')+1]='127.0.0.1'
assert command[command.index('--max-num-seqs')+1]=='4'
assert json.loads(command[command.index('--compilation-config')+1])['cudagraph_capture_sizes']==[1,2]
assert not any('speculative' in x for x in command)
command+=['--profiler-config',json.dumps({'profiler':'torch','torch_profiler_dir':'/experiment/profiles','torch_profiler_with_stack':False,'ignore_frontend':True,'torch_profiler_record_shapes':False})]
manifest['env']['PYTHONPATH']='/observe'
manifest['env']['DECODE_OBSERVE_DIR']='/experiment/observation'
manifest['port']=18106
(out/'observation').mkdir();(out/'cache').mkdir();(out/'replica-manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'provenance.json').write_text(json.dumps({'base_manifest_sha256':hashlib.sha256((root/'production-manifest.json').read_bytes()).hexdigest(),'observation_only':True,'profile_mode':'Level1; no argument capture; shapes from separate mapped forwards','inference_logic_changes':False,'image':image},indent=2))
args=['podman','run','-d','--pull=never','--name','decode-2-observed-20261104','--network','host','--group-add','keep-groups','--ipc','shareable','--shm-size','32g']
for device in ['davinci0','davinci1','davinci_manager','devmm_svm','hisi_hdc']:args+=['--device','/dev/'+device]
mounts=[(root/'model','/models/Qwen3.6-35B-A3B-w8a8','ro'),(out,'/experiment','rw'),(out/'cache','/root/.cache','rw'),(root/'start-production-replica.sh','/workspace/start_vllm_ascend.sh','ro'),(out/'replica-manifest.json','/workspace/backend-primary.manifest.json','ro'),(root/'observe','/observe','ro')]
for path in ['/usr/local/bin/npu-smi','/etc/ascend_install.info','/usr/local/Ascend/driver/version.info','/usr/local/Ascend/driver/lib64','/usr/local/dcmi']:mounts.append((path,path,'ro'))
for src,dst,mode in mounts:args+=['-v',f'{src}:{dst}:{mode}']
args += [image,'bash','/workspace/start_vllm_ascend.sh']
subprocess.run(args,check=True)
