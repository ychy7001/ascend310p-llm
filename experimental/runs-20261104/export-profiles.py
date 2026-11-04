#!/usr/bin/env python3
"""Offline CPU export of completed dual-rank windows; interrupted captures stay untouched."""
import argparse,json,pathlib,re,shlex,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--experiment',default='experiment');a=p.parse_args();out=a.root.resolve()/a.experiment
image=json.loads((out/'replica-manifest.json').read_text())['image']
events=[json.loads(line) for line in (out/'http-events.jsonl').read_text().splitlines()]
starts=[r for r in events if r['kind']=='profile_start'];completed={r['label'] for r in events if r['kind']=='profile_stop'}
selection=[]
for rank in [0,1]:
 dirs=sorted(d for d in (out/'profiles').glob('*_ascend_pt') if ('_rank%d_'%rank) in d.name)
 assert len(dirs)==len(starts),(rank,len(dirs),len(starts))
 selection.extend({'label':event['label'],'rank':rank,'directory':directory.name} for event,directory in zip(starts,dirs) if event['label'] in completed)
(out/'profile-selection.json').write_text(json.dumps(selection,indent=2))
code='from torch_npu.profiler.profiler import analyse; analyse("/experiment/profiles", max_process_number=1)'
container='decode-2-export-'+a.experiment.replace('-','')[-24:]
command=['podman','run','--rm','--cpus','4','--memory','8g','--memory-swap','8g','--pids-limit','256','--timeout','1800','--name',container,'--network','none','--entrypoint','/bin/bash','--group-add','keep-groups','-e','OMP_NUM_THREADS=1','-v',f'{out}:/experiment:rw','-v','/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro']
for device in ['davinci0','davinci1','davinci_manager','devmm_svm','hisi_hdc']:command+=['--device','/dev/'+device]
for path in ['/etc/ascend_install.info','/usr/local/Ascend/driver/version.info','/usr/local/dcmi']:command+=['-v',f'{path}:{path}:ro']
command+=[image,'--noprofile','--norc','-c','source /usr/local/Ascend/ascend-toolkit/set_env.sh && python3 -c '+shlex.quote(code)]
start=time.monotonic()
with (out/'offline-export.log').open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=2400)
files=[out/'profiles'/s['directory']/'ASCEND_PROFILER_OUTPUT'/'kernel_details.csv' for s in selection]
report={'returncode':result.returncode,'elapsed_seconds':time.monotonic()-start,'nonempty_kernel_exports':sum(f.exists() and f.stat().st_size>0 for f in files),'expected_kernel_exports':len(selection),'scope':'CPU-only offline parse; no model or inference'}
(out/'offline-export-result.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
assert result.returncode==0 and report['nonempty_kernel_exports']==len(selection),report
