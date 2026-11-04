#!/usr/bin/env python3
"""固定 PR17640 H/O 成对算子门禁；不加载模型、不修改生产 OPP。"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--attempt', default='')
p.add_argument('--runtime', choices=('production', 'cann921'), default='production')
a = p.parse_args()
root = a.root.resolve()
if a.attempt and not re.fullmatch(r'[a-zA-Z0-9-]+', a.attempt):
    raise ValueError('Invalid attempt suffix')
new_runtime = a.runtime == 'cann921'
image921 = 'd103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a'
vendor = 'hyperqwen_gdn_pair921' if new_runtime else 'hyperqwen_gdn_pair'
out = root/(('upstream-gdn-pair921' if new_runtime else 'upstream-gdn-pair')+('-'+a.attempt if a.attempt else ''))
out.mkdir(exist_ok=False)
shutil.copy2(__file__, out/'supervisor-used.py')
steps = []
paused = False
primary_error = None
container = None

def state(value, **extra):
    temporary = out/'status.tmp'
    temporary.write_text(json.dumps(dict(state=value, steps=steps, **extra), indent=2)+'\n')
    temporary.replace(out/'status.json')

def run(label, command, timeout):
    global container
    container = command[command.index('--name')+1] if '--name' in command else None
    state('running', current=label)
    (out/(label+'-command.json')).write_text(json.dumps(command, indent=2)+'\n')
    begin = time.monotonic()
    try:
        with (out/(label+'.log')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        if container:
            subprocess.run(['podman','stop','--time','5',container], capture_output=True, timeout=30)
        steps.append(dict(label=label, returncode=None, timed_out=True, elapsed_s=time.monotonic()-begin))
        raise
    steps.append(dict(label=label, returncode=result.returncode, elapsed_s=time.monotonic()-begin))
    return result.returncode

try:
    prior_path = root/('cann921-upgrade/gdn-focused-v1/status.json' if new_runtime else 'gdn-runtime-control/status.json')
    previous = json.loads(prior_path.read_text())
    if previous['state'] != ('diagnostics_complete_requires_adjudication' if new_runtime else 'complete'):
        raise RuntimeError('Prior runtime control has not completed')
    state('preparing')
    source = out/'source'
    original_source = root/'native-gdn/source'
    def ignore(path, names):
        excluded = {'.git', '__pycache__'}
        if Path(path) == original_source/'csrc':
            excluded.update({'build', 'build_out', 'build_cache', 'output'})
        return set(names) & excluded
    shutil.copytree(original_source, source, symlinks=True, ignore=ignore)
    tree = json.loads((root/'native-gdn/tree.json').read_text())
    verified = []
    prefixes = ('csrc/moe/chunk_gated_delta_rule_fwd_h_vllm/', 'csrc/moe/chunk_fwd_o_vllm/',
                'csrc/moe/common/', 'csrc/common/', 'csrc/aclnn_torch_adapter/')
    for entry in tree['tree']:
        name = entry['path']
        if entry['type'] != 'blob' or not (name.startswith(prefixes) or name == 'csrc/torch_binding.cpp'):
            continue
        data = (source/name).read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if actual != entry['sha']:
            raise RuntimeError('Pinned source mismatch: '+name)
        verified.append(name)
    catlass = source/'csrc/third_party/catlass'
    if catlass.exists():shutil.rmtree(catlass)
    shutil.copytree(root/'fwd-o-maskfix/source/csrc/third_party/catlass', catlass, symlinks=True)
    tools = out/'tools'
    tools.mkdir()
    for name in ('build_pr17640_gdn_pair_extension.py', 'check_pr17640_gdn_pair.py',
                 'probe_gdn_intermediates.py', 'native_wy.py', 'check_native_wy.py',
                 'check_cann921_gdn_pair.py', 'check_default_gdn.py', 'repro_chunk_gdn_nondeterminism.py'):
        shutil.copy2(root/name, tools/name)
    (out/'source-provenance.json').write_text(json.dumps(dict(
        commit='18163f6b1d208ab3b4fb06e9752225611c37f40e', upstream_author='GruntovDima',
        runtime=a.runtime, build_image=image921 if new_runtime else json.loads((root/'production-manifest.json').read_text())['image'],
        wy='original Torch WY only' if new_runtime else 'Torch/native variants',
        verified_git_blobs=verified, algorithm_changes=[], scope='Paired upstream H/O with original binding NZ state contract; no model',
        probes_sha256={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in tools.iterdir()}), indent=2)+'\n')
    cmd = json.loads((root/'upstream-fwd-o/build-command.json').read_text())
    cmd[cmd.index(str(root/'upstream-fwd-o')+':/work:rw')] = str(out)+':/work:rw'
    cmd[cmd.index('--name')+1] = 'gdn-pair-build-'+str(time.time_ns())
    if new_runtime:
        old_image = json.loads((root/'production-manifest.json').read_text())['image']
        cmd[cmd.index(old_image)] = image921
        # Build without devices; no competing NPU work while the temporary service runs.
        for index in reversed([i for i, value in enumerate(cmd) if value == '--device']):
            del cmd[index:index+2]
    cmd[-1] = '''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /work/source/csrc
bash build.sh --pkg --ops=chunk_gated_delta_rule_fwd_h_vllm,chunk_fwd_o_vllm --soc=ascend310p --vendor_name=hyperqwen_gdn_pair -j4
python3 /work/tools/build_pr17640_gdn_pair_extension.py --source-root /work/source --build /work/extension
'''
    if new_runtime:
        cmd[-1] = cmd[-1].replace('source /usr/local/Ascend/ascend-toolkit/set_env.sh', 'source /usr/local/Ascend/cann/set_env.sh').replace('--vendor_name=hyperqwen_gdn_pair ', '--vendor_name='+vendor+' ')
    if run('build', cmd, 1200):raise RuntimeError('Build failed')
    packages = list((source/'csrc/build_out').glob('*.run'))
    if len(packages) != 1:raise RuntimeError('Ambiguous package')
    cmd[cmd.index('--name')+1] = 'gdn-pair-install-'+str(time.time_ns())
    cmd[-1] = 'bash /work/'+str(packages[0].relative_to(out))+' --quiet --install-path=/work/opp'
    if run('install', cmd, 120):raise RuntimeError('Install failed')
    lib = out/('opp/vendors/'+vendor+'_transformer/op_api/lib/libcust_opapi.so')
    symbols = subprocess.check_output(['nm','-D','--defined-only',str(lib)], text=True)
    for api in ('aclnnChunkGatedDeltaRuleFwdHVllm', 'aclnnChunkFwdOVllm'):
        if not re.search(r'\b'+api+r'$', symbols, re.M):raise RuntimeError('Missing paired API: '+api)
    if re.search(r'\baclnnChunkFwdO$|\baclnnChunkGatedDeltaRuleFwdH$', symbols, re.M):
        raise RuntimeError('Production API unexpectedly exported')
    (out/'package-isolation.json').write_text(json.dumps(dict(symbols=symbols,sha256=hashlib.sha256(lib.read_bytes()).hexdigest()),indent=2)+'\n')
    paused = True
    subprocess.run(['sudo','-n','systemctl','stop','llm-backend-primary.service'], check=True)
    idle = 0
    for _ in range(90):
        info = subprocess.run(['npu-smi','info'],capture_output=True,text=True,check=True)
        idle = idle+1 if 'No running processes found' in info.stdout else 0
        if idle >= 2:break
        time.sleep(2)
    else:raise RuntimeError('NPU not idle')
    template = json.loads((root/'native-gdn/validation-command.json').read_text())
    if new_runtime:
        old_image = json.loads((root/'production-manifest.json').read_text())['image']
        template[template.index(old_image)] = image921
    image = image921 if new_runtime else json.loads((root/'production-manifest.json').read_text())['image']
    prefix = '''set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH="/pair/opp/vendors/hyperqwen_gdn_pair_transformer:/work/opp/vendors/hyperqwen_wy_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"
cd /pair/tools
'''
    if new_runtime:
        # No old9.1 WY library or vendor is mounted into the9.2.1 validation process.
        prefix921 = 'set -eo pipefail\nsource /usr/local/Ascend/cann/set_env.sh\n'
        prefix921 += 'export ASCEND_CUSTOM_OPP_PATH="/pair/opp/vendors/'+vendor+'_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"\ncd /pair/tools\n'
        clean = []
        i = 0
        while i < len(template):
            if template[i] == '-v' and template[i+1].endswith('/native-gdn:/work:rw'):
                i += 2
                continue
            clean.append(template[i]); i += 1
        template = clean
        summaries = []
        for device in (0, 1):
            for hv in (16, 24):
                label = f't512-hv{hv}-device{device}'
                cmd = template.copy()
                cmd[cmd.index('--name')+1] = 'gdn-pair921-'+label+'-'+str(time.time_ns())
                index = cmd.index(image)
                cmd[index:index] = ['-v', str(out)+':/pair:rw']
                cmd[-1] = prefix921+f'python3 check_cann921_gdn_pair.py --library /pair/extension/pr17640_gdn_pair.so --probe repeat --out /pair/{label}.jsonl --tokens 512 --seed 318 --value-heads {hv} --device {device} --repeats 48\n'
                if run(label, cmd, 240): raise RuntimeError(label+' execution failed')
                rows = [json.loads(line) for line in (out/(label+'.jsonl')).read_text().splitlines()]
                summary = rows[-1]
                summaries.append(dict(label=label, summary=summary))
                (out/'focused-summaries.json').write_text(json.dumps(summaries, indent=2)+'\n')
                if any(summary.get(key)!=48 for key in ('samples','within_tolerance','output_equal_first','state_equal_first')):
                    raise RuntimeError(label+' failed independent numeric/repeatability gate; no expanded tests or model')
        for device in (0, 1):
            label = f'strict-device{device}'
            cmd = template.copy()
            cmd[cmd.index('--name')+1] = 'gdn-pair921-'+label+'-'+str(time.time_ns())
            index = cmd.index(image)
            cmd[index:index] = ['-v', str(out)+':/pair:rw']
            cmd[-1] = prefix921+f'python3 check_cann921_gdn_pair.py --library /pair/extension/pr17640_gdn_pair.so --probe strict --out /pair/{label}.jsonl --device {device}\n'
            if run(label, cmd, 300): raise RuntimeError(label+' numerical gate failed')
            rows = [json.loads(line) for line in (out/(label+'.jsonl')).read_text().splitlines()]
            if rows[-1].get('passed')!=15 or rows[-1].get('cases')!=15:
                raise RuntimeError('Incomplete strict gate')
        state('passed_operator_validation', scope='Pinned upstream paired H/O on9.2.1, original Torch WY unchanged; no model/performance acceptance')
    else:
        summaries = []
        for device in (0,1):
            for wy in ('torch','native'):
                label = f'chain-{wy}-device{device}'
                cmd = template.copy()
                cmd[cmd.index('--name')+1] = 'gdn-pair-'+label+'-'+str(time.time_ns())
                index = cmd.index(image)
                cmd[index:index] = ['-v',str(out)+':/pair:rw']
                cmd[-1] = prefix + f'python3 probe_gdn_intermediates.py --work /work --out-dir /pair/{label} --device {device} --fwd-h-library /pair/extension/pr17640_gdn_pair.so --fwd-o-library /pair/extension/pr17640_gdn_pair.so --fwd-o-namespace pr17640_gdn_pair --fwd-o-entry fwd_o' + (' --wy-library /work/wy-extension/pr17640_wy.so' if wy == 'native' else '')+'\n'
                if run(label, cmd, 240):raise RuntimeError(label+' execution failed')
                summaries.append(dict(label=label, summary=json.loads((out/label/'summary.json').read_text())))
        (out/'chain-summaries.json').write_text(json.dumps(summaries,indent=2)+'\n')
        failures = [row['label'] for row in summaries if any(
            x['tested']!=24 or x['outside_tolerance_runs'] or x['raw_output_differs_final_cpu'] or any(x['interface_variation'].values())
            for x in row['summary'].values())]
        if failures:
            state('rejected_numerical_gate', failed_chains=failures, scope='No model launch; paired H/O not accepted')
        else:
            for repeat in range(2):
                label = 'strict-gate-'+str(repeat)
                cmd = template.copy()
                cmd[cmd.index('--name')+1] = 'gdn-pair-'+label+'-'+str(time.time_ns())
                index = cmd.index(image)
                cmd[index:index] = ['-v',str(out)+':/pair:rw']
                cmd[-1] = prefix+f'python3 check_pr17640_gdn_pair.py --library /pair/extension/pr17640_gdn_pair.so --wy-library /work/wy-extension/pr17640_wy.so --out /pair/{label}.jsonl\n'
                if run(label,cmd,240):raise RuntimeError(label+' numerical gate failed; inspect raw rows')
            state('passed_operator_validation',scope='Paired H/O operator gate only; no model/performance acceptance')
except Exception as error:
    primary_error = str(error)
    state('failed',error=primary_error)
    raise
finally:
    if paused:
        with (out/'restore-a2-service.log').open('x') as log:
            subprocess.run(['sudo','-n','systemctl','reset-failed','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
            result = subprocess.run(['sudo','-n','systemctl','start','llm-backend-primary.service'],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:state('failed',error='Temporary unit restore failed',primary_error=primary_error)
