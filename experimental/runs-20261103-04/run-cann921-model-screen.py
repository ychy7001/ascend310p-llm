#!/usr/bin/env python3
"""新版默认栈模型短请求/事实/业务质量筛选；语义复核前不运行性能。"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

IMAGE = 'd103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--model', choices=['a3b', 'dense'], required=True)
p.add_argument('--attempt', default='v1')
p.add_argument('--util', type=float, default=.9)
p.add_argument('--max-model-len', type=int, help='Explicit capacity screen; omitted preserves production context')
p.add_argument('--strict-gdn', action='store_true')
p.add_argument('--short-format-diagnostic', action='store_true', help='Diagnostic only: preserve failed short gate, no facts/business/performance')
p.add_argument('--gdn-pair-dir', type=Path, help='Explicit9.2.1 paired H/O candidate; original Torch WY')
a = p.parse_args()
if a.gdn_pair_dir and a.strict_gdn:
    p.error('paired candidate uses its own completed strict operator gate')
if a.max_model_len is not None and not 4096 <= a.max_model_len <= 262144:
    p.error('max-model-len must be4096..262144')
if not a.attempt.replace('-', '').isalnum() or not 0 < a.util < 1:
    p.error('invalid attempt/util')
root = a.root.resolve()
out = root / 'cann921-upgrade' / f'{a.model}-quality-{a.attempt}'
out.mkdir(exist_ok=False)
shutil.copy2(__file__, out / 'supervisor-used.py')
steps = []; paused = False; launched = False
container = f'cann921-{a.model}-{time.time_ns()}'
port = 18114 if a.model == 'a3b' else 18115
base = f'http://127.0.0.1:{port}'
model = 'Qwen3.6-35B-A3B-w8a8' if a.model == 'a3b' else 'Qwen3.8-27B-w8a8-310p'
common = ['--base-url', base, '--model', model]

def state(value, **extra):
    tmp = out / 'status.tmp'
    tmp.write_text(json.dumps(dict(state=value, steps=steps, **extra), indent=2) + '\n')
    tmp.replace(out / 'status.json')

def run(label, cmd, timeout=900):
    state('running', current=label, container=container)
    (out / (label + '-command.json')).write_text(json.dumps(cmd, indent=2) + '\n')
    begin = time.monotonic()
    with (out / (label + '.log')).open('x') as log:
        r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    steps.append(dict(label=label, returncode=r.returncode, elapsed_s=time.monotonic() - begin))
    if r.returncode:
        raise RuntimeError(label + ' failed; inspect raw output')

def quality(label, require_pass=True):
    run(label, ['python3', str(root / 'check_cache_request_reuse.py'), *common,
                '--out', str(out / (label + '.json'))], 600)
    q = json.loads((out / (label + '.json')).read_text())
    passed = q['strict_matches'] == 8 and q['reuse']['identical_text'] and not q['reuse']['canary_leaked']
    if require_pass and not passed:
        raise RuntimeError(label + ' strict short/ABA gate failed; no further quality or performance runs')
    return passed

def interrupted(signum, frame):
    raise RuntimeError(f'Supervisor interrupted: {signum}')

signal.signal(signal.SIGTERM, interrupted)
try:
    prior = root / 'cann921-upgrade/operator-screen-v3'
    if json.loads((prior / 'status.json').read_text())['state'] != 'diagnostics_complete_requires_adjudication' or not (prior / 'restore-a2-service.log').exists():
        raise RuntimeError('Prior operator screen or cleanup incomplete')
    for device in (0, 1):
        rows = [json.loads(x) for x in (prior / f'gdn-device{device}.jsonl').read_text().splitlines()]
        if rows[-1].get('within_tolerance') != 48 or rows[-1].get('state_equal_first') != 48:
            raise RuntimeError('GDN repeatability gate failed')
    for seed in range(310, 315):
        for allocator in ('False', 'True'):
            if json.loads((prior / f'dense-seed{seed}-{allocator}.json').read_text())['status'] != 'PASS':
                raise RuntimeError('Synthetic quant gate failed')
    if a.gdn_pair_dir:
        pair = a.gdn_pair_dir.resolve()
        if json.loads((pair/'status.json').read_text())['state'] != 'passed_operator_validation':
            raise RuntimeError('Paired H/O operator gate incomplete')
        if not (pair/'operator-artifacts.json').is_file():
            raise RuntimeError('Missing frozen paired H/O operator artifacts')
    manifest = json.loads((root / 'production-manifest.json').read_text())
    manifest['image'] = IMAGE
    manifest['experiment_kind'] = 'runtime_compatibility_screen'
    manifest['experiment_scope'] = 'Whole official CANN9.2.1/post7/vLLM0.30.0 runtime default path; no experimental WY/H/O/gating; quality only'
    if a.gdn_pair_dir:
        manifest['experiment_kind'] = 'runtime_gdn_pair_correctness'
        manifest['experiment_scope'] = 'Fixed official9.2.1 runtime + pinned PR17640 paired H/O; original Torch WY; quality only'
    command = manifest['command']
    command[command.index('--gpu-memory-utilization') + 1] = str(a.util)
    if a.model == 'dense':
        command[command.index('serve') + 1] = '/models/' + model
        command[command.index('--served-model-name') + 1] = model
    if a.max_model_len is not None:
        command[command.index('--max-model-len') + 1] = str(a.max_model_len)
        manifest['capacity_screen'] = {'max_model_len': a.max_model_len, 'scope': 'Separate quality/capacity screen; no production context equivalence'}
    manifest['env'].pop('ASCEND_CUSTOM_PATH', None)
    (out / 'candidate-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    model_dir = root / ('production-w8a8/model' if a.model == 'a3b' else 'qwen38-27b-w8a8-310p/model')
    if a.model == 'dense':
        weights = json.loads((root/'qwen38-27b-w8a8-310p/status.json').read_text())
        if weights['state'] != 'complete':
            raise RuntimeError('Dense weights not verified')
        for item in weights['files']:
            if (model_dir/item['name']).stat().st_size != item['bytes']:
                raise RuntimeError('Weight file size changed: '+item['name'])
        shutil.copy2(root/'qwen38-27b-w8a8-310p/status.json', out/'weights-verification-source.json')
    tools = out/'tools'; tools.mkdir()
    tool_names = ['launch-production-replica.py', 'check_cache_request_reuse.py',
                  'check_business_facts.py', 'check_business_quality.py',
                  'start-cann921-gdn-pair.sh' if a.gdn_pair_dir else 'start-cann921.sh']
    if a.short_format_diagnostic:
        tool_names.append('check_dense_format_diagnostic.py')
    for name in tool_names:
        shutil.copy2(root/name, tools/name)
    (out/'tool-freeze.json').write_text(json.dumps({name: hashlib.sha256((tools/name).read_bytes()).hexdigest() for name in tool_names}, indent=2)+'\n')
    with urllib.request.urlopen('http://127.0.0.1:18081/health', timeout=10) as response:
        if response.status != 200: raise RuntimeError('Original service not healthy before trial')
    paused = True
    subprocess.run(['sudo', '-n', 'systemctl', 'stop', 'llm-backend-primary.service'], check=True)
    idle = 0
    for _ in range(90):
        info = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True)
        # Driver may retain a process table without the old empty-table sentence.
        busy = re.search(r'\|\s*\d+\s+\d+\s+\d+\s+\S+', info.stdout) is not None
        idle = idle + 1 if not busy else 0
        if idle >= 2: break
        time.sleep(2)
    else: raise RuntimeError('NPU not idle')
    if a.strict_gdn:
        strict = out / 'strict-gdn'; strict.mkdir()
        for name in ('check_default_gdn.py', 'check_native_wy.py', 'native_wy.py'):
            shutil.copy2(root / 'native-gdn/tools' / name, strict / name)
        (strict / 'freeze.json').write_text(json.dumps({f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in strict.glob('*.py')}, indent=2) + '\n')
        cmd = json.loads((prior / 'gdn-device0-command.json').read_text())
        cmd[cmd.index('--name') + 1] = container + '-strict'
        idx = cmd.index(IMAGE)
        cmd[idx:idx] = ['-v', str(strict) + ':/strict:rw']
        cmd[-1] = cmd[-1][:cmd[-1].index('python3 ')] + 'python3 /strict/check_default_gdn.py --out /strict/results.jsonl\n'
        try:
            run('strict-gdn', cmd, 300)
        finally:
            subprocess.run(['podman', 'stop', '--time', '5', container + '-strict'], capture_output=True, timeout=30)
    model_dir = root / ('production-w8a8/model' if a.model == 'a3b' else 'qwen38-27b-w8a8-310p/model')
    launched = True
    launch = ['python3', str(root / 'launch-production-replica.py'), '--manifest', str(out / 'candidate-manifest.json'),
                  '--start-script', str(root / ('start-cann921-gdn-pair.sh' if a.gdn_pair_dir else 'start-cann921.sh')), '--model-dir', str(model_dir), '--out', str(out),
                  '--container', container, '--port', str(port), '--compatibility-image-id', IMAGE]
    if a.gdn_pair_dir:
        launch += ['--cann921-gdn-pair-dir', str(pair)]
    run('launch', launch, 120)
    state('waiting_health', container=container)
    for _ in range(240):
        try:
            with urllib.request.urlopen(base + '/health', timeout=3) as r:
                if r.status == 200: break
        except Exception: pass
        status = subprocess.run(['podman', 'inspect', '--format', '{{.State.Status}}', container], capture_output=True, text=True)
        if status.stdout.strip() == 'exited': raise RuntimeError('Candidate exited during startup')
        time.sleep(5)
    else: raise TimeoutError('Startup health not ready within budget')
    if a.gdn_pair_dir:
        probe = """import json,pathlib
rows=[]
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit(): continue
 try: maps=(proc/'maps').read_text()
 except (OSError,PermissionError): continue
 if '/gdn-pair/extension/pr17640_gdn_pair.so' in maps:
  libs=sorted(set(line.split()[-1] for line in maps.splitlines() if 'libcust_opapi' in line or 'pr17640_gdn_pair.so' in line))
  rows.append({'pid':int(proc.name),'libraries':libs})
assert len(rows)==2,rows
assert all(any('/gdn-pair/opp/vendors/hyperqwen_gdn_pair921_transformer/' in lib for lib in row['libraries']) for row in rows),rows
print(json.dumps(rows,indent=2))
"""
        run('worker-library-maps', ['podman','exec',container,'python3','-c',probe], 30)
    if a.short_format_diagnostic:
        short_passed = quality('quality-before', require_pass=False)
        run('format-diagnostics', ['python3', str(tools/'check_dense_format_diagnostic.py'), *common,
                                 '--out', str(out/'format-diagnostics.jsonl')], 600)
        state('format_diagnostics_complete_not_accepted', short_gate_passed=short_passed,
              scope='Controlled punctuation probes only; short failure remains, no business/performance acceptance')
    else:
        quality('quality-before')
        run('business-facts', ['python3', str(root / 'check_business_facts.py'), *common, '--cases', '8', '--concurrencies', '1', '3', '--out', str(out / 'business-facts.jsonl')])
        run('business-quality', ['python3', str(root / 'check_business_quality.py'), '--url', base, '--model', model,
                                 '--fixtures-dir', '/path/to/lab-runtime/a1-310p-tune-20261101', '--out', str(out / 'business-quality.jsonl')])
        quality('quality-after')
        state('quality_screen_complete_requires_semantic_review', scope='No GSM/performance before semantic review; HTTP200 and keywords are not acceptance')
except Exception as error:
    state('failed', error=str(error)); raise
finally:
    if launched:
        subprocess.run(['podman', 'stop', '--time', '30', container], capture_output=True, timeout=45)
        with (out / 'server.log').open('w') as log:
            subprocess.run(['podman', 'logs', container], stdout=log, stderr=subprocess.STDOUT)
        with (out / 'container-inspect.json').open('w') as log:
            subprocess.run(['podman', 'inspect', container], stdout=log, stderr=subprocess.STDOUT)
    if paused:
        with (out / 'restore-a2-service.log').open('x') as log:
            subprocess.run(['sudo', '-n', 'systemctl', 'reset-failed', 'llm-backend-primary.service'], stdout=log, stderr=subprocess.STDOUT)
            r = subprocess.run(['sudo', '-n', 'systemctl', 'start', 'llm-backend-primary.service'], stdout=log, stderr=subprocess.STDOUT)
        if r.returncode: state('failed', error='Could not restore a2 temporary service')
