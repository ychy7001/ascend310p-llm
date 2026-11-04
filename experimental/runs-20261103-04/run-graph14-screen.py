#!/usr/bin/env python3
"""W8 TP2单变量筛选：两张Graph由[1,2]改为[1,4]，其余生产参数不变。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out-dir', default='production-graph14')
parser.add_argument('--container', default='qwen36-production-graph14-20261103')
parser.add_argument('--port', type=int, default=18107)
parser.add_argument('--faulthandler', action='store_true', help='Record Python stacks for startup crash diagnosis')
args = parser.parse_args()
root = Path(__file__).resolve().parent
baseline = root / 'production-w8a8'
output = root / args.out_dir
if output.exists():
    previous = json.loads((output / 'status.json').read_text())
    if previous['state'] != 'not_started' or previous['steps'] or {p.name for p in output.iterdir()} != {'status.json'}:
        raise RuntimeError('Refusing to overwrite an executed candidate')
else:
    output.mkdir()
container = args.container
state_path = output / 'status.json'
steps = []
outcome = None
base = f'http://127.0.0.1:{args.port}'
model = 'Qwen3.6-35B-A3B-w8a8'
common = ['--base-url', base, '--model', model]
fixtures = Path('/path/to/lab-runtime/a1-310p-tune-20261101')


def state(value, **extra):
    temporary = state_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(state=value, steps=steps, **extra), indent=2) + '\n')
    temporary.replace(state_path)


def run(label, command):
    state('running', current=label)
    begin = time.monotonic()
    with (output / (label + '.log')).open('x') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    steps.append(dict(label=label, returncode=result.returncode, elapsed_s=time.monotonic()-begin))
    if result.returncode:
        raise RuntimeError(f'{label}: returncode={result.returncode}')


def quality(label):
    run(label, ['python3', str(root / 'check_cache_request_reuse.py'), *common,
                '--out', str(output / (label + '.json'))])
    result = json.loads((output / (label + '.json')).read_text())
    if result['strict_matches'] != 8 or not result['reuse']['identical_text'] or result['reuse']['canary_leaked']:
        raise RuntimeError(f'{label}: short quality gate failed')


def aggregate(path):
    return [row['aggregate_output_tok_s'] for row in
            map(json.loads, path.read_text().splitlines()) if row['kind'] == 'round']


state('waiting_for_production_baseline')
for _ in range(2160):
    previous = json.loads((baseline / 'status.json').read_text())
    if previous['state'] in ('failed', 'not_started'):
        state('not_started', reason='baseline failed; inspect before starting candidate')
        raise SystemExit(1)
    if previous['state'] == 'complete':
        break
    time.sleep(5)
else:
    state('not_started', reason='baseline did not finish within three hours')
    raise SystemExit(1)

try:
    # State.Running may become false before the previous container fully exits.
    # Require terminal status and two NPU-idle snapshots before any new launch.
    state('waiting_for_devices_idle')
    idle = 0
    for _ in range(60):
        inspect = subprocess.run(['podman', 'inspect', '--format', '{{.State.Status}}',
                                  'qwen36-production-replica-20261103'], capture_output=True, text=True, check=True)
        smi = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, check=True)
        idle = idle + 1 if inspect.stdout.strip() == 'exited' and 'No running processes found' in smi.stdout else 0
        if idle >= 2:
            break
        time.sleep(2)
    else:
        raise RuntimeError('previous service/device cleanup incomplete; candidate not launched')
    manifest = json.loads((root / 'production-manifest.json').read_text())
    if args.faulthandler:
        manifest['env']['PYTHONFAULTHANDLER'] = '1'
    command = manifest['command']
    position = command.index('--compilation-config') + 1
    old_config = json.loads(command[position])
    new_config = dict(old_config, cudagraph_capture_sizes=[1, 4])
    command[position] = json.dumps(new_config)
    manifest_path = output / 'candidate-manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'candidate-definition.json').write_text(json.dumps(dict(
        primary_variable='cudagraph_capture_sizes [1,2] → [1,4]',
        rationale='keep two captured graphs while covering gateway concurrency 3 via padding to 4; event resources and padded-state correctness require measurement',
        original_compilation_config=old_config, candidate_compilation_config=new_config,
        production_manifest_sha256=hashlib.sha256((root / 'production-manifest.json').read_bytes()).hexdigest(),
        diagnostic_python_faulthandler=args.faulthandler,
        scope='P0 configuration screening; no production change or upstream-engine patch'), indent=2) + '\n')
    run('launch', ['python3', str(root / 'launch-production-replica.py'),
                   '--manifest', str(manifest_path), '--start-script', str(root / 'start-production-replica.sh'),
                   '--model-dir', str(baseline / 'model'), '--out', str(output),
                   '--container', container, '--port', str(args.port), '--profiler'])
    state('waiting_for_health')
    for _ in range(240):
        try:
            with urllib.request.urlopen(base + '/health', timeout=3) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        inspect = subprocess.run(['podman', 'inspect', '--format', '{{.State.Status}}', container], capture_output=True, text=True)
        if inspect.stdout.strip() == 'exited':
            raise RuntimeError('Graph[1,4] replica exited during startup')
        time.sleep(5)
    else:
        raise RuntimeError('Graph[1,4] health timeout')
    quality('quality-before')
    comparison = {}
    for concurrency in (1, 3, 4):
        label = f'c{concurrency}-p1024-o256-stream'
        run(label, ['python3', str(root / 'bench_http_stream.py'), *common,
                    '--prompt-tokens', '1024', '--output-tokens', '256', '--repeats', '3', '--seed', '0',
                    '--concurrency', str(concurrency), '--out', str(output / (label + '.jsonl'))])
        old = aggregate(baseline / (label + '.jsonl'))
        new = aggregate(output / (label + '.jsonl'))
        comparison[str(concurrency)] = dict(baseline_rounds=old, candidate_rounds=new,
            median_ratio=statistics.median(new)/statistics.median(old),
            scope='three rounds screening; not a reliable P95 or final business speedup')
    (output / 'screening-comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    run('business-facts', ['python3', str(root / 'check_business_facts.py'), *common,
                           '--cases', '8', '--concurrencies', '1', '3',
                           '--out', str(output / 'business-facts.jsonl')])
    quality('quality-after-screen')
    if comparison['3']['median_ratio'] < 1.05 or comparison['1']['median_ratio'] < 0.95:
        outcome = dict(decision='not_selected', reason='screening gain insufficient or C1 regressed', comparison=comparison)
    else:
        run('business-quality', ['python3', str(root / 'check_business_quality.py'),
                                '--url', base, '--model', model, '--fixtures-dir', str(fixtures),
                                '--out', str(output / 'business-quality.jsonl')])
        shutil.copy2(baseline / 'replay-prompts.json', output / 'replay-prompts.json')
        run('business-replay-long', ['python3', str(root / 'replay_stage_workload.py'),
              '--workload', str(fixtures / 'a1-310p-tune.workload.json'),
              '--recording', 'chinese_company_annual_meeting_planning', '--urls', base, '--model', model,
              '--prompts', str(output / 'replay-prompts.json'), '--out', str(output / 'business-replay-long.json')])
        old = json.loads((baseline / 'business-replay-long.json').read_text())
        new = json.loads((output / 'business-replay-long.json').read_text())
        business_ratio = old['elapsed_s'] / new['elapsed_s']
        (output / 'business-comparison.json').write_text(json.dumps(dict(
            baseline_seconds=old['elapsed_s'], candidate_seconds=new['elapsed_s'],
            wall_clock_speed_ratio=business_ratio, scope='one sequential replay per config; repeat A/B and review meeting semantics before deployment'), indent=2) + '\n')
        if business_ratio < 1.05:
            outcome = dict(decision='not_selected', reason='no sufficient full-workload gain', comparison=comparison)
        else:
            run('gsm64', ['python3', str(root / 'eval_gsm8k_http.py'), *common,
                '--dataset', '/path/to/lab-runtime/qwen36-public-release-20261012/gsm8k-test.jsonl',
                '--limit', '64', '--seed', '310', '--max-tokens', '512', '--label', 'production-graph14',
                '--out', str(output / 'gsm64.jsonl')])
            old_gsm = json.loads((baseline / 'gsm64.jsonl').read_text().splitlines()[-1])
            new_gsm = json.loads((output / 'gsm64.jsonl').read_text().splitlines()[-1])
            acceptable = new_gsm['correct'] >= old_gsm['correct'] and new_gsm['truncated'] <= old_gsm['truncated']
            outcome = dict(decision='promising_requires_repeat_and_semantic_review' if acceptable else 'rejected_quality',
                           business_wall_clock_speed_ratio=business_ratio, comparison=comparison,
                           baseline_gsm=old_gsm, candidate_gsm=new_gsm)
    state('finishing', **outcome)
except Exception as error:
    state('failed', error=str(error))
    raise
finally:
    with (output / 'service-final.log').open('w') as log:
        subprocess.run(['podman', 'logs', container], stdout=log, stderr=subprocess.STDOUT)
    stopped = subprocess.run(['podman', 'stop', '-t', '90', container])
    if outcome is not None:
        state('complete' if stopped.returncode == 0 else 'failed', **outcome)
