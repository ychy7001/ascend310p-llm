#!/usr/bin/env python3
"""3090隔离35B筛选；只暂停固定测试容器，finally恢复并独立检查原端点。"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
IMAGE = '4d9fb561ad6a866c0828a67b1e4a0c3f35f3373daa02f2bb2a689acc49e76457'
ORIGINAL = 'cb2c12cdf22b73fec58be417399e8dea0ce2f4cd1ea627ce1895d385a7ed9316'
MODEL = 'qwen3.6-35b-a3b-gptq-int4'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--probe-script', type=Path)
    p.add_argument('--scheduler-hook', type=Path)
    p.add_argument('--verified-inventory', type=Path)
    a = p.parse_args()
    a.download = a.download.resolve()
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    source = ROOT/'results/local-gpu-business-20261104/v1'
    name = 'ascend310p-gpu35b-screen-'+a.out.name
    switched = False
    launched = False
    state = {'state': 'preflight', 'image': IMAGE, 'original_container_id': ORIGINAL,
             'container_name': name, 'model': MODEL, 'scope': 'W4 GPTQ GPU behavior reference; not FP16 equivalence or performance acceptance'}

    def save():
        state['updated_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tmp = a.out/'status.json.tmp'
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        tmp.replace(a.out/'status.json')

    def command(label, args, timeout=3600):
        (a.out/(label+'-command.json')).write_text(json.dumps(args, ensure_ascii=False, indent=2))
        with (a.out/(label+'.log')).open('x') as log:
            result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'{label} exit={result.returncode}')

    def json_command(args):
        return json.loads(subprocess.check_output(args, text=True, timeout=30))

    def health(url):
        try:
            with urllib.request.urlopen(url+'/health', timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def wait_health(url, duration, container=None):
        deadline = time.monotonic()+duration
        while time.monotonic() < deadline:
            if health(url):
                return
            if container:
                obj = json_command(['podman', 'inspect', container])[0]
                if not obj['State']['Running']:
                    raise RuntimeError('Experiment container exited before healthy')
            time.sleep(5)
        raise RuntimeError('Timed out waiting for health: '+url)

    def stop_signal(signum, frame):
        raise RuntimeError(f'Controller interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    save()
    try:
        download = json.loads((a.download/'status.json').read_text())
        assert download['state'] == 'complete', download['state']
        assert download['revision'] == '00a66983516f8f8057741221277eed4141c9e431'
        if a.verified_inventory:
            inventory_source = a.verified_inventory.resolve()
            (a.out/'checkpoint-inventory.json').write_bytes(inventory_source.read_bytes())
            (a.out/'inventory-reuse.json').write_text(json.dumps({'source':str(inventory_source), 'sha256':hashlib.sha256(inventory_source.read_bytes()).hexdigest(), 'scope':'Reuse previously verified pinned cache; file sizes rechecked. No new full-weight hashing claim.'},indent=2))
        else:
            command('checkpoint-inventory', [sys.executable, str(ROOT/'benchmarks/ascend310p/inventory-checkpoint.py'), str(a.download/'model'), '--out', str(a.out/'checkpoint-inventory.json')])
        expected = {x['name']:x for x in download['files']}
        inventory = json.loads((a.out/'checkpoint-inventory.json').read_text())
        actual = {x['name']:x for x in inventory['files']}
        assert expected.keys() == actual.keys()
        for key in expected:
            assert expected[key]['bytes'] == actual[key]['bytes'] == (a.download/'model'/key).stat().st_size
            assert expected[key]['sha256'] == actual[key]['sha256']
        original = json_command(['podman', 'inspect', 'GPU_SERVICE'])[0]
        assert original['Id'] == ORIGINAL
        assert original['Image'].removeprefix('sha256:') == IMAGE
        assert original['State']['Running'] and health('http://127.0.0.1:18020')
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1', 18021)) != 0, 'Port 18021 already in use'
        (a.out/'gpu-before.csv').write_text(subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free', '--format=csv'], text=True))
        tools = {}
        for filename in ('check_cache_request_reuse.py', 'check_business_facts.py', 'check_business_quality.py', 'check_action_state_regression.py'):
            path = source/'tools'/filename
            tools[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        (a.out/'tool-freeze.json').write_text(json.dumps(tools, indent=2))
        fixtures_freeze = {}
        for path in [source/'semantic-review-oracle.json', source/'fixtures/meeting-prompts.jsonl', source/'fixtures/tool-prompts.jsonl', source/'action-state-fixtures.json']:
            fixtures_freeze[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        (a.out/'fixture-freeze.json').write_text(json.dumps(fixtures_freeze, indent=2))
        (a.out/'semantic-review-oracle.json').write_bytes((source/'semantic-review-oracle.json').read_bytes())
        # Own writable compilation cache; no changes to the original cache/model mount.
        cache = a.download/'screen-cache'
        cache.mkdir(exist_ok=True)
        launch = ['podman', 'run', '-d', '--name', name, '--network', 'host', '--ipc', 'host',
            '--device', 'nvidia.com/gpu=0', '--security-opt', 'label=disable',
            '-v', f'{a.download / "model"}:/model:ro', '-v', f'{cache}:/cache',
            '-e', 'HOME=/cache', '-e', 'NVIDIA_VISIBLE_DEVICES=void', '-e', 'CUDA_VISIBLE_DEVICES=0', '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
            '-e', 'VLLM_NO_USAGE_STATS=1', '-e', 'VLLM_USE_FLASHINFER_SAMPLER=0',
            '--entrypoint', '/app/venv/bin/vllm', IMAGE, 'serve', '/model',
            '--host', '127.0.0.1', '--port', '18021', '--served-model-name', MODEL,
            '--language-model-only', '--dtype', 'half', '--kv-cache-dtype', 'auto',
            '--mamba-ssm-cache-dtype', 'float16', '--max-model-len', '4096',
            '--max-num-seqs', '3', '--max-num-batched-tokens', '1024',
            '--gpu-memory-utilization', '0.95', '--enforce-eager',
            '--enable-prefix-caching', '--reasoning-parser', 'qwen3',
            '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder',
            '--enable-prompt-tokens-details']
        if a.probe_script:
            launch += ['--logprobs-mode', 'raw_logprobs', '--max-logprobs', '20']
        if a.scheduler_hook:
            hooks = a.out/'hooks'; hooks.mkdir()
            (hooks/'sitecustomize.py').write_bytes(a.scheduler_hook.read_bytes())
            index = launch.index(IMAGE)
            launch[index:index] = ['-v',f'{hooks}:/probe:ro','-v',f'{a.out}:/probe-out:rw','-e','PYTHONPATH=/probe','-e','GPU_SCHEDULE_PROBE=/probe-out/scheduler.jsonl']
        (a.out/'launch-command.json').write_text(json.dumps(launch, indent=2))
        state['state'] = 'stopping_original_test_instance'; save()
        switched = True
        command('stop-original', ['podman', 'stop', '--time', '30', 'GPU_SERVICE'], timeout=90)
        state['state'] = 'starting_gpu35b'; save()
        # Set before run: even a timed-out CLI may have created the container.
        launched = True
        command('start-experiment', launch, timeout=180)
        wait_health('http://127.0.0.1:18021', 1200, name)
        (a.out/'gpu-loaded.csv').write_text(subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free', '--format=csv'], text=True))
        urllib.request.urlretrieve('http://127.0.0.1:18021/v1/models', a.out/'served-models.json')
        if a.probe_script:
            state['state'] = 'focused-probe'; save()
            command('focused-probe', [sys.executable, str(a.probe_script.resolve()), '--base-url', 'http://127.0.0.1:18021', '--model', MODEL, '--fixtures', str(source/'action-state-fixtures.json'), '--out', str(a.out/'probe')])
            state['state'] = 'focused_probe_complete_requires_adjudication'; save()
            return
        state['state'] = 'quality-before'; save()
        base = [sys.executable, str(source/'tools/check_cache_request_reuse.py'), '--base-url', 'http://127.0.0.1:18021', '--model', MODEL]
        command('quality-before', base+['--out', str(a.out/'quality-before.json')])
        quality = json.loads((a.out/'quality-before.json').read_text())
        state['strict_matches'] = quality['strict_matches']
        if quality['strict_matches'] != 8 or not quality['reuse']['identical_text'] or quality['reuse']['canary_leaked']:
            state['state'] = 'quality_gate_failed'; save()
            return
        state['state'] = 'business-facts'; save()
        command('business-facts', [sys.executable, str(source/'tools/check_business_facts.py'), '--base-url', 'http://127.0.0.1:18021', '--model', MODEL, '--cases', '8', '--concurrencies', '1', '3', '--out', str(a.out/'business-facts.jsonl')])
        state['state'] = 'business-quality'; save()
        command('business-quality', [sys.executable, str(source/'tools/check_business_quality.py'), '--url', 'http://127.0.0.1:18021', '--model', MODEL, '--fixtures-dir', str(source/'fixtures'), '--out', str(a.out/'business-quality.jsonl')])
        state['state'] = 'action-state-short'; save()
        original_fixtures = source/'action-state-fixtures.json'
        fixtures = json.loads(original_fixtures.read_text())
        fixtures['cases'] = [case for case in fixtures['cases'] if case['id'].startswith('short-')]
        assert len(fixtures['cases']) == 8
        fixtures['subset_source_sha256'] = hashlib.sha256(original_fixtures.read_bytes()).hexdigest()
        fixtures['subset_scope'] = 'Original 8 short cases only; 4K model screen cannot cover original 4K/16K/32K long prompts'
        short_fixture = a.out/'action-state-short-fixtures.json'
        short_fixture.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2))
        command('action-state-short', [sys.executable, str(source/'tools/check_action_state_regression.py'), '--base-url', 'http://127.0.0.1:18021', '--model', MODEL, '--fixtures', str(short_fixture), '--out', str(a.out/'action-state-short.jsonl')])
        state['state'] = 'quality-after'; save()
        command('quality-after', base+['--out', str(a.out/'quality-after.json')])
        after = json.loads((a.out/'quality-after.json').read_text())
        state['strict_matches_after'] = after['strict_matches']
        state['state'] = ('diagnostics_complete_requires_adjudication' if after['strict_matches'] == 8 and after['reuse']['identical_text'] and not after['reuse']['canary_leaked'] else 'quality_after_gate_failed')
        save()
    except Exception as exc:
        state.update(state='failed', error_type=type(exc).__name__, error=str(exc)); save()
    finally:
        if launched:
            try:
                with (a.out/'experiment-container.log').open('w') as log:
                    subprocess.run(['podman', 'logs', name], stdout=log, stderr=subprocess.STDOUT, timeout=30)
                stopped = subprocess.run(['podman', 'stop', '--time', '30', name], capture_output=True, text=True, timeout=90)
                (a.out/'stop-experiment.json').write_text(json.dumps({'returncode': stopped.returncode, 'stdout': stopped.stdout, 'stderr': stopped.stderr}, indent=2))
            except Exception as exc:
                state['experiment_stop_error'] = str(exc)
                # Restoration below must still be attempted if the log/stop CLI fails.
        if switched:
            restore = subprocess.run(['podman', 'start', 'GPU_SERVICE'], capture_output=True, text=True, timeout=90)
            restoration = {'start_returncode': restore.returncode, 'stdout': restore.stdout, 'stderr': restore.stderr}
            try:
                wait_health('http://127.0.0.1:18020', 1200, 'GPU_SERVICE')
                restoration['independent_health_200'] = True
            except Exception as exc:
                restoration.update(independent_health_200=False, error=str(exc))
            (a.out/'service-restoration.json').write_text(json.dumps(restoration, indent=2))
            state['original_service_restored'] = restoration['independent_health_200']
        else:
            state['original_service_unchanged_health_200'] = health('http://127.0.0.1:18020')
        save()
        print(json.dumps(state, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    main()
