"""固定RC 112K配置，比较draft Graph与请求历史；自动恢复原服务。"""
import gzip
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path('/data/qwen36-deploy/results/mtp2-history-20261016')
PREVIOUS = Path('/data/qwen36-deploy/results/mtp2-fullgraph-long-20261015')
spec = importlib.util.spec_from_file_location('previous_run', PREVIOUS / 'run.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
from bench_http_stream import post, request
from bench_server_phases import prompt_hash, prefix_snapshot, METRICS

BASE = previous.BASE
SOURCE = 'qwen36-mtp2-matched-fullgraph-112k-20261015'


def stage(value):
    (ROOT / 'state.txt').write_text(value)
    print(value, flush=True)


def snapshot():
    raw = previous.run('curl', '-fsS', '--max-time', '30', BASE + '/metrics', capture_output=True).stdout
    values = {}
    for line in raw.splitlines():
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split()[0]
        if name in METRICS or name in previous.meeting.SPEC_METRICS:
            values[name] = values.get(name, 0) + float(line.split()[-1])
        if name == 'vllm:spec_decode_num_accepted_tokens_per_pos_total':
            pos = re.search(r'position="(\d+)"', line).group(1)
            values['accepted_pos_' + pos] = float(line.split()[-1])
    assert all(k in values for k in (*METRICS, *previous.meeting.SPEC_METRICS, 'accepted_pos_0', 'accepted_pos_1'))
    return values


def clone(name, draft_eager):
    meta = json.loads(previous.run('docker', 'inspect', SOURCE, capture_output=True).stdout)[0]
    hc = meta['HostConfig']
    args = ['docker', 'create', '--name', name, '--network', hc['NetworkMode'],
            '--ipc', hc['IpcMode'], '--pid', hc['PidMode'], '--shm-size', str(hc['ShmSize'])]
    if hc['Privileged']:
        args += ['--privileged']
    for d in hc.get('Devices') or []:
        args += ['--device', f"{d['PathOnHost']}:{d['PathInContainer']}:{d['CgroupPermissions']}"]
    for bind in hc.get('Binds') or []:
        args += ['-v', bind]
    env = dict(e.split('=', 1) for e in meta['Config']['Env'])
    env['MTP_DRAFT_EAGER'] = str(int(draft_eager))
    for k, v in env.items():
        args += ['-e', f'{k}={v}']
    args += ['--entrypoint', 'bash', meta['Config']['Image'], *meta['Config']['Cmd']]
    previous.run(*args)
    (ROOT / f'{name}.env.json').write_text(json.dumps(env, indent=2))


def tokens():
    saved = ROOT / 'inputs.json.gz'
    if saved.exists():
        with gzip.open(saved, 'rt') as f:
            return json.load(f)
    data = []
    for variant in (1, 2):
        unit = f'Benchmark cold-prefill variant {variant}. Explain how quantization affects memory bandwidth and numerical precision. '
        with post(BASE, '/tokenize', {'model': 'qwen36-w4a8', 'prompt': unit, 'add_special_tokens': False}) as response:
            ids = json.load(response)['tokens']
        data.append((ids * ((32256 + len(ids) - 1) // len(ids)))[:32256])
    with gzip.open(saved, 'xt') as f:
        json.dump(data, f)
    return data


def measured(out, ids, salt):
    before, cache_before = snapshot(), prefix_snapshot(BASE)
    result = request(BASE, 'qwen36-w4a8', ids, 128, cache_salt=salt)
    after, cache_after = snapshot(), prefix_snapshot(BASE)
    delta = {k: after[k] - before[k] for k in before}
    assert delta[METRICS[0]] == 1 and cache_after['vllm:prefix_cache_hits_total'] == cache_before['vllm:prefix_cache_hits_total']
    row = {'prompt_ids_sha256': prompt_hash(ids), 'response': result, 'metric_delta': delta,
           'cache_before': cache_before, 'cache_after': cache_after,
           'prefill_tok_s': len(ids) / delta[METRICS[1]], 'decode_tok_s': 127 / delta[METRICS[2]]}
    out.write_text(json.dumps(row, ensure_ascii=False))
    print(json.dumps({k: v for k, v in row.items() if k not in ('response', 'cache_before', 'cache_after')}, ensure_ascii=False), flush=True)


def main():
    ROOT.mkdir(exist_ok=True)
    active = None
    passed = False
    try:
        previous.run('docker', 'stop', '-t', '30', previous.STABLE)
        for label, eager in (('fullgraph', False), ('draft-eager', True)):
            active = f'qwen36-mtp2-history-{label}-20261016'
            clone(active, eager)
            out = ROOT / label
            out.mkdir()
            stage(label + ':startup')
            previous.run('docker', 'start', active)
            previous.health()
            previous.quality(out / 'quality-before.json')
            data = tokens()
            # Explicit short warmup, identical for both modes; all long inputs are cold.
            request(BASE, 'qwen36-w4a8', data[0][:3072], 32, cache_salt=label + '-warmup')
            for phase in ('before', 'after'):
                if phase == 'after':
                    stage(label + ':meeting32256')
                    previous.canary(out / 'meeting-32256.json', 32256)
                for i, ids in enumerate(data):
                    stage(f'{label}:{phase}:{i}')
                    measured(out / f'{phase}-{i}.json', ids, f'history-{label}-{phase}-{i}')
            previous.quality(out / 'quality-after.json')
            previous.run('docker', 'stop', '-t', '30', active)
            logs = previous.run('docker', 'logs', active, capture_output=True)
            with gzip.open(out / 'service.log.gz', 'wt') as f:
                f.write(logs.stdout + logs.stderr)
            active = None
        passed = True
    except Exception as exc:
        (ROOT / 'failure.txt').write_text(repr(exc))
        raise
    finally:
        if active:
            subprocess.run(['docker', 'stop', '-t', '30', active], timeout=90)
            logs = previous.run('docker', 'logs', active, capture_output=True)
            with gzip.open(ROOT / (active + '-failure.log.gz'), 'wt') as f:
                f.write(logs.stdout + logs.stderr)
        stage('restoring-mtp1')
        previous.run('docker', 'start', previous.STABLE)
        previous.health()
        previous.quality(ROOT / 'restored-mtp1-quality.json')
        stage('completed-restored' if passed else 'failed-restored')


if __name__ == '__main__':
    main()
