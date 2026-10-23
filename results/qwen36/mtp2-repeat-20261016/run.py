"""RC MTP2 draft eager重复冷请求；复用冻结工具，结束恢复原服务。"""
import gzip
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path('/data/qwen36-deploy/results/mtp2-repeat-20261016')
HISTORY = Path('/data/qwen36-deploy/results/mtp2-history-20261016')
spec = importlib.util.spec_from_file_location('history_run', HISTORY / 'run.py')
history = importlib.util.module_from_spec(spec)
spec.loader.exec_module(history)
previous = history.previous


def stage(value):
    (ROOT / 'state.txt').write_text(value)
    print(value, flush=True)


def main():
    ROOT.mkdir(exist_ok=True)
    name = 'qwen36-mtp2-repeat-eager-20261016'
    active = False
    passed = False
    try:
        history.ROOT = ROOT
        history.clone(name, True)
        previous.run('docker', 'stop', '-t', '30', previous.STABLE)
        active = True
        stage('startup')
        previous.run('docker', 'start', name)
        previous.health()
        previous.quality(ROOT / 'quality-before.json')
        with gzip.open(HISTORY / 'inputs.json.gz', 'rt') as f:
            data = json.load(f)
        history.request(history.BASE, 'qwen36-w4a8', data[0][:3072], 32,
                        cache_salt='repeat-eager-warmup')
        for cycle in range(3):
            if cycle == 1:
                stage('meeting32256')
                previous.canary(ROOT / 'meeting-32256.json', 32256)
            for index, ids in enumerate(data):
                stage(f'cycle{cycle}:input{index}')
                history.measured(ROOT / f'cycle{cycle}-input{index}.json', ids,
                                 f'repeat-eager-{cycle}-{index}')
        previous.quality(ROOT / 'quality-after.json')
        passed = True
    except Exception as exc:
        (ROOT / 'failure.txt').write_text(repr(exc))
        raise
    finally:
        if active:
            subprocess.run(['docker', 'stop', '-t', '30', name], timeout=90)
            logs = previous.run('docker', 'logs', name, capture_output=True)
            with gzip.open(ROOT / 'service.log.gz', 'wt') as f:
                f.write(logs.stdout + logs.stderr)
        stage('restoring-mtp1')
        previous.run('docker', 'start', previous.STABLE)
        previous.health()
        previous.quality(ROOT / 'restored-mtp1-quality.json')
        stage('completed-restored' if passed else 'failed-restored')


if __name__ == '__main__':
    main()
