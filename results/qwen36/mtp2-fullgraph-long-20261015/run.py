"""RC 独占顺序验收；失败自动恢复稳定服务，所有候选保留。"""
import argparse
import gzip
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path('/data/qwen36-deploy/results/mtp2-fullgraph-long-20261015')
TOOLS = ROOT / 'tools'
BASE = 'http://127.0.0.1:18024'
STABLE = 'qwen36-orangepi'
SOURCE = 'qwen36-mtp2-slot-device-fullgraph-32k-20261015'
sys.path.insert(0, str(TOOLS))
import check_meeting_long_context as meeting


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def stage(name):
    (ROOT / 'state.txt').write_text(name)
    print(name, flush=True)


def health():
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(BASE + '/health', timeout=3):
                return
        except Exception:
            time.sleep(5)
    raise TimeoutError('health startup timeout')


def quality(path):
    run(sys.executable, str(TOOLS / 'check_cache_request_reuse-fixed.py'),
        '--base-url', BASE, '--model', 'qwen36-w4a8', '--out', str(path), timeout=240)
    d = json.loads(path.read_text())
    assert d['strict_matches'] == 8 and d['reuse']['identical_text'] and not d['reuse']['canary_leaked']


def canary(out, count):
    saved = ROOT / f'meeting-{count}.prompt.json.gz'
    if saved.exists():
        with gzip.open(saved, 'rt') as f:
            tokens = json.load(f)['prompt_ids']
    else:
        tokens, positions = meeting.make_prompt(BASE, 'qwen36-w4a8', count)
        with gzip.open(saved, 'xt') as f:
            json.dump({'prompt_ids': tokens, 'record_positions': positions}, f)
    before = meeting.snapshot(BASE, include_spec=True)
    result, elapsed = meeting.complete(BASE, 'qwen36-w4a8', tokens)
    after = meeting.snapshot(BASE, include_spec=True)
    delta = {k: after[k] - before[k] for k in before}
    assert delta[meeting.METRICS[0]] == 1 and result['usage']['prompt_tokens'] == count
    text = result['choices'][0]['text']
    row = {'prompt_ids_sha256': meeting.prompt_hash(tokens), 'result': result,
           'elapsed_s': elapsed, 'metric_delta': delta, 'score': meeting.score(text),
           'prefill_tok_s': count / delta[meeting.METRICS[1]],
           'decode_tok_s': (result['usage']['completion_tokens'] - 1) / delta[meeting.METRICS[2]]}
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2))
    print(json.dumps({k: row[k] for k in ('score', 'prefill_tok_s', 'decode_tok_s')}, ensure_ascii=False), flush=True)
    assert row['score']['passed'], 'meeting canary failed'


def clone(name, mtp):
    meta = json.loads(run('docker', 'inspect', SOURCE, capture_output=True).stdout)[0]
    args = ['docker', 'create', '--name', name, '--network', 'host']
    hc = meta['HostConfig']
    for key, flag in (('IpcMode', '--ipc'), ('PidMode', '--pid')):
        if hc.get(key):
            args += [flag, hc[key]]
    if hc['Privileged']:
        args += ['--privileged']
    if hc.get('ShmSize'):
        args += ['--shm-size', str(hc['ShmSize'])]
    for d in hc.get('Devices') or []:
        args += ['--device', f"{d['PathOnHost']}:{d['PathInContainer']}:{d['CgroupPermissions']}"]
    for bind in hc.get('Binds') or []:
        args += ['-v', bind]
    env = dict(e.split('=', 1) for e in meta['Config']['Env'])
    env.update(MTP=str(mtp), MAX_MODEL_LEN='114688', KV_BYTES=str(4 * 1024**3), MAX_BATCHED_TOKENS='2048')
    for k, v in env.items():
        args += ['-e', f'{k}={v}']
    args += ['--entrypoint', 'bash', meta['Config']['Image'], *meta['Config']['Cmd']]
    run(*args)
    (ROOT / f'{name}.env.json').write_text(json.dumps(env, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mtp', type=int, choices=(1, 2), nargs='+', default=[2, 1])
    parser.add_argument('--resume-startup', action='store_true', help='复用已创建且尚未开始质量测试的候选')
    parser.add_argument('--skip-long-bench', action='store_true', help='长门禁失败后，仅保留112K会议诊断')
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    active = None
    failure = None
    try:
        for index, mtp in enumerate(args.mtp):
            name = f'qwen36-mtp{mtp}-matched-fullgraph-112k-20261015'
            if not args.resume_startup:
                clone(name, mtp)
            if index == 0:
                run('docker', 'stop', '-t', '30', STABLE)
            active = name
            out = ROOT / f'mtp{mtp}'
            out.mkdir(exist_ok=args.resume_startup)
            stage(f'mtp{mtp}:startup')
            run('docker', 'start', name)
            health()
            quality(out / 'quality-before.json')
            stage(f'mtp{mtp}:meeting32k')
            canary(out / 'meeting-32256.json', 32256)
            for count in ((32256,) if args.skip_long_bench else (32256, 114176)):
                stage(f'mtp{mtp}:bench{count}')
                run(sys.executable, str(TOOLS / 'bench_server_phases.py'), '--base-url', BASE,
                    '--model', 'qwen36-w4a8', '--prompt-tokens', str(count), '--output-tokens', '128',
                    '--repeats', '2', '--unique-prompts', '--spec-metrics',
                    '--cache-salt-prefix', f'fullgraph-mtp{mtp}-{count}',
                    '--out', str(out / f'bench-{count}.jsonl'), timeout=1800)
            stage(f'mtp{mtp}:meeting112k')
            canary(out / 'meeting-114176.json', 114176)
            quality(out / 'quality-after.json')
            run('docker', 'stop', '-t', '30', name)
            with gzip.open(out / 'service.log.gz', 'wt') as log:
                logs = run('docker', 'logs', name, capture_output=True)
                log.write(logs.stdout + logs.stderr)
            active = None
    except Exception as exc:
        failure = repr(exc)
        (ROOT / 'failure.txt').write_text(failure)
        raise
    finally:
        if active:
            subprocess.run(['docker', 'stop', '-t', '30', active], timeout=90)
            logs = subprocess.run(['docker', 'logs', active], capture_output=True, text=True)
            with gzip.open(ROOT / f'{active}-failure.log.gz', 'wt') as log:
                log.write(logs.stdout + logs.stderr)
        stage('restoring-mtp1')
        run('docker', 'start', STABLE)
        health()
        quality(ROOT / 'restored-mtp1-quality.json')
        stage('failed-restored' if failure else 'completed-restored')


if __name__ == '__main__':
    main()
