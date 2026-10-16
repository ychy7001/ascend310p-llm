"""Baseline and standalone-image gates for EP310P1, restoring the original container."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path('/home/clt/qwen36-runtime-image-20261016')
TOOLS = ROOT / 'tools'
MODEL = 'qwen36-w4a8-c8'
IMAGE = 'registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend:qwen36-c9fd1da-amd64'
STABLE = 'qwen36-c16-112k-20261015'
CANDIDATE = 'qwen36-image-c9fd1da-clt-20261016'
SERVICE = 'qwen36-a2.service'
BASE = 'http://127.0.0.1:18016'
TEST = 'http://127.0.0.1:18026'
sys.path.insert(0, str(TOOLS))
import check_meeting_long_context as meeting


def run(*args, **kwargs):
    return subprocess.run(args, text=True, check=True, **kwargs)


def stage(name):
    (ROOT / 'state.txt').write_text(name)
    print(name, flush=True)


def health(base):
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base + '/health', timeout=3) as response:
                assert response.status == 200
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError('health timeout: ' + base)


def quality(base, path):
    run(sys.executable, str(TOOLS / 'check_cache_request_reuse.py'), '--base-url', base,
        '--model', MODEL, '--out', str(path), timeout=300)
    result = json.loads(path.read_text())
    assert result['strict_matches'] == 8 and result['reuse']['identical_text'] and not result['reuse']['canary_leaked']


def benchmark(base, path):
    run(sys.executable, str(TOOLS / 'bench_server_phases.py'), '--base-url', base,
        '--model', MODEL, '--out', str(path), '--prompt-tokens', '32768', '--output-tokens', '128',
        '--repeats', '2', '--unique-prompts', '--spec-metrics', '--cache-salt-prefix', 'image-ab', timeout=1200)


def canary(path):
    tokens, positions = meeting.make_prompt(TEST, MODEL, 112000)
    with gzip.open(ROOT / 'meeting-112000.prompt.json.gz', 'xt') as f:
        json.dump([tokens, positions], f)
    before = meeting.snapshot(TEST, True)
    response, elapsed = meeting.complete(TEST, MODEL, tokens)
    after = meeting.snapshot(TEST, True)
    delta = {k: after[k] - before[k] for k in before}
    usage = response['usage']
    score = meeting.score(response['choices'][0]['text'])
    result = dict(response=response, elapsed_s=elapsed, score=score, positions=positions,
                  prompt_ids_sha256=meeting.prompt_hash(tokens), metric_delta=delta,
                  prefill_tok_s=len(tokens)/delta[meeting.METRICS[1]],
                  decode_tok_s=(usage['completion_tokens']-1)/delta[meeting.METRICS[2]])
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    assert usage['prompt_tokens'] == 112000 and score['passed']


def create():
    meta = json.loads(run('docker', 'inspect', STABLE, capture_output=True).stdout)[0]
    (ROOT / 'baseline-inspect.json').write_text(json.dumps(meta, indent=2))
    env = dict(x.split('=', 1) for x in meta['Config']['Env'])
    env.update(PORT='18026', ASYNC_SCHEDULING='0', LANGUAGE_MODEL_ONLY='1', QWEN36_SYNC_METADATA='1')
    args = ['docker', 'create', '--name', CANDIDATE, '--network', 'host', '--shm-size', '16g']
    if meta['HostConfig'].get('Privileged'):
        args += ['--privileged']
    allowed = {'/model', '/usr/local/Ascend/driver/lib64', '/usr/local/Ascend/driver/version.info', '/usr/local/dcmi', '/etc/ascend_install.info'}
    for bind in meta['HostConfig']['Binds']:
        if bind.split(':')[1] in allowed:
            args += ['-v', bind]
    for device in ('/dev/davinci0', '/dev/davinci_manager', '/dev/devmm_svm', '/dev/hisi_hdc'):
        args += ['--device', device]
    for key, value in env.items():
        args += ['-e', key + '=' + value]
    # No command override: use the image's versioned entrypoint.
    run(*args, IMAGE)
    candidate = json.loads(run('docker', 'inspect', CANDIDATE, capture_output=True).stdout)[0]
    destinations = {x['Destination'] for x in candidate['Mounts']}
    assert destinations == allowed, destinations
    (ROOT / 'candidate-runtime.json').write_text(json.dumps({
        'image_id': candidate['Image'], 'binds': candidate['HostConfig']['Binds'],
        'entrypoint': candidate['Config']['Entrypoint'], 'cmd': candidate['Config']['Cmd'],
        'environment': {k: v for k, v in env.items() if k.startswith(('QWEN36_', 'GRAPH_', 'MAX_', 'NUM_', 'MEMORY_', 'ENABLE_', 'ASYNC_', 'LANGUAGE_', 'MTP', 'OMP_', 'MKL_', 'ENFORCE_'))},
    }, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('baseline', 'candidate'))
    args = parser.parse_args()
    if args.phase == 'baseline':
        (ROOT / 'baseline').mkdir()
        health(BASE)
        stage('baseline:quality')
        quality(BASE, ROOT / 'baseline/quality-before.json')
        stage('baseline:32k')
        benchmark(BASE, ROOT / 'baseline/p32768-o128.jsonl')
        quality(BASE, ROOT / 'baseline/quality-after.json')
        (ROOT / 'baseline-passed.json').write_text('{"passed":true}')
        stage('baseline-completed')
        return
    assert json.loads((ROOT / 'baseline-passed.json').read_text())['passed']
    active = False
    result = {'passed': False}
    try:
        (ROOT / 'image').mkdir()
        create()
        active = True
        run('docker', 'stop', '-t', '30', STABLE)
        stage('image:startup')
        run('docker', 'start', CANDIDATE)
        health(TEST)
        stage('image:quality')
        quality(TEST, ROOT / 'image/quality-before.json')
        stage('image:32k')
        benchmark(TEST, ROOT / 'image/p32768-o128.jsonl')
        stage('image:112k-meeting')
        canary(ROOT / 'image/meeting-112k.json')
        quality(TEST, ROOT / 'image/quality-after.json')
        result['passed'] = True
    except Exception as error:
        result['failed_stage'] = (ROOT / 'state.txt').read_text()
        result['error'] = repr(error)
        print(repr(error), flush=True)
    finally:
        if active:
            run('docker', 'stop', '-t', '30', CANDIDATE, timeout=90)
            logs = run('docker', 'logs', CANDIDATE, capture_output=True)
            with gzip.open(ROOT / 'image-service.log.gz', 'wt') as f:
                f.write(logs.stdout + logs.stderr)
            stage('restoring-baseline')
            run('docker', 'start', STABLE)
            health(BASE)
        quality(BASE, ROOT / 'restored-quality.json')
        result['restored_quality_passed'] = True
        result['restored_health_status'] = 200
        result['running_containers'] = run('docker', 'ps', '--format', '{{.Names}}', capture_output=True).stdout.splitlines()
        (ROOT / 'outcome.json').write_text(json.dumps(result, indent=2))
        stage('completed-restored' if result['passed'] else 'failed-restored')


if __name__ == '__main__':
    main()
