"""clt EP收束复验；原服务基准后测试独立MTP2候选，最后恢复原容器。"""
import gzip
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path('/home/clt/qwen36-ep-convergence-20261016')
TOOLS = ROOT / 'tools'
BASE = 'http://127.0.0.1:18016'
MODEL = 'qwen36-w4a8-c8'
STABLE = 'qwen36-c16-112k-20261015'
CANDIDATE = 'qwen36-ep-mtp2-slot-20261016'
sys.path.insert(0, str(TOOLS))
import check_meeting_long_context as meeting


def run(*args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def stage(value):
    (ROOT / 'state.txt').write_text(value)
    print(value, flush=True)


def health():
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(BASE + '/health', timeout=3) as response:
                assert response.status == 200
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError('health startup timeout')


def quality(out):
    run(sys.executable, str(TOOLS / 'check_cache_request_reuse.py'),
        '--base-url', BASE, '--model', MODEL, '--out', str(out), timeout=300)
    d = json.loads(out.read_text())
    assert d['strict_matches'] == 8 and d['reuse']['identical_text'] and not d['reuse']['canary_leaked'], 'quality gate failed'


def benchmark(out, tokens, repeats):
    run(sys.executable, str(TOOLS / 'bench_server_phases.py'), '--base-url', BASE,
        '--model', MODEL, '--out', str(out), '--prompt-tokens', str(tokens),
        '--output-tokens', '128', '--repeats', str(repeats), '--unique-prompts',
        '--spec-metrics', '--cache-salt-prefix', out.stem, timeout=1800)


def canary(out):
    saved = ROOT / 'meeting-112000.prompt.json.gz'
    if saved.exists():
        with gzip.open(saved, 'rt') as f:
            tokens, positions = json.load(f)
    else:
        tokens, positions = meeting.make_prompt(BASE, MODEL, 112000)
        with gzip.open(saved, 'xt') as f:
            json.dump([tokens, positions], f)
    before = meeting.snapshot(BASE, True)
    response, elapsed = meeting.complete(BASE, MODEL, tokens)
    after = meeting.snapshot(BASE, True)
    delta = {k: after[k] - before[k] for k in before}
    usage = response['usage']
    scored = meeting.score(response['choices'][0]['text'])
    result = dict(prompt_ids_sha256=meeting.prompt_hash(tokens), response=response,
                  score=scored, elapsed_s=elapsed, positions=positions, metric_delta=delta,
                  prefill_tok_s=len(tokens)/delta[meeting.METRICS[1]],
                  decode_tok_s=(usage['completion_tokens']-1)/delta[meeting.METRICS[2]])
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    assert usage['prompt_tokens'] == 112000 and scored['passed']


def clone():
    meta = json.loads((ROOT / 'baseline-inspect.json').read_text())[0]
    hc = meta['HostConfig']
    args = ['docker', 'create', '--name', CANDIDATE, '--network', hc['NetworkMode'],
            '--ipc', hc['IpcMode'], '--shm-size', str(hc['ShmSize'])]
    if hc.get('PidMode'):
        args += ['--pid', hc['PidMode']]
    if hc['Privileged']:
        args += ['--privileged']
    for d in hc.get('Devices') or []:
        args += ['--device', f"{d['PathOnHost']}:{d['PathInContainer']}:{d['CgroupPermissions']}"]
    for bind in hc.get('Binds') or []:
        args += ['-v', bind]
    args += ['-v', f'{ROOT}:/experiment:ro', '-v',
             f'{ROOT}/proposer-slot.py:/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py:ro']
    env = dict(e.split('=', 1) for e in meta['Config']['Env'])
    env.update(NUM_SPECULATIVE_TOKENS='2', GRAPH_CAPTURE_SIZES='1,2,3',
               QWEN36_C8_ALLOW_MTP2='1', PYTHONPATH='/experiment/candidate_bootstrap:/experiment:/c8bootstrap')
    for k, v in env.items():
        args += ['-e', f'{k}={v}']
    args += ['--entrypoint', 'bash', meta['Image'], *meta['Config']['Cmd']]
    run(*args)
    (ROOT / 'candidate-env.json').write_text(json.dumps(env, indent=2))


def validate(label):
    out = ROOT / label
    out.mkdir()
    stage(label + ':quality-before')
    quality(out / 'quality-before.json')
    stage(label + ':32k')
    benchmark(out / 'p32768-o128.jsonl', 32768, 2)
    stage(label + ':112k')
    benchmark(out / 'p112000-o128.jsonl', 112000, 1)
    stage(label + ':meeting112k')
    canary(out / 'meeting-112k.json')
    quality(out / 'quality-after.json')
    (out / 'passed.json').write_text(json.dumps({'passed': True, 'scope': 'short;32K warmup+2;112K warmup+1;112K meeting;short'}))


def main():
    active = False
    result = {'passed': False}
    try:
        health()
        validate('mtp1')
        clone()
        run('docker', 'stop', '-t', '30', STABLE)
        active = True
        stage('mtp2:startup')
        run('docker', 'start', CANDIDATE)
        health()
        validate('mtp2')
        result['passed'] = True
    except Exception as error:
        result['error'] = repr(error)
        print(repr(error), flush=True)
    finally:
        if active:
            run('docker', 'stop', '-t', '30', CANDIDATE, timeout=90)
            logs = run('docker', 'logs', CANDIDATE, capture_output=True)
            with gzip.open(ROOT / 'candidate-service.log.gz', 'wt') as f:
                f.write(logs.stdout + logs.stderr)
            stage('restoring-mtp1')
            run('docker', 'start', STABLE)
            health()
        quality(ROOT / 'restored-mtp1-quality.json')
        with urllib.request.urlopen(BASE + '/health', timeout=10) as response:
            result['restored_health_status'] = response.status
        result['restored_quality_passed'] = True
        result['running_containers'] = run('docker', 'ps', '--format', '{{.Names}}', capture_output=True).stdout.splitlines()
        (ROOT / 'outcome.json').write_text(json.dumps(result, indent=2))
        stage('completed-restored' if result['passed'] else 'candidate-failed-restored')


if __name__ == '__main__':
    main()
