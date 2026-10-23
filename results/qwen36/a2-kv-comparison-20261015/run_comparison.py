"""本轮 a2 device0 顺序对照；在仓库根目录执行，不触碰 device1。"""
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from check_meeting_long_context import complete, score
from bench_server_phases import snapshot, prefix_snapshot, METRICS, SPEC_METRICS, prompt_hash

OUT = Path(__file__).resolve().parent
BASE = 'http://ychy-a2:18024'
MODEL = 'qwen36-w4a8-c8'  # 兼容原服务名；以实际环境判断 KV 格式。
REMOTE = '/home/ychy/services/qwen36/kv-ab-20261015'


def ssh(command):
    return subprocess.check_output(['ssh', 'ychy-a2', command], text=True)


def meeting(name):
    with gzip.open(ROOT / 'results/orangepi-rc-batch4096-20261015/meeting-112000-prompt.json.gz', 'rt') as stream:
        tokens = json.load(stream)['prompt_ids']
    before = snapshot(BASE, True)
    cache_before = prefix_snapshot(BASE)
    result, elapsed = complete(BASE, MODEL, tokens)
    after = snapshot(BASE, True)
    cache_after = prefix_snapshot(BASE)
    delta = {key: after[key] - before[key] for key in before}
    row = {
        'scope': 'one synthetic meeting-update canary, not formal accuracy/P95',
        'prompt_tokens': len(tokens), 'prompt_ids_sha256': prompt_hash(tokens),
        'result': result, 'elapsed_s': elapsed, 'score': score(result['choices'][0]['text']),
        'metric_delta': delta,
        'prefix_cache_delta': {key: cache_after[key] - cache_before[key] for key in cache_before},
        'prefill_tok_s': len(tokens) / delta[METRICS[1]],
        'decode_tok_s': (result['usage']['completion_tokens'] - 1) / delta[METRICS[2]],
        'mtp_acceptance': delta[SPEC_METRICS[2]] / delta[SPEC_METRICS[1]] if delta[SPEC_METRICS[1]] else None,
    }
    (OUT / f'{name}-meeting-112000.json').write_text(json.dumps(row, ensure_ascii=False, indent=2) + '\n')
    assert row['score']['passed'] and result['usage']['prompt_tokens'] == len(tokens)
    assert delta[METRICS[0]] == 1 and row['prefix_cache_delta']['vllm:prefix_cache_hits_total'] == 0
    print(name, 'meeting passed', flush=True)


def main():
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        file = OUT / 'c8-p112000-o128.jsonl'
        if file.exists():
            rows = [json.loads(line) for line in file.read_text().splitlines()]
            if any(row.get('kind') == 'request' and row.get('valid') for row in rows):
                break
        time.sleep(5)
    else:
        raise RuntimeError('C8 timing did not complete')
    meeting('c8')
    (OUT / 'c8-startup-and-validation.txt').write_text(ssh('podman logs qwen36-tp1-48g-112k 2>&1'))
    (OUT / 'c8-memory-after.txt').write_text(ssh('npu-smi info'))
    changed = passed = False
    try:
        # 必须先停管理单元，否则Restart=on-failure会反复创建旧C8容器。
        active = subprocess.run(['ssh', 'ychy-a2', 'systemctl --user is-active qwen36-a2.service'],
                                capture_output=True, text=True).stdout.strip() == 'active'
        if active:
            ssh('systemctl --user stop qwen36-a2.service')
        else:
            ssh('podman stop -t 20 qwen36-tp1-48g-112k')
        changed = True
        print(ssh(f'bash {REMOTE}/start-c16.sh'), flush=True)
        deadline = time.monotonic() + 1000
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(BASE + '/health', timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(5)
        else:
            raise RuntimeError('C16 startup timeout')
        subprocess.run([sys.executable, str(ROOT / 'tools/check_cache_request_reuse.py'),
                        '--base-url', BASE, '--model', MODEL, '--out', str(OUT / 'c16-quality.json')], check=True)
        quality = json.loads((OUT / 'c16-quality.json').read_text())
        assert quality['strict_matches'] == 8 and quality['reuse']['identical_text'] and not quality['reuse']['canary_leaked']
        for length, repeats in ((8192, 3), (32768, 3), (112000, 1)):
            subprocess.run([sys.executable, str(ROOT / 'tools/bench_server_phases.py'),
                            '--base-url', BASE, '--model', MODEL,
                            '--out', str(OUT / f'c16-p{length}-o128.jsonl'),
                            '--prompt-tokens', str(length), '--output-tokens', '128',
                            '--repeats', str(repeats), '--unique-prompts', '--spec-metrics',
                            '--cache-salt-prefix', f'a2-kv-c16-{length}'], check=True, timeout=1500)
        meeting('c16')
        passed = True
    finally:
        if changed:
            for file, command in (
                ('c16-startup-and-validation.txt', 'podman logs qwen36-c16-kv-ab-20261015'),
                ('c16-memory-after.txt', 'npu-smi info'),
            ):
                log = subprocess.run(['ssh', 'ychy-a2', command], text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                (OUT / file).write_text(log.stdout)
            if not passed:
                ssh(f'podman stop -t 20 qwen36-c16-kv-ab-20261015; bash {REMOTE}/restore-c8.sh')
                print('C16 did not pass; original C8 command restored, health still needs checking', flush=True)
    print('A2 C16 passed; candidate remains on device0:18024; device1 unchanged', flush=True)


if __name__ == '__main__':
    main()
