"""Run on a2 after the separate graph launcher; preserve failures and raw results."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request

work = Path('/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018')


def mark(state):
    (work / 'graph-gate-status.json').write_text(json.dumps({
        'state': state, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}))


try:
    mark('waiting_health')
    for _ in range(120):
        try:
            with urllib.request.urlopen('http://127.0.0.1:18038/health', timeout=3) as response:
                assert response.status == 200
            break
        except Exception:
            state = subprocess.run([
                'podman', 'inspect', '--format', '{{.State.Running}}',
                'qwen38-tp2-text-4k-graph'], capture_output=True, text=True)
            if state.stdout.strip() != 'true':
                raise RuntimeError('Graph container stopped before health')
            time.sleep(5)
    else:
        raise TimeoutError('Health timed out')
    mark('smoke')
    subprocess.run(['python3', str(work / 'qwen38_serving_smoke.py'),
                    '--output', str(work / 'graph-smoke.json')], check=True)
    for tokens in (512, 3968):
        mark(f'benchmark_{tokens}')
        subprocess.run([
            'python3', str(work / 'bench_http_stream.py'),
            '--base-url', 'http://127.0.0.1:18038',
            '--model', 'qwen38-w4a8-tp2-text',
            '--out', str(work / f'graph-bench-{tokens}-128.jsonl'),
            '--prompt-tokens', str(tokens), '--output-tokens', '128',
            '--repeats', '3'], check=True)
    mark('all_gates_completed')
except Exception as error:
    mark('failed: ' + repr(error))
    raise
