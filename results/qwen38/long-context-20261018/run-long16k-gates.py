"""Run on a2 after the 16K graph launcher; preserve failures and raw results."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request

work = Path('/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018')


def mark(state):
    (work / 'long16k-gate-status.json').write_text(json.dumps({
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
                'qwen38-tp2-text-16k-graph'], capture_output=True, text=True)
            if state.stdout.strip() != 'true':
                raise RuntimeError('Graph container stopped before health')
            time.sleep(5)
    else:
        raise TimeoutError('Health timed out')
    mark('smoke')
    subprocess.run(['python3', str(work / 'qwen38_serving_smoke.py'),
                    '--output', str(work / 'long16k-smoke.json')], check=True)
    for tokens in (8064, 16256):
        mark(f'benchmark_{tokens}')
        subprocess.run([
            'python3', str(work / 'bench_http_stream.py'),
            '--base-url', 'http://127.0.0.1:18038',
            '--model', 'qwen38-w4a8-tp2-text',
            '--out', str(work / f'long16k-bench-{tokens}-128.jsonl'),
            '--prompt-tokens', str(tokens), '--output-tokens', '128',
            '--repeats', '3'], check=True)
    mark('needle_16256')
    needle = work / 'long16k-needle-16256.jsonl'
    subprocess.run([
        'python3', str(work / 'check_long_context_needle.py'),
        '--base-url', 'http://127.0.0.1:18038', '--model', 'qwen38-w4a8-tp2-text',
        '--out', str(needle), '--prompt-tokens', '16256',
        '--needle-fraction', '.5', '--max-tokens', '64'], check=True)
    row = json.loads(needle.read_text().splitlines()[-1])
    assert row['kind'] == 'result' and row['retrieved'], '16K retrieval failed'
    mark('refresh_after_long')
    subprocess.run(['python3', str(work / 'qwen38_graph_request_gate.py'),
                    '--output', str(work / 'long16k-request-refresh.json')], check=True)
    mark('all_gates_completed')
except Exception as error:
    mark('failed: ' + repr(error))
    raise
