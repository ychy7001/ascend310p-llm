"""Verify the exact main-Graph/eager-draft image after a clean full window."""
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

ROOT = Path('/home/ychy/ascend-lab-runtime/hyperqwen-20261103')
base = ROOT / 'graph-localize'
work = ROOT / 'wy-mtp-production-20261108'
full = base / 'main-graph-release-full'
out = base / 'main-graph-release-packaged'
report = base / 'main-graph-release-closure.json'
image = 'localhost/qwen36-w8-wy-mtp-main-graph:20261108'


def record(**data):
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


try:
    deadline = time.monotonic() + 3600
    while True:
        status = json.loads((full / 'status.json').read_text())
        if status['state'] == 'failed':
            raise RuntimeError('full qualification failed; packaged window canceled')
        if status['state'] == 'complete':
            assert status['cleanup_passed']
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('qualification wait budget exceeded; no new device action')
        record(state='waiting-full-qualification')
        time.sleep(20)
    markers = []
    seen = set()
    for line in (full / 'server.log').read_text(errors='replace').splitlines():
        if '[graph-lifecycle] ' in line:
            row = json.loads(line.split('[graph-lifecycle] ', 1)[1])
            rank = re.search(r'Worker_TP([01])', line)
            assert rank is not None
            row['rank'] = int(rank.group(1))
            markers.append(row)
            seen.add((row['rank'], row['role'], row['event']))
    assert all((rank, 'main', event) in seen for rank in (0, 1) for event in ('capture', 'replay')), seen
    assert not any(row['role'] == 'draft' for row in markers)
    (full / 'graph-lifecycle.json').write_text(json.dumps(markers, indent=2) + '\n')
    record(state='running-packaged')
    with (base / 'main-graph-release-packaged-driver.log').open('w') as log:
        subprocess.run(['python3', str(work / 'run_profile_graph_packaged.py'),
                        '--out', str(out), '--image', image, '--mode', 'packaged',
                        '--eager', '0', '--draft-eager', '1', '--seqs', '4',
                        '--served-model-name', 'Qwen3.6-35B-A3B-w8a8',
                        '--cooldown-after', str(full / 'window-closed.json')],
                       check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    status = json.loads((out / 'status.json').read_text())
    assert status['state'] == 'complete' and status['cleanup_passed']
    config = json.loads((out / 'runtime-cli.json').read_text())
    assert not config['enforce_eager']
    assert config['speculative_config']['enforce_eager'] is True
    assert config['speculative_config']['num_speculative_tokens'] == 2
    assert config['max_num_seqs'] == '4' and config['max_model_len'] == '262144'
    assert config['compilation_config']['cudagraph_capture_sizes'] == [3, 6, 12]
    freeze = json.loads((out / 'freeze.json').read_text())
    full_freeze = json.loads((full / 'freeze.json').read_text())
    assert freeze['entry_origin'] == 'image'
    assert freeze['entry_sha256'] == full_freeze['entry_sha256']
    assert freeze['image']['Id'] == full_freeze['image']['Id']
    assert 'QWEN36_GRAPH_AUDIT' not in freeze['env']
    record(state='complete', full_window=full.name, packaged_window=out.name,
           same_image_and_entry=True, main_capture_replay_both_ranks=True,
           draft_eager=True, packaged_entry_unmounted=True,
           packaged_audit_disabled=True, image=image, image_id=freeze['image']['Id'])
except Exception:
    record(state='failed', error=traceback.format_exc())
    raise
