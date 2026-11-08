"""Finish packaged verification only after clean full qualification of identical math."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import traceback

ROOT = Path('/home/ychy/ascend-lab-runtime/hyperqwen-20261103')
work = ROOT / 'wy-mtp-production-20261108'
base = ROOT / 'graph-localize'
full = base / 'graph-release-r1-full'
out = base / 'graph-release-r2-packaged'
report = base / 'graph-release-r2-closure.json'
image = 'localhost/qwen36-w8-wy-mtp-graph-r2:20261108'


def record(**data):
    report.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


try:
    deadline = time.monotonic() + 3600
    while True:
        state = json.loads((full / 'status.json').read_text())
        if state['state'] == 'failed':
            raise RuntimeError('full qualification failed; packaged window canceled')
        if state['state'] == 'complete':
            assert state['cleanup_passed']
            break
        if time.monotonic() >= deadline:
            raise TimeoutError('qualification wait budget exceeded; no new device action')
        record(state='waiting-full-qualification')
        time.sleep(20)

    expected = json.loads((base / 'pr18730-candidate-context/graph-release-artifact-sha256.json').read_text())
    paths = list(expected)
    paths.append('/vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py')
    code = 'import hashlib,json,pathlib; paths=' + repr(paths) + '; print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in paths}))'
    actual = json.loads(subprocess.check_output(['podman', 'run', '--rm', '--pull=never', '--network=none', '--entrypoint', 'python3', image, '-S', '-c', code], text=True))
    audit_path = '/opt/qwen36/wy-compat/graph_lifecycle_audit.py'
    assert all(actual[path] == sha for path, sha in expected.items() if path != audit_path)
    assert actual[paths[-1]] == '3b9bc0eb2e2e9d944316858df1efaea72776e7221b5eb1640db4a719b985accb'
    (base / 'graph-release-r2-artifact-sha256.json').write_text(json.dumps(actual, indent=2) + '\n')
    record(state='running-packaged', core_files_match_full=True, only_changed_file=audit_path)
    with (base / 'graph-release-r2-packaged-driver.log').open('w') as log:
        subprocess.run(['python3', str(work / 'run_profile_graph_packaged.py'),
                        '--out', str(out), '--image', image, '--mode', 'packaged',
                        '--eager', '0', '--draft-eager', '0', '--seqs', '4', '--graph-audit',
                        '--served-model-name', 'Qwen3.6-35B-A3B-w8a8',
                        '--cooldown-after', str(full / 'window-closed.json')],
                       check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    status = json.loads((out / 'status.json').read_text())
    assert status['state'] == 'complete' and status['cleanup_passed']
    config = json.loads((out / 'runtime-cli.json').read_text())
    assert not config['enforce_eager']
    assert config['speculative_config']['enforce_eager'] is False
    assert config['speculative_config']['num_speculative_tokens'] == 2
    assert config['max_num_seqs'] == '4' and config['max_model_len'] == '262144'
    assert config['compilation_config']['cudagraph_capture_sizes'] == [12]
    freeze = json.loads((out / 'freeze.json').read_text())
    assert freeze['entry_origin'] == 'image'
    assert freeze['entry_sha256'] == expected['/opt/qwen36/start-wy-mtp.sh']
    markers = []
    seen = set()
    for line in (out / 'server.log').read_text(errors='replace').splitlines():
        if '[graph-lifecycle] ' in line:
            row = json.loads(line.split('[graph-lifecycle] ', 1)[1])
            rank = re.search(r'Worker_TP([01])', line)
            assert rank is not None
            row['rank'] = int(rank.group(1))
            markers.append(row)
            seen.add((row['rank'], row['role'], row['event']))
    assert all((rank, role, event) in seen for rank in (0, 1)
               for role in ('main', 'draft') for event in ('capture', 'replay')), seen
    (out / 'graph-lifecycle.json').write_text(json.dumps(markers, indent=2) + '\n')
    record(state='complete', core_files_match_full=True, only_changed_file=audit_path,
           full_window=full.name, packaged_window=out.name,
           main_and_draft_capture_replay_both_ranks=True,
           packaged_entry_unmounted=True, audit_enabled_for_verification=True,
           production_audit_default=False, image=image, image_id=freeze['image']['Id'])
except Exception:
    record(state='failed', error=traceback.format_exc())
    raise
