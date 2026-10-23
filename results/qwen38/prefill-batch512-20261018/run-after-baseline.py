"""a2: require completed 16K b256 gates before switching to the b512 experiment."""
import json
from pathlib import Path
import subprocess
import time

work = Path('/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018')
container = 'qwen38-tp2-text-16k-graph'
status = work / 'b512-transition-status.json'


def mark(state):
    status.write_text(json.dumps({'state': state,
        'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}))


try:
    mark('waiting_b256_gates')
    for _ in range(180):
        gate = json.loads((work / 'long16k-gate-status.json').read_text())
        if gate['state'] == 'all_gates_completed':
            break
        if gate['state'].startswith('failed'):
            raise RuntimeError('Baseline failed: ' + gate['state'])
        time.sleep(5)
    else:
        raise TimeoutError('Baseline gates did not finish')
    # Save evidence before stopping the validated baseline.
    with (work / 'long16k-serving-final.log').open('w') as output:
        subprocess.run(['podman', 'logs', container], stdout=output,
                       stderr=subprocess.STDOUT, check=True)
    with (work / 'long16k-npu-final.txt').open('w') as output:
        subprocess.run(['npu-smi', 'info'], stdout=output, check=True)
    mark('switching_to_b512')
    subprocess.run(['podman', 'stop', '--time', '20', container], check=True)
    mark('b512_gdn_numerical_gate')
    subprocess.run(['bash', str(work / 'run-b512-gdn-on-a2.sh')], check=True)
    for rank in (0, 1):
        gdn = json.loads((work / 'b512-gdn' / f'gdn-tp2-rank{rank}.json').read_text())
        assert gdn['hccl_sum'] == 3 and len(gdn['cases']) == 1
        case = gdn['cases'][0]
        assert case['tokens'] == 512 and case['output_pass']
        assert case['state_max_abs_error'] <= .002
        native = json.loads((work / 'b512-native' / f'native-rank{rank}.json').read_text())
        assert len(native) == 4 and all(case['pass'] for case in native)
    subprocess.run(['bash', str(work / 'launch-text-tp2-graph-16k-b512-on-a2.sh')], check=True)
    mark('b512_gates')
    subprocess.run(['python3', str(work / 'run-b512-long16k-gates.py')], check=True)
    mark('b512_gates_completed')
except Exception as error:
    mark('failed: ' + repr(error))
    raise
