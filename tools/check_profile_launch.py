# SPDX-License-Identifier: Apache-2.0
"""不用 NPU，验证候选校验输出与实际容器参数一致。"""
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        (path / 'model.safetensors.index.json').write_text('{}')
        artifact = path / 'placeholder.so'
        artifact.write_text('parameter-check fixture, not a real binary')
        podman = path / 'podman'
        podman.write_text('#!/usr/bin/env python3\nimport sys,json\nprint(json.dumps(sys.argv[1:]))\n')
        podman.chmod(0o755)
        env = {k: v for k, v in os.environ.items() if not k.startswith('QWEN36_')}
        for key in ('MAX_MODEL_LEN', 'MAX_BATCHED_TOKENS', 'MEMORY_UTILIZATION', 'ENFORCE_EAGER'):
            env.pop(key, None)
        env['PATH'] = directory + ':' + env['PATH']
        env.update({key: directory for key in ('MODEL_DIR', 'OPP_DIR', 'GDN_OPP_DIR')})
        env.update({key: str(artifact) for key in ('EXTENSION', 'C8_LIBRARY', 'GDN_LIBRARY', 'GDN_SOURCE', 'MINDIEQ_LIBRARY')})
        for profile, drafts, batch in (('compat', 1, 1024), ('decode', 2, 4096), ('long-prefill', 2, 2048)):
            selected = {**env, 'QWEN36_MINDIEQ_ENABLED': '1' if profile == 'long-prefill' else '0'}
            proc = subprocess.run(['bash', str(ROOT / 'scripts/run-candidate.sh'), '--profile', profile],
                                  env=selected, capture_output=True, text=True, check=True)
            config, end = json.JSONDecoder().raw_decode(proc.stdout)
            args = json.loads(proc.stdout[end:].strip())
            actual = dict(args[i + 1].split('=', 1) for i, value in enumerate(args[:-1]) if value == '-e')
            effective = config['effective_config']
            assert int(effective['NUM_SPECULATIVE_TOKENS']) == drafts
            assert int(effective['MAX_BATCHED_TOKENS']) == batch
            for key, value in effective.items():
                if key != 'QWEN36_PROFILE':
                    assert actual[key] == value, (profile, key, value, actual.get(key))
            if profile == 'long-prefill':
                assert config['mindieq_min_tokens'] == 128
                assert actual['QWEN36_MINDIEQ_SHARED_GATE'] == actual['QWEN36_MINDIEQ_CUSTOM_QUANT'] == '1'
            print(f'{profile}: effective_config matches captured container arguments')
        for overrides in ({'QWEN36_MINDIEQ_ENABLED': '0'},
                          {'QWEN36_MINDIEQ_ENABLED': '1', 'QWEN36_MINDIEQ_MIN_TOKENS': '17'},
                          {'QWEN36_MINDIEQ_ENABLED': '1', 'QWEN36_MINDIEQ_SHARED_GATE': '0'}):
            proc = subprocess.run(['bash', str(ROOT / 'scripts/run-candidate.sh'), '--check', '--profile', 'long-prefill'],
                                  env={**env, **overrides}, capture_output=True, text=True)
            assert proc.returncode == 2, (overrides, proc.stderr)
        print('invalid long-prefill configurations rejected; no NPU initialized')
        proc = subprocess.run(['bash', str(ROOT / 'scripts/run-candidate.sh'), '--check'],
                              env=env, capture_output=True, text=True, check=True)
        default = json.loads(proc.stdout)
        assert default['profile_name'] == 'long-prefill'
        assert default['mindieq_prefill'] == default['mindieq_custom_quant'] == default['mindieq_shared_gate'] == '1'
        assert default['mindieq_min_tokens'] == 128
        assert default['effective_config']['MAX_BATCHED_TOKENS'] == '2048'
        assert default['effective_config']['QWEN36_C8_ENABLED'] == '0'
        proc = subprocess.run(['bash', str(ROOT / 'scripts/run-candidate.sh'), '--check'],
                              env={**env, 'QWEN36_C8_ENABLED': '1'}, capture_output=True,
                              text=True, check=True)
        assert json.loads(proc.stdout)['effective_config']['QWEN36_C8_ENABLED'] == '1'
        print('48G/112K defaults to C16; explicit C8 override preserved')
        print('default candidate selects long-prefill E5; no NPU initialized')


if __name__ == '__main__':
    main()
