#!/usr/bin/env python3
"""a2-only controlled MTP comparison; immutable image, default entry, no source overlays."""
import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.request

ROOT = Path('/home/ychy/ascend-lab-runtime/hyperqwen-20261103')
sys.path[:0] = [str(ROOT), str(ROOT / 'wy-boundary-20261106/tools')]
import run_w8_mask_fix_20261108 as common
import bench_http_stream as http
import bench_server_phases as phases

IMAGE = 'localhost/qwen36-w8-wy-mtp-main-graph:20261108'
IMAGE_ID = 'bb0fa66f1c715a94abf56afc17a6bd02e3f993c4804e200dfd229e22b0d5cab4'
ENTRY_SHA = 'ca8391c85f1a89dae6f71eb61233c06eec0f849dedd0be0453b800cb4f60388a'
MODEL = 'Qwen3.6-35B-A3B-w8a8'
BASE = common.BASE
OUTPUT = 512


def write(path, data):
    common.write(path, data)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def option(argv, key):
    return argv[argv.index(key) + 1] if key in argv else None


def metrics(include_spec):
    return {**phases.snapshot(BASE, include_spec), **phases.prefix_snapshot(BASE)}


def measured_round(tokens, concurrency, output, salt, mtp):
    before = metrics(bool(mtp))
    start = time.perf_counter()
    barrier = threading.Barrier(concurrency)

    def one(index):
        barrier.wait(timeout=10)
        offset = time.perf_counter() - start
        result = http.measured_request(BASE, MODEL, tokens, output,
                                       cache_salt=f'{salt}-{index}', seed=310)
        result['start_offset_s'] = offset
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(one, range(concurrency)))
    elapsed = time.perf_counter() - start
    after = metrics(bool(mtp))
    # Server counters may be exported just after the terminal stream event.
    for _ in range(6):
        if after[phases.METRICS[0]] - before[phases.METRICS[0]] == concurrency:
            break
        time.sleep(.5)
        after = metrics(bool(mtp))
    delta = {key: after[key] - before[key] for key in before}
    successful = [row for row in rows if row['success']]
    valid = (len(successful) == concurrency
             and delta[phases.METRICS[0]] == concurrency
             and delta['vllm:prefix_cache_hits_total'] == 0)
    total = sum(row['usage']['completion_tokens'] for row in successful)
    server_decode = delta[phases.METRICS[2]]
    spec = None
    if mtp:
        drafts, drafted, accepted = (delta[key] for key in phases.SPEC_METRICS)
        spec = {'drafts': drafts, 'drafted_tokens': drafted,
                'accepted_tokens': accepted,
                'acceptance_rate': accepted / drafted if drafted else None,
                'mean_acceptance_length': 1 + accepted / drafts if drafts else None}
    return {'valid': valid, 'elapsed_s': elapsed, 'concurrency': concurrency,
            'prompt_sha256': phases.prompt_hash(tokens), 'output_tokens_per_request': output,
            'success': len(successful), 'requests': rows,
            'metrics_before': before, 'metrics_after': after, 'metrics_delta': delta,
            'e2e_aggregate_output_tok_s': total / elapsed,
            'client_tpot_s_median': statistics.median(row['mean_tpot_s'] for row in successful) if successful else None,
            'ttft_s_median': statistics.median(row['ttft_s'] for row in successful) if successful else None,
            'server_decode_tpot_s': server_decode / (total - concurrency) if valid else None,
            'server_decode_tok_s_per_request_weighted': (total - concurrency) / server_decode if valid and server_decode > 0 else None,
            'spec_decode': spec}


def container_command(out, name, arm, reference):
    cmd = ['podman', 'run', '-d', '--pull=never', '--name', name, '--network', 'host',
           '--group-add', 'keep-groups', '--ipc', 'shareable', '--shm-size', '32g']
    for device in ('davinci0', 'davinci1', 'davinci_manager', 'devmm_svm', 'hisi_hdc'):
        cmd += ['--device', '/dev/' + device]
    for path in ('/usr/local/bin/npu-smi', '/etc/ascend_install.info',
                 '/usr/local/Ascend/driver/version.info', '/usr/local/Ascend/driver/lib64', '/usr/local/dcmi'):
        cmd += ['-v', path + ':' + path + ':ro']
    cmd += ['-v', str(out / 'cache') + ':/root/.cache:rw']
    if arm == 'reference':
        manifest = copy.deepcopy(reference)
        argv = manifest['command']
        argv[argv.index('--host') + 1] = '127.0.0.1'
        argv[argv.index('--port') + 1] = '18126'
        # Archived production model path is preserved inside the container.
        model_path = argv[argv.index('serve') + 1]
        cmd += ['-v', str(ROOT / 'production-w8a8/model') + ':' + model_path + ':ro']
        write(out / 'reference-manifest.json', manifest)
        cmd += ['-v', str(out / 'reference-manifest.json') + ':/workspace/backend-primary.manifest.json:ro',
                '-v', str(ROOT / 'wy-boundary-20261106/tools/start-production-replica.sh') + ':/workspace/start.sh:ro',
                '--entrypoint', '/bin/bash', manifest['image'], '/workspace/start.sh']
        return cmd, manifest['image'], {'scope': 'archived-a1-profile on a2; endpoint only changed'}, 0
    mtp = int(arm)
    env = {'HOST': '127.0.0.1', 'PORT': '18126', 'SERVED_MODEL_NAME': MODEL,
           'MAX_MODEL_LEN': '262144', 'MAX_NUM_SEQS': '4', 'MAX_BATCHED_TOKENS': '2048',
           'MEMORY_UTILIZATION': '0.90', 'ENABLE_PREFIX_CACHING': '0', 'ASYNC_SCHEDULING': '1',
           'ENFORCE_EAGER': '0', 'DRAFT_ENFORCE_EAGER': '1', 'NUM_SPECULATIVE_TOKENS': str(mtp)}
    for key, value in env.items():
        cmd += ['-e', key + '=' + value]
    cmd += ['-v', str(ROOT / 'production-w8a8/model') + ':/model:ro', IMAGE]
    return cmd, IMAGE, env, mtp


def run_window(base, index, arm, args, reference):
    out = base / f'w{index:02d}-mtp{arm}'
    out.mkdir()
    (out / 'cache').mkdir()
    begin = time.monotonic()
    name = f'wy-decode-{time.time_ns()}'
    state = None
    error = None
    started = False
    rounds = []

    def status(stage, **extra):
        write(out / 'status.json', {'state': stage, 'arm': arm,
                                    'elapsed_s': time.monotonic() - begin, **extra})

    before = common.pdf.snapshot()
    write(out / 'window-start.json', before)
    assert before['rootless']['stdout'] == before['rootful']['stdout'] == ''
    assert 'No running processes' in before['npu']['stdout']
    cmd, image, env, mtp = container_command(out, name, arm, reference)
    detail = json.loads(subprocess.check_output(['podman', 'image', 'inspect', image], text=True))[0]
    freeze = {'arm': arm, 'env': env, 'image': image, 'image_id': detail['Id'],
              'runner_sha256': sha(Path(__file__)), 'reference_source_sha256': sha(args.reference),
              'http_source_sha256': sha(Path(http.__file__)), 'phase_source_sha256': sha(Path(phases.__file__)),
              'launch_command': cmd, 'source_mounts': arm == 'reference'}
    if arm != 'reference':
        assert detail['Id'] == IMAGE_ID, detail['Id']
        entry = subprocess.check_output(['podman', 'run', '--rm', '--pull=never', '--network=none',
                                         '--entrypoint', 'cat', image, '/opt/qwen36/start-wy-mtp.sh'])
        freeze['entry_sha256'] = hashlib.sha256(entry).hexdigest()
        assert freeze['entry_sha256'] == ENTRY_SHA
    write(out / 'freeze.json', freeze)
    try:
        common.run(cmd, capture_output=True, text=True)
        started = True
        status('launching')
        common.wait_health(name)
        inspect_cli = "import json,pathlib; print(json.dumps(pathlib.Path('/proc/1/cmdline').read_bytes().decode().strip(chr(0)).split(chr(0))))"
        argv = json.loads(subprocess.check_output(['podman', 'exec', name, 'python3', '-S', '-c', inspect_cli], text=True))
        compile_config = json.loads(option(argv, '--compilation-config') or '{}')
        spec = json.loads(option(argv, '--speculative-config') or '{}')
        runtime = {'argv': argv, 'compilation_config': compile_config, 'speculative_config': spec}
        write(out / 'runtime-cli.json', runtime)
        assert option(argv, '--max-model-len') == '262144'
        assert option(argv, '--max-num-seqs') == '4'
        assert '--enforce-eager' not in argv
        assert compile_config.get('cudagraph_mode') == 'FULL_DECODE_ONLY'
        if arm != 'reference':
            assert option(argv, '--max-num-batched-tokens') == '2048'
            assert compile_config['cudagraph_capture_sizes'] == [n * (mtp + 1) for n in (1, 2, 4)]
            assert '--no-enable-prefix-caching' in argv
            if mtp:
                assert spec['num_speculative_tokens'] == mtp and spec['enforce_eager'] is True
            else:
                assert not spec
        else:
            assert not spec
            assert option(argv, '--max-num-batched-tokens') == '4096'
            assert '--enable-prefix-caching' in argv
        status('smoke')
        common.pdf.MODEL = MODEL
        common.bench(out, mtp, out, smoke_only=True)
        with http.post(BASE, '/tokenize', {'model': MODEL,
                'prompt': 'Explain how quantization affects memory bandwidth and numerical precision. ' * 4,
                'add_special_tokens': False}) as response:
            unit = json.load(response)['tokens']
        tokens = (unit * ((4096 + len(unit) - 1) // len(unit)))[:4096]
        prompt_path = base / 'prompt.json'
        if prompt_path.exists():
            assert json.loads(prompt_path.read_text()) == tokens, 'tokenizer/input drift'
        else:
            write(prompt_path, tokens)
        # Exact same prompt and seed in every arm. Salt is always unique.
        for concurrency in ((1, 4) if index % 2 == 0 else (4, 1)):
            status('warmup', concurrency=concurrency)
            warm = measured_round(tokens, concurrency, 64, out.name + f'-warm-c{concurrency}', mtp)
            write(out / f'warmup-c{concurrency}.json', warm)
            assert warm['valid'], warm
            for repeat in range(args.repeats):
                status('measuring', concurrency=concurrency, repeat=repeat)
                row = measured_round(tokens, concurrency, OUTPUT,
                                     out.name + f'-c{concurrency}-r{repeat}', mtp)
                row.update({'repeat': repeat, 'arm': arm, 'window': out.name})
                write(out / f'c{concurrency}-r{repeat}.json', row)
                rounds.append(row)
                if not row['valid']:
                    raise RuntimeError('request/counter/cache failure; see raw round')
        # Compact summary deliberately excludes raw text/events and source paths containing secrets.
        compact = [{key: value for key, value in row.items()
                    if key not in ('requests', 'metrics_before', 'metrics_after')}
                   for row in rounds]
        write(out / 'summary.json', {'arm': arm, 'rounds': compact,
              'image_id': detail['Id'], 'runtime': runtime,
              'measurement_scope': 'synthetic fixed 4K prompt, 512 output, ignore_eos; no business quality claim'})
    except BaseException:
        error = traceback.format_exc()
    finally:
        if started:
            try:
                state = common.stop(name, out)
            except BaseException:
                error = (error or '') + '\nCLEANUP ERROR\n' + traceback.format_exc()
        after = common.pdf.snapshot()
        clean = (after['rootless']['stdout'] == after['rootful']['stdout'] == ''
                 and 'No running processes' in after['npu']['stdout'])
        write(out / 'window-closed.json', {'elapsed_s': time.monotonic() - begin,
              'cleanup_passed': clean, 'error': error, 'container_state': state, 'snapshot': after})
        status('failed' if error or not clean else 'complete', error=error, cleanup_passed=clean)
    if error or not clean:
        raise RuntimeError(error or 'cleanup failed')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--sequence', default='0,1,2,2,1,0,reference,reference')
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    sequence = args.sequence.split(',')
    assert set(sequence) <= {'0', '1', '2', 'reference'}
    assert args.repeats > 0
    args.out.mkdir()
    (args.out / 'runner-source.py').write_bytes(Path(__file__).read_bytes())
    reference = json.loads(args.reference.read_text())
    write(args.out / 'protocol.json', {
        'sequence': sequence, 'repeats_per_concurrency_per_window': args.repeats,
        'prompt_tokens': 4096, 'output_tokens': OUTPUT, 'concurrency': [1, 4],
        'warmup': '64 tokens at each concurrency, excluded',
        'same_image_arms': ['0', '1', '2'], 'reference_scope': 'archived a1 profile reproduced on a2',
        'capture_policy': 'same bounded three sizes: [1,2,4]*(MTP+1)',
        'metric_scope': 'e2e aggregate includes prefill; client TPOT is stream-first-text to final usage; server decode TPOT sums request-phase durations, not pure kernel time',
        'scope_limits': 'one synthetic fixed prompt; mirrored order, two boots per arm; not workload-wide optimality or long-duration load test',
        'no_device_code_or_image_changes': True, 'runner_sha256': sha(Path(__file__))})
    completed = []
    try:
        with open('/tmp/a310-npu.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for index, arm in enumerate(sequence):
                if index:
                    write(args.out / 'status.json', {'state': 'cooldown', 'next_arm': arm, 'completed': completed})
                    time.sleep(60)
                write(args.out / 'status.json', {'state': 'running', 'index': index, 'arm': arm, 'completed': completed})
                out = run_window(args.out, index, arm, args, reference)
                completed.append(out.name)
        write(args.out / 'status.json', {'state': 'complete', 'completed': completed})
    except BaseException:
        write(args.out / 'status.json', {'state': 'failed', 'completed': completed, 'error': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
