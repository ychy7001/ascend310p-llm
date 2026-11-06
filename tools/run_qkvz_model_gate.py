#!/usr/bin/env python3
"""Run fresh A/A-prime/B gates while owning the shared NPU flock.

The controller acquires the lock itself; do not wrap it in a second flock.
The window ends with experiment containers stopped and both services inactive.
Inputs, weights, evaluator tools and the dataset are frozen before launch.
"""
import argparse
import copy
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

from compare_token_logprobs import compare, digest

BASE = 'http://localhost:18142'
MODEL = 'Qwen3.6-35B-A3B-w8a8'
SERVICE = 'llm-backend-primary.service'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def read(path):
    return json.loads(path.read_text())


def health(base):
    try:
        with urllib.request.urlopen(base+'/health', timeout=4) as r:
            return r.status == 200
    except OSError:
        return False


def positions(choice):
    ids, probs = choice['token_ids'], choice['logprobs']
    assert len(ids) == len(probs['top_logprobs']) == len(probs['token_logprobs']) == 256
    result = []
    for token, actual, top in zip(ids, probs['token_logprobs'], probs['top_logprobs'], strict=True):
        assert all(k.startswith('token_id:') for k in top)
        values = {k.removeprefix('token_id:'): v for k, v in top.items()}
        values[str(token)] = actual
        result.append({'top1': token, 'logprobs': values})
    return result


class Gate:
    def __init__(self, args):
        self.args = args
        self.root = args.root.resolve()
        self.tools = args.tools.resolve()
        self.inputs = read(self.root/'inputs.json')
        self.rule = read(self.root/'threshold-rule.json')
        assert self.rule['multiplier'] == 1 and self.rule['absolute_floor'] == 0
        assert sha(self.root/'inputs.json') == self.rule['input_sha256']
        assert len(self.inputs['cases']) == 37
        assert sha(args.dataset) == '3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'
        self.manifest = read(self.root/'production-manifest.json')
        self.container = None
        self.container_arm = None
        self.started_unix_ns = time.time_ns()
        self.started = time.monotonic()
        self.error = None

    def state(self, phase, **kw):
        temp = self.root/'status.tmp'
        temp.write_text(json.dumps({'phase': phase, 'elapsed_s': time.monotonic()-self.started, **kw}, indent=2)+'\n')
        temp.replace(self.root/'status.json')

    def remaining(self):
        value = 6600 - (time.monotonic()-self.started)
        if value <= 0:
            raise TimeoutError('NPU work budget reached; close experiment window')
        return value

    def command(self, label, cmd):
        write(self.root/(label+'-command.json'), cmd)
        with (self.root/(label+'.log')).open('x') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=self.remaining())

    def wait_health(self, base):
        for _ in range(240):
            if health(base):
                return
            if base == BASE:
                self.remaining()
                status = subprocess.check_output(['sudo', '-n', 'podman', 'inspect', '--format', '{{.State.Status}}', self.container], text=True).strip()
                if status == 'exited':
                    raise RuntimeError('Experimental model exited during startup')
            time.sleep(5)
        raise TimeoutError('Service startup health budget exhausted')

    def stop(self, label):
        if self.container is None:
            return
        name = self.container
        label = self.container_arm or label
        subprocess.run(['sudo', '-n', 'podman', 'stop', '--time', '30', name], check=True, capture_output=True, timeout=90)
        with (self.root/(label+'-server.log')).open('x') as log:
            subprocess.run(['sudo', '-n', 'podman', 'logs', name], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=30)
        assert subprocess.check_output(['sudo', '-n', 'podman', 'inspect', '--format', '{{.State.Status}}', name], text=True).strip() == 'exited'
        inspected = json.loads(subprocess.check_output(
            ['sudo', '-n', 'podman', 'inspect', name], text=True, timeout=30))
        write(self.root/(label+'-inspect.json'), inspected)
        self.container = None
        self.container_arm = None
        if inspected[0]['State']['ExitCode'] != 0:
            raise RuntimeError(f'{label} container exited with code {inspected[0]["State"]["ExitCode"]}')

    def launch(self, label, candidate=False):
        self.remaining()
        directory = self.root/label
        directory.mkdir()
        (directory/'cache').mkdir()
        manifest = copy.deepcopy(self.manifest)
        c = manifest['command']
        assert c[c.index('--max-num-seqs')+1] == '4'
        assert json.loads(c[c.index('--compilation-config')+1])['cudagraph_capture_sizes'] == [1, 2]
        assert c[c.index('--max-model-len')+1] == '262144'
        assert c[c.index('--max-num-batched-tokens')+1] == '4096'
        assert '--speculative-config' not in c and '--enable-prefix-caching' in c
        c[c.index('--host')+1] = 'localhost'
        c[c.index('--port')+1] = '18142'
        if candidate:
            assert (self.root/'concrete-thresholds.json').exists()
            c.extend(['--worker-cls', 'qkvz_w8a8_worker.QKVZW8A8Worker'])
        write(directory/'manifest.json', manifest)
        self.container_arm = label
        self.container = 'qkvz-'+label.lower().replace('_', '-')+'-'+str(time.time_ns())
        cmd = ['sudo', '-n', 'podman', 'run', '-d', '--pull=never', '--privileged',
               '--name', self.container, '--network', 'host', '--shm-size', '32g', '-w', '/workspace']
        for dev in ('davinci0', 'davinci1', 'davinci_manager', 'devmm_svm', 'hisi_hdc'):
            cmd += ['--device', '/dev/'+dev]
        mounts = [(self.args.model_root, '/models/Qwen3.6-35B-A3B-w8a8', 'ro'),
                  (directory, '/experiment', 'rw'), (directory/'cache', '/root/.cache', 'rw'),
                  (self.root/'start-production-replica.sh', '/workspace/start_vllm_ascend.sh', 'ro'),
                  (directory/'manifest.json', '/workspace/backend-primary.manifest.json', 'ro')]
        if candidate:
            mounts += [(self.args.weights, '/qkvz-weights', 'ro'), (self.root/'worker', '/qkvz-tools', 'ro')]
        for path in ('/usr/local/bin/npu-smi', '/etc/ascend_install.info', '/usr/local/Ascend/driver/version.info',
                     '/usr/local/Ascend/driver/lib64', '/usr/local/dcmi'):
            mounts.append((Path(path), path, 'ro'))
        for src, dest, mode in mounts:
            cmd += ['-v', f'{src}:{dest}:{mode}']
        cmd += [self.args.image]
        if candidate:
            cmd += ['bash', '-c', 'export PYTHONPATH="/qkvz-tools${PYTHONPATH:+:$PYTHONPATH}"; exec bash /workspace/start_vllm_ascend.sh']
        else:
            cmd += ['bash', '/workspace/start_vllm_ascend.sh']
        write(directory/'launch-metadata.json', {'image': self.args.image,
                                                'started_unix_ns': time.time_ns(),
                                                'controller_sha256': sha(Path(__file__))})
        self.state('launching', arm=label)
        self.command(label+'-launch', cmd)
        self.wait_health(BASE)

    def prefix(self):
        keys = ('vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total')
        with urllib.request.urlopen(BASE+'/metrics', timeout=30) as response:
            lines = response.read().decode().splitlines()
        values = {}
        for line in lines:
            if not line or line.startswith('#'):
                continue
            name = line.split('{', 1)[0].split()[0]
            if name in keys:
                values[name] = values.get(name, 0.) + float(line.split()[-1])
        assert set(values) == set(keys)
        return values

    def g1(self, label, reference=None):
        directory = self.root/(label+'-G1-raw')
        directory.mkdir()
        record = {'mode': 'generated', 'input_sha256': self.rule['input_sha256'], 'cases': []}
        before = self.prefix()
        for i, case in enumerate(self.inputs['cases']):
            self.state('G1', arm=label, completed=i, case=case['id'])
            body = {'model': MODEL, 'prompt': case['token_ids'], 'temperature': 0, 'seed': 0,
                    'max_tokens': 256, 'ignore_eos': True, 'logprobs': 20,
                    'return_token_ids': True, 'return_tokens_as_token_ids': True,
                    'cache_salt': 'qkvz-g1-'+case['id']}
            req = urllib.request.Request(BASE+'/v1/completions', data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=min(1800, self.remaining())) as response:
                result = json.load(response)
            with gzip.open(directory/(case['id']+'.json.gz'), 'wt') as f:
                json.dump(result, f)
            assert result['usage']['prompt_tokens'] == case['tokens']
            choice = result['choices'][0]
            record['cases'].append({'id': case['id'], 'greedy_ids': choice['token_ids'], 'positions': positions(choice)})
            print(json.dumps({'arm': label, 'G1_cases': len(record['cases']), 'id': case['id']}), flush=True)
        after = self.prefix()
        record['prefix_cache_delta'] = {k: after[k]-before[k] for k in before}
        assert record['prefix_cache_delta']['vllm:prefix_cache_hits_total'] == 0
        sequences = [c['greedy_ids'] for c in (record if reference is None else reference)['cases']]
        record['reference_sequence_sha256'] = digest(sequences)
        write(self.root/(label+'-G1.json'), record)
        return record

    def g2(self, label):
        common = ['--base-url', BASE, '--model', MODEL]
        counts = {}
        for phase in ('before', 'after'):
            if phase == 'after':
                self.command(label+'-gsm64', ['python3', str(self.tools/'eval_gsm8k_http.py'), *common,
                    '--dataset', str(self.args.dataset), '--limit', '64', '--seed', '310', '--max-tokens', '512',
                    '--label', label, '--out', str(self.root/(label+'-gsm64.jsonl'))])
                # This evaluator's frozen source model already equals MODEL.
                # Its older deployed CLI does not accept --model.
                self.command(label+'-fields', ['python3', str(self.tools/'check_action_state_regression.py'),
                    '--base-url', BASE,
                    '--fixtures', str(self.args.fields), '--out', str(self.root/(label+'-fields.jsonl'))])
            self.state('G2', arm=label, step=phase)
            self.command(label+'-short-'+phase, ['python3', str(self.tools/'check_cache_request_reuse.py'), *common,
                '--out', str(self.root/(label+'-short-'+phase+'.json'))])
            result = read(self.root/(label+'-short-'+phase+'.json'))
            assert result['reuse']['identical_text'] and not result['reuse']['canary_leaked']
            counts['short_'+phase] = result['strict_matches']
        counts['GSM64'] = json.loads((self.root/(label+'-gsm64.jsonl')).read_text().splitlines()[-1])['correct']
        rows = [json.loads(l) for l in (self.root/(label+'-fields.jsonl')).read_text().splitlines()]
        for concurrency in (1, 3):
            group = [r for r in rows if r.get('kind') == 'request' and r['concurrency_limit'] == concurrency]
            assert len(group) == 13 and not any('error_type' in row for row in group)
            counts['fields_C'+str(concurrency)] = sum(r['correct'] for r in group)
        write(self.root/(label+'-G2-counts.json'), counts)
        return counts

    def snapshot(self):
        def capture(cmd):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            return {'returncode': result.returncode, 'stdout': result.stdout,
                    'stderr': result.stderr}
        return {'unix_ns': time.time_ns(), 'free': capture(['free', '-g']),
                'npu': capture(['npu-smi', 'info']),
                'containers': capture(['sudo', '-n', 'podman', 'ps', '-q']),
                'user_containers': capture(['podman', 'ps', '-q']),
                'services': {unit: capture(['systemctl', 'is-active', unit])
                             for unit in (SERVICE, 'qwen36-image.service')}}

    def preflight_evaluators(self):
        # Run the actual parsers but stop before loading fixtures or making HTTP
        # requests. This checks the deployed files, not synthetic CLI stubs.
        commands = [
            [str(self.tools/'eval_gsm8k_http.py'), '--base-url', BASE, '--model', MODEL,
             '--dataset', str(self.args.dataset), '--limit', '64', '--seed', '310',
             '--max-tokens', '512', '--label', self.args.arm,
             '--out', str(self.root/'preflight-gsm.jsonl')],
            [str(self.tools/'check_action_state_regression.py'), '--base-url', BASE,
             '--fixtures', str(self.args.fields), '--out', str(self.root/'preflight-fields.jsonl')],
            [str(self.tools/'check_cache_request_reuse.py'), '--base-url', BASE,
             '--model', MODEL, '--out', str(self.root/'preflight-short.json')],
        ]
        code = """import argparse,runpy,sys
class Parsed(BaseException): pass
original=argparse.ArgumentParser.parse_args
def parse(self,*args,**kwargs):
    original(self,*args,**kwargs)
    raise Parsed()
argparse.ArgumentParser.parse_args=parse
script=sys.argv.pop(1)
sys.argv[0]=script
try:
    runpy.run_path(script,run_name='__main__')
except Parsed:
    pass
else:
    raise RuntimeError('Evaluator never parsed arguments')
"""
        for index, cmd in enumerate(commands):
            self.command(self.args.arm+'-cli-preflight-'+str(index), ['python3', '-c', code, *cmd])

    def execute(self):
        try:
            write(self.root/(self.args.arm+'-tool-freeze.json'), {p.name: sha(p) for p in self.tools.iterdir() if p.is_file()})
            self.preflight_evaluators()
            initial = self.snapshot()
            write(self.root/(self.args.arm+'-window-open.json'), initial)
            assert initial['services']['qwen36-image.service']['stdout'].strip() == 'inactive'
            primary = initial['services'][SERVICE]['stdout'].strip()
            if primary == 'active':
                subprocess.run(['sudo', '-n', 'systemctl', 'stop', SERVICE], check=True, timeout=180)
                subprocess.run(['sudo', '-n', 'systemctl', 'reset-failed', SERVICE], check=True, timeout=30)
            else:
                assert primary == 'inactive', primary
            ready = self.snapshot()
            assert ready['services'][SERVICE]['stdout'].strip() == 'inactive'
            for key in ('containers', 'user_containers'):
                assert ready[key]['returncode'] == 0 and not ready[key]['stdout'].strip(), ready
            assert ready['npu']['returncode'] == 0 and 'No running processes found' in ready['npu']['stdout'], ready
            if self.args.arm == 'A':
                self.launch('A')
                self.g1('A')
                self.g2('A')
                self.stop('A')
            elif self.args.arm == 'A_prime':
                a = read(self.root/'A-G1.json')
                self.launch('A_prime')
                ap = self.g1('A_prime', a)
                self.g2('A_prime')
                self.stop('A_prime')
                noise = compare(a, ap)
                write(self.root/'A_prime-vs-A.json', noise)
                frozen = {'thresholds': noise['metrics'], 'multiplier': 1, 'absolute_floor': 0,
                          'A_sha256': sha(self.root/'A-G1.json'), 'A_prime_sha256': sha(self.root/'A_prime-G1.json'),
                          'input_sha256': sha(self.root/'inputs.json'), 'rule_sha256': sha(self.root/'threshold-rule.json'),
                          'frozen_unix_ns': time.time_ns(), 'B_started': False}
                write(self.root/'concrete-thresholds.json', frozen)
                self.state('thresholds_frozen_before_first_model_B', thresholds_sha256=sha(self.root/'concrete-thresholds.json'))
            else:
                a = read(self.root/'A-G1.json')
                ga = read(self.root/'A-G2-counts.json')
                gap = read(self.root/'A_prime-G2-counts.json')
                noise = read(self.root/'A_prime-vs-A.json')
                frozen = read(self.root/'concrete-thresholds.json')
                assert frozen['thresholds'] == noise['metrics']
                assert frozen['rule_sha256'] == sha(self.root/'threshold-rule.json')
                self.launch('B', candidate=True)
                b = self.g1('B', a)
                gb = self.g2('B')
                self.stop('B')
                paired = compare(a, b)
                write(self.root/'B-vs-A.json', paired)
                checks = {}
                for key, value in paired['metrics'].items():
                    passed = value <= frozen['thresholds'][key]
                    if key == 'max_absolute_target_logprob_difference' and paired['target_logprob_censoring']['count'] and passed:
                        passed = None
                    checks[key] = {'noise': noise['metrics'][key], 'threshold': frozen['thresholds'][key],
                                   'B_vs_A': value, 'passed': passed}
                passed = all(c['passed'] is True for c in checks.values())
                write(self.root/'G1-adjudication.json', {'passed': passed, 'checks': checks,
                    'concrete_thresholds_sha256': sha(self.root/'concrete-thresholds.json')})
                task_checks = {k: {'A': ga[k], 'A_prime': gap[k], 'B': gb[k],
                                  'B_vs_A_difference': abs(gb[k]-ga[k]),
                                  'B_vs_A_prime_difference': abs(gb[k]-gap[k]),
                                  'passed': abs(gb[k]-ga[k]) <= 1 and abs(gb[k]-gap[k]) <= 1}
                               for k in ga}
                write(self.root/'G2-adjudication.json', {'passed': all(c['passed'] for c in task_checks.values()), 'checks': task_checks})
                self.state('model_gates_complete', G1_passed=passed, G2_passed=all(c['passed'] for c in task_checks.values()))
            if self.args.arm != 'B':
                self.state(self.args.arm+'_complete')
        except Exception as e:
            self.error = repr(e)
            self.state('failed', error=self.error)
        finally:
            try:
                self.stop(self.args.arm+'-emergency')
            finally:
                closed = self.snapshot()
                closed.update(start_unix_ns=self.started_unix_ns,
                              elapsed_s=time.monotonic()-self.started,
                              error=self.error)
                closed['passed'] = (
                    self.container is None and closed['elapsed_s'] <= 10800
                    and all(value['stdout'].strip() == 'inactive'
                            for value in closed['services'].values())
                    and all(closed[key]['returncode'] == 0 and not closed[key]['stdout'].strip()
                            for key in ('containers', 'user_containers'))
                    and closed['npu']['returncode'] == 0
                    and 'No running processes found' in closed['npu']['stdout'])
                write(self.root/(self.args.arm+'-window-closed.json'), closed)
                if not closed['passed']:
                    raise RuntimeError('Experiment window cleanup failed; inspect window-closed evidence')
                print(json.dumps({'error': self.error, 'window_closed': True}), flush=True)

        if self.error:
            raise RuntimeError(self.error)


def interrupted(signum, frame):
    raise RuntimeError('Interrupted signal '+str(signum))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('root', 'tools', 'model-root', 'weights', 'dataset', 'fields'):
        p.add_argument('--'+arg, type=Path, required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--arm', choices=('A', 'A_prime', 'B'), required=True)
    a = p.parse_args()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with open('/tmp/a310-npu.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        Gate(a).execute()


if __name__ == '__main__':
    main()
