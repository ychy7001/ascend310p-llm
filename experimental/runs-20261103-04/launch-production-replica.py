#!/usr/bin/env python3
"""在a2复制a1生产命令，仅隔离端口、模型路径与缓存；可显式加离线Profiler。"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--manifest', type=Path, required=True)
p.add_argument('--start-script', type=Path, required=True)
p.add_argument('--model-dir', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--container', default='qwen36-production-replica-20261103')
p.add_argument('--port', type=int, default=18106)
p.add_argument('--profiler', action='store_true')
p.add_argument('--gdn-pair-dir', type=Path, help='operator-validated upstream paired FwdH/FwdO package')
p.add_argument('--cann921-gdn-pair-dir', type=Path, help='9.2.1-only paired H/O correctness candidate, original Torch WY')
p.add_argument('--maskfix-dir', type=Path, help='isolated, operator-validated FwdO package and worker')
p.add_argument('--native-wy-dir', type=Path, help='isolated WY-only OPP, extension and worker tools')
p.add_argument('--compatibility-image-id', choices=['b381d8e5ed45c11633a7bd07a50a0f5fcf8cf71bc797f1b543d589ab07dd759d', 'd103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a'], help='separate audited complete-runtime screen; never a single-variable production A/B')
a = p.parse_args()
runtime_pair = a.cann921_gdn_pair_dir is not None
if runtime_pair:
    if (a.compatibility_image_id != 'd103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a'
            or a.gdn_pair_dir or a.native_wy_dir or a.maskfix_dir or a.profiler):
        raise RuntimeError('9.2.1 paired H/O requires fixed9.2.1 image and original Torch WY only')
    a.gdn_pair_dir = a.cann921_gdn_pair_dir
pair_vendor = 'hyperqwen_gdn_pair921_transformer' if runtime_pair else 'hyperqwen_gdn_pair_transformer'
pair_worker = 'cann921_gdn_pair_worker.py' if runtime_pair else 'gdn_pair_worker.py'
a.out.mkdir(parents=True, exist_ok=True)
manifest = json.loads(a.manifest.read_text())
expected = 'registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:1a193ded0d75832321019542326740c7fc4509853556d032e773631466453b38'
if a.compatibility_image_id:
    if manifest.get('experiment_kind') != ('runtime_gdn_pair_correctness' if runtime_pair else 'runtime_compatibility_screen') or a.profiler or a.native_wy_dir or a.maskfix_dir or (a.gdn_pair_dir and not runtime_pair):
        raise RuntimeError('Compatibility image requires separate runtime scope without kernel candidates')
    expected = a.compatibility_image_id
    inspected = json.loads(subprocess.check_output(['podman', 'image', 'inspect', expected], text=True))
    if inspected[0]['Id'] != expected:
        raise RuntimeError('Compatibility image identity mismatch')
if manifest['image'] != expected:
    raise RuntimeError('Unexpected production image digest')
command = manifest['command']
model_mount = command[command.index('serve') + 1]
model_path = PurePosixPath(model_mount)
if not model_path.is_absolute() or model_path.parts[:2] != ('/', 'models') or '..' in model_path.parts:
    raise RuntimeError('Model mount must be an absolute path under /models')
for flag, value in (('--port', str(a.port)), ('--host', '127.0.0.1')):
    command[command.index(flag) + 1] = value
if a.profiler:
    command += ['--profiler-config', json.dumps(dict(profiler='torch',
        torch_profiler_dir='/experiment/profiles', torch_profiler_with_stack=False,
        ignore_frontend=True))]
if a.gdn_pair_dir and a.maskfix_dir:
    raise RuntimeError('Paired H/O cannot be combined with a separate FwdO patch')
if a.native_wy_dir:
    native = a.native_wy_dir.resolve()
    required = ('wy-extension/pr17640_wy.so', 'opp/vendors/hyperqwen_wy_transformer/op_api/lib/libcust_opapi.so',
                'tools/native_wy.py', 'tools/native_wy_worker.py')
    if any(not (native / item).is_file() for item in required):
        raise RuntimeError('Native WY package incomplete')
    if not a.maskfix_dir and not a.gdn_pair_dir:
        command += ['--worker-cls', 'native_wy_worker.NativeWYWorker']
if a.maskfix_dir:
    maskfix = a.maskfix_dir.resolve()
    status = json.loads((maskfix / 'status.json').read_text())
    if status['state'] != 'passed_operator_validation':
        raise RuntimeError('FwdO package has not passed operator validation')
    required = ('extension/maskfix_fwd_o.so', 'opp/vendors/hyperqwen_maskfix_transformer/op_api/lib/libcust_opapi.so',
                'tools/maskfix_worker.py')
    if any(not (maskfix / item).is_file() for item in required):
        raise RuntimeError('Maskfix package incomplete')
    if (manifest['env'].get('EXPERIMENT_NATIVE_WY') == '1') != bool(a.native_wy_dir):
        raise RuntimeError('Native WY environment flag and mount disagree')
    command += ['--worker-cls', 'maskfix_worker.MaskFixWorker']
if a.gdn_pair_dir:
    pair = a.gdn_pair_dir.resolve()
    status = json.loads((pair/'status.json').read_text())
    if status['state'] != 'passed_operator_validation':
        raise RuntimeError('Paired GDN has not passed operator validation')
    required = ('extension/pr17640_gdn_pair.so', f'opp/vendors/{pair_vendor}/op_api/lib/libcust_opapi.so',
                'tools/'+pair_worker, 'source-provenance.json', 'package-isolation.json')
    if any(not (pair/item).is_file() for item in required):
        raise RuntimeError('Paired GDN package incomplete')
    provenance = json.loads((pair/'source-provenance.json').read_text())
    if provenance['commit'] != '18163f6b1d208ab3b4fb06e9752225611c37f40e' or provenance['algorithm_changes']:
        raise RuntimeError('Unexpected paired GDN source')
    package = json.loads((pair/'package-isolation.json').read_text())
    lib = pair/f'opp/vendors/{pair_vendor}/op_api/lib/libcust_opapi.so'
    if hashlib.sha256(lib.read_bytes()).hexdigest() != package['sha256']:
        raise RuntimeError('Paired GDN API library changed after validation')
    if (manifest['env'].get('EXPERIMENT_NATIVE_WY') == '1') != bool(a.native_wy_dir):
        raise RuntimeError('Paired GDN native WY flag and mount disagree')
    if runtime_pair:
        if provenance['runtime'] != 'cann921' or provenance['build_image'] != a.compatibility_image_id:
            raise RuntimeError('Paired H/O was built for another runtime')
        artifacts = json.loads((pair/'operator-artifacts.json').read_text())
        for item, expected_sha in artifacts.items():
            if hashlib.sha256((pair/item).read_bytes()).hexdigest() != expected_sha:
                raise RuntimeError('Validated operator artifact changed: '+item)
        command += ['--worker-cls', 'cann921_gdn_pair_worker.CANN921GDNPairWorker']
    else:
        command += ['--worker-cls', 'gdn_pair_worker.GDNPairWorker']
manifest['port'] = a.port
frozen = a.out / 'replica-manifest.json'
with frozen.open('x') as stream:
    json.dump(manifest, stream, indent=2)
(a.out / 'cache').mkdir(exist_ok=False)
(a.out / 'freeze.json').write_text(json.dumps({
    'production_manifest_sha256': hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
    'start_script_sha256': hashlib.sha256(a.start_script.read_bytes()).hexdigest(),
    'paired_gdn_artifacts_sha256': ({item:hashlib.sha256((a.gdn_pair_dir/item).read_bytes()).hexdigest()
        for item in ('extension/pr17640_gdn_pair.so','tools/'+pair_worker,'source-provenance.json','package-isolation.json','status.json')} if a.gdn_pair_dir else {}),
    'scope': manifest.get('experiment_scope', 'a1 production command reproduced on a2; host driver differs; offline measurements only'),
    'differences': (['Entire audited runtime image differs; not a single-variable production A/B', 'Explicit bash entrypoint and empty initial PYTHONPATH prevent inherited bootstrap hooks'] if a.compatibility_image_id else []) + ['host=127.0.0.1', f'port={a.port}', 'isolated model and compilation cache paths',
                    'explicit device exposure in rootless Podman'] +
                   (['WY OPP/extension; '+('paired upstream H/O' if a.gdn_pair_dir else 'original FwdH/FwdO retained')] if a.native_wy_dir else []) +
                   (['paired upstream FwdH/FwdO and NZ state binding; changes both kernels'] if a.gdn_pair_dir else []) +
                   (['isolated FwdO synchronization package and guarded worker'] if a.maskfix_dir else []) + (['profiler configured, inactive during benchmark'] if a.profiler else []),
}, indent=2) + '\n')
args = ['podman', 'run', '-d', '--pull=never', '--name', a.container,
        '--network', 'host', '--group-add', 'keep-groups', '--ipc', 'shareable', '--shm-size', '32g']
if a.compatibility_image_id:
    # This budget image defaults to an unrelated C8 sitecustomize hook.
    args += ['--env', 'PYTHONPATH=', '--env', 'QWEN36_C8_ENABLED=0']
for device in ('davinci0', 'davinci1', 'davinci_manager', 'devmm_svm', 'hisi_hdc'):
    args += ['--device', '/dev/' + device]
mounts = [(a.model_dir.resolve(), model_mount, 'ro'),
          (a.out.resolve(), '/experiment', 'rw'),
          ((a.out / 'cache').resolve(), '/root/.cache', 'rw'),
          (a.start_script.resolve(), '/workspace/start_vllm_ascend.sh', 'ro'),
          (frozen.resolve(), '/workspace/backend-primary.manifest.json', 'ro')]
if a.gdn_pair_dir:
    mounts.append((a.gdn_pair_dir.resolve(), '/gdn-pair', 'ro'))
if a.maskfix_dir:
    mounts.append((a.maskfix_dir.resolve(), '/maskfix', 'ro'))
if a.native_wy_dir:
    mounts.append((a.native_wy_dir.resolve(), '/native-wy', 'ro'))
for source in ('/usr/local/bin/npu-smi', '/etc/ascend_install.info',
               '/usr/local/Ascend/driver/version.info', '/usr/local/Ascend/driver/lib64', '/usr/local/dcmi'):
    mounts.append((source, source, 'ro'))
for source, destination, mode in mounts:
    args += ['-v', f'{source}:{destination}:{mode}']
args += ['--entrypoint', '/bin/bash', expected, '/workspace/start_vllm_ascend.sh'] if a.compatibility_image_id else [expected, 'bash', '/workspace/start_vllm_ascend.sh']
subprocess.run(args, check=True)
