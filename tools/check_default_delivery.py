#!/usr/bin/env python3
"""Check that the default delivery, evidence and deploy entry points agree offline."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def require(condition, reason):
    if not condition:
        raise SystemExit(reason)


def main():
    variants = load('deploy/variants/qwen36.json')
    key = variants['default_variant']
    variant = variants['variants'][key]
    require(variant['status'] == 'default-delivery', 'Default variant must be released')
    release = load(variant['evidence'])
    require(release['status'] == 'default-delivery' and release['default_variant'] == key,
            'Default release registration differs')
    require(variant['registry_image'] == release['image'], 'Default image differs')
    require(variant['image_id'] == release['image_id'], 'Default config ID differs')
    require(variant['registry_tag'] == release['version_tag'], 'Version tag differs')
    require(variant['registry_default_tag'] == release['default_tag'], 'Default tag differs')
    require(variant['rollback_registry_image'] == release['rollback_image'], 'Rollback differs')
    for path, expected in {**release['sources'], **release['evidence_sha256']}.items():
        require(sha(path) == expected, f'Source/evidence hash changed: {path}')
    registry = release['registry']
    require(release['image'].endswith('@' + registry['manifest_digest']), 'Manifest reference differs')
    require('sha256:' + sha(registry['manifest_file']) == registry['manifest_digest'],
            'Remote manifest bytes do not match registry digest')
    manifest = load(registry['manifest_file'])
    require(manifest['config']['digest'] == registry['config_digest'] == 'sha256:' + release['image_id'],
            'Remote config does not match qualified image')
    for gate in ('version_and_default_manifests_equal', 'pull_by_digest_verified',
                 'pull_by_default_tag_verified', 'rollback_pull_verified'):
        require(registry[gate] is True, f'Registry gate missing: {gate}')
    qualification = release['qualification']
    require(qualification['performance'] == qualification['quality'] == 'passed',
            'Performance and quality gates must pass')
    quality = load(qualification['quality_record'])
    require(quality['same_call_bitwise_equal'] is True and quality['audit_process_exit_code'] == 0,
            'Real operator parity gate failed')
    require(sum(row['checks'] for row in quality['same_call_per_rank']) ==
            qualification['same_call_fp16_bitwise_equal_checks'], 'Parity check count differs')
    package = load(qualification['packaged_record'])
    require(package['image_id'] == release['image_id'], 'Qualified package differs')
    smoke = package['runtime_smoke']
    for gate in ('decode_graph_capture', 'decode_graph_replay', 'runtime_cli_matches_qualified_overlay'):
        require(smoke[gate] is True, f'Smoke gate failed: {gate}')
    require(smoke['container_exit_code'] == 0 and smoke['needle']['retrieved'] is True,
            'Packaged inference failed')
    base = ROOT / 'deploy/w8a8-tp2-wy-mtp'
    config = json.loads((base / 'qwen36-wy-mtp-production.manifest.json').read_text())
    require(config['image'] == release['image'], 'Deployment manifest differs')
    env = dict(line.split('=', 1) for line in (base / 'qwen36-wy-mtp-image.env.example').read_text().splitlines()
               if line and not line.startswith('#'))
    for name, value in env.items():
        if name in config['env'] and name not in ('MODEL_DIR', 'PORT'):
            require(config['env'][name] == value, f'Default profile differs: {name}')
    require(config['env']['QWEN36_EAGER_MASK_REUSE'] == '1', 'Mask reuse must be enabled by default')
    for path, prefix in [(base / 'qwen36-wy-mtp-image.env.example', 'IMAGE='),
                         (base / 'qwen36-wy-mtp-production.container', 'Image=')]:
        refs = [line[len(prefix):] for line in path.read_text().splitlines() if line.startswith(prefix)]
        require(refs == [release['image']], f'Deployment image differs: {path.name}')
    for path in ('deploy/README.md', 'docs/qwen36/w8a8-262k.md', 'docs/qwen36/w8a8-wy-delivery.md'):
        refs = re.findall(r'registry\.cn-hangzhou\.aliyuncs\.com/meetai/llm-service-vllm-ascend@sha256:[a-f0-9]{64}',
                          (ROOT / path).read_text())
        require(refs and refs[0] == release['image'], f'Default documentation differs: {path}')
    for path in ('README.md', 'README.en.md'):
        require('docs/qwen36/w8a8-wy-delivery.md' in (ROOT / path).read_text(),
                f'Default delivery link missing: {path}')
    checkpoint = (ROOT / release['model']['checkpoint_sha256_file']).read_text().splitlines()
    require(len(checkpoint) == release['model']['files'] and
            sum('.safetensors' in line and not line.endswith('.json') for line in checkpoint)
            == release['model']['weight_shards'], 'Checkpoint inventory differs')
    print(f"Default delivery verified: {release['release_id']} ({release['image']})")


if __name__ == '__main__':
    main()
