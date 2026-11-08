#!/usr/bin/env python3
"""Create a reproducible build context for the TP2 W8A8 PR17640 WY image.

The binary inputs are deliberately supplied by path. This tool copies them
into a self-contained context and records SHA256 values; it does not download
models, CANN, or registry images.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def copy_file(source: Path, dest: Path) -> str:
    source = source.resolve(strict=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return sha256(source)


def copy_tree(source: Path, dest: Path) -> dict[str, str]:
    source = source.resolve(strict=True)
    result = {}
    for item in sorted(source.rglob('*')):
        if item.is_file():
            relative = item.relative_to(source)
            result[str(relative)] = copy_file(item, dest / relative)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-image', required=True)
    parser.add_argument('--wrapper', type=Path, required=True,
                        help='PR17640 chunk_gated_delta_rule.py')
    parser.add_argument('--binding', type=Path, required=True,
                        help='vllm_ascend_C extension')
    parser.add_argument('--custom-opp', type=Path, required=True)
    parser.add_argument('--tensor-cache-stub', type=Path)
    parser.add_argument('--skip-tensor-cache-stub', action='store_true',
                        help='Use the base image FLA utils unchanged')
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'output already exists: {args.output}')

    root = args.output
    artifacts = root / 'artifacts'
    start_script = Path(__file__).resolve().parents[1] / 'deploy/w8a8-tp2-wy/start-wy.sh'
    start_sha = copy_file(start_script, root / 'start-wy.sh')
    wrapper_sha = copy_file(args.wrapper, artifacts / 'chunk_gated_delta_rule.py')
    binding_sha = copy_file(args.binding, artifacts / args.binding.name)
    opp_hashes = copy_tree(args.custom_opp, artifacts / 'custom_transformer')
    if args.skip_tensor_cache_stub and args.tensor_cache_stub:
        raise SystemExit('--tensor-cache-stub and --skip-tensor-cache-stub are mutually exclusive')
    if not args.skip_tensor_cache_stub and not args.tensor_cache_stub:
        raise SystemExit('provide --tensor-cache-stub or --skip-tensor-cache-stub')
    stub_hash = None
    if not args.skip_tensor_cache_stub:
        stub_hash = copy_file(args.tensor_cache_stub,
                              root / 'vllm_stub/third_party/flash_linear_attention/ops/utils.py')

    dockerfile = f'''ARG BASE_IMAGE={args.base_image}\nFROM ${{BASE_IMAGE}}\n\n'''
    dockerfile += '''COPY start-wy.sh /opt/qwen36/start-wy.sh
COPY artifacts/chunk_gated_delta_rule.py /vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py
COPY artifacts/''' + args.binding.name + ''' /vllm-workspace/vllm-ascend/vllm_ascend/''' + args.binding.name + '''
COPY artifacts/custom_transformer /pr17640-opp/vendors/custom_transformer
COPY vllm_stub/third_party/flash_linear_attention/ops/utils.py /vllm-workspace/vllm/vllm/third_party/flash_linear_attention/ops/utils.py

ENV ASCEND_CUSTOM_OPP_PATH=/pr17640-opp/vendors/custom_transformer \\
    PR17640_WY_ENABLED=1

RUN mkdir -p /vllm-workspace/vllm/vllm/third_party/flash_linear_attention/ops && \\
    touch /vllm-workspace/vllm/vllm/third_party/__init__.py /vllm-workspace/vllm/vllm/third_party/flash_linear_attention/__init__.py /vllm-workspace/vllm/vllm/third_party/flash_linear_attention/ops/__init__.py

ENTRYPOINT ["bash", "/opt/qwen36/start-wy.sh"]
'''
    if args.skip_tensor_cache_stub:
        dockerfile = dockerfile.replace(
            'COPY vllm_stub/third_party/flash_linear_attention/ops/utils.py /vllm-workspace/vllm/vllm/third_party/flash_linear_attention/ops/utils.py\n',
            '')
        dockerfile = dockerfile.replace(
            'RUN mkdir -p /vllm-workspace/vllm/vllm/third_party/flash_linear_attention/ops && \\\n    touch /vllm-workspace/vllm/vllm/third_party/__init__.py /vllm-workspace/vllm/vllm/third_party/flash_linear_attention/__init__.py /vllm-workspace/vllm/vllm/third_party/flash_linear_attention/ops/__init__.py\n',
            '')
    (root / 'Containerfile').write_text(dockerfile)
    manifest = {
        'schema_version': 1,
        'variant': 'tp2-w8a8-wy',
        'base_image': args.base_image,
        'source_revision': args.source_revision,
        'artifacts': {
            'wrapper': wrapper_sha,
            'binding': binding_sha,
            'custom_opp': opp_hashes,
            'tensor_cache_stub': stub_hash,
            'start_script': start_sha,
        },
        'build_policy': 'staging-only; TP1 W4A8 runtime is packaged separately',
    }
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'context': str(root), 'manifest': manifest}, indent=2))


if __name__ == '__main__':
    main()
