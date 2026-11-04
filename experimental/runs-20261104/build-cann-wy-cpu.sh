#!/usr/bin/env bash
# 固定源码下的本机CPU构建；不挂NPU设备或驱动，各构建步骤分别记录退出码。
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 /audit/inspect-cann-wy-rootfs.py > /evidence/local-image-rootfs.json
python3 -m pip install build wheel setuptools ninja > /evidence/build-dependencies.log 2>&1
cd /source
set +e
bash build.sh --pkg --experimental --soc=ascend310p --vendor_name=custom --ops=fused_gdn_gating,chunk_gated_delta_rule_compute_wy -j16 > /evidence/custom-package-build.log 2>&1
package_rc=$?
printf '%s\n' "$package_rc" > /evidence/custom-package-build.returncode
bash build.sh --torch_extension --experimental > /evidence/python-wheel-build.log 2>&1
wheel_rc=$?
printf '%s\n' "$wheel_rc" > /evidence/python-wheel-build.returncode
bash build.sh --torch_extension --experimental --ops=fused_gdn_gating,chunk_gated_delta_rule_compute_wy --vendor_name=wy921_cann > /evidence/isolated-python-wheel-build.log 2>&1
isolated_wheel_rc=$?
printf '%s\n' "$isolated_wheel_rc" > /evidence/isolated-python-wheel-build.returncode
set -e
python3 - <<'PY'
import hashlib,json,pathlib
p=pathlib.Path('/source/build_out')
artifacts=[{'name':f.name,'size_bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(p.glob('*')) if f.is_file() and (f.suffix=='.whl' or f.suffix=='.run')]
pathlib.Path('/evidence/build-artifacts.json').write_text(json.dumps(artifacts,indent=2)+'\n')
PY
test "$package_rc" -eq 0
test "$wheel_rc" -eq 0
test "$isolated_wheel_rc" -eq 0
