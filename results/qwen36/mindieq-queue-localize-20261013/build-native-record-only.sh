#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
release=/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012
# 复现时输出目录必须全新；先同步最新tools/patch_native_queue_lifetime.py到code。
podman run --rm --network none -v "$work:/work:rw" -v "$code:/project:ro" \
 -v "$release/clean-build/community:/community:ro" \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default -e MAX_JOBS=4 \
 localhost/qwen36-mindieq-embedded:20261013 bash -lc '
 set -e
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 python3 /project/tools/patch_native_queue_lifetime.py --adapter-source /community/csrc/aclnn_torch_adapter/op_api_common.h --extension-source /project/tools/qwen36_int4_custom.cpp --mode record-only --output-dir /work/native-record-source-only
 mkdir /work/native-record-build-only
 python3 /project/tools/build_qwen36_int4_extension.py --community-root /community --source /work/native-record-source-only/qwen36_int4_custom.cpp --build /work/native-record-build-only > /work/native-record-build-only.txt 2>&1
 '
