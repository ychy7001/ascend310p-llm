#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
# 先同步本仓库到code；源码副本和构建输出目录须全新。
: "${BUILD_DIR:?fresh directory under work}"
: "${SOURCE_DIR:?fresh source directory under work}"
[[ "$BUILD_DIR" == "$work/"* && "$SOURCE_DIR" == "$work/"* && ! -e "$BUILD_DIR" && ! -e "$SOURCE_DIR" ]]
podman run --rm --network none -v "$work:/work:rw" -v "$code:/project:ro" \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default \
 -e SOURCE_DIR="/work/${SOURCE_DIR#"$work"/}" \
 -e BUILD_DIR="/work/${BUILD_DIR#"$work"/}" \
 localhost/qwen36-mindieq-embedded:20261013 bash -lc '
 set -e
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 cp -r /project/experimental/mindieq_grouped_kernel "$SOURCE_DIR"
 python3 /project/tools/prepare_mindieq_grouped_header.py --source /work/ops-nn/build/tbe/ascendc/quant_batch_matmul_v3/arch20/quant_batch_matmul_v3_pertoken_arch20.h --output "$SOURCE_DIR/mindieq_grouped_arch20.h"
 cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -DPROJECT_REPO=/project -DSOC_VERSION=Ascend310P3 -DASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest -DCMAKE_BUILD_TYPE=Release
 cmake --build "$BUILD_DIR" -j4
 '
