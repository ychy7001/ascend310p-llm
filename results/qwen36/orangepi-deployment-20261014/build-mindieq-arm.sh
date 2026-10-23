#!/usr/bin/env bash
set -euo pipefail
# 外部 MindIEQ/CANN 头来自已固定的 a2 构建树，版权与许可证见原文件。
# 外部头仅置于实验机，不纳入本公开仓库。
docker run --rm --name qwen36-arm-mindieq-build --entrypoint bash \
 -v /data/qwen36-deploy:/work -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=2 \
 local/vllm-ascend:v0.23.0-310p-openeuler-orangepi-rcfix -lc '
 set -e
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 src=/work/artifacts/mindieq-arm-source
 test ! -e "$src"
 cp -r /work/repo/experimental/mindieq_grouped_kernel "$src"
 python3 /work/repo/tools/prepare_mindieq_grouped_header.py \
  --source /work/mindieq-headers/quant_batch_matmul_v3/arch20/quant_batch_matmul_v3_pertoken_arch20.h \
  --native-layout --output "$src/mindieq_grouped_arch20.h"
 cmake -S "$src" -B /work/artifacts/mindieq-arm-build \
  -DPROJECT_REPO=/work/repo -DSOC_VERSION=Ascend310P1 \
  -DASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest -DCMAKE_BUILD_TYPE=Release
 cmake --build /work/artifacts/mindieq-arm-build -j2
 '
