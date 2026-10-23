#!/usr/bin/env bash
set -euo pipefail
docker run --rm --name qwen36-arm-c8-build --entrypoint bash \
 -v /data/qwen36-deploy:/work \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e OMP_NUM_THREADS=2 \
 local/vllm-ascend:v0.23.0-310p-openeuler-orangepi-rcfix -lc '
 set -e
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 cmake -S /work/repo/experimental/ascend_c8_kernel -B /work/artifacts/c8-build \
  -DSOC_VERSION=Ascend310P1 -DCMAKE_BUILD_TYPE=Release
 cmake --build /work/artifacts/c8-build -j2
 '
