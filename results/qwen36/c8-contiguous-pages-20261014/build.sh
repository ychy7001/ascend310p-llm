#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
release=/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012
[[ ! -e "$work/c8-contiguous-pages-build" ]]
podman run --rm --network none -v /home/ychy/ascend-lab-runtime/pr17640-product-20261013/code/experimental/ascend_c8_kernel:/source:ro -v "$work:/work:rw" \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default \
 localhost/qwen36-mindieq-embedded:20261013 bash -lc '
 set -e
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 cmake -S /source -B /work/c8-contiguous-pages-build -DSOC_VERSION=Ascend310P3 -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0 -DC8_NZ_TRANSDATA=ON -DC8_NZ_COMPACT_SCALE=ON -DC8_NZ_MAX_TILE=64 -DC8_NZ_PIPELINE=OFF -DC8_NZ_CONTIGUOUS_PAGES=ON -DCMAKE_BUILD_TYPE=Release
 cmake --build /work/c8-contiguous-pages-build -j4
 '
