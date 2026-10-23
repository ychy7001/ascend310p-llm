#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /experiment/community
cmake -S . -B /experiment/runtime-bridge-build \
 -DSOC_VERSION=ascend310p3 -DCMAKE_BUILD_TYPE=Release \
 -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
 -DTORCH_NPU_PATH="$(python3 -c 'import importlib.util; print(next(iter(importlib.util.find_spec("torch_npu").submodule_search_locations)))')" \
 -Dpybind11_DIR="$(python3 -m pybind11 --cmakedir)" \
 -DPYTHON_EXECUTABLE="$(command -v python3)" \
 -DPYTHON_INCLUDE_PATH="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')" \
 -DCMAKE_INSTALL_PREFIX=/experiment/community/vllm_ascend
cmake --build /experiment/runtime-bridge-build -j2
cmake --install /experiment/runtime-bridge-build
PYTHONPATH=/experiment/community:/experiment/vllm-fork python3 -c 'import torch; import torch_npu; import vllm_ascend.vllm_ascend_C; print("BRIDGE_IMPORT_OK")'
