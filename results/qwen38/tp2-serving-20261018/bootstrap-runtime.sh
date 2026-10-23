#!/usr/bin/env bash
# Run inside the isolated qwen38-runtime-build container; no NPU required.
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export SOC_VERSION=ascend310p3 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4
python3 -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'setuptools-rust==1.13.0' 'transformers==5.14.1' 'fastapi==0.136.3' 'starlette==1.8.0' 'fastokens==0.2.0'
cd /experiment/vllm-fork
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.30.0+qwen38.3ab5dda python3 -m pip install --no-deps --no-build-isolation -e .
cd /experiment/community
SETUPTOOLS_SCM_PRETEND_VERSION=0.30.0+qwen38.ffa980c COMPILE_CUSTOM_KERNELS=0 python3 -m pip install --no-deps --no-build-isolation -e .
python3 -m pip freeze > /experiment/runtime-installed-freeze.txt
python3 -c 'import vllm; import transformers; print("VLLM", vllm.__version__, vllm.__file__); print("transformers", transformers.__version__)'
