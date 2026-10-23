#!/usr/bin/env bash
# 隔离生成外部绑定；工作目录需全新，原始外部源码不改写。
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
: "${SOURCE_DIR:?set fresh source directory under $work}"
: "${BUILD_DIR:?set fresh build directory under $work}"
[[ "$SOURCE_DIR" == "$work/"* && "$BUILD_DIR" == "$work/"* ]]
[[ ! -e "$SOURCE_DIR" && ! -e "$BUILD_DIR" ]]
python3 "$code/tools/patch_mindieq_queue_binding.py" --caller-workspace \
 --source "$work/ops-nn/matmul/quant_batch_matmul_v4/tools/torch_extension/qmm_v5_ext.cpp" \
 --output "$SOURCE_DIR/qmm_v5_ext.cpp"
cp "$work/ops-nn/matmul/quant_batch_matmul_v4/tools/torch_extension/CMakeLists.txt" "$SOURCE_DIR/CMakeLists.txt"
# 适配容器x86_64 CANN布局及固定外部vendor，不修改原CMake。
python3 - "$SOURCE_DIR/CMakeLists.txt" <<'PYCM'
import sys
from pathlib import Path
p=Path(sys.argv[1]);s=p.read_text()
s=s.replace('aarch64-linux','x86_64-linux').replace('${VENDOR_INC}/aclnnop','${VENDOR_INC}')
s=s.replace('${CANN_ROOT}/include','${CANN_ROOT}/x86_64-linux/include').replace('${CANN_ROOT}/lib64','${CANN_ROOT}/x86_64-linux/lib64')
s=s.replace('set(VENDOR_LIB "${CANN_ROOT}/opp/vendors/custom_transformer/op_api/lib")','set(VENDOR_LIB "/work/ops-nn/build")')
s=s.replace('set(VENDOR_INC "${CANN_ROOT}/opp/vendors/custom_transformer/op_api/include")','set(VENDOR_INC "/work/ops-nn/matmul/quant_batch_matmul_v4/op_host/op_api")')
p.write_text(s)
PYCM
# 使用固定外部源码、x86_64目录和/work/ops-nn/build；无NPU设备挂载。
podman run --rm --network none -v "$work:/work:rw" \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default \
 -e SOURCE_DIR="/work/${SOURCE_DIR#"$work"/}" -e BUILD_DIR="/work/${BUILD_DIR#"$work"/}" \
 localhost/qwen36-mindieq-embedded:20261013 bash -lc '
 set -e
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 cmake -S "$SOURCE_DIR" -B "$BUILD_DIR"
 cmake --build "$BUILD_DIR" -j4
 sha256sum "$BUILD_DIR/libqmm_v5_ext.so"
 '
