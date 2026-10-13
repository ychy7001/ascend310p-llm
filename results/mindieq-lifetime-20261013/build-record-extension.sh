#!/usr/bin/env bash
# 隔离生成外部绑定；工作目录需全新，原始外部源码不改写。
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
: "${SOURCE_DIR:?set fresh source directory under $work}"
: "${BUILD_DIR:?set fresh build directory under $work}"
: "${RECORD_SCOPE:=all}"
[[ "$SOURCE_DIR" == "$work/"* && "$BUILD_DIR" == "$work/"* ]]
[[ ! -e "$SOURCE_DIR" && ! -e "$BUILD_DIR" ]]
python3 "$code/tools/patch_mindieq_queue_binding.py" --record-stream --record-stream-scope "$RECORD_SCOPE" \
 --source "$work/ops-nn/matmul/quant_batch_matmul_v4/tools/torch_extension/qmm_v5_ext.cpp" \
 --output "$SOURCE_DIR/qmm_v5_ext.cpp"
cp "$work/ops-nn/matmul/quant_batch_matmul_v4/tools/torch_extension/CMakeLists.txt" "$SOURCE_DIR/CMakeLists.txt"
# CMake与外部固定源码一致，引用/work/ops-nn/build；无NPU设备挂载。
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
