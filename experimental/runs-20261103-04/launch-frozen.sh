#!/usr/bin/env bash
# 固定已保留镜像，隔离实验服务；不删除/重启历史容器。
set -euo pipefail
: "${EXPERIMENT_DIR:?Set an absolute experiment output directory}"
: "${IMAGE:=b381d8e5ed45c11633a7bd07a50a0f5fcf8cf71bc797f1b543d589ab07dd759d}"
: "${MODEL_DIR:=/path/to/lab-runtime/qwen36-public-release-20261012/model}"
: "${CONTAINER_NAME:=qwen36-baseline-20261103}"
: "${PORT:=18103}"
entrypoint=()
command_args=()
if [[ "${EXPERIMENT_START_SCRIPT:-0}" == 1 ]]; then
  entrypoint=(--entrypoint bash)
  command_args=(/experiment/start-experiment.sh)
fi
mkdir -p "$EXPERIMENT_DIR"
podman run -d --pull=never --name "$CONTAINER_NAME" --network host \
  --group-add keep-groups --shm-size 16g \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e QWEN36_PROFILE=long-prefill -e QWEN36_C8_ENABLED=0 \
  -e NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-2}" \
  -e MAX_MODEL_LEN=114688 -e MAX_BATCHED_TOKENS=2048 \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" -e MEMORY_UTILIZATION=0.95 \
  -e ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}" -e ASYNC_SCHEDULING=1 \
  -e LANGUAGE_MODEL_ONLY=0 -e QWEN36_SYNC_METADATA=0 -e OMP_NUM_THREADS=1 \
  -e HOST=127.0.0.1 -e PORT="$PORT" \
  -e EXPERIMENT_MTP_TOKENS="${EXPERIMENT_MTP_TOKENS:-2}" \
  -e EXPERIMENT_PROFILE="${EXPERIMENT_PROFILE:-0}" \
  -v "$MODEL_DIR:/model:ro" -v "$EXPERIMENT_DIR:/experiment" \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  "${entrypoint[@]}" "$IMAGE" "${command_args[@]}"
