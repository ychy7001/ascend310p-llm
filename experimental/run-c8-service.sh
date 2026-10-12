#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${MODEL_DIR:?}" "${OPP_DIR:?}" "${EXTENSION:?}" "${C8_LIBRARY:?}"
MTP=${MTP:-0}
if [[ "$MTP" != 0 && "$MTP" != 1 ]]; then echo 'MTP must be 0 or 1' >&2; exit 2; fi
podman run --rm --name "${CONTAINER_NAME:-qwen36-c8-bringup}" --network host \
  --group-add keep-groups --shm-size 16g \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e PYTHONPATH=/c8bootstrap \
  -e QWEN36_C8_LIBRARY=/custom/libqwen36_c8.so \
  -e QWEN36_C8_ENABLED="${QWEN36_C8_ENABLED:-1}" \
  -e QWEN36_C8_FUSED_WRITE="${QWEN36_C8_FUSED_WRITE:-0}" \
  -e MTP="$MTP" \
  -e PORT="${PORT:-18016}" -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" \
  -e MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.90}" \
  -v "$MODEL_DIR:/model:ro" -v "$root/runtime:/w4only:ro" \
  -v "$root/bootstrap:/bootstrap:ro" -v "$root/experimental:/c8adapter:ro" \
  -v "$root/experimental/c8_bootstrap:/c8bootstrap:ro" \
  -v "$EXTENSION:/custom/qwen36_int4_custom.so:ro" -v "$C8_LIBRARY:/custom/libqwen36_c8.so:ro" \
  -v "$OPP_DIR:/opp:ro" \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  quay.io/ascend/vllm-ascend:v0.23.0.post1-310p bash -lc '
    set -e
    triton_path=/usr/local/python3.12.13/lib/python3.12/site-packages/triton
    if [[ -d "$triton_path" ]]; then mv "$triton_path" /opt/triton-disabled; fi
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /opp/bin/set_env.bash
    export ASCEND_CUSTOM_OPP_PATH=/opp
    python3 -c "import sitecustomize; print(sitecustomize.__file__)"
    args=()
    if [[ "$MTP" == 1 ]]; then
      args+=(--speculative-config '\''{"method":"mtp","num_speculative_tokens":1}'\'')
    fi
    exec vllm serve /model "${args[@]}" --served-model-name qwen36-w4a8-c8 \
      --tensor-parallel-size 1 --dtype float16 --enforce-eager --block-size 64 \
      --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$MEMORY_UTILIZATION" \
      --max-num-seqs 1 --max-num-batched-tokens 256 --no-enable-prefix-caching \
      --host 0.0.0.0 --port "$PORT"
  '
