#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${MODEL_DIR:?Set MODEL_DIR to a standalone checkpoint}"
: "${OPP_DIR:?Set OPP_DIR to the built vendors/custom_transformer directory}"
: "${EXTENSION:?Set EXTENSION to qwen36_int4_custom.so}"
TP=${TP:-1}
MTP=${MTP:-0}
if [[ "$TP" != 1 && "$TP" != 2 ]]; then echo 'TP must be 1 or 2' >&2; exit 2; fi
if [[ "$MTP" != 0 && "$MTP" != 1 ]]; then echo 'MTP must be 0 or 1' >&2; exit 2; fi
devices=(--device /dev/davinci0)
if [[ "$TP" == 2 ]]; then devices+=(--device /dev/davinci1); fi
podman run --rm --name "${CONTAINER_NAME:-qwen36-public}" --network host \
  --group-add keep-groups --shm-size 16g \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e PYTHONPATH=/bootstrap \
  -e TP="$TP" -e MTP="$MTP" -e PORT="${PORT:-18015}" \
  -e QWEN36_GROUPED_CHUNK_TOKENS="${QWEN36_GROUPED_CHUNK_TOKENS:-512}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" \
  -e MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.90}" \
  -v "$MODEL_DIR:/model:ro" -v "$project_root/runtime:/w4only:ro" \
  -v "$project_root/bootstrap:/bootstrap:ro" \
  -v "$EXTENSION:/custom/qwen36_int4_custom.so:ro" -v "$OPP_DIR:/opp:ro" \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  "${devices[@]}" --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  "${IMAGE:-quay.io/ascend/vllm-ascend:v0.23.0.post1-310p}" bash -lc '
    set -e
    triton_path=/usr/local/python3.12.13/lib/python3.12/site-packages/triton
    if [[ -d "$triton_path" ]]; then mv "$triton_path" /opt/triton-disabled; fi
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /opp/bin/set_env.bash
    export ASCEND_CUSTOM_OPP_PATH=/opp
    python3 -c "import sitecustomize; print(sitecustomize.__file__)"
    args=(--served-model-name qwen36-w4a8 --tensor-parallel-size "$TP"
      --dtype float16 --max-model-len "$MAX_MODEL_LEN"
      --gpu-memory-utilization "$MEMORY_UTILIZATION" --enforce-eager
      --host 0.0.0.0 --port "$PORT")
    if [[ "$MTP" == 1 ]]; then
      args+=(--speculative-config '\''{"method":"mtp","num_speculative_tokens":1}'\'')
    fi
    exec vllm serve /model "${args[@]}"
  '
