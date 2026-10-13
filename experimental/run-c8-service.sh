#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${MODEL_DIR:?}" "${OPP_DIR:?}" "${EXTENSION:?}" "${C8_LIBRARY:?}"
MTP=${MTP:-0}
NUM_SPECULATIVE_TOKENS=${NUM_SPECULATIVE_TOKENS:-1}
ENFORCE_EAGER=${ENFORCE_EAGER:-1}
QWEN36_C8_ALLOW_GRAPH=${QWEN36_C8_ALLOW_GRAPH:-0}
QWEN36_C8_FORCE_UPDATABLE_GRAPH=${QWEN36_C8_FORCE_UPDATABLE_GRAPH:-0}
GRAPH_MODE=${GRAPH_MODE:-FULL_DECODE_ONLY}
COMPILATION_MODE=${COMPILATION_MODE:-3}
GRAPH_CAPTURE_SIZES=${GRAPH_CAPTURE_SIZES:-1}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-256}
if [[ "$MTP" != 0 && "$MTP" != 1 ]]; then echo 'MTP must be 0 or 1' >&2; exit 2; fi
if [[ "$NUM_SPECULATIVE_TOKENS" != 1 && "$NUM_SPECULATIVE_TOKENS" != 2 ]]; then
  echo 'NUM_SPECULATIVE_TOKENS must be 1 or 2' >&2; exit 2
fi
if [[ "$MTP" == 1 && "$NUM_SPECULATIVE_TOKENS" == 2 && "${QWEN36_C8_ENABLED:-1}" == 1 ]]; then
  [[ "${QWEN36_C8_ALLOW_MTP2:-0}" == 1 ]] || { echo 'C8 MTP2 is experimental; set QWEN36_C8_ALLOW_MTP2=1' >&2; exit 2; }
fi
if [[ "$ENFORCE_EAGER" != 0 && "$ENFORCE_EAGER" != 1 ]]; then echo 'ENFORCE_EAGER must be 0 or 1' >&2; exit 2; fi
if [[ "$ENABLE_PREFIX_CACHING" != 0 && "$ENABLE_PREFIX_CACHING" != 1 ]]; then
  echo 'ENABLE_PREFIX_CACHING must be 0 or 1' >&2; exit 2
fi
if [[ "$ENABLE_PREFIX_CACHING" == 1 && "$MAX_BATCHED_TOKENS" -lt 1152 ]]; then
  echo 'ENABLE_PREFIX_CACHING=1 requires MAX_BATCHED_TOKENS>=1152 on the hybrid 310P path' >&2; exit 2
fi
if [[ "$ENFORCE_EAGER" == 0 && "$QWEN36_C8_ALLOW_GRAPH" != 1 ]]; then
  echo 'ENFORCE_EAGER=0 requires QWEN36_C8_ALLOW_GRAPH=1' >&2; exit 2
fi
profile_mount=()
if [[ -n "${PROFILE_DIR:-}" ]]; then
  mkdir -p "$PROFILE_DIR"
  profile_mount=(-v "$PROFILE_DIR:/profile:rw")
fi
gdn_mount=()
if [[ "${QWEN36_GDN_ENABLED:-0}" == 1 ]]; then
  : "${GDN_OPP_DIR:?}" "${GDN_LIBRARY:?}" "${GDN_SOURCE:?}"
  gdn_mount=(-v "$GDN_OPP_DIR:/gdn-opp:ro" -v "$GDN_LIBRARY:/custom/pr17640_gdn.so:ro" -v "$GDN_SOURCE:/gdn-source.py:ro")
fi
mindieq_mount=()
if [[ "${QWEN36_MINDIEQ_ENABLED:-0}" == 1 ]]; then
  : "${MINDIEQ_LIBRARY:?set native-layout grouped experiment library}"
  mindieq_mount=(-v "$MINDIEQ_LIBRARY:/custom/mindieq_grouped.so:ro")
fi
podman run --rm --name "${CONTAINER_NAME:-qwen36-c8-bringup}" --network host \
  "${profile_mount[@]}" "${gdn_mount[@]}" "${mindieq_mount[@]}" \
  --group-add keep-groups --shm-size 16g \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e PYTHONPATH=/c8bootstrap \
  -e QWEN36_C8_LIBRARY=/custom/libqwen36_c8.so \
  -e QWEN36_C8_ENABLED="${QWEN36_C8_ENABLED:-1}" \
  -e QWEN36_C8_FUSED_WRITE="${QWEN36_C8_FUSED_WRITE:-0}" \
  -e QWEN36_C8_POOL_MODE="${QWEN36_C8_POOL_MODE:-independent}" \
  -e QWEN36_GROUPED_CHUNK_TOKENS="${QWEN36_GROUPED_CHUNK_TOKENS:-512}" \
  -e QWEN36_ROUTE_TRACE_PATH="${QWEN36_ROUTE_TRACE_PATH:-}" \
  -e QWEN36_ROUTE_TRACE_CALLS="${QWEN36_ROUTE_TRACE_CALLS:-256}" \
  -e QWEN36_C8_ALLOW_GRAPH="$QWEN36_C8_ALLOW_GRAPH" \
  -e QWEN36_C8_FORCE_UPDATABLE_GRAPH="$QWEN36_C8_FORCE_UPDATABLE_GRAPH" \
  -e QWEN36_C8_DEVICE_LENGTH="${QWEN36_C8_DEVICE_LENGTH:-0}" \
  -e QWEN36_C8_ALLOW_MTP2="${QWEN36_C8_ALLOW_MTP2:-0}" \
  -e ENFORCE_EAGER="$ENFORCE_EAGER" \
  -e GRAPH_MODE="$GRAPH_MODE" -e GRAPH_CAPTURE_SIZES="$GRAPH_CAPTURE_SIZES" \
  -e COMPILATION_MODE="$COMPILATION_MODE" \
  -e QWEN36_MINDIEQ_CUSTOM_QUANT="${QWEN36_MINDIEQ_CUSTOM_QUANT:-0}" \
  -e QWEN36_MINDIEQ_MIN_TOKENS="${QWEN36_MINDIEQ_MIN_TOKENS:-17}" \
  -e QWEN36_MINDIEQ_ENABLED="${QWEN36_MINDIEQ_ENABLED:-0}" \
  -e QWEN36_MINDIEQ_LIBRARY=/custom/mindieq_grouped.so \
  -e QWEN36_GDN_ENABLED="${QWEN36_GDN_ENABLED:-0}" \
  -e QWEN36_GDN_LIBRARY=/custom/pr17640_gdn.so -e QWEN36_GDN_SOURCE=/gdn-source.py \
  -e MTP="$MTP" \
  -e NUM_SPECULATIVE_TOKENS="$NUM_SPECULATIVE_TOKENS" \
  -e ENABLE_PROFILER="${PROFILE_DIR:+1}" \
  -e PORT="${PORT:-18016}" -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" \
  -e MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.90}" \
  -e MAX_BATCHED_TOKENS="$MAX_BATCHED_TOKENS" \
  -e ENABLE_PREFIX_CACHING="$ENABLE_PREFIX_CACHING" \
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
    if [[ "$QWEN36_GDN_ENABLED" == 1 ]]; then
      export ASCEND_CUSTOM_OPP_PATH="/gdn-opp:/opp"
      export LD_LIBRARY_PATH="/gdn-opp/op_api/lib:$LD_LIBRARY_PATH"
    fi
    python3 -c "import sitecustomize; print(sitecustomize.__file__)"
    args=()
    if [[ "$ENABLE_PROFILER" == 1 ]]; then
      args+=(--profiler-config '\''{"profiler":"torch","torch_profiler_dir":"/profile","torch_profiler_with_stack":false,"ignore_frontend":true}'\'')
    fi
    if [[ "$ENFORCE_EAGER" == 1 ]]; then
      args+=(--enforce-eager)
    else
      args+=(--compilation-config "{\"mode\":$COMPILATION_MODE,\"cudagraph_mode\":\"$GRAPH_MODE\",\"cudagraph_capture_sizes\":[$GRAPH_CAPTURE_SIZES]}")
    fi
    if [[ "$MTP" == 1 ]]; then
      args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}")
    fi
    if [[ "$ENABLE_PREFIX_CACHING" == 1 ]]; then
      args+=(--enable-prefix-caching)
    else
      args+=(--no-enable-prefix-caching)
    fi
    exec vllm serve /model "${args[@]}" --served-model-name qwen36-w4a8-c8 \
      --tensor-parallel-size 1 --dtype float16 --block-size 64 \
      --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$MEMORY_UTILIZATION" \
      --max-num-seqs 1 --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
      --host 0.0.0.0 --port "$PORT"
  '
