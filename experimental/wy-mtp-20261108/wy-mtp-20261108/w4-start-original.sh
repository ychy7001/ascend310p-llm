#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export QWEN36_PROFILE=${QWEN36_PROFILE:-long-prefill}
export QWEN36_C8_ENABLED=${QWEN36_C8_ENABLED:-0}
export ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-0}
export PORT=${PORT:-18024}
export ASYNC_SCHEDULING=${ASYNC_SCHEDULING:-1}
export LANGUAGE_MODEL_ONLY=${LANGUAGE_MODEL_ONLY:-0}
export QWEN36_SYNC_METADATA=${QWEN36_SYNC_METADATA:-0}
source /opt/qwen36/configs/tp1-profile.sh
export QWEN36_GDN_ENABLED=1
export QWEN36_C8_LIBRARY=/custom/libqwen36_c8.so
export QWEN36_GDN_LIBRARY=/custom/pr17640_gdn.so
export QWEN36_GDN_SOURCE=/gdn-source.py
export QWEN36_MINDIEQ_LIBRARY=/custom/mindieq_grouped.so
export QWEN36_EXTENSION=/custom/qwen36_int4_custom.so
export QWEN36_RUNTIME=/w4only
export PYTHONPATH=/c8bootstrap
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
for flag in ENABLE_PREFIX_CACHING ASYNC_SCHEDULING LANGUAGE_MODEL_ONLY QWEN36_SYNC_METADATA; do
  value=${!flag:-0}
  [[ "$value" == 0 || "$value" == 1 ]] || { echo "$flag must be 0 or 1" >&2; exit 2; }
done
if [[ "$ENABLE_PREFIX_CACHING" == 1 && "$MAX_BATCHED_TOKENS" -lt 1152 ]]; then
  echo 'prefix caching requires MAX_BATCHED_TOKENS>=1152 on this hybrid 310P path' >&2
  exit 2
fi
# CANN environment scripts are not compatible with bash nounset.
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /opp/bin/set_env.bash
set -u
export ASCEND_CUSTOM_OPP_PATH=/gdn-opp:/opp
export LD_LIBRARY_PATH=/gdn-opp/op_api/lib:${LD_LIBRARY_PATH:-}
args=()
if [[ "$ENFORCE_EAGER" == 1 ]]; then
  args+=(--enforce-eager)
else
  args+=(--compilation-config "{\"mode\":$COMPILATION_MODE,\"cudagraph_mode\":\"$GRAPH_MODE\",\"cudagraph_capture_sizes\":[$GRAPH_CAPTURE_SIZES]}")
fi
args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}")
if [[ "$ENABLE_PREFIX_CACHING" == 1 ]]; then args+=(--enable-prefix-caching); else args+=(--no-enable-prefix-caching); fi
if [[ "${ASYNC_SCHEDULING:-1}" == 0 ]]; then args+=(--no-async-scheduling); fi
if [[ "${LANGUAGE_MODEL_ONLY:-0}" == 1 ]]; then args+=(--language-model-only); fi
if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"); fi
exec vllm serve /model "${args[@]}" --served-model-name "${SERVED_MODEL_NAME:-qwen36-w4a8-c8}" \
  --tensor-parallel-size 1 --dtype float16 --block-size 64 \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$MEMORY_UTILIZATION" \
  --max-num-seqs "${MAX_NUM_SEQS:-1}" --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --host "${HOST:-0.0.0.0}" --port "$PORT"
