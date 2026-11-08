#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

: "${MODEL_DIR:=/model}"
: "${PORT:=18025}"
: "${SERVED_MODEL_NAME:=qwen36-w8a8-wy}"
: "${MAX_MODEL_LEN:=114688}"
: "${MAX_BATCHED_TOKENS:=2048}"
: "${MEMORY_UTILIZATION:=0.90}"
: "${MAX_NUM_SEQS:=1}"
: "${GRAPH_CAPTURE_SIZES:=1,2}"
: "${COMPILATION_MODE:=0}"
: "${GRAPH_MODE:=FULL_DECODE_ONLY}"
: "${ENFORCE_EAGER:=0}"
: "${ASCEND_CUSTOM_OPP_PATH:=/pr17640-opp/vendors/custom_transformer}"
: "${PR17640_WY_ENABLED:=1}"

[[ "$PR17640_WY_ENABLED" == 1 ]] || { echo 'PR17640_WY_ENABLED must be 1 for this variant' >&2; exit 2; }
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export ASCEND_CUSTOM_OPP_PATH
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PYTHONPATH="${PYTHONPATH:-}:/pr17640-wy"

args=(--no-enable-prefix-caching --tensor-parallel-size 2 --dtype float16
      --block-size 64 --max-model-len "$MAX_MODEL_LEN"
      --gpu-memory-utilization "$MEMORY_UTILIZATION"
      --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED_TOKENS"
      --served-model-name "$SERVED_MODEL_NAME" --host "${HOST:-0.0.0.0}" --port "$PORT")
if [[ "$ENFORCE_EAGER" == 1 ]]; then
  args+=(--enforce-eager)
else
  args+=(--compilation-config "{\"mode\":$COMPILATION_MODE,\"cudagraph_mode\":\"$GRAPH_MODE\",\"cudagraph_capture_sizes\":[$GRAPH_CAPTURE_SIZES]}")
fi
exec vllm serve "$MODEL_DIR" "${args[@]}"
