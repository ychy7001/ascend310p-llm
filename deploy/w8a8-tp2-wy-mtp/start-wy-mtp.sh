#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
: "${MODEL_DIR:=/model}"
: "${PORT:=18025}"
: "${SERVED_MODEL_NAME:=qwen36-w8a8-wy-mtp}"
: "${MAX_MODEL_LEN:=262144}"
: "${MAX_BATCHED_TOKENS:=2048}"
: "${MEMORY_UTILIZATION:=0.90}"
: "${MAX_NUM_SEQS:=4}"
: "${NUM_SPECULATIVE_TOKENS:=2}"
: "${DRAFT_ENFORCE_EAGER:=1}"
if [[ ! "$NUM_SPECULATIVE_TOKENS" =~ ^[012]$ ]]; then
  echo 'NUM_SPECULATIVE_TOKENS must be 0, 1 or 2' >&2; exit 2
fi
: "${ENABLE_PREFIX_CACHING:=1}"
: "${ASYNC_SCHEDULING:=1}"
: "${ENFORCE_EAGER:=1}"
if [[ ! "$ENABLE_PREFIX_CACHING" =~ ^[01]$ ]]; then
  echo 'ENABLE_PREFIX_CACHING must be 0 or 1' >&2; exit 2
fi
# Keep capture count bounded: 310P event resources cannot hold every batch size.
# Intermediate batches use vLLM's standard padding to the next captured size.
capture_sizes=$((NUM_SPECULATIVE_TOKENS + 1))
for ((n=2; n<MAX_NUM_SEQS; n*=2)); do
  capture_sizes+=,$((n * (NUM_SPECULATIVE_TOKENS + 1)))
done
if (( MAX_NUM_SEQS > 1 )); then
  capture_sizes+=,$((MAX_NUM_SEQS * (NUM_SPECULATIVE_TOKENS + 1)))
fi
capture_sizes=${GRAPH_CAPTURE_SIZES:-$capture_sizes}
extra_args=()
if [[ "$ENFORCE_EAGER" == 1 ]]; then
  extra_args+=(--enforce-eager)
elif [[ "$ENFORCE_EAGER" == 0 ]]; then
  extra_args+=(--compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":[$capture_sizes]}")
else
  echo 'ENFORCE_EAGER must be 0 or 1' >&2; exit 2
fi
if [[ "$ASYNC_SCHEDULING" == 1 ]]; then
  extra_args+=(--async-scheduling)
elif [[ "$ASYNC_SCHEDULING" == 0 ]]; then
  extra_args+=(--no-async-scheduling)
else
  echo 'ASYNC_SCHEDULING must be 0 or 1' >&2; exit 2
fi
if [[ "$ENABLE_PREFIX_CACHING" == 1 ]]; then
  extra_args+=(--enable-prefix-caching)
else
  extra_args+=(--no-enable-prefix-caching)
fi
if [[ "$NUM_SPECULATIVE_TOKENS" != 0 ]]; then
  if [[ "$DRAFT_ENFORCE_EAGER" == 1 ]]; then
    extra_args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS,\"enforce_eager\":true}")
  elif [[ "$DRAFT_ENFORCE_EAGER" == 0 ]]; then
    extra_args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS,\"enforce_eager\":false}")
  else
    echo 'DRAFT_ENFORCE_EAGER must be 0 or 1' >&2; exit 2
  fi
fi
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export ASCEND_CUSTOM_OPP_PATH=/pr17640-opp/vendors/custom_transformer
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0,1}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export SOC_VERSION="${SOC_VERSION:-ascend310p1}"
export ASCEND_CUSTOM_PATH="${ASCEND_CUSTOM_PATH:-/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom}"
export QWEN36_C8_FORCE_UPDATABLE_GRAPH=1
export PYTHONPATH="/opt/qwen36/wy-compat:${PYTHONPATH:-}"
exec vllm serve "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL_NAME" --host "${HOST:-0.0.0.0}" --port "$PORT" \
  --tensor-parallel-size 2 --quantization ascend --dtype float16 \
  --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --kv-cache-dtype auto --mamba-ssm-cache-dtype float16 --trust-remote-code \
  --enable-chunked-prefill --enable-prompt-tokens-details \
  --reasoning-parser qwen3 --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder --allowed-local-media-path / \
  --additional-config '{"ascend_compilation_config":{"fuse_norm_quant":false}}' \
  "${extra_args[@]}"
