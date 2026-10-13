#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reproducible TP1 112K product-candidate launcher.
# The underlying C8 service remains experimental until the graph backend fix
# is accepted upstream.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
profile=${QWEN36_PROFILE:-mtp1}

: "${MODEL_DIR:?set MODEL_DIR to the standalone checkpoint}"
: "${OPP_DIR:?set OPP_DIR to the custom OPP directory}"
: "${EXTENSION:?set EXTENSION to qwen36_int4_custom.so}"
: "${C8_LIBRARY:?set C8_LIBRARY to libqwen36_c8.so}"

case "$profile" in
  mtp1)
    mtp=1
    drafts=1
    capture=1,2
    allow_mtp2=0
    ;;
  mtp2)
    mtp=1
    drafts=2
    capture=1,2,3
    allow_mtp2=1
    ;;
  *)
    echo "QWEN36_PROFILE must be mtp1 or mtp2 (got: $profile)" >&2
    exit 2
    ;;
esac

: "${PORT:=18020}"
: "${CONTAINER_NAME:=qwen36-c8-112k-$profile}"

export CONTAINER_NAME PORT
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-114688}"
export MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.95}"
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-1024}"
export MTP="$mtp"
export NUM_SPECULATIVE_TOKENS="$drafts"
export QWEN36_C8_ALLOW_MTP2="$allow_mtp2"
export QWEN36_C8_ENABLED="${QWEN36_C8_ENABLED:-1}"
export QWEN36_C8_ALLOW_GRAPH=1
export QWEN36_C8_FORCE_UPDATABLE_GRAPH=1
export QWEN36_C8_DEVICE_LENGTH=1
export QWEN36_C8_FUSED_WRITE=1
export QWEN36_C8_POOL_MODE=split_by_type
export QWEN36_GROUPED_CHUNK_TOKENS="${QWEN36_GROUPED_CHUNK_TOKENS:-512}"
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
export COMPILATION_MODE=0
export GRAPH_MODE=FULL_DECODE_ONLY
export GRAPH_CAPTURE_SIZES="$capture"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"

echo "QWEN36_PROFILE=$profile MAX_MODEL_LEN=$MAX_MODEL_LEN " \
  "MTP=$MTP NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS " \
  "GRAPH_CAPTURE_SIZES=$GRAPH_CAPTURE_SIZES" >&2
exec bash "$root/experimental/run-c8-service.sh"
