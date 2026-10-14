#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reproducible TP1 112K product-candidate launcher.
# The underlying C8 service remains experimental until the graph backend fix
# is accepted upstream.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
profile=${QWEN36_PROFILE:-compat}

: "${MODEL_DIR:?set MODEL_DIR to the standalone checkpoint}"
: "${OPP_DIR:?set OPP_DIR to the custom OPP directory}"
: "${EXTENSION:?set EXTENSION to qwen36_int4_custom.so}"
: "${C8_LIBRARY:?set C8_LIBRARY to libqwen36_c8.so}"

case "$profile" in
  compat|mtp1)
    profile=compat
    mtp=1
    drafts=1
    capture=1,2
    allow_mtp2=0
    profile_description='compatibility baseline: MTP1, decode graph, conservative batching'
    default_batched_tokens=1024
    ;;
  decode|mtp2)
    profile=decode
    mtp=1
    drafts=2
    capture=1,2,3
    allow_mtp2=1
    profile_description='single-stream decode candidate: MTP2, decode graph'
    default_batched_tokens=4096
    ;;
  long-prefill)
    mtp=1
    drafts=2
    capture=1,2,3
    allow_mtp2=1
    profile_description='long-prefill candidate: MTP2, grouped prefill, optional MindIEQ'
    default_batched_tokens=4096
    ;;
  *)
    echo "QWEN36_PROFILE must be compat, decode, or long-prefill (aliases: mtp1, mtp2; got: $profile)" >&2
    exit 2
    ;;
esac

: "${PORT:=18020}"
: "${CONTAINER_NAME:=qwen36-c8-112k-$profile}"

export CONTAINER_NAME PORT
export QWEN36_PROFILE="$profile"
export QWEN36_PROFILE_DESCRIPTION="$profile_description"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-114688}"
export MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.95}"
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$default_batched_tokens}"
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

if [[ "$profile" == long-prefill && "${QWEN36_MINDIEQ_ENABLED:-0}" != 1 ]]; then
  echo 'long-prefill requires QWEN36_MINDIEQ_ENABLED=1; use decode for the W4 baseline' >&2
  exit 2
fi

echo "QWEN36_PROFILE=$profile ($profile_description) MAX_MODEL_LEN=$MAX_MODEL_LEN " \
  "MTP=$MTP NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS " \
  "GRAPH_CAPTURE_SIZES=$GRAPH_CAPTURE_SIZES MAX_BATCHED_TOKENS=$MAX_BATCHED_TOKENS " \
  "MINDIEQ=${QWEN36_MINDIEQ_ENABLED:-0}" >&2
exec bash "$root/experimental/run-c8-service.sh"
