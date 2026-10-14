#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reproducible TP1 112K product-candidate launcher.
# The underlying C8 service remains experimental until the graph backend fix
# is accepted upstream.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

: "${MODEL_DIR:?set MODEL_DIR to the standalone checkpoint}"
: "${OPP_DIR:?set OPP_DIR to the custom OPP directory}"
: "${EXTENSION:?set EXTENSION to qwen36_int4_custom.so}"
: "${C8_LIBRARY:?set C8_LIBRARY to libqwen36_c8.so}"

source "$root/configs/tp1-profile.sh"

echo "QWEN36_PROFILE=$profile ($profile_description) MAX_MODEL_LEN=$MAX_MODEL_LEN " \
  "MTP=$MTP NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS " \
  "GRAPH_CAPTURE_SIZES=$GRAPH_CAPTURE_SIZES MAX_BATCHED_TOKENS=$MAX_BATCHED_TOKENS " \
  "MINDIEQ=${QWEN36_MINDIEQ_ENABLED:-0}" >&2
exec bash "$root/experimental/run-c8-service.sh"
