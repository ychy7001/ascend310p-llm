#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# TP1 GDN 性能候选；外部 PR 的 OPP/扩展/固定源码必须显式提供。
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${GDN_OPP_DIR:?set pinned PR OPP vendor directory}"
: "${GDN_LIBRARY:?set independent pr17640_gdn.so}"
: "${GDN_SOURCE:?set pinned PR chunk_gated_delta_rule.py}"
export QWEN36_GDN_ENABLED=1
export QWEN36_PROFILE="${QWEN36_PROFILE:-mtp2}"
export QWEN36_GROUPED_CHUNK_TOKENS="${QWEN36_GROUPED_CHUNK_TOKENS:-512}"
export PORT="${PORT:-18024}"
export CONTAINER_NAME="${CONTAINER_NAME:-qwen36-gdn-112k-$QWEN36_PROFILE}"
exec bash "$root/experimental/run-c8-112k-product.sh"
