#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 固定MTP1/C8/32K，分别测eager、Graph全范围和Graph有效长度；独占davinci0。
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${MODEL_DIR:?}" "${OPP_DIR:?}" "${EXTENSION:?}" "${C8_LIBRARY:?}" "${RESULT_DIR:?}"
mkdir -p "$RESULT_DIR"
PORT=${PORT:-18020}
base="http://127.0.0.1:$PORT"
for mode in ${MODES:-eager graph-full graph-device}; do
  eager=0; device_length=0
  case "$mode" in
    eager) eager=1 ;;
    graph-full) ;;
    graph-device) device_length=1 ;;
    *) echo "Unknown mode: $mode" >&2; exit 2 ;;
  esac
  name="qwen36-c8-${mode}-32k"
  log="$RESULT_DIR/$name.log"
  [[ ! -e "$log" ]] || { echo "Evidence exists: $log" >&2; exit 2; }
  if curl --max-time 2 -sf "$base/health" >/dev/null; then
    echo 'Endpoint occupied; stop existing service explicitly' >&2; exit 2
  fi
  env CONTAINER_NAME="$name" PORT="$PORT" MTP=1 NUM_SPECULATIVE_TOKENS=1 \
    QWEN36_C8_ENABLED=1 QWEN36_C8_FUSED_WRITE=1 QWEN36_C8_POOL_MODE=split_by_type \
    ENFORCE_EAGER="$eager" QWEN36_C8_ALLOW_GRAPH=1 COMPILATION_MODE=0 \
    GRAPH_MODE=FULL_DECODE_ONLY GRAPH_CAPTURE_SIZES=1,2 \
    QWEN36_C8_DEVICE_LENGTH="$device_length" MAX_MODEL_LEN=32768 MEMORY_UTILIZATION=0.90 \
    MAX_BATCHED_TOKENS=1024 QWEN36_GROUPED_CHUNK_TOKENS=512 \
    bash "$root/experimental/run-c8-service.sh" >"$log" 2>&1 &
  launcher=$!
  ready=0
  for ((attempt=0; attempt<180; attempt++)); do
    if curl --max-time 3 -sf "$base/health" >/dev/null; then ready=1; break; fi
    if ! kill -0 "$launcher" 2>/dev/null; then
      wait "$launcher" || true
      echo "$name terminated; inspect $log" >&2; exit 1
    fi
    sleep 5
  done
  [[ "$ready" == 1 ]] || { echo 'Readiness expired; preserve live container' >&2; exit 1; }
  python3 "$root/tools/check_cache_request_reuse.py" --base-url "$base" \
    --model qwen36-w4a8-c8 --out "$RESULT_DIR/$name-reuse.json"
  python3 - "$RESULT_DIR/$name-reuse.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
if r['strict_matches'] != 8 or r['reuse']['canary_leaked'] or not r['reuse']['identical_text']:
    raise RuntimeError('Toy/reuse quality gate failed; preserve service')
PY
  for tokens in ${PROMPT_LENGTHS:-128 8192 32256}; do
    before="$RESULT_DIR/$name-p${tokens}-metrics-before.txt"
    after="$RESULT_DIR/$name-p${tokens}-metrics-after.txt"
    curl --max-time 30 -sf "$base/metrics" >"$before"
    python3 "$root/tools/bench_server_phases.py" --base-url "$base" \
      --model qwen36-w4a8-c8 --prompt-tokens "$tokens" --output-tokens 256 \
      --repeats 3 --out "$RESULT_DIR/$name-p${tokens}-o256.jsonl"
    curl --max-time 30 -sf "$base/metrics" >"$after"
  done
  needle="$RESULT_DIR/$name-needle32256.jsonl"
  python3 "$root/tools/check_long_context_needle.py" --base-url "$base" \
    --model qwen36-w4a8-c8 --prompt-tokens 32256 --max-tokens 512 --out "$needle"
  python3 - "$needle" <<'PY'
import json, sys
if not [json.loads(line) for line in open(sys.argv[1])][-1]['retrieved']:
    raise RuntimeError('Long retrieval gate failed; preserve service')
PY
  podman stop -t 20 "$name"
  wait "$launcher"
done
