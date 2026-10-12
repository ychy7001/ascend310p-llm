#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 独占davinci0；固定32K上限，对比长输入Prefill和持续Decode。
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${MODEL_DIR:?}" "${OPP_DIR:?}" "${EXTENSION:?}" "${C8_LIBRARY:?}" "${RESULT_DIR:?}"
mkdir -p "$RESULT_DIR"
PORT=${PORT:-18020}
BASE_URL="http://127.0.0.1:$PORT"
PAIRS=${PAIRS:-"1024:512 4096:640"}
for pair in $PAIRS; do
  batch=${pair%:*}; chunk=${pair#*:}
  [[ "$batch" =~ ^[0-9]+$ && "$chunk" =~ ^[0-9]+$ ]] || exit 2
  name="qwen36-long-b${batch}-g${chunk}"
  log="$RESULT_DIR/$name.log"
  [[ ! -e "$log" ]] || { echo "Evidence exists: $log" >&2; exit 2; }
  if curl --max-time 2 -sf "$BASE_URL/health" >/dev/null; then
    echo "Port $PORT already serves a model; stop it explicitly" >&2; exit 2
  fi
  env CONTAINER_NAME="$name" PORT="$PORT" MAX_MODEL_LEN=32768 \
    MEMORY_UTILIZATION=0.90 MTP=0 QWEN36_C8_ENABLED=0 ENFORCE_EAGER=1 \
    MAX_BATCHED_TOKENS="$batch" QWEN36_GROUPED_CHUNK_TOKENS="$chunk" \
    bash "$root/experimental/run-c8-service.sh" >"$log" 2>&1 &
  launcher=$!
  ready=0
  for ((attempt=0; attempt<180; attempt++)); do
    if curl --max-time 3 -sf "$BASE_URL/health" >/dev/null; then ready=1; break; fi
    if ! kill -0 "$launcher" 2>/dev/null; then
      wait "$launcher" || true
      echo "$name terminated; inspect $log" >&2; exit 1
    fi
    sleep 5
  done
  [[ "$ready" == 1 ]] || { echo 'Readiness expired; preserve live container' >&2; exit 1; }
  for tokens in 8192 32256; do
    for output in 1 256; do
      python3 "$root/tools/bench_server_phases.py" --base-url "$BASE_URL" \
        --model qwen36-w4a8-c8 --prompt-tokens "$tokens" --output-tokens "$output" \
        --repeats 3 --out "$RESULT_DIR/$name-p${tokens}-o${output}.jsonl"
    done
  done
  needle="$RESULT_DIR/$name-needle32256.jsonl"
  python3 "$root/tools/check_long_context_needle.py" --base-url "$BASE_URL" \
    --model qwen36-w4a8-c8 --prompt-tokens 32256 --max-tokens 512 --out "$needle"
  python3 - "$needle" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
if not rows[-1].get('retrieved'):
    raise RuntimeError('Retrieval canary failed; preserve container for diagnosis')
PY
  podman stop -t 20 "$name"
  wait "$launcher"
done
