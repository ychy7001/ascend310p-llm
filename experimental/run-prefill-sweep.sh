#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 独占davinci0；顺序启动隔离服务，固定FP16 KV/MTP=0/eager，保存原始日志。
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${MODEL_DIR:?}" "${OPP_DIR:?}" "${EXTENSION:?}" "${C8_LIBRARY:?}" "${RESULT_DIR:?}"
mkdir -p "$RESULT_DIR"
PORT=${PORT:-18020}
BASE_URL="http://127.0.0.1:$PORT"
# Each pair changes only scheduling or expert chunk relative to another pair.
PAIRS=${PAIRS:-"1024:256 1024:512 1024:640 4096:512 4096:640"}
for pair in $PAIRS; do
  batch=${pair%:*}
  chunk=${pair#*:}
  [[ "$batch" =~ ^[0-9]+$ && "$chunk" =~ ^[0-9]+$ ]] || exit 2
  name="qwen36-prefill-b${batch}-g${chunk}"
  log="$RESULT_DIR/$name.log"
  result="$RESULT_DIR/$name-p3072-o1.jsonl"
  # Existing evidence is never overwritten; reruns use a new RESULT_DIR.
  [[ ! -e "$log" && ! -e "$result" ]] || { echo "Evidence exists for $name" >&2; exit 2; }
  if curl --max-time 2 -sf "$BASE_URL/health" >/dev/null; then
    echo "Port $PORT already serves a model; stop it explicitly before the sweep" >&2
    exit 2
  fi
  # Record the source and artifact identity beside the raw measurements.
  {
    git -C "$root" rev-parse HEAD
    printf 'TP=1 MAX_MODEL_LEN=4096 MEMORY_UTILIZATION=0.90 MTP=0 C8=0 EAGER=1 PREFIX_CACHE=0\n'
    printf 'MAX_BATCHED_TOKENS=%s QWEN36_GROUPED_CHUNK_TOKENS=%s\n' "$batch" "$chunk"
    sha256sum "$EXTENSION" "$C8_LIBRARY"
    podman image inspect quay.io/ascend/vllm-ascend:v0.23.0.post1-310p --format '{{.Id}}'
  } > "$RESULT_DIR/$name-environment.txt"
  env CONTAINER_NAME="$name" PORT="$PORT" MAX_MODEL_LEN=4096 \
    MEMORY_UTILIZATION=0.90 MTP=0 QWEN36_C8_ENABLED=0 ENFORCE_EAGER=1 \
    MAX_BATCHED_TOKENS="$batch" QWEN36_GROUPED_CHUNK_TOKENS="$chunk" \
    bash "$root/experimental/run-c8-service.sh" >"$log" 2>&1 &
  launcher=$!
  ready=0
  for ((attempt=0; attempt<180; attempt++)); do
    if curl --max-time 3 -sf "$BASE_URL/health" >/dev/null; then ready=1; break; fi
    if ! kill -0 "$launcher" 2>/dev/null; then
      wait "$launcher" || true
      echo "$name terminated; inspect $log" >&2
      exit 1
    fi
    sleep 5
  done
  if [[ "$ready" != 1 ]]; then
    echo "$name readiness observation expired; inspect live container without restarting" >&2
    exit 1
  fi
  python3 "$root/tools/bench_server_phases.py" --base-url "$BASE_URL" \
    --model qwen36-w4a8-c8 --prompt-tokens 3072 --output-tokens 1 \
    --repeats 3 --out "$result"
  # One quality canary per shape; separate from performance counter window.
  python3 "$root/tools/check_long_context_needle.py" --base-url "$BASE_URL" \
    --model qwen36-w4a8-c8 --prompt-tokens 3072 --max-tokens 512 \
    --out "$RESULT_DIR/$name-needle.jsonl"
  python3 - "$RESULT_DIR/$name-needle.jsonl" <<'PY'
import json,sys
rows=[json.loads(line) for line in open(sys.argv[1])]
if not rows[-1].get('retrieved'):raise RuntimeError('Retrieval canary failed; no accepted speed result')
PY
  podman stop -t 20 "$name"
  wait "$launcher"
done
