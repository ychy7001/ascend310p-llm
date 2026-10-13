#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# 对已启动、独占的112K C8/MTP2/Graph服务验收；失败保留服务和证据。
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${RESULT_DIR:?set a new evidence directory}"
base=${BASE_URL:-http://127.0.0.1:18023}
model=qwen36-w4a8-c8
mkdir -p "$RESULT_DIR"
python3 "$root/tools/check_cache_request_reuse.py" --base-url "$base" \
  --model "$model" --out "$RESULT_DIR/reuse.json"
python3 - "$RESULT_DIR/reuse.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r['strict_matches'] == 8
assert r['reuse']['identical_text'] and not r['reuse']['canary_leaked']
PY
for tokens in 8192 32256; do
  curl --max-time 30 -sf "$base/metrics" >"$RESULT_DIR/p${tokens}-metrics-before.txt"
  python3 "$root/tools/bench_server_phases.py" --base-url "$base" \
    --model "$model" --prompt-tokens "$tokens" --output-tokens 128 \
    --repeats 3 --unique-prompts --out "$RESULT_DIR/p${tokens}-o128.jsonl"
  curl --max-time 30 -sf "$base/metrics" >"$RESULT_DIR/p${tokens}-metrics-after.txt"
done
python3 "$root/tools/check_long_context_needle.py" --base-url "$base" \
  --model "$model" --prompt-tokens 114560 --max-tokens 128 \
  --out "$RESULT_DIR/needle114560.jsonl"
python3 - "$RESULT_DIR/needle114560.jsonl" <<'PY'
import json, sys
r = json.loads(open(sys.argv[1]).read().splitlines()[-1])
assert r['retrieved'] and r['usage']['prompt_tokens'] == 114560
PY
echo '112K组合初步门禁通过；不等于正式质量或尾延迟验收。'
