#!/usr/bin/env bash
set -euo pipefail
: "${PAIR_CODE:?}" "${PAIR_BASE:?}" "${PAIR_MODEL:?}" "${PAIR_OUT:?}"
for ((attempt=0; attempt<180; attempt++)); do
 if curl -sf --max-time 2 "$PAIR_BASE/health" >/dev/null; then break; fi
 if (( attempt==179 )); then exit 1; fi
 sleep 10
done
python3 "$PAIR_CODE/tools/check_cache_request_reuse.py" --base-url "$PAIR_BASE" \
 --model "$PAIR_MODEL" --out "$PAIR_OUT/quality.json"
python3 - "$PAIR_OUT/quality.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['strict_matches']==8 and r['reuse']['identical_text'] and not r['reuse']['canary_leaked']
PY
for tokens in 2304 8192 32000; do
 python3 "$PAIR_CODE/tools/bench_server_phases.py" --base-url "$PAIR_BASE" \
  --model "$PAIR_MODEL" --prompt-tokens "$tokens" --output-tokens 128 --repeats 3 \
  --cache-salt-prefix "rc-ep-pair-$tokens" --spec-metrics --out "$PAIR_OUT/p$tokens.jsonl"
done
