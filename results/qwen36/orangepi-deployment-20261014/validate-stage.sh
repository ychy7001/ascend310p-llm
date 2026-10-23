#!/usr/bin/env bash
set -euo pipefail
label=${1:?stage label}
base=/data/qwen36-deploy
trap 'docker logs qwen36-orangepi > "$base/startup-$label.log" 2>&1' EXIT
for ((i=0;i<180;i++)); do
 if curl -sf --max-time 2 http://127.0.0.1:18024/health >/dev/null; then break; fi
 if (( i==179 )); then docker logs --tail 60 qwen36-orangepi; exit 1; fi
 sleep 10
done
python3 "$base/smoke_service.py"
python3 "$base/repo/tools/check_cache_request_reuse.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8 --out "$base/quality-$label.json"
python3 - "$base/quality-$label.json" <<'CHECK'
import json,sys
r=json.load(open(sys.argv[1]))
assert r["strict_matches"]==8 and r["reuse"]["identical_text"] and not r["reuse"]["canary_leaked"], "Quality gate failed"
CHECK
spec=()
if [[ "$label" == *mtp1* || "$label" == *mtp2* ]]; then spec=(--spec-metrics); fi
python3 "$base/repo/tools/bench_server_phases.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8 \
 --prompt-tokens 2304 --output-tokens 128 --repeats 3 --cache-salt-prefix "opi-$label" "${spec[@]}" --out "$base/bench-$label.jsonl"
docker logs qwen36-orangepi > "$base/startup-$label.log" 2>&1
