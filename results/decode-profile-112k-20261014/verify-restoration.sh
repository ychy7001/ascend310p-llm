#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
out=$work/decode-profile-112k-20261014
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
controller=${1:?controller PID required}
while kill -0 "$controller" 2>/dev/null; do sleep 10; done
[[ $(cat "$out/state.txt") == CAPTURE_DONE ]]
ready=0
for attempt in $(seq 1 120); do
 if curl -fsS http://127.0.0.1:18024/health >/dev/null 2>&1; then ready=1; break; fi
 sleep 5
done
[[ "$ready" == 1 ]]
python3 "$code/tools/bench_generation_sanity.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --max-tokens 64 --out "$out/restored-sanity.json" > "$out/restored-sanity.txt" 2>&1
printf 'DONE\n' > "$out/restoration-state.txt"
