#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
out=$work/decode-profile-20261014
restore() {
 podman stop -t 30 qwen36-tp1-48g-112k || true
 nohup env -u PROFILE_DIR bash "$work/start-shared-profile-base.sh" > "$out/restored-service.txt" 2>&1 < /dev/null &
 printf '%s\n' "$!" > "$out/restored-pid.txt"
}
trap restore EXIT
ready=0
for attempt in $(seq 1 120); do
 if curl -fsS http://127.0.0.1:18024/health > /dev/null 2>&1; then ready=1; break; fi
 sleep 5
done
[[ "$ready" == 1 ]]
python3 "$code/tools/profile_http_decode.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --prompt-tokens 8192 --output-tokens 128 --out "$out/p8192-o128.jsonl" > "$out/request.txt" 2>&1
printf 'CAPTURE_DONE\n' > "$out/state.txt"
