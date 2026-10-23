#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
release=/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012
out="$work/paired-20261014"
mkdir "$out"
health() {
 python3 - <<'PY'
import time, urllib.request
start=time.monotonic()
while time.monotonic()-start < 900:
    try:
        with urllib.request.urlopen('http://127.0.0.1:18024/health',timeout=3) as r:
            if r.status==200:
                print('health200 elapsed',round(time.monotonic()-start),flush=True)
                break
    except Exception:
        pass
    time.sleep(10)
else:
    raise SystemExit('health timeout; inspect existing container, do not restart blindly')
PY
}
trap 'printf "FAILED stage=%s\n" "$stage" > "$out/state.txt"' ERR
stage=initial
for label in B0 A1 B2; do
 stage="$label/start"
 printf '%s\n' "$stage" > "$out/state.txt"
 if [[ "$label" != B0 ]]; then
  podman stop -t 30 qwen36-tp1-48g-112k > "$out/$label-stop.txt"
  if [[ "$label" == A1 ]]; then
   QWEN36_MINDIEQ_ENABLED=0 bash "$work/start-candidate.sh" > "$out/$label-service.txt" 2>&1 &
  else
   bash "$work/start-native-layout-service.sh" > "$out/$label-service.txt" 2>&1 &
  fi
  printf '%s\n' "$!" > "$out/$label-start.pid"
 fi
 health
 for tokens in 8192 32256; do
  stage="$label/p$tokens"
  printf '%s\n' "$stage" > "$out/state.txt"
  python3 "$code/tools/bench_server_phases.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --prompt-tokens "$tokens" --output-tokens 128 --repeats 3 --unique-prompts --spec-metrics --out "$out/$label-p${tokens}-o128.jsonl" > "$out/$label-p${tokens}-o128.txt" 2>&1
 done
 if [[ "$label" != B2 ]]; then
  stage="$label/gsm64"
  printf '%s\n' "$stage" > "$out/state.txt"
  python3 "$code/tools/eval_gsm8k_http.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --label "$label" --dataset "$release/gsm8k-test.jsonl" --limit 64 --seed 310 --max-tokens 512 --out "$out/$label-gsm64.jsonl" > "$out/$label-gsm64.txt" 2>&1
 fi
 curl -fsS http://127.0.0.1:18024/metrics > "$out/$label-metrics-final.txt"
 npu-smi info > "$out/$label-npu-smi.txt"
done
stage=DONE
printf '%s\n' "$stage" > "$out/state.txt"
