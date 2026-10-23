#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
release=/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012
out="$work/a8-service-20261014"
mkdir "$out"
python3 - <<'PY'
import time,urllib.request
start=time.monotonic()
while time.monotonic()-start<900:
    try:
        with urllib.request.urlopen('http://127.0.0.1:18024/health',timeout=3) as r:
            if r.status==200:
                print('health200 elapsed',round(time.monotonic()-start),flush=True)
                break
    except Exception:
        pass
    time.sleep(10)
else:raise SystemExit('health timeout; inspect live service without blind restart')
PY
printf 'sanity\n' > "$out/state.txt"
python3 "$code/tools/bench_generation_sanity.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --max-tokens 64 --out "$out/sanity.json" > "$out/sanity.txt" 2>&1
printf 'target711\n' > "$out/state.txt"
python3 "$code/tools/eval_gsm8k_http.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --label C3-target711 --dataset "$release/gsm8k-test.jsonl" --indices 711 --max-tokens 512 --out "$out/target711.jsonl" > "$out/target711.txt" 2>&1
printf 'gsm64\n' > "$out/state.txt"
python3 "$code/tools/eval_gsm8k_http.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --label C3-custom-a8 --dataset "$release/gsm8k-test.jsonl" --limit 64 --seed 310 --max-tokens 512 --out "$out/gsm64.jsonl" > "$out/gsm64.txt" 2>&1
python3 "$code/tools/compare_gsm8k_runs.py" "$work/paired-20261014/A1-gsm64.jsonl" "$out/gsm64.jsonl" > "$out/quality-comparison.json"
for tokens in 8192 32256; do
 printf 'p%s\n' "$tokens" > "$out/state.txt"
 python3 "$code/tools/bench_server_phases.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --prompt-tokens "$tokens" --output-tokens 128 --repeats 3 --unique-prompts --spec-metrics --out "$out/p${tokens}-o128.jsonl" > "$out/p${tokens}-o128.txt" 2>&1
done
printf 'needle114560\n' > "$out/state.txt"
python3 "$code/tools/check_long_context_needle.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --prompt-tokens 114560 --max-tokens 128 --out "$out/needle114560.jsonl" > "$out/needle114560.txt" 2>&1
python3 "$code/tools/bench_generation_sanity.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --max-tokens 64 --out "$out/post-needle-sanity.json" > "$out/post-needle-sanity.txt" 2>&1
curl -fsS http://127.0.0.1:18024/metrics > "$out/metrics-final.txt"
npu-smi info > "$out/npu-smi.txt"
printf 'DONE\n' > "$out/state.txt"
