#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
release=/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012
out="$work/shared-gate-service-20261014"
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
printf 'chat\n' > "$out/state.txt"
python3 "$code/tools/bench_mtp_chat.py" --base-url http://127.0.0.1:18024 --label E5-shared-gate --out "$out/chat.jsonl" > "$out/chat.txt" 2>&1
printf 'p8192\n' > "$out/state.txt"
python3 "$code/tools/bench_server_phases.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --prompt-tokens 8192 --output-tokens 128 --repeats 3 --unique-prompts --spec-metrics --out "$out/p8192-o128.jsonl" > "$out/p8192-o128.txt" 2>&1
printf 'p32256\n' > "$out/state.txt"
python3 "$code/tools/bench_server_phases.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --prompt-tokens 32256 --output-tokens 128 --repeats 3 --unique-prompts --spec-metrics --out "$out/p32256-o128.jsonl" > "$out/p32256-o128.txt" 2>&1
printf 'quality-boundary\n' > "$out/state.txt"
python3 "$code/tools/eval_gsm8k_http.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --label E5-boundary --dataset "$release/gsm8k-test.jsonl" --indices 711,205 --max-tokens 512 --out "$out/quality-boundary.jsonl" > "$out/quality-boundary.txt" 2>&1
printf 'gsm64\n' > "$out/state.txt"
python3 "$code/tools/eval_gsm8k_http.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --label E5-shared-gate --dataset "$release/gsm8k-test.jsonl" --limit 64 --seed 310 --max-tokens 512 --out "$out/gsm64.jsonl" > "$out/gsm64.txt" 2>&1
curl -fsS http://127.0.0.1:18024/metrics > "$out/metrics-final.txt"
npu-smi info > "$out/npu-smi.txt"
printf 'DONE\n' > "$out/state.txt"
