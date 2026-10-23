#!/usr/bin/env bash
set -euo pipefail
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
out=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013/shared-gate-service-20261014
python3 "$code/tools/bench_server_phases.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --prompt-tokens 114560 --output-tokens 128 --repeats 3 --unique-prompts --spec-metrics --out "$out/p114560-o128.jsonl" > "$out/p114560-o128.txt" 2>&1
python3 "$code/tools/bench_generation_sanity.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --max-tokens 64 --out "$out/post-phases-sanity.json" > "$out/post-phases-sanity.txt" 2>&1
curl -fsS http://127.0.0.1:18024/metrics > "$out/long-metrics-final.txt"
npu-smi info > "$out/long-npu-smi.txt"
printf 'DONE\n' > "$out/long-state.txt"
