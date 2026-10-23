#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
python3 "$code/tools/check_long_context_needle.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --prompt-tokens 114560 --max-tokens 128 --out "$work/mindieq-native-layout-needle114560.jsonl" > "$work/mindieq-native-layout-needle114560.txt" 2>&1
python3 "$code/tools/bench_generation_sanity.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --max-tokens 64 --out "$work/mindieq-native-layout-post-needle-sanity.json" > "$work/mindieq-native-layout-post-needle-sanity.txt" 2>&1
curl -fsS http://127.0.0.1:18024/metrics > "$work/native-layout-metrics-final.txt"
