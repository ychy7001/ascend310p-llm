#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
for tokens in 8192 32256; do
 python3 "$code/tools/bench_server_phases.py" --base-url http://127.0.0.1:18024 --model qwen36-w4a8-c8 --prompt-tokens "$tokens" --output-tokens 128 --repeats 3 --unique-prompts --out "$work/mindieq-native-layout-phases-p${tokens}-o128.jsonl" > "$work/mindieq-native-layout-phases-p${tokens}-o128.txt" 2>&1
done
