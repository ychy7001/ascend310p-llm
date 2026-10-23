#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
export QWEN36_MINDIEQ_ENABLED=1 QWEN36_MINDIEQ_CUSTOM_QUANT=1
export QWEN36_MINDIEQ_MIN_TOKENS=128
export QWEN36_MINDIEQ_SHARED_GATE=1
export MINDIEQ_LIBRARY=$work/grouped-shared-build-v1/output/libqwen36_mindieq_grouped.so
exec bash "$work/start-candidate.sh"
