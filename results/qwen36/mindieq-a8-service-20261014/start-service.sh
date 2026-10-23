#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
export QWEN36_MINDIEQ_ENABLED=1 QWEN36_MINDIEQ_CUSTOM_QUANT=1
export MINDIEQ_LIBRARY=$work/grouped-a8-build-v2/output/libqwen36_mindieq_grouped.so
exec bash "$work/start-candidate.sh"
