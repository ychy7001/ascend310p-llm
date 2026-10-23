#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
export QWEN36_MINDIEQ_ENABLED=1
export MINDIEQ_LIBRARY=$work/grouped-native-build-v1/output/libqwen36_mindieq_grouped.so
exec bash "$work/start-candidate.sh"
