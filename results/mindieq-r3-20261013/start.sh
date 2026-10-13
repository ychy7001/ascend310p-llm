#!/usr/bin/env bash
set -e
cd /home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
release=/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
export MODEL_DIR=$release/model OPP_DIR=$release/n128-candidate/artifacts/opp/vendors/custom_transformer EXTENSION=$release/clean-build/artifacts/qwen36_int4_custom.so C8_LIBRARY=$release/c8-device-length-build/output/libqwen36_c8.so
export GDN_OPP_DIR=$work/pr17640-opp/vendors/custom_transformer GDN_LIBRARY=$work/gdn-extension/pr17640_gdn.so GDN_SOURCE=$work/pr17640/vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py
export QWEN36_GDN_ENABLED=1 QWEN36_PROFILE=mtp2 PORT=18024 CONTAINER_NAME=qwen36-pr17640-product
exec bash experimental/run-c8-112k-product.sh
