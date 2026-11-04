#!/usr/bin/env bash
# 在同一固定源码上验证独立vendor；只做本机CPU构建。
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd /source
bash build.sh --pkg --experimental --soc=ascend310p --vendor_name=wy921_cann --ops=fused_gdn_gating,chunk_gated_delta_rule_compute_wy -j16
