#!/usr/bin/env bash
set -euo pipefail
mode=${1:-c8}
docker run --rm --name qwen36-arm-npu-check --privileged --pid host --ipc host --network host --entrypoint bash \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
 -v /var/driver:/var/driver -v /var/dmp:/var/dmp \
 -v /etc/hdcBasic.cfg:/etc/hdcBasic.cfg:ro \
 -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -v /data/qwen36-deploy:/work \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e QWEN36_C8_FUSED_WRITE=1 -e CHECK_MODE="$mode" \
 local/vllm-ascend:v0.23.0-310p-openeuler-orangepi-rcfix -lc '
 set -e
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 export PYTHONPATH=/work/repo/experimental:$PYTHONPATH
 if [[ "$CHECK_MODE" == gdn ]]; then
  export ASCEND_CUSTOM_OPP_PATH=/work/artifacts/gdn-opp/vendors/custom_transformer
  export LD_LIBRARY_PATH=/work/artifacts/gdn-opp/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
  python3 /work/check-gdn-eager.py
 elif [[ "$CHECK_MODE" == c8 ]]; then
  python3 /work/check-c8-eager.py
 else
  python3 /work/check-c8-graph-cpu-input.py
 fi
 '
