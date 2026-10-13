#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Requires isolated build/install described in docs/huawei-followup-20261013.md.
set -euo pipefail
: "${WORK_DIR:?directory containing external ops-nn, extension-build, opp and test-source}"
: "${IMAGE:=localhost/qwen36-mindieq-embedded:20261013}"
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
podman run --name "${CONTAINER_NAME:-qwen36-mindieq-nz}" --network none --group-add keep-groups \
  --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v "$WORK_DIR:/work" -v "$root:/project:ro" \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default -e ASCEND_RT_VISIBLE_DEVICES=0 \
  "$IMAGE" bash -lc '
    set -e
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export ASCEND_CUSTOM_OPP_PATH="$ASCEND_HOME_PATH/opp/vendors/custom_nn"
    export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
    python /project/tools/check_mindieq_qmm_310.py \
      --upstream-test /work/test-source/test_qmm_v5_ext.py --out /work/real-shapes-nz.jsonl \
      > /work/real-shapes-nz.txt 2>&1
    cat /work/real-shapes-nz.txt
  '
