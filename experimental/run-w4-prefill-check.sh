#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Standalone grouped-kernel check; run after stopping services on davinci0.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${MODEL_DIR:?set standalone checkpoint directory}"
: "${OPP_DIR:?set built vendors/custom_transformer directory}"
: "${EXTENSION:?set independent Torch extension}"
: "${RESULT_DIR:?set a new result directory}"
: "${LABEL:?set artifact label}"
if [[ -e "$RESULT_DIR" ]]; then echo 'RESULT_DIR must not exist' >&2; exit 2; fi
mkdir -p "$RESULT_DIR"
podman run --rm --name "${CONTAINER_NAME:-qwen36-w4-prefill-check}" --network none \
  --group-add keep-groups --shm-size 2g \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default \
  -e LABEL="$LABEL" -e REPEATS="${REPEATS:-20}" \
  -v "$root:/project:ro" -v "$MODEL_DIR:/model:ro" \
  -v "$OPP_DIR:/opp:ro" -v "$EXTENSION:/custom/qwen36_int4_custom.so:ro" \
  -v "$RESULT_DIR:/results" \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  --device /dev/davinci0 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  "${IMAGE:-quay.io/ascend/vllm-ascend:v0.23.0.post1-310p}" bash -lc '
    set -e
    triton_path=/usr/local/python3.12.13/lib/python3.12/site-packages/triton
    if [[ -d "$triton_path" ]]; then mv "$triton_path" /opt/triton-disabled; fi
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /opp/bin/set_env.bash
    export ASCEND_CUSTOM_OPP_PATH=/opp
    python3 /project/tools/check_w4_grouped_prefill_310.py --model /model \
      --library /custom/qwen36_int4_custom.so --label "$LABEL" \
      --repeats "$REPEATS" --out /results/measurements.jsonl
  '
