#!/usr/bin/env bash
# 在空闲 310P 上运行独立诊断容器；不会停止或修改模型服务。
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${GDN_OPP_DIR:?set GDN_OPP_DIR to the patched OPP vendor directory}"
wrapper_mount=()
if [[ -n "${GDN_WRAPPER_SOURCE:-}" ]]; then
  wrapper_mount=(-v "$GDN_WRAPPER_SOURCE:/wrapper.py:ro")
fi
podman run --rm --name gdn-state-table-check --network none --shm-size 2g \
  "${wrapper_mount[@]}" \
  --group-add keep-groups \
  --device "${NPU_DEVICE:-/dev/davinci1}" --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default \
  -v "$GDN_OPP_DIR:/gdn-opp:ro" \
  -v "$root/tools/check_gdn_state_table_310.py:/check.py:ro" \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  "${IMAGE:-quay.io/ascend/vllm-ascend:v0.23.0.post1-310p}" bash -lc '
    set -e
    triton_path=/usr/local/python3.12.13/lib/python3.12/site-packages/triton
    if [[ -d "$triton_path" ]]; then mv "$triton_path" /opt/triton-disabled; fi
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export ASCEND_CUSTOM_OPP_PATH=/gdn-opp
    export LD_LIBRARY_PATH="/gdn-opp/op_api/lib:$LD_LIBRARY_PATH"
    python3 /check.py "$@"
  ' bash "$@"
