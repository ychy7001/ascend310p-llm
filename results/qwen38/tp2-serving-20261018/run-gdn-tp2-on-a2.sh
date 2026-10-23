#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018
# Existing services were inspected before stopping. The systemd unit must stop
# first so it cannot immediately recreate the chip0 container.
systemctl --user stop qwen36-image.service
if podman container exists qwen36-tp1-chip1-w4-c8-original-112k; then
 podman stop --time 15 qwen36-tp1-chip1-w4-c8-original-112k
fi
podman run --rm --name qwen38-tp2-gdn-gate --network host --group-add keep-groups --shm-size 2g \
 --device /dev/davinci0 --device /dev/davinci1 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
 -v "$work:/experiment:rw" \
 -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
 -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
 -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -e QWEN38_OPP="${QWEN38_OPP:-/experiment/community/vllm_ascend/_cann_ops_custom/vendors/custom_transformer}" \
 -e QWEN38_PROBE_OUTPUT="${QWEN38_PROBE_OUTPUT:-/experiment}" \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ASCEND_RT_VISIBLE_DEVICES=0,1 -e OMP_NUM_THREADS=1 \
 localhost/qwen38-runtime:ffa980c-3ab5dda-20261018 bash -lc '
 set -e
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 opp="$QWEN38_OPP"
 source "$opp/bin/set_env.bash"
 export ASCEND_CUSTOM_OPP_PATH="$opp" PYTHONPATH=/experiment/community:/experiment/vllm-fork:${PYTHONPATH:-}
 export LD_LIBRARY_PATH="$opp/op_api/lib:$LD_LIBRARY_PATH"
 mkdir -p "$QWEN38_PROBE_OUTPUT"
 torchrun --standalone --nproc-per-node=2 /experiment/qwen38_gdn_tp2_probe.py
 ' 2>&1 | tee "$work/gdn-tp2-gate.log"
