#!/usr/bin/env bash
set -euo pipefail
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
release=/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
# 独立使用空闲芯片1，避免与芯片0服务争用NPU。
podman run --rm --name qwen36-a8-quant-diag --network none --group-add keep-groups --shm-size 4g \
 --device /dev/davinci1 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
 -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
 -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
 -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -v "$code:/project:ro" -v "$work:/work:rw" \
 -v "$work/grouped-a8-build-v2/output/libqwen36_mindieq_grouped.so:/custom/grouped.so:ro" \
 -v "$release/n128-candidate/artifacts/opp/vendors/custom_transformer:/opp:ro" \
 -e QUANT_DIAG_OUTPUT="${QUANT_DIAG_OUTPUT:-activation-quantizers-20261014.json}" -e QUANT_DIAG_FP32_FLAG="${QUANT_DIAG_FP32_FLAG:-}" \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default -e ASCEND_RT_VISIBLE_DEVICES=0 \
 quay.io/ascend/vllm-ascend:v0.23.0.post1-310p bash -lc '
 set -e
 triton_path=/usr/local/python3.12.13/lib/python3.12/site-packages/triton
 if [[ -d "$triton_path" ]]; then mv "$triton_path" /opt/triton-disabled; fi
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 source /opp/bin/set_env.bash
 export ASCEND_CUSTOM_OPP_PATH=/opp
 export LD_LIBRARY_PATH="/opp/op_api/lib:$LD_LIBRARY_PATH"
 python3 /project/tools/check_a8_quant_310.py --library /custom/grouped.so --out /work/a8-quant-v2-20261014.jsonl
 '
