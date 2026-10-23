#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
release=/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012
podman run --rm --name qwen36-projection-parity --network none --group-add keep-groups --shm-size 16g \
 --device /dev/davinci1 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
 -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
 -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
 -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -v "$work/grouped-a8-build-v2/output/libqwen36_mindieq_grouped.so:/custom/grouped.so:ro" -v "$code:/project:ro" -v "$release/model:/model:ro" -v "$work:/work:rw" \
 -v "$work/native-record-build-only/qwen36_int4_custom.so:/custom/qwen36_int4_custom.so:ro" \
 -v "$release/n128-candidate/artifacts/opp/vendors/custom_transformer:/opp:ro" \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default -e ASCEND_RT_VISIBLE_DEVICES=0 \
 localhost/qwen36-mindieq-embedded:20261013 bash -lc '
 set -e
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 source /opp/bin/set_env.bash
 export ASCEND_CUSTOM_OPP_PATH="$ASCEND_HOME_PATH/opp/vendors/custom_nn:/opp"
 export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/opp/vendors/custom_nn/op_api/lib:$LD_LIBRARY_PATH"
 python3 /project/tools/check_mindieq_projection_parity_310.py --model /model --native-library /custom/qwen36_int4_custom.so --grouped-library /custom/grouped.so --out /work/projection-parity-group128-20261014.jsonl > /work/projection-parity-group128-20261014.txt 2>&1
 cat /work/projection-parity-group128-20261014.txt
 '
