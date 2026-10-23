#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
release=/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012
podman run --rm --name qwen36-three-backends --network none --group-add keep-groups --shm-size 16g \
 --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
 -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
 -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
 -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -v "$code:/project:ro" -v "$release/model:/model:ro" -v "$work:/work:rw" \
 -v "$release/clean-build/artifacts/qwen36_int4_custom.so:/custom/qwen36_int4_custom.so:ro" \
 -v "$release/n128-candidate/artifacts/opp/vendors/custom_transformer:/opp:ro" \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default -e ASCEND_RT_VISIBLE_DEVICES=0 \
 localhost/qwen36-mindieq-embedded:20261013 bash -lc '
 set -e
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 source /opp/bin/set_env.bash
 export ASCEND_CUSTOM_OPP_PATH="$ASCEND_HOME_PATH/opp/vendors/custom_nn:/opp"
 export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/opp/vendors/custom_nn/op_api/lib:$LD_LIBRARY_PATH"
 python3 /project/tools/bench_mindieq_mlp_310.py --model /model --library /custom/qwen36_int4_custom.so --upstream-test /work/test-source/test_qmm_v5_record_none.py --experts 256 --route-percentiles .95 --activation-draw 2 --validate-only --route-trace /work/route-occupancy/combined.jsonl --out /work/mindieq-record-none-tail.jsonl > /work/mindieq-record-none-tail.txt 2>&1
 cat /work/mindieq-record-none-tail.txt
 '
