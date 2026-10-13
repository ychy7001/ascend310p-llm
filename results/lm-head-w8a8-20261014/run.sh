#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/mindieq-r3-20261013
code=/home/ychy/ascend-lab-runtime/pr17640-product-20261013/code
model=/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012/model
podman run --rm --name qwen36-lm-head-w8a8 --network none --group-add keep-groups --shm-size 8g \
 --device /dev/davinci1 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
 -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -v "$code:/project:ro" -v "$model:/model:ro" -v "$work:/work:rw" -v /home/ychy/ascend-lab-runtime/qwen36-public-release-20261012/n128-candidate/artifacts/opp/vendors/custom_transformer:/opp:ro -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default -e ASCEND_RT_VISIBLE_DEVICES=0 \
 localhost/qwen36-mindieq-embedded:20261013 bash -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh; source /opp/bin/set_env.bash; export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_nn:/opp; export ASCEND_LAUNCH_BLOCKING=1; python3 /project/tools/bench_real_lm_head_w8a8_310.py --model /model --out /work/lm-head-w8a8-20261014.jsonl'
