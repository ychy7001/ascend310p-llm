#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018
art=/home/ychy/ascend-lab-runtime/qwen36-public-release-20261012
service=qwen36-tp1-chip1-w4-c8-original-112k
restore(){ python3 - "$work/chip1-before.json" > "$work/restore.log" 2>&1 <<'RESTORE'
import json,subprocess,sys
cmd=json.load(open(sys.argv[1]))[0]['Config']['CreateCommand'];cmd.insert(2,'-d');subprocess.run(cmd,check=True)
RESTORE
}
trap restore EXIT
if podman container exists "$service"; then podman stop --time 10 "$service"; fi
podman run --rm --name qwen38-real-expert-gate --network none --group-add keep-groups --shm-size 2g \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e PYTHONPATH= -e ATEN_CPU_CAPABILITY=default \
 -v "$work:/experiment:rw" \
 -v "$work/probe-build/qwen36_int4_custom.so:/custom/qwen36_int4_custom.so:ro" \
 -v "$work/grouped-opp/vendors/custom_transformer:/opp:ro" \
 -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
 -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
 -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 --device /dev/davinci1 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
 quay.io/ascend/vllm-ascend:v0.23.0.post1-310p bash -lc '
 set -e
 if [[ -d /usr/local/python3.12.13/lib/python3.12/site-packages/triton ]]; then mv /usr/local/python3.12.13/lib/python3.12/site-packages/triton /opt/triton-disabled; fi
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 source /opp/bin/set_env.bash
 export ASCEND_CUSTOM_OPP_PATH=/opp
 python3 /experiment/qwen38_verify_fixture.py --fixture /experiment --library /custom/qwen36_int4_custom.so --output /experiment/npu-experts.json
 python3 /experiment/qwen38_qsa_probe.py --library /custom/qwen36_int4_custom.so --output /experiment/qsa-npu.json
 ' 2>&1 | tee "$work/npu-experts.log"
