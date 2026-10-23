#!/usr/bin/env bash
set -euo pipefail
mkdir -p /data/qwen36-deploy/pr17640
tar -xzf /data/qwen36-deploy/pr17640-source.tar.gz -C /data/qwen36-deploy/pr17640
tar -xzf /data/qwen36-deploy/qwen36-pr17640-cmake-build-scripts.tar.gz -C /data/qwen36-deploy/pr17640
docker run --rm --name qwen36-arm-gdn-build --privileged --pid host --ipc host --network host --entrypoint bash \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
 -v /var/driver:/var/driver -v /var/dmp:/var/dmp \
 -v /etc/hdcBasic.cfg:/etc/hdcBasic.cfg:ro \
 -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -v /data/qwen36-deploy:/work \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e MAX_JOBS=2 -e CMAKE_BUILD_PARALLEL_LEVEL=2 -e TE_PARALLEL_COMPILER=2 \
 local/vllm-ascend:v0.23.0-310p-openeuler-orangepi-rcfix -lc '
 set -e
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 export CPATH=/work/pr17640/csrc/third_party/catlass/include
 cd /work/pr17640/csrc
 # a2 source snapshot already includes the upstream protobuf patch.
 if (cd third_party/ascend_protobuf && patch -R --dry-run -p1 < ../../cmake/third_party/build/modules/patch/protobuf_25.1_change_version.patch >/dev/null); then
  sed -i "s@PATCH_COMMAND patch -p1 < .*protobuf_25.1_change_version.patch@PATCH_COMMAND \${CMAKE_COMMAND} -E true@" cmake/third_party/ascend_protobuf.cmake
 fi
 bash build.sh --pkg --ops="chunk_gated_delta_rule_compute_wy;chunk_gated_delta_rule_fwd_h_vllm;chunk_fwd_o_vllm" --soc=ascend310p -j2
 bash build/cann-ops-transformer-custom_linux-aarch64.run --install-path=/work/artifacts/gdn-opp
 python3 /work/repo/tools/build_pr17640_wy_extension.py --source-root /work/pr17640 --build /work/artifacts/gdn-extension --full-gdn
 '
