#!/usr/bin/env bash
set -euo pipefail
work=/home/ychy/ascend-lab-runtime/qwen38-tp2-validation-20261018
cp -an /home/ychy/ascend-lab-runtime/qwen36-public-release-20261012/clean-build/community/csrc/third_party/. "$work/community/csrc/third_party/"
podman run --rm --name qwen38-grouped-build --network host --shm-size 2g -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e MAX_JOBS=4 -e CMAKE_BUILD_PARALLEL_LEVEL=4 -v "$work:/experiment:rw" quay.io/ascend/vllm-ascend:v0.23.0.post1-310p bash -lc '
 set -e
 if [[ -d /usr/local/python3.12.13/lib/python3.12/site-packages/triton ]]; then mv /usr/local/python3.12.13/lib/python3.12/site-packages/triton /opt/triton-disabled; fi
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 cd /experiment/community/csrc
 bash build.sh --pkg --ops="qwen_w4_a8_pack_v310;qwen_w4_a8_int4_matmul_v310;qwen_w4_a8_swiglu_pack_v310;qwen_w4_a8_int4_down_reduce_v310;qsa_sparse_attention_v310" --soc=ascend310p -j4
 mkdir -p /experiment/probe-build
 python3 /experiment/build_qwen36_int4_extension.py --community-root /experiment/community --source /experiment/qwen38_probe.cpp --build /experiment/probe-build
 for installer in build/cann-ops-transformer*.run; do bash "$installer" --install-path=/experiment/grouped-opp; done
 ' 2>&1 | tee "$work/build-grouped.log"
