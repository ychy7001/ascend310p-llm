#!/usr/bin/env bash
set -euo pipefail
base=/data/qwen36-deploy
docker run -d --name qwen36-orangepi --privileged --pid host --ipc host --network host \
  --entrypoint bash \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /var/driver:/var/driver -v /var/dmp:/var/dmp \
  -v /etc/hdcBasic.cfg:/etc/hdcBasic.cfg:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /data/models/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P:/model:ro \
  -v "$base:/work" \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e OMP_NUM_THREADS=4 \
  -e VLLM_ASCEND_FORCE_RC=1 \
  local/vllm-ascend:v0.23.0-310p-openeuler-orangepi-rcfix -lc '
    set -e
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /work/artifacts/opp/vendors/custom_transformer/bin/set_env.bash
    export ASCEND_CUSTOM_OPP_PATH=/work/artifacts/opp/vendors/custom_transformer
    export QWEN36_EXTENSION=/work/artifacts/extension-build/qwen36_int4_custom.so
    export QWEN36_RUNTIME=/work/repo/runtime
    export PYTHONPATH=/work/repo/bootstrap:$PYTHONPATH
    triton_path=/usr/local/python3.12.13/lib/python3.12/site-packages/triton
    if [[ -d "$triton_path" ]]; then mv "$triton_path" /opt/triton-disabled; fi
    exec vllm serve /model --served-model-name qwen36-w4a8 \
      --tensor-parallel-size 1 --dtype float16 --max-model-len 4096 \
      --gpu-memory-utilization 0.45 --enforce-eager \
      --enable-prefix-caching --kv-cache-memory-bytes 1073741824 \
      --max-num-seqs 1 --max-num-batched-tokens 2048 \
      --host 0.0.0.0 --port 18024
  '
