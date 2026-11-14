#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/pr17640-opp/vendors/custom_transformer
export ASCEND_RT_VISIBLE_DEVICES=0,1 ASCEND_VISIBLE_DEVICES=0,1
export ATEN_CPU_CAPABILITY=default TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}"
profile=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":['"${GRAPH_SIZES:-3,6,12}"']}'
         --speculative-config '{"method":"mtp","num_speculative_tokens":2,"enforce_eager":true}')
if [[ ${WY_PROFILE:-graph-mtp} == eager ]]; then profile=(--enforce-eager); fi
exec vllm serve /model --host 127.0.0.1 --port "$PORT" --served-model-name Qwen3.6-35B-A3B-w8a8 \
  --tensor-parallel-size 2 --quantization ascend --dtype float16 --max-model-len "${LIMIT:-262144}" \
  --max-num-seqs 4 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90 \
  --mamba-ssm-cache-dtype float16 --trust-remote-code --enable-prefix-caching \
  --enable-chunked-prefill --enable-prompt-tokens-details --async-scheduling "${profile[@]}"
