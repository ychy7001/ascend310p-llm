#!/usr/bin/env bash
set -eo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export ATEN_CPU_CAPABILITY=default TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:${LD_LIBRARY_PATH:-}"
extra=(--enforce-eager)
if [[ ${MAIN_AB_GRAPH:-0} == 2 ]]; then
  extra+=(--speculative-config '{"method":"mtp","num_speculative_tokens":2,"enforce_eager":true}')
elif [[ ${MAIN_AB_GRAPH:-0} == 1 ]]; then
  extra=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,9,12]}'
         --speculative-config '{"method":"mtp","num_speculative_tokens":2,"enforce_eager":false}')
fi
exec vllm serve /model --host 127.0.0.1 --port "$PORT" --served-model-name qwen36-main \
  --tensor-parallel-size 2 --quantization ascend --dtype float16 --max-model-len "$LIMIT" \
  --max-num-seqs 4 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.90 \
  --mamba-ssm-cache-dtype float16 --trust-remote-code --enable-prefix-caching \
  --enable-chunked-prefill --enable-prompt-tokens-details "${extra[@]}"
