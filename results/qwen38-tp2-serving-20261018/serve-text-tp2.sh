#!/usr/bin/env bash
# Run inside the pinned runtime image with /experiment and /model mounted.
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
opp=/experiment/community/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
source "$opp/bin/set_env.bash"
export ASCEND_CUSTOM_OPP_PATH="$opp"
export LD_LIBRARY_PATH="$opp/op_api/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/experiment/community:/experiment/vllm-fork:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=0,1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
# The checkpoint defaults to eager_dequant. An explicit native backend is
# essential: eager FP16 expansion of every expert cannot fit on this DUO.
overrides=$(cat <<'JSON'
{"text_config":{"ascend_expert_quantization":{"backend":"cube_310_int4_a8","activation_quantization":"int8_per_group","bits":4,"format":"qwen4exp_w4a16_group_v1","group_size":128,"offset_dtype":"int8","packing":"signed_int4_low_nibble_first_in_axis","scale_dtype":"float16","shared_expert_execution":"tp_sharded","symmetric":false,"lm_head_execution":"float16","ple_projection_execution":"float16","grouped_prefill_chunk_tokens":256}}}
JSON
)
exec python3 -m vllm.entrypoints.cli.main serve /model \
 --served-model-name qwen38-w4a8-tp2-text \
 --host 0.0.0.0 --port 18038 --dtype float16 \
 --tensor-parallel-size 2 --enable-expert-parallel --enable-ep-weight-filter \
 --disable-custom-all-reduce --no-async-scheduling \
 --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 256 \
 --gpu-memory-utilization 0.95 --language-model-only \
 --mamba-cache-mode align --enable-chunked-prefill \
 --limit-mm-per-prompt '{"image":0,"video":0}' \
 --enforce-eager --hf-overrides "$overrides"
