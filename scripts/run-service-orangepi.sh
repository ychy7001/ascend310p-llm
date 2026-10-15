#!/usr/bin/env bash
set -euo pipefail
base=/data/qwen36-deploy
: "${ENABLE_PROFILER:=0}" "${PROFILE_OUTPUT_REL:=profile}" "${MAX_BATCHED_TOKENS:=2048}" "${MINDIEQ:=0}" "${RC_DEVICE_BLOCK_INDICES:=0}" "${MTP_DRAFT_EAGER:=0}" "${TASK_QUEUE_ENABLE:=1}" "${CPU_AFFINITY_CONF:=0}"
: "${MAX_MODEL_LEN:=8192}" "${KV_BYTES:=2147483648}" "${MTP:=0}" "${C8:=0}" "${GRAPH:=0}" "${GDN:=0}" "${ASYNC_SCHEDULING:=0}" "${C8_REUSE_INT32_METADATA:=0}" "${C8_LIBRARY:=/work/artifacts/c8-transdata-build/output/libqwen36_c8.so}"
: "${C8_PERSISTENT_NZ:=0}" "${C8_PERSISTENT_NZ_ASSUME_STABLE:=0}" "${C8_PERSISTENT_NZ_MAX_GIB:=4}"
: "${RC_BLOCKING_MM_H2D:=0}"
[[ "$MTP" =~ ^[012]$ && "$C8" =~ ^[01]$ && "$GRAPH" =~ ^[01]$ ]] || exit 2
docker run -d --name qwen36-orangepi --privileged --pid host --ipc host --network host \
 --entrypoint bash \
 -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
 -v /var/driver:/var/driver -v /var/dmp:/var/dmp \
 -v /etc/hdcBasic.cfg:/etc/hdcBasic.cfg:ro \
 -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
 -v /data/models/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P:/model:ro \
 -v "$base:/work" \
 -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
 -e TE_PARALLEL_COMPILER=2 -e OMP_NUM_THREADS=4 -e VLLM_ASCEND_FORCE_RC=1 \
 -e TASK_QUEUE_ENABLE="$TASK_QUEUE_ENABLE" \
 -e CPU_AFFINITY_CONF="$CPU_AFFINITY_CONF" \
 -e RC_DEVICE_BLOCK_INDICES="$RC_DEVICE_BLOCK_INDICES" \
 -e RC_BLOCKING_MM_H2D="$RC_BLOCKING_MM_H2D" \
 -e MTP_DRAFT_EAGER="$MTP_DRAFT_EAGER" \
 -e ENABLE_PROFILER="$ENABLE_PROFILER" -e PROFILE_OUTPUT_REL="$PROFILE_OUTPUT_REL" -e MAX_BATCHED_TOKENS="$MAX_BATCHED_TOKENS" -e GDN="$GDN" -e ASYNC_SCHEDULING="$ASYNC_SCHEDULING" -e MAX_MODEL_LEN="$MAX_MODEL_LEN" -e KV_BYTES="$KV_BYTES" -e MTP="$MTP" -e C8="$C8" -e GRAPH="$GRAPH" \
 -e QWEN36_C8_ENABLED="$C8" -e QWEN36_C8_ALLOW_MTP2=1 \
 -e QWEN36_C8_ALLOW_GRAPH="$GRAPH" -e QWEN36_C8_FORCE_UPDATABLE_GRAPH="$GRAPH" \
 -e QWEN36_C8_DEVICE_LENGTH="${QWEN36_C8_DEVICE_LENGTH:-1}" -e QWEN36_C8_FUSED_WRITE=1 -e QWEN36_C8_POOL_MODE=split_by_type \
 -e QWEN36_C8_REUSE_INT32_METADATA="$C8_REUSE_INT32_METADATA" \
 -e QWEN36_C8_PERSISTENT_NZ="$C8_PERSISTENT_NZ" -e QWEN36_C8_PERSISTENT_NZ_ASSUME_STABLE="$C8_PERSISTENT_NZ_ASSUME_STABLE" \
 -e QWEN36_C8_PERSISTENT_NZ_MAX_GIB="$C8_PERSISTENT_NZ_MAX_GIB" \
 -e QWEN36_C8_LIBRARY="$C8_LIBRARY" \
 -e QWEN36_MINDIEQ_ENABLED="$MINDIEQ" \
 -e QWEN36_MINDIEQ_LIBRARY=/work/artifacts/mindieq-arm-build/output/libqwen36_mindieq_grouped.so \
 -e QWEN36_MINDIEQ_MIN_TOKENS=128 -e QWEN36_MINDIEQ_CUSTOM_QUANT=1 -e QWEN36_MINDIEQ_SHARED_GATE=1 \
 local/vllm-ascend:v0.23.0-310p-openeuler-orangepi-rcfix -lc '
 set -e
 source /usr/local/Ascend/ascend-toolkit/set_env.sh
 source /work/artifacts/opp/vendors/custom_transformer/bin/set_env.bash
 export ASCEND_CUSTOM_OPP_PATH=/work/artifacts/opp/vendors/custom_transformer
 if [[ "$GDN" == 1 ]]; then
  export ASCEND_CUSTOM_OPP_PATH=/work/artifacts/gdn-opp/vendors/custom_transformer:$ASCEND_CUSTOM_OPP_PATH
  export LD_LIBRARY_PATH=/work/artifacts/gdn-opp/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
  export QWEN36_GDN_LIBRARY=/work/artifacts/gdn-extension/pr17640_gdn.so
  export QWEN36_GDN_SOURCE=/work/pr17640/vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py
 fi
 export QWEN36_EXTENSION=/work/artifacts/extension-build/qwen36_int4_custom.so
 export QWEN36_RUNTIME=/work/repo/runtime
 export PYTHONPATH=/work/repo/bootstrap:/work/repo/experimental:$PYTHONPATH
 triton_path=/usr/local/python3.12.13/lib/python3.12/site-packages/triton
 if [[ -d "$triton_path" ]]; then mv "$triton_path" /opt/triton-disabled; fi
 args=()
 if [[ "$ENABLE_PROFILER" == 1 ]]; then
  [[ "$PROFILE_OUTPUT_REL" != /* && "$PROFILE_OUTPUT_REL" != */* && "$PROFILE_OUTPUT_REL" != "" ]] || { echo "PROFILE_OUTPUT_REL must be a non-empty direct child name" >&2; exit 2; }
  mkdir -p "/work/$PROFILE_OUTPUT_REL"
  args+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/work/$PROFILE_OUTPUT_REL\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}")
 fi
 if [[ "$ASYNC_SCHEDULING" == 0 ]]; then args+=(--no-async-scheduling); fi
 if [[ "$GRAPH" == 0 ]]; then args+=(--enforce-eager); else
  # max-num-seqs is one on the product profile. Capturing unused batch sizes
  # makes the MTP0 C8 path enter metadata combinations it never executes.
  # Keep the capture set explicit and tied to the speculative path.
  case "$MTP" in
   0) capture_sizes='[1]' ;;
   1) capture_sizes='[1,2]' ;;
   2) capture_sizes='[1,2,3]' ;;
  esac
  args+=(--compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"cudagraph_capture_sizes\":$capture_sizes}")
 fi
 if [[ "$MTP" != 0 ]]; then
  draft_extra=""
  if [[ "$MTP_DRAFT_EAGER" == 1 ]]; then draft_extra=",\"enforce_eager\":true"; fi
  args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP$draft_extra}")
 fi
 exec vllm serve /model --served-model-name qwen36-w4a8 \
  --tensor-parallel-size 1 --dtype float16 --block-size 64 --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.45 --kv-cache-memory-bytes "$KV_BYTES" \
  --enable-prefix-caching --mm-processor-cache-gb 0.5 \
  --mm-processor-kwargs '\''{"max_pixels":1048576}'\'' \
  --limit-mm-per-prompt '\''{"image":1,"video":0}'\'' \
  --max-num-seqs 1 --max-num-batched-tokens "$MAX_BATCHED_TOKENS" "${args[@]}" \
  --host 0.0.0.0 --port 18024
 '
