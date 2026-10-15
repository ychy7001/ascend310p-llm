#!/usr/bin/env bash
set -euo pipefail
podman run -d --name qwen36-c16-kv-ab-20261015 --network host -v /home/ychy/ascend-lab-runtime/mindieq-r3-20261013/pr17640-opp/vendors/custom_transformer:/gdn-opp:ro -v /home/ychy/ascend-lab-runtime/mindieq-r3-20261013/gdn-extension/pr17640_gdn.so:/custom/pr17640_gdn.so:ro -v /home/ychy/ascend-lab-runtime/mindieq-r3-20261013/pr17640/vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py:/gdn-source.py:ro -v /home/ychy/ascend-lab-runtime/mindieq-r3-20261013/grouped-shared-build-v1/output/libqwen36_mindieq_grouped.so:/custom/mindieq_grouped.so:ro --group-add keep-groups --shm-size 16g -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 -e ATEN_CPU_CAPABILITY=default -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e PYTHONPATH=/c8bootstrap -e QWEN36_C8_LIBRARY=/custom/libqwen36_c8.so -e QWEN36_C8_ENABLED=0 -e QWEN36_C8_FUSED_WRITE=1 -e QWEN36_C8_POOL_MODE=split_by_type -e QWEN36_GROUPED_CHUNK_TOKENS=512 -e QWEN36_ROUTE_TRACE_PATH= -e QWEN36_ROUTE_TRACE_CALLS=256 -e QWEN36_C8_ALLOW_GRAPH=1 -e QWEN36_C8_FORCE_UPDATABLE_GRAPH=1 -e QWEN36_C8_DEVICE_LENGTH=1 -e QWEN36_C8_ALLOW_MTP2=1 -e ENFORCE_EAGER=0 -e GRAPH_MODE=FULL_DECODE_ONLY -e GRAPH_CAPTURE_SIZES=1,2,3 -e COMPILATION_MODE=0 -e QWEN36_MINDIEQ_CUSTOM_QUANT=1 -e QWEN36_MINDIEQ_SHARED_GATE=1 -e QWEN36_MINDIEQ_MIN_TOKENS=128 -e QWEN36_MINDIEQ_ENABLED=1 -e QWEN36_MINDIEQ_LIBRARY=/custom/mindieq_grouped.so -e QWEN36_GDN_ENABLED=1 -e QWEN36_GDN_LIBRARY=/custom/pr17640_gdn.so -e QWEN36_GDN_SOURCE=/gdn-source.py -e MTP=1 -e NUM_SPECULATIVE_TOKENS=2 -e ENABLE_PROFILER= -e PORT=18024 -e MAX_MODEL_LEN=114688 -e MEMORY_UTILIZATION=0.95 -e MAX_BATCHED_TOKENS=4096 -e MAX_NUM_SEQS=1 -e ENABLE_PREFIX_CACHING=0 -v /home/ychy/ascend-lab-runtime/qwen36-public-release-20261012/model:/model:ro -v /home/ychy/services/qwen36/repo/runtime:/w4only:ro -v /home/ychy/services/qwen36/repo/bootstrap:/bootstrap:ro -v /home/ychy/services/qwen36/repo/experimental:/c8adapter:ro -v /home/ychy/services/qwen36/kv-ab-20261015/bootstrap:/c8bootstrap:ro -v /home/ychy/ascend-lab-runtime/qwen36-public-release-20261012/clean-build/artifacts/qwen36_int4_custom.so:/custom/qwen36_int4_custom.so:ro -v /home/ychy/ascend-lab-runtime/qwen36-public-release-20261012/c8-device-length-build/output/libqwen36_c8.so:/custom/libqwen36_c8.so:ro -v /home/ychy/ascend-lab-runtime/qwen36-public-release-20261012/n128-candidate/artifacts/opp/vendors/custom_transformer:/opp:ro -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro -v /usr/local/dcmi:/usr/local/dcmi:ro -v /etc/ascend_install.info:/etc/ascend_install.info:ro --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc quay.io/ascend/vllm-ascend:v0.23.0.post1-310p bash -lc '
    set -e
    triton_path=/usr/local/python3.12.13/lib/python3.12/site-packages/triton
    if [[ -d "$triton_path" ]]; then mv "$triton_path" /opt/triton-disabled; fi
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /opp/bin/set_env.bash
    export ASCEND_CUSTOM_OPP_PATH=/opp
    if [[ "$QWEN36_GDN_ENABLED" == 1 ]]; then
      export ASCEND_CUSTOM_OPP_PATH="/gdn-opp:/opp"
      export LD_LIBRARY_PATH="/gdn-opp/op_api/lib:$LD_LIBRARY_PATH"
    fi
    python3 -c "import sitecustomize; print(sitecustomize.__file__)"
    args=()
    if [[ "$ENABLE_PROFILER" == 1 ]]; then
      args+=(--profiler-config '"'"'{"profiler":"torch","torch_profiler_dir":"/profile","torch_profiler_with_stack":false,"ignore_frontend":true}'"'"')
    fi
    if [[ "$ENFORCE_EAGER" == 1 ]]; then
      args+=(--enforce-eager)
    else
      args+=(--compilation-config "{\"mode\":$COMPILATION_MODE,\"cudagraph_mode\":\"$GRAPH_MODE\",\"cudagraph_capture_sizes\":[$GRAPH_CAPTURE_SIZES]}")
    fi
    if [[ "$MTP" == 1 ]]; then
      args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS}")
    fi
    if [[ "$ENABLE_PREFIX_CACHING" == 1 ]]; then
      args+=(--enable-prefix-caching)
    else
      args+=(--no-enable-prefix-caching)
    fi
    exec vllm serve /model "${args[@]}" --served-model-name qwen36-w4a8-c8 \
      --tensor-parallel-size 1 --dtype float16 --block-size 64 \
      --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$MEMORY_UTILIZATION" \
      --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
      --host 0.0.0.0 --port "$PORT"
  '
