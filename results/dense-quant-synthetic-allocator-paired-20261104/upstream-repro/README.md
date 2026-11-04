# Ascend310P intermittent QuantBatchMatmulV3 507015, no checkpoint required

Unposted reproduction bundle. This probe needs torch, torch-npu and CANN, and imports no vLLM model code. It generates every tensor locally and reads no capture/checkpoint when invoked with --synthetic.

Tested:310P3, driver26.1.1, torch2.10.0+cpu, torch-npu2.10.0.post2, CANN9.1.0-beta.1. Production image digest:registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:1a193ded0d75832321019542326740c7fc4509853556d032e773631466453b38.

After loading the matching CANN environment and exposing a device, execute each attempt in a fresh process/context:

```bash
export ASCEND_LAUNCH_BLOCKING=1
export SOC_VERSION=ascend310p1
export TASK_QUEUE_ENABLE=1
for allocator in False True; do
  for seed in 310 311 312 313 314; do
    PYTORCH_NPU_ALLOC_CONF=expandable_segments:$allocator \
      timeout -k 5s 180s python3 probe.py --synthetic --seed "$seed" --device 0 \
      --out "case-$allocator-$seed.json"
  done
done
```

Intermittent, so retry several fresh processes. The archived experiment alternated False/True per seed (310-314). False4PASS/1FAIL507015; True2PASS/2FAIL507015; last True initialization timed out before a result/preparation marker and is counted separately. Both allocators can fail; no allocator fix is established.

Shapes:activation[4096,8704]int8 ND, weight[8704,5120]int8 NZ stride[1,8704], scale[5120]float32 ND, pertoken_scale[4096]float32 ND, outputfloat16. Weight is first created as[N,K] and converted to NZ before transpose. Activation and token scale are zero, so every output must be finite and exactly zero.

Failures report507015, kernel QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21, hash5247287448945562503, matching the complete Dense startup failure. A passing run does not rule it out. Initialization timeout is distinct from an operator failure.

All bundled tensors are generated at runtime. JSON contains metadata/errors only; no model weights or business data are distributed. There is no confirmed source fix and no accepted performance improvement.
