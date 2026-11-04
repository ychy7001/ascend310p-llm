# 上游反馈草稿：310P动态W8量化Matmul间歇507015

状态：本地兼容性反馈草稿，未发布。已取得无权重复现，尚无确定源码修复。

## 环境与最小复现

310P3、驱动26.1.1，torch2.10.0+cpu、torch-npu2.10.0.post2、CANN9.1.0-beta.1；vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，Ascend `88eac2713b14beedc7a48803274f8d6cee3e9f30`。

[合成探针](../../benchmarks/ascend310p/probe_dense_quant_matmul.py)不读checkpoint或捕获文件。加载匹配CANN环境并暴露所选设备，每次以新进程运行，seed310～314分别重复，交替allocator False/True：

```bash
ASCEND_LAUNCH_BLOCKING=1 SOC_VERSION=ascend310p1 TASK_QUEUE_ENABLE=1 PYTORCH_NPU_ALLOC_CONF=expandable_segments:False python3 benchmarks/ascend310p/probe_dense_quant_matmul.py --synthetic --seed 310 --device 0 --out case.json
```

| 参数 | Shape | Stride | 类型/格式 |
|---|---|---|---|
| activation | [4096,8704] | [8704,1] | INT8 ND(2) |
| weight | [8704,5120] | [1,8704] | INT8 FRACTAL_NZ(29) |
| scale | [5120] | [1] | F32 ND |
| per-token scale | [4096] | [1] | F32 ND |

权重先按物理[N,K]生成、转NZ后transpose；激活和逐token scale全零且finite。预期FP16输出全零且finite。

## 实际结果与证据

旧栈完整Dense TP2在两rank各加载19.2416GB后、health之前报507015；同步eager定位`AscendW8A8DynamicLinearMethod310.apply`的MLP down_proj，故障kernel为`QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21`、hash5247287448945562503。合成随机INT8权重也能复现相同错误。

冻结合成复现的allocator交替5次：False4PASS/1FAIL507015、True2PASS/2FAIL507015；最后True初始化超时、未到算子，单列。关闭expandable、重建权重存储、关闭executor cache均不是已证实修复。一次重放通过不足以排除间歇故障。

可分享的生成数据、代码、元数据与完整命令见[无权重复现包说明](../../results/dense-quant-synthetic-allocator-paired-20261104/upstream-repro/README.md)。真实捕获布局/设备/分配对照统一见[Profiling](../ascend310p-profiling.md)，完整运行栈容量和质量筛选见[结果](../ascend310p-results.md)，不将多组件变化归因单独torch-npu。
