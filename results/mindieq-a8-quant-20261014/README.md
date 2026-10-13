# MindIEQ 独立 FP32算术 A8准备原型（2026-10-14）

解决FP16 dynamic_quant对微小幅值输入的量化边界，同时减少与原native量化器的差异。新activation_quant.cpp读取FP16，每行用FP32 absmax/div、CAST_RINT、clamp[-127,127]，转INT8 ND。每核处理8行一组，scale按8float对齐写出；多分配对齐行及16零padding，返回有效M+16行。零输入scale=1、量化码0；padding scale=0。没有host观察激活或同步，沿用recordStream及RunOpApiV2排队生命周期处理。

本轮扩展在grouped库新增quantize接口，原matmul接口和旧模型路径不变。构建复用既有CANN9.1/310P3环境，初次Duplicate<int8>编译失败，改按int16清零后完成；编译日志均保留。新库SHA256为246cc4656cf401cc01586628bf018c2132251d9e22db2fb93ad17ddba8685de0。外部MindIEQ类许可与固定来源不变，自有量化代码沿用仓库Apache2.0许可。

## 独立真实NPU门禁

芯片1独占，K512/2048 × M1/2/4/8/17/128/512/1024/4096共18案例：整数码对FP32 CPU参考完全一致、scale rtol1e-6/atol0，16行padding为零、8次连续提交全部通过、Graph replay更换输入通过。包含约1e-5量级激活、零行、常数和线性边界。原始结果a8-quant-v2-20261014.jsonl。

warmup5/repeat20、两后端交错、host dispatch+synchronize，比较双方完整A8+16行padding准备，不是纯kernel latency。大M新算子更慢：4096×512约515 vs285us，4096×2048约920 vs391us；小M部分更快，M8/K512出现250/394us噪声。不能只取小shape声称总体加速。芯片0同时进行服务启动，属于工程诊断数据，后续正式计时应在整机静默时复核。

## 全部256真实专家完整MLP

芯片1，真实layer0全部expert权重、三份512token×top8实际route counts。新grouped使用新A8准备，原native使用既有fused pack；warmup5/repeat20，两后端×8异步输出×3案例=48输出通过。激活为合成FP16，计入两投影/SwiGLU/量化，不含router/shared expert/attention/combine。p05/p50/p95为路由选样，非耗时分位。

| counts | 原native ms | 新A8+MindIEQ grouped ms | 相对L2误差 |
|---|---:|---:|---:|
| p05 |19.338|14.000|0.00490%|
| p50 |19.219|13.935|0.00888%|
| p95 |17.285|11.192|0.00358%|

相对原native减少约28%/27%/35%耗时。此前dynamic_quant路径相对L2约1%，本轮显著降低；尚非模型质量证明。整机并行服务加载存在CPU/IO干扰，不能与前轮不同芯片的13.2/10.4ms直接构成严格A/B。新量化准备有成本，最终服务收益待验证。

## 可选服务接入准备

QWEN36_MINDIEQ_CUSTOM_QUANT=1开启新quantize；同时必须开启QWEN36_MINDIEQ_ENABLED并挂载本轮新库。默认仍关闭，不自动替换旧库。仅在当前B0/A1/B2测试结束后更新服务代码并启动新的C3候选，避免混入配对结果。B/A/B结束后已启动C3新量化服务，正在加载，尚待服务质量与性能门禁。

复现：build-a8.sh使用新的SOURCE_DIR/BUILD_DIR；run-quant.sh和run-mlp.sh中的库路径须改为对应输出。只挂载davinci1，在该容器环境逻辑编号为0。代码check_a8_quant_310.py、bench_mindieq_mlp_310.py --grouped-custom-quant；原始数值/耗时与构建日志全部保留。服务质量需要64题同协议、第711题以及8K/32K性能复测后再判定。


后续真实服务已完成64题、8K/32K、近112K及业务补测，详见[服务门禁与局限](../mindieq-a8-service-20261014/README.md)。单算子数值通过不代表整模型无差异；8K仍有MTP接受率及Decode下降。
