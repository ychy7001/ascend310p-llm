# orangepi RC：MindIEQ同prompt对照（2026-10-15）

本目录只对应orangepi的310P1/RC。固定TP1、W4A8、C8 device-length/fused-write/split-by-type、MTP1、Decode Graph、GDN、同步调度、Prefix Cache、114688上限、KV6GiB、batch2048、max_num_seqs=1。

关闭候选仅设置 `MINDIEQ=0`。源码包含C8 DecodeOnly设备长度修复c3e9895；该修复不改变本次MTP1的SpecDecoding路径。开启参考来自[RC stage1](../orangepi-rc-baseline-20261015/README.md)。两者使用相同prompt IDs、128输出、warmup1、重复3轮和独立cache salt，检查Prefix Cache命中为零，记录draft/accept计数。基准完整usage校验来自e3fc3f9。

关闭候选通过8题8/8、A/B/A文本一致且无canary泄漏；两档各三轮计时均有效、Prefix Cache命中均为零、MTP接受率均100%。正式业务质量、P95和完整长上下文集合仍待完成。关闭候选已停止并保留，开始恢复MTP1 + MindIEQ，并测试batch4096。

## 同协议中位数

|输入/输出|关闭 Prefill|开启 Prefill|开启变化|关闭 Decode|开启 Decode|开启变化|
|---|---:|---:|---:|---:|---:|---:|
|8192/128|490.16|563.62|+14.99%|24.10|23.87|-0.95%|
|32768/128|448.99|511.15|+13.84%|19.37|19.29|-0.42%|

单位tok/s，warmup后3次中位数。MindIEQ在当前RC上有明确Prefill收益；Decode差异小于1%，尚不能证明有稳定的Decode影响。测试为顺序跨服务对照，没有正式P95/长时间交替门禁。输出逐字一致数量见summary.json；不能将重复文本下的输出一致或100%接受率当作正式精度验证。

原始数据off-p{8192,32768}-o128.jsonl；开启参考保存在stage1目录。off-env.json、off-startup-and-validation.txt与off-memory.json保留配置、日志和RC共享内存快照。
