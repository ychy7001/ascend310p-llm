# Ascend 310P 可复现实验入口

仅用于独立实验容器，不操作历史服务。第一次运行先确认设备/端口空闲；文件和容器名重复时拒绝覆盖，不自动删除旧结果。

## 固定基线

在a2新实验目录复制本目录的launch-frozen.sh、run-baseline.py，以及tools目录内bench_http_stream.py、bench_server_phases.py、check_cache_request_reuse.py、eval_gsm8k_http.py、validate_checkpoint.py。脚本中的镜像ID/路径锁定本次a2制品，换机器须显式提供IMAGE/MODEL_DIR并重新冻结来源。

```bash
EXPERIMENT_DIR=/path/to/lab-runtime/hyperqwen-20261103 \
  bash launch-frozen.sh
python3 run-baseline.py
```

基线默认TP1/W4A8/C16/MTP2/Graph/MindIEQ、112K、seqs1、batch2048、prefix off、async。输出status.json和每请求原始JSONL。run-baseline.py绑定18103端口/模型名，本轮不可与另一runner共用端点。GSM数据使用已有固定SHA，未下载/重分发数据集。

1K/4K/16K/32K的phase基准采样seed0、温度0、ignore_eos、256输出、每档warmup1+正式3。token数组写入各config，unique salt排除前缀复用。正式三次只作筛选，不能称可靠P95。phase counters仅用于独占端点；客户端mean TPOT、首个非空文本块TTFT和服务端Decode时间分开解释。MTP每步可能发送多token，SSE块间隔不是逐token ITL。

```bash
python3 benchmarks/ascend310p/summarize.py results/baseline-20261103/c1-*.jsonl
```

汇总采用nearest-rank分位数；延迟/阶段吞吐分位数只包含成功请求，失败保留在成功率分母中。总输出吞吐仅从有批次墙钟的stream rounds计算，不能把多请求速率求和当聚合吞吐。`bench_http_stream.py`默认seed310、每次运行生成salt前缀，并为每个请求生成独立salt；与phase基准seed0不同必须如实记录。

## Profile

run-profile-stage.py等待基线完成，再核验全权重哈希、停止本会话基线容器，启动同镜像的独立Profile容器。start-experiment.sh从镜像固定入口派生，只增加MTP0/1/2和profiler开关。Profile阶段仍用MTP2；采集4K Prefill、4K完整请求和32K首个输出后的Decode窗口；随后检查输出与可解析kernel CSV。Profile时间不参与普通Benchmark。全部原始trace留a2，归档时保留原始CSV压缩及分析摘要，避免提交数GB trace。

sample-resources.py约每5秒外加查询耗时采样npu-smi；报告“采样峰值”，不能称准确的HBM/allocator峰值。

## 生产W8A8 TP2副本（当前主线）

production-manifest.json保存a1生产命令。launch-production-replica.py将host/port改为127.0.0.1:18106，使用a2同digest镜像、独立W8权重/缓存，保留TP2/MTP0/Graph[1,2]/262144/batch4096/seqs4/prefix on。可显式配置Profiler，采集关闭期间才计常规Benchmark。start-production-replica.sh复制生产wrapper。a2驱动与a1不同，结果标记为生产配置副本而非原机线上性能。

run-production-baseline.py只在a2使用：等待transfer-complete.json后校验权重index/headers与SHA指纹，依次执行正确性、四档C1、C1/3/4并发、业务质量和阶段回放、GSM64，最后Profile与前后门禁。业务fixtures使用a2已保存的a1-310p-tune-20261101；未向生产端点发送Benchmark。check_business_quality.py对会议仅检查完成/格式，候选纪要仍需语义配对；schema另校验预算、批准、负责人/截止时间和采购任务，不只检查结构。

Prefix Cache生产已开启，仅记录；W4预算方案的基线与Profile仍保留，其收益不作为W8生产优化收益。


## Graph单变量筛选

`run-graph14-screen.py`等待W8基线结束、容器退出且两次NPU空闲采样通过，再在18107启动Graph `[1,4]`，保持两张捕获图、其他生产参数不变。先跑C1/3/4各三轮与事实门禁；C3中位吞吐至少提高5%、C1不下降超过5%才继续完整回放。业务墙钟改善不足5%即不选择；达到门槛仍需GSM64、交替A/B重复和纪要语义审阅。阈值只用于筛选，不代表显著性或部署验收。

第一轮事实门禁的“负责”前缀误报已保留原始状态和输出；`resume-production-baseline.py`只重启同参数实验服务、补做剩余阶段，不重跑有效性能。回放与Profile使用重启后的同配置服务；缓存协议使用独立salt，冷缓存区间必须命中增量为0。资源采样及原始服务日志与实验结果一起保留。
