# clt C16/C8 配对（2026-10-15）

目标：同一x86 EP/310P1、44215MB设备内存，同W4权重、TP1/MTP1/Decode Graph/GDN/MindIEQ、112K上限、batch2048、util.90、单流、Prefix off、同步调度，仅切换KV存储。服务名为兼容测试客户端仍使用qwen36-w4a8-c8，C16实际由QWEN36_C8_ENABLED=0决定，不能从服务名推断KV格式。

C8原始数据见 ../clt-deployment-20261015。顺序为C8 A→C16 B→恢复C8 A；恢复后8K三轮用于检查漂移。8K/32K三轮使用完全相同prompt IDs，112000两次使用与C8完全相同的10%/90%位置检索和固定128token输出。所有输入/输出长度及服务计数须验证，检索门禁与速度分开记录。

C16为独立容器qwen36-c16-112k-20261015、同端口18016；仅单颗NPU，测试期间停止原C8容器。后台脚本无论C16成功与否均尝试停止C16、恢复原C8并验证健康。

内存审计：KV_AUDIT_CONFIG记录配置预算、共享池映射和scheduler所需最低blocks；KV_AUDIT_STORAGE遍历原生allocator返回张量，按底层storage地址去重，记录实际唯一存储字节和Torch分配增量。原生310P allocator对同一个逻辑shared_by池的linear state和Attention分别创建物理tensor，因此不能把配置sum(t.size)当作全部实际存储。C8则拆分类型且预留共享FP16/NZ scratch，须对比完整内存而非仅INT8数据payload。

C16已成功启动相同112K配置。对比结果由summary.json从原始指标重算；112K两条不同位置检索各一次，属于有限样本，不能当作统计稳定的生产吞吐或完整精度评测。

## 配对实测结论

48G规格、TP1、112K单流且优先响应速度时，建议优先C16；C8保留为内存约束档。本次在clt的310P1上完成，不能直接当作a2的310P3绝对速度。两者均通过112000输入+128输出的前/后位置检索，以及长请求后的9/9固定答案回归。配置上限114688是输入+输出总长；这不是最大可承载上下文的极限测试。

| 输入token | C16 Prefill tok/s | C8 Prefill tok/s | C16 Decode tok/s | C8 Decode tok/s | C8 Decode下降 |
|---|---:|---:|---:|---:|---:|
| 8192 | 589.54 | 586.84 | 30.11 | 28.72 | 4.6% |
| 32768 | 527.05 | 523.34 | 24.35 | 21.28 | 12.6% |
| 112000，10%位置 | 395.76 | 392.62 | 13.13 | 10.78 | 17.8% |
| 112000，90%位置 | 397.69 | 394.23 | 13.82 | 10.63 | 23.1% |

8K/32K为一次预热+三次独立prompt重复的中位数；112K为服务已预热、两个不同检索位置各一次，没有单独同长预热，也没有三轮同输入重复。所有请求128输出token、prefix disabled；Prefill和Decode用单请求Prometheus计数差值，Decode为127/服务decode时间，TTFT另保留。表中百分比定义为1−C8速度/C16速度；换成“C16相对C8加速”会是另一个百分比。

Prefill各档差异均不到1%；主要代价在Decode且随历史增长。8K/32K两档MTP接受率均100%；112K early的C16为53/75=70.7%，C8为57/70=81.4%，late两者都是56/71=78.9%。即使early的C8接受率较高仍然更慢，late接受率相同也慢23.1%，支持历史转换成本的解释，但没有声称所有差异均来自一个kernel。

112K检索门禁只要求首段命中准确值；为测速强制ignore_eos输出128token，答案后的重复模板文本不代表自然对话质量通过。两条合成检索和固定算术不替代正式精度集、真实会议摘要评测或持续并发压测。

## 内存口径与净收益

C16启动审计：逻辑预算5,787,353,088字节，223 scheduler blocks；实际32个唯一底层storage共11,048,583,168字节（10.29GiB）。Attention与linear state物理分开，不能直接采用逻辑预算5.40GiB作为实际内存。

C8原服务：148blocks、21类型池共5,532,254,208字节（5.15GiB），另有234,881,024字节（224MiB）共享FP16/NZ转换工作区。默认快照C16约40175MB、C8约35514MB（npu-smi口径）；长测过程另记录memory-samples.jsonl。两者实际预留页数不同，不能将设备差值全部当作同容量量化收益。

长请求结束后的同口径npu-smi快照：C16为40560/40561MB，C8为35514/35514MB，总量44215MB；当前自动配置C8少占约5046–5047MB，C16仍余约3654MB。该数字同时包含预留容量、allocator、工作区等差异，不是相同容量下净省5GB的证明。C16 Graph日志增量0.18GiB，C8为0.40GiB；C8工作区在capture阶段分配，不能再次重复相加。

按审计所得最低107blocks核算相同配置容量（仅核算，未以107blocks启动验收）：C16每block实际49,545,216字节，C8每block37,380,096字节。C16池5,301,338,112字节；C8池3,999,670,272字节+234,881,024字节scratch；差1,066,786,816字节（0.994GiB）。该核算不是整机最小运行内存，也不能据此声称32G产品档已可用。C8转换工作区在Graph capture时创建，Graph日志的显存增量可能包含它，不应把Graph数字再次全部叠加到这项差值。

有效112000历史token本身，11个Full Attention层（含MTP）：C16每token共11×2048字节，约2.350GiB；C8含padded scales为11×1088字节，约1.248GiB。有效KV数据减小约46.9%，但加上共享工作区、GDN state、padding和预留页后，完整缓存与整机的节省比例远低于46.9%。

## 延迟机制与判断边界

C8当前计算路径是动态INT8存储→有效历史页反量化到共享FP16/NZ工作区→原生Attention；不是原生INT8 KV Attention。历史越长，反量化和额外读写的绝对成本越大。device_length避免短请求转换整个最大上下文，但不消除长历史转换。模型权重、GDN state及绝大部分算子保持不变。

单层机制证据见 [C8长上下文组件微基准](../c8-long-components-20261014/README.md)：114688历史时转换3.340ms、原生PA2.802ms、组合6.028ms。这个单层合成结果不能当作整个服务减速倍数；端到端还包括MoE/GDN/MTP和调度。

48G、单流112K场景若C16正确性通过且有内存余量，优先保留C16性能档；C8适合确实受KV容量约束、并能接受实测Decode损失的档位。当前实验不能证明32G可用、不能证明更大上下文上限、不能证明并发稳定性。默认预留页不同是现有allocator行为，不是本次偷偷改变参数；相同容量核算用来避免把额外预留差异包装成量化收益。

## 复现方式

本次直接复用clt已校验权重、镜像和定版算子制品，没有转换/覆盖权重、升级依赖。环境差异保存在c8-baseline.env/c16.env。CLT_KV_AUDIT=1只在C16启动时打印allocator配置和storage，推理路径不变；clt_kv_audit.py仅对分配入口做一次审计。

在clt上查看后台实验进度与结果：

```bash
cat /home/clt/qwen36-kv-comparison-20261015/comparison-progress.jsonl
cat /home/clt/qwen36-kv-comparison-20261015/c16/logs/validation-progress.jsonl
```

本次C16实验启动命令（必须先停止占同颗NPU的C8服务）：

```bash
docker stop qwen36-optimized-112k-20261015
bash /home/clt/qwen36-kv-comparison-20261015/start-c16.sh
python3 /home/clt/qwen36-optimized-20261015/validate_deployment.py \
  --deployment /home/clt/qwen36-kv-comparison-20261015/c16
```

恢复C8：

```bash
docker stop qwen36-c16-112k-20261015
docker start qwen36-optimized-112k-20261015
curl -fsS http://127.0.0.1:18016/health
```

自动流程run_comparison.py的finally执行上述恢复，随后8K预热+三轮验证漂移。源码中含本次实验的绝对路径；需要先具备[clt部署制品](../clt-deployment-20261015/README.md)，不是独立安装器。若重跑已存在的C16容器，使用docker start而不是直接重用docker run名称。测试源码validate_deployment.py在前序部署结果目录，完整协议及原始指标均保留。

在仓库根目录重算对比：

```bash
python3 results/qwen36/clt-kv-comparison-20261015/summarize.py
```

## 服务恢复与复核

自动流程已完成，原C8容器qwen36-optimized-112k-20261015恢复至18016并通过health。C16测试容器已停止、保留以便复现；没有把C8默认服务静默切成C16。

恢复后8K预热+三轮中位数：Prefill 586.74 tok/s、Decode 28.81 tok/s；相对实验前C8 Decode中位数变化+0.29%。恢复后三轮MTP接受率分别100%、100%、89.55%，不能声称逐token输出或所有MTP计数完全确定；本次没有据此扩展成状态正确性根因排查。恢复回测用于检查明显环境漂移，主要对照表仍采用原C8与C16相同输入的原始结果。

恢复后五类生成sanity均PASS，最终容器状态和health见final-service-state.txt；它仍然只是有限生成检查，不代表正式精度验收。原始容器日志以gzip无损保存，sha256sums.txt可校验全部本次制品。
