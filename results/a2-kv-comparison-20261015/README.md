# a2 TP1 W4A8：C16/C8 配对（已完成）

仅对应 ychy-a2 第一颗310P3，device0/18024。第二颗芯片原服务不变。固定当前部署的W4A8、MTP2、GDN、MindIEQ（custom quant/shared gate/min_tokens128）、Decode Graph、batch4096、max_model_len114688、单流、prefix关闭、gpu_memory_utilization=.95。日志确认该既有a2服务启用了异步调度；本次C16保留相同调度条件。RC/clt采用同步调度，因此不得混算为硬件同条件对比。`MTP=1` 是启动开关，实际 speculative tokens=2。本轮不能与orangepi RC的MTP1/prefix开启配置合并成同条件硬件比较。

C8既有容器带auto-remove，已在停止前保存完整创建命令、镜像/挂载/环境、日志和restore-c8.sh。C16使用独立名字qwen36-c16-kv-ab-20261015，同设备/同端口，仅关闭QWEN36_C8_ENABLED；独立bootstrap仅在启动期增加storage审计，原挂载源文件和权重不变。服务名仍为qwen36-w4a8-c8以兼容客户端，KV类型必须看环境而非名字。

8K/32K各warmup1+正式3轮、112000/128为warmup1+正式1轮方向性测试；逐请求记录phase metrics、输入SHA、最终usage、MTP接受率和prefix命中。两种KV另各跑一次相同112K合成会议更新检查，不等于正式精度评测或P95。最小质量检查8题、A/B/A；原始工具位于tools/。run_comparison.py在仓库根目录执行，等待C8测速完成后继续质量检查、顺序切换C16及测试，失败则调用原创建命令恢复C8，恢复健康需检查。

内存必须区分自动配置的预留页、原生allocator实际storage以及同容量核算。C8额外的FP16 NZ共享工作区计入代价；最终计时使用清理旧systemd重试后的C16复测；公开48G/112K入口改为C16默认，显式C8覆盖保留。另一台clt的EP310P1结论不替代本轮a2结果。

## 本轮纠偏与解释边界

初次C16期间原systemd单元仍在Restart=on-failure重试创建C8容器，在sitecustomize导入阶段以basic_string::_S_construct错误退出，未成功加载第二份模型。已停止该管理单元，原C16容器继续运行，并新增c16-clean-*完整复测；最终统计只使用clean复测。旧数据和journal保留以追溯，不作为默认选择的最终计时依据。run_comparison.py已修正先停管理单元的复现顺序。

112K会议canary两种KV均通过，C16长请求后8题8/8、A/B/A一致且未泄漏；普通文本JSON通过不代表grammar/response_format已验收。C16与C8固定128输出的逐字结果存在差异，MTP接受率也有差别，因此不宣称逐字或正式精度等价。8K差异包含MTP接受率变化，32K/112K接受率全为1，更适合观察KV路径代价。

## 最终配对结果

单位tok/s；8K/32K为3轮中位数，112000为单轮方向性结果，每档先同长度预热。

|输入token|C16 Prefill|C8 Prefill|C16 Decode|C8 Decode|C8 Decode降低|
|---|---:|---:|---:|---:|---:|
|8192|627.63|627.69|38.08|34.92|8.29%|
|32768|548.25|547.69|32.31|29.11|9.92%|
|112000|383.60|382.22|21.66|17.12|20.95%|

同输入SHA、输出usage=128、prefix hits=0均经summarize.py断言。32K/112K两侧MTP接受率均100%、固定输出逐字一致；8K仅1/3配对输出逐字一致，接受率不同，不能将8K全部差值视为KV算子成本。112K普通文本会议决议canary两侧均正确；不是grammar、全业务精度或P95验收。

## 内存代价

当前自动配置：C16 177块、实际storage 8,769,503,232B（8.17GiB）；C8 116块、storage 4,336,091,136B，加224MiB FP16 NZ工作区后约4.26GiB。由于预留容量不相等，这个约3.91GiB池差不是同容量节省。设备0长请求后npu-smi分别39800/44277MB、35736/44277MB。10秒采样最大分别39801和35736MB；C8只覆盖晚期会议尾段，C16覆盖启动和初轮长测，不宣称对称完整峰值。

按相同110块调度契约估算，C16约5.08GiB，C8含NZ工作区约4.05GiB，净省**1.03GiB**。这是storage核算，尚未收紧两边预算运行验证，也不包含权重、Graph、激活和运行时；不能据此宣称32G可用。原生allocator中同标称shared_by池仍可能分别创建linear raw状态和Full Attention K/V，因此不能用标称配置字节替代实际分配。

## 选择与后续

48G、TP1、112K单流优先C16：112K的约21% Decode代价不值得用同容量约1GiB换取。当前C8先解码历史选页到FP16 NZ后调用原生Attention，仍保留为更紧内存约束/独立算子优化对照。下一步优先测试收紧C16预留预算及prefix业务复用，保留视觉功能；当前a2两项业务验收未完成，prefix仍关闭。

配置入口`configs/tp1-48g-112k.env`默认`QWEN36_C8_ENABLED=0`；`QWEN36_C8_ENABLED=1 bash scripts/run-candidate.sh`可切回C8。先`--check`，停止管理单元与旧实例后重建容器；不能用docker/podman start更新已有容器环境。clt是独立EP310P1/MTP1/batch2048/prefix-off基线，不把其吞吐与本机MTP2/batch4096直接相减归因为硬件差异。

复算：`python3 results/a2-kv-comparison-20261015/summarize.py`。原始c8-*与c16-clean-*为最终配对；c16-p*保留初轮干扰窗口数据，不用于最终表。独立启动与回滚命令为start-c16.sh/restore-c8.sh，运行前先停止qwen36-a2.service，保证device0独占。当前运行与管理状态将在部署完成后追加快照。
