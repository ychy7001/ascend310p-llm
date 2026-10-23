# orangepi RC：C16/C8 长上下文取舍

本轮C16计时与长会议/图片/prefix最小业务检查均通过。仅对应orangepi/310P1/RC，物理整机87.51GiB统一内存，不能据本轮宣称48G或32G硬件通过。固定W4A8、TP1、MTP1、MindIEQ E5、GDN、Decode Graph、同步调度、prefix开启、seq1、max_model_len114688、batch4096。

## 性能结果与边界

C8在重启前完成，C16在重启及blocking-MM-H2D兼容开关后完成，**方向性结果，不是交替同启动条件A/B**。C8标称KV预算6GiB/165块，C16改为3GiB/124块避免原生allocator过度预留；两者均满足112K调度预算。固定输入SHA、最终usage和零prefix命中均已断言。8K/32K各warmup1+正式3轮，112000/128为warmup1+正式1轮；不是P95。

|输入token|C16 Prefill|C8 Prefill|C16 Decode|C8 Decode|C8 Decode降低|
|---|---:|---:|---:|---:|---:|
|8192|627.48|625.90|24.18|23.90|1.18%|
|32768|568.13|567.07|21.63|19.36|10.53%|
|112000|420.53|419.40|14.84|11.70|21.15%|

单位tok/s。32K和112K两边接受率100%、固定输出逐字一致；8K接受率不同且仅1/3配对逐字一致，不能把全部差异视为KV成本。原始C8 8K/32K记录引用前一batch4096目录；112K在本目录。详情见paired-summary.json。

## 物理内存审计

原生C16标称3GiB，124blocks实际KV张量storage为6,143,606,784B（5.72GiB）；诊断按untyped_storage设备/地址去重，不修改分配或逐token执行。旧6GiB原生配置248块实际12,287,213,568B（11.44GiB），已放在c16-6gib-allocation-report.json，不作为当前配置。原因是原生allocator对shared_by标称池仍分别创建linear raw和Full Attention K/V，标称值不能替代物理分配。

C8标称6GiB，split后165blocks实际6,167,715,840B，加234,881,024B（224MiB）FP16 NZ工作区共5.96GiB。本轮没有避免额外padding。两者实际预留页数不同，不能相减宣称同容量收益。相同110块调度契约的核算：C16约5.08GiB，C8含NZ工作区约4.05GiB，C8净省1.03GiB。**这是估算，不是最低预算运行验证**，未包括权重、激活、Graph、allocator rounding和OS。

cgroup、HugeTLB、/proc/meminfo、KV storage会重叠，不能相加。RC整机87.5GiB的结果不证明物理48G机器足够。a2和clt是独立48G EP对照，配置/硬件不同，不把吞吐差值归因于RC硬件。

## 启动失败与兼容处理

1. 两次原C16启动都在视觉profile dummy batch H2D等待，尚未分配KV；保留blocked/retry日志及栈，没有性能结果。
2. 仅添加skip-mm-profiling的诊断完成KV分配后在Graph metadata D2H等待，保留skip-mm日志；该配置没有作为有效部署。没有关闭视觉编码器来冒充验收。
3. 独立无模型拷贝探针也等待，驱动记录SDMA channel0 quiescent0复位。npu-smi reset返回不支持。保存证据并停止实验进程后重启两次；没有升级驱动/固件。
4. 第二次冷启动先通过1KiB non_blocking=True，随后4MiB等待stream1/task4；再次冷启动仅改non_blocking=False，1KiB/4/16/64MiB全部正确回读。证据说明本环境存在大块异步H2D兼容问题，不宣称已穷尽驱动根因。
5. 显式RC_BLOCKING_MM_H2D=1只替换vllm多模态输入H2D函数，不改普通Tensor.to、W4计算、KV和Decode精度。默认关闭。C16新候选保留原生视觉profile，完成Graph捕获和8题8/8/A-B-A，随后正式计时；不包含skip-mm-profiling。

全部探针、失败日志与启动bootstrap快照保留；copy_probe_blocking.py与copy_probe.py只有阻塞语义区别。当前正式源码为experimental/rc_blocking_mm_h2d.py。rc-sitecustomize.py、rc_kv_budget.py为RC部署现场快照；不能用EP bootstrap覆盖RC初始化兼容设置。

## 复现与回滚

本机镜像/模型/OPP已就绪时，在orangepi上先独占芯片、停止并重命名旧容器保留日志，再执行：

```bash
MTP=1 C8=0 GRAPH=1 GDN=1 MINDIEQ=1 RC_BLOCKING_MM_H2D=1 \
MAX_MODEL_LEN=114688 KV_BYTES=3221225472 MAX_BATCHED_TOKENS=4096 \
ASYNC_SCHEDULING=0 bash /data/qwen36-deploy/run-service.sh
```

确保experimental模块在部署PYTHONPATH上、bootstrap显式按开关导入，现场快照见本目录。C8回滚使用C8=1/KV_BYTES=6442450944，其他已测条件保持；需要重建容器，start旧容器不会更新参数。旧C16失败容器保留，不能直接当作可用服务启动。

纯计时在本仓根目录执行bash results/qwen36/orangepi-rc-kv-20261015/run-c16-timing.sh。它依赖独占C16服务。112K会议/128与1024像素图片/prefix热命中检查由post_timing_validation.py在计时结束后顺序执行。原运行期间修改过shell错误提示，Python已正常打印fixed-output comparison complete后shell因读取旧偏移返回2；JSONL各项最终usage/计数/样本均另行校验。现存脚本bash -n通过，不以shell退出码掩盖这次记录。

clt资料只读检查，18016健康、默认C16与其报告相符，没有更改其服务；同容量约0.99GiB同样是核算而非收紧预算实测。

## 当前可用候选与最后验收

orangepi:18024、容器qwen36-orangepi，当前C16/MTP1/Graph/GDN/MindIEQ/batch4096/同步/prefix-on，视觉1图/max_pixels1048576，max_model_len114688、标称KV3GiB、RC_BLOCKING_MM_H2D=1。112K普通文本会议决议更新正确，Prefill378.45/Decode14.68 tok/s；这与固定128输出的重复文本prompt不同，不能混算速度提升。128×128和1024×1024红色图片均识别正确；4223-token聊天prompt冷命中0，热命中2304，两次答案均703。长文和多模态后8题8/8及A/B/A一致、无canary泄漏。

本轮选择C16作为可用RC候选，C8保留对照；性能差值仍标注重启/兼容修复/池预算边界。后续若需要精确量化RC收益，应补同修复后的C8交替复测。当前不支持RC MTP2的有效性能声明，物理48G RC、并发、正式业务精度和P95仍待验证。
