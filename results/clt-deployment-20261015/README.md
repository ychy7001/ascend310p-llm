# clt 优化版部署与 RC 对照（2026-10-15）

目标 clt 为 x86_64、PCIe EP、单颗310P1、44215MB设备内存；orangepi 为 ARM64 RC、单颗310P1、89610MB共享内存。两者不能互用主机架构不同的扩展库，也不能将 RC 内存预算套用于 EP。

OrangePi现有记录显示，MTP1异步调度固定题4/8、关闭异步后8/8；MTP2仍有AI CPU异常。因此clt首轮采用同步调度、MTP1，不据此推断两台底层问题相同。

clt公钥和设备组已配置，SSH别名 `clt-192-168-29-49` 经 `ychy-a2` 跳转。部署在 `/home/clt/qwen36-optimized-20261015`，复用目标既有 Docker 与固定版本v0.23.0.post1-310p镜像，避免重建环境；只读复用原有59文件完整校验通过的权重，无需重新下载。

旧容器 `qwen36-w4a8-tp1` 保留，原端口18015；当前候选容器 `qwen36-optimized-112k-20261015` 端口18016；4K候选已停止并保留。回滚须先停止候选，随后 `docker start qwen36-w4a8-tp1`，单芯片不能同时加载两套35B服务。回滚前确认测试进程已完成或停止，避免请求打到已停候选。

候选：TP1/W4/C8/device-length/MTP1/FULL_DECODE_ONLY Graph/GDN/MindIEQ，4K、batch2048、单请求、util0.90、text-only、prefix off、async off。保留目标特有同步元数据上传补丁，移除旧CPU WY和逐步recurrent CPU检查。部署修改仅在目标归档副本，不改共享运行时默认配置。

初步结果：五类生成基础检查通过（仅检查空输出/明显退化，不是精度评测）；独立固定答案8/8。8K/32K重复测速、112K两位置检索和长请求后的短请求回归已完成；本轮仅构成单请求部署门禁，不是正式精度评测。

4K候选2304输入/128输出，warmup1+repeat3，独立prompt，无prefix：Prefill553.19–555.90、中位553.24 tok/s；Decode28.74–30.38、中位29.19 tok/s。MTP接受率80.28%–89.55%，按服务端阶段指标统计；不同于HTTP总输出吞吐。112K组合结果见下表。

## 112K 配置与复现

运行时核心源码94个Python/Shell文件按Git blob校验匹配 `e3fc3f9e85d7cd372d8a225317f24abea6f67861`；见 `source-provenance.json`。扩展库校验见 `artifact-sha256.txt`，OPP及所有制品文件校验见 `artifact-tree-sha256.txt`，目标源文件SHA256清单见 `deployment-source-sha256.txt`。这不表示对全部文档或二进制来源做了同等源码校验。

部署使用该提交的源码，覆盖 `experimental/run-c8-service.sh` 为本目录 `run-c8-service-docker.sh`，覆盖 `experimental/c8_bootstrap/sitecustomize.py` 为 `c8-sitecustomize.py`，并复制 `clt_sync_metadata.py` 到 `experimental/`。HTTP客户端使用该提交中的 usage 事件结束处理，防止服务端没有发送DONE导致客户端长时间等待。`candidate.env` 和 `start-candidate.sh` 记录目标机路径与启动配置；原权重和稳定容器均保留。

当前参数：`max_model_len=114688`、`gpu_memory_utilization=0.90`、`max_num_batched_tokens=2048`、`max_num_seqs=1`、MTP1、C8设备有效长度、Decode Graph、GDN、MindIEQ；仅文本、关闭异步调度和Prefix Cache。GDN ragged MTP PR18730与本轮部署分别验证；当前服务仅单请求，没有宣称并发变长批处理已验收。

启动日志报告KV预算5.40GiB，C8池实际分配5,532,254,208字节、共享scratch234,881,024字节、Graph约0.40GiB。vLLM通用容量241277、C8 Attention槽位170496、服务上限114688是不同数字；本轮只按真实请求确认上下文，不以通用容量作为产品规格。`npu-smi` 空闲常驻约35472MB/44215MB，属于设备总体占用，不能与Torch分配量混用。

```bash
ssh clt-192-168-29-49
# 启动入口：仅在该名称容器不存在、芯片已释放时使用
bash /home/clt/qwen36-optimized-20261015/start-candidate.sh
# 已存在容器停止后重启
docker start qwen36-optimized-112k-20261015
curl --fail http://127.0.0.1:18016/health
curl --fail http://127.0.0.1:18016/v1/models
# 测试状态
tail -f /home/clt/qwen36-optimized-20261015/logs/validation-worker.log
# 回滚：先确认测试worker已结束
docker stop qwen36-optimized-112k-20261015
docker start qwen36-w4a8-tp1
```

接口为 `http://192.168.29.49:18016/v1`，模型名 `qwen36-w4a8-c8`。本机通过a2跳板访问，远程用户需有相应网络可达性。

测试脚本 `validate_deployment.py` 使用固定单流端点：8K/32K各warmup1+repeat3、112000-token两位置检索各一次、每次128输出，再做A/B/A与固定答案。长检索强制输出128token便于计时，检索门禁单独检查输出开头是否精确包含对应代码；后续文本不当作自然生成质量评估。112K没有相同长度单独warmup，数据与8K/32K重复中位数分开报告。Prefix已关闭，当前不存在热缓存加速声明。

目标设备原有 `Warning A419321C lp pmbus error` 已记录，不将当前软件测试通过解释为硬件健康问题已经解除。

## 最终结果

112K配置服务持续运行，端口18016，健康检查200，容器重启策略 `unless-stopped`。该策略已设置到当前容器和Docker启动副本；未执行宿主机重启测试。

| 输入token | 输出token | Prefill tok/s | Decode tok/s | TTFT | 样本与检索 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8192 | 128 | 586.84 | 28.72 | 13.99秒 | warmup1+repeat3，中位数 |
| 32768 | 128 | 523.34 | 21.28 | 62.70秒 | warmup1+repeat3，中位数 |
| 112000 | 128 | 392.62 | 10.78 | 285.46秒 | 独立10%位置检索，通过 |
| 112000 | 128 | 394.23 | 10.63 | 284.31秒 | 独立90%位置检索，通过 |

两次长检索均精确输出对应代码开头；随后9/9固定答案通过，包括A/B/A复访，未发现该组测试中的明显状态串扰。8K/32K合成重复输入的MTP接受率100%；112K检索两轮接受率分别为81.4%、78.9%，不将前者外推为自然文本的接受率。

10秒间隔共56个设备内存样本，观测范围35514–35518MB /44215MB（npu-smi口径）；采样覆盖长请求区间，不是全启动峰值。原始阶段数据、流式事件、检索输出、固定题与设备采样均保存在本目录。

服务配置上限114688包含输入和输出。实际完成112000输入+128输出；114688输入+128输出被正确以HTTP400拒绝，见context-boundary.json。没有测试硬件极限，不应将170496槽位或241277通用容量作为已验收上下文。

112K冷输入TTFT约4.7–4.8分钟，Decode约10.6–10.8tok/s，相对8K下降约62%–63%。本轮确认部署与有限检索门禁可用，尚未确认持续业务负载、并发、结构化输出、视觉或完整精度评估。下一轮可独立验证Prefix Cache的冷/热正确性和收益，再决定长会议服务默认配置；本轮不声称热缓存优化已经验收。

模型公开来源：[ModelScope W4A8 SSZ MTP权重](https://www.modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P)。没有重写或重新量化目标已有权重。旧服务未做同条件阶段测速，因此不从历史12–14tok/s数据计算本轮加速比。

终端原始日志以 `.log.gz` 无损压缩保存，保留ANSI/回车和原始空白；用 `gzip -cd 112k-complete.log.gz` 查看。
