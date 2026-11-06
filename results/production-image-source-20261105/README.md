# 生产镜像源码化：候选发布与设备窗口（2026-11-05）

后续同日启动复现和冻结阈值复核见 [后续窗口](evening/README.md)；本页保留早先窗口的结果。八格基准的失败与规则修正在 [基准窗口](bench/README.md)。

**候选已发布，设备验收未完成。** 生产 A、生产新进程 A′ 完成 G1/G2；候选 B 在 TP1 权重加载之前的 Python 子进程 CPU FP16 zeros 初始化中段错误退出，退出码 1，未达到健康状态。按计划外问题停止规则，未重试、未改参数、未采 B 数据、未跑性能矩阵。根因尚未确定，不能把 CPU 文件一致视为设备验收通过。

## 发布与固定运行条件

候选：`${REGISTRY}/${NAMESPACE}/ascend310p-llm-candidate@sha256:c590aebfa21e59f75a7d845c5b5563bce51292539c53b6b438057ffc24d167f0`，tag `image-src-f46116b-amd64-310p`。恢复基底：`${REGISTRY}/${NAMESPACE}/vllm-ascend-310p-base@sha256:f7b14f2deb44b7558cc5506bbd7a329126bd260491d35a88249f94119b154a4f`。

两者均已回读原始远端 manifest，其 SHA256 与本地推送输出一致。候选远端运行 Config、全部 diff_id 与本地候选一致；格式转换及压缩导致 manifest/config 外层摘要不同，详见 [发布记录](../../deploy/image/production-publication.json)。未写入生产镜像仓库的任何 tag。

三臂固定 Qwen3.6-35B-A3B W8A8、TP2、MTP0、maxlen262144、batch4096、seqs4、Graph[1,2]、prefix on、float16、gpu-memory-utilization0.9，来自归档生产命令。同一只读权重挂载、同一启动脚本及环境，仅镜像和独立缓存目录不同；隔离实验服务的 host/port。A、A′ 使用锁定生产 digest，B 使用上述候选 digest。未注入实验算子或 worker。

## 冻结方法与生产噪声

先采 A/A′ 再启动 B，B 前冻结 G1 阈值为 A′ 对 A 的实测误差 ×1.0，无绝对误差地板；G2 各项差 ≤1。35 条固定输入共236375 token：26 条原样回放、5 条边界输入、16K/32K 合成会议各两条。每条 greedy256、temperature0、seed0、生成 top-20 logprob。使用既有生成观测协议，未做已知 OOM 的 prompt-logprob 探测。

输入 SHA256 `e8fa2ed60bb717b2cad731988a0f85ba9dd07f39f0e614b2fb2e9b333b9e8c13`。A、A′ 各生成8960 token，前缀缓存查询各236375、命中各0。

具体阈值文件 SHA256 `870c0b398c23c9804605f6170f01f5de86521f8c979bbc94b8a62848246dc0e0`。冻结时间 `1793849418237940314` unix ns，早于 B 容器启动 `2026-11-05T03:30:20.374250986Z`；冻结值与本地重新计算的生产噪声完全一致。

| G1 指标 | A′ 对 A / 冻结阈值 | B |
| --- | ---: | --- |
| Top-1 不一致率 | 2.803738% | 无数据 |
| 平均共同 top-k 条件 KL | 0.0372147682 | 无数据 |
| 最大共同 top-k 条件 KL | 6.1917063657 | 无数据 |
| 最大绝对 A 目标 logprob 差 | ≥12.7878983021；冻结数值12.7878983021 | 无数据 |
| Top-1 首分歧后剩余比例，均值 | 87.232143% | 无数据 |
| Top-1 首分歧后剩余比例，最大 | 100% | 无数据 |
| Greedy 首分歧后剩余比例，均值 | 87.232143% | 无数据 |
| Greedy 首分歧后剩余比例，最大 | 100% | 无数据 |

生产两次进程自身已有33/35条 greedy 分歧；生成观测只比较相同条件前缀并包含首分歧位置，共1177位置，Top-1一致率97.196262%。5处 A 目标 token 未出现在 A′ top-20，最大绝对 logprob 差仅是保守下界。KL 仅对共同显式 top-k 支持分别归一化，不是完整词表 KL。B 未启动成功，不能进行 B 对该噪声底的判定。

## G2 与基准状态

| 项目 | A | A′ | B |
| --- | ---: | ---: | --- |
| GSM64 | 63/64 | 63/64 | 未执行 |
| 结构化 C1 | 13/13 | 13/13 | 未执行 |
| 结构化 C3 | 13/13 | 13/13 | 未执行 |
| 短题前测 | 8/8 | 8/8 | 未执行 |
| 短题后测 | 8/8 | 8/8 | 未执行 |

两个生产臂的 GSM 各有1题在512-token上限截断且未解析答案；结构化 usage token 不匹配均为0，短题复用一致、无 canary 泄漏。G2 未完成三臂判定。

1K/4K/16K/32K × C1/C4 八格基准均未执行，无 TTFT、TPOT、吞吐或不劣于生产的结论。会议回放和 RAG Prefill 也无本轮性能收益数据。

## 失败证据、窗口与限制

B 在启动后约64.5秒退出，容器 `ExitCode=1`、`OOMKilled=false`。更正后的日志定位是TP1 权重加载之前，一个 `python -c` 子进程在对象初始化中调用 CPU FP16 `torch.zeros`，原生栈经过 ATen AVX2 `fill_non_native_type<c10::Half>`、`fill_kernel`、`zero_` 和 `THPVariable_zeros`。`tvm_ffi` 是信号报告者；后续 `WorkerProc initialization failed` / `Engine core initialization failed` 是退出结果。TP1 没有打印 `Starting to load model`，也没有 `Loading model weights took` 或可用 KV cache 日志；另一个 TP0 在段错误之后才打印开始加载，不能将其当成崩溃子进程的阶段。原始 B 未单独保存完整日志，仅保留 emergency 日志；启动区间内核日志没有 OOM 记录，根因仍未确定。aarch64 loader 警告在成功的 A、A′ 中也出现，未据此归因或添加兼容处理。没有分析或改动 Ascend 算子/内核实现。

阻塞排队4318.97秒后取锁，取锁时原服务 active、health=200；全窗口持有共享 NPU 锁。结束时恢复原单元，在新的独立 SSH 连接确认 health=200 后放锁，持锁3132.73秒（52分13秒），未超过三小时。健康结论对应解锁前的恢复确认。

现有 CPU 证据仍是16869个源码/安装文件相同、全部非缓存差异0、pip freeze/环境/入口/运行 Config相同；65项缓存差异单独记录。CPU 复现通过与本轮候选启动失败是两个独立事实。没有重试或性能/业务验收结论，没有切换生产引用，没有推送 Git 分支。

可复用命令为 `tools/compare_token_logprobs.py`、`tools/eval_gsm8k_http.py`、`benchmarks/ascend310p/check_action_state_regression.py`、`tools/check_cache_request_reuse.py`；比较工具支持冻结倍率，9项单测通过。只读原始响应、退出日志、启动命令、冻结文件和窗口恢复证据留在仓库外，摘要及证据 SHA256 见 [adjudication.json](adjudication.json)。
