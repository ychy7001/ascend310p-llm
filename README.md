# Ascend 310P — Qwen3.6 W4A8 / MTP / C8 实验

在Atlas 300I DUO / Ascend310P3上研究Qwen3.6-35B-A3B低比特推理。首版提供可运行的Routed Experts W4A8与BF16 MTP适配，以及独立C8分页KV算子实验。**W4A8+MTP+C8+Decode Graph已在TP1/310P3上通过4K、32K和112K启动、正确性及长检索门禁；MTP1/MTP2同协议32K持续生成对照已完成。** Graph仍使用实验性updatable-graph开关，不能视为官方默认实现。本项目不是华为、Qwen或vLLM官方发布。

| 能力 | 当前状态 |
| --- | --- |
| W4A8 TP1/TP2、FP16 KV | 已在a2启动、生成 |
| W4A8 + Qwen3.6自身BF16 MTP | 一枚草稿TP1/TP2已运行；两枚草稿C8+Graph已完成4K/32K/112K门禁 |
| INT8 KV / C8 | TP1 C8/MTP1/MTP2 Graph均已完成启动、数值和长needle；仍是实验后端 |
| Decode Graph | mode0/FULL_DECODE_ONLY/[1,2,3]已在310P capture/replay；正式updatable-graph backend修复待上游确认 |
| 长上下文 | TP1/W4A8/C8/MTP1/MTP2 Graph均完成112K近上限检索；不宣称256K实际验收 |
| 全面质量及稳定加速 | 核心门禁完成；正式P95、多模态和TP2 C8仍待补齐；INT4不保证更快 |

模型制品：[ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P](https://www.modelscope.cn/models/ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P)。发布状态及独立加载证据以[验证记录](docs/release-validation.md)为准。

后续实验原始数据、关键日志和复现记录保存在本仓库的 `results/`，结论与参数选择更新到 `docs/`。此前群聊对接文档已交付；不再向 `/home/ychy/workspace/data/ascend` 写入或同步新报告。关键日志使用可追踪的 `.txt` 文件，避免被 `*.log` 忽略规则排除。

阅读顺序：[整体思路历程](docs/development-history.md) → [代码入口与维护边界](docs/code-map.md) → [华为会议评审报告](docs/huawei-adaptation-review-20261013.md)。最新N128产品组合对照见[原始记录](results/n128-product-baseline-n64-20261013/README.md)，固定32题质量回归见[结果与限制](results/n128-quality-20261013/README.md)。

## 后端交付

a2可用服务、权重、调用示例、重启和运行边界统一见[后端交付说明](docs/backend-handoff.md)。后端接入默认只使用18024端口及long-prefill组合。

## 最新单流部署候选

下面的 `TP1/48G/MTP2` 配置属于 **ychy-a2 EP**。`orangepi` 是另一台
单颗 310P1 的 RC 共享内存设备，其已验证基线为 **MTP1、C8、Decode Graph、
GDN、MindIEQ、prefix 开启、batch2048、114688 上限和固定6GiB KV**；
当前显存比例为0.45，不能直接套用EP的0.90/0.95。RC的四档重复数据见
[RC基线](results/orangepi-rc-baseline-20261015/README.md)，MTP兼容性和调优进展见
[RC MTP对照](results/orangepi-rc-mtp-20261015/README.md)。

当前业务候选收敛为TP1/48G、W4A8、动态C8、MTP2、Decode Graph、固定GDN PR、MindIEQ E5长Prefill分流、4096/512、112K，prefix默认关闭。唯一默认部署档为 `long-prefill`，短输入按阈值128回退原 W4；`compat`/`decode` 仅作为显式内部诊断和 A/B 对照。现行参数与长上下文门禁见[Profile说明](docs/profiles-20261014.md)和[部署矩阵](docs/deployment-matrix-20261014.md)。配置事实源为 `configs/tp1-48g-112k.env`；设置 MODEL_DIR、OPP_DIR、EXTENSION、C8_LIBRARY、GDN_OPP_DIR、GDN_LIBRARY、GDN_SOURCE、MINDIEQ_LIBRARY 后执行：

```bash
bash scripts/run-candidate.sh --check > /tmp/qwen36-artifacts.json
bash scripts/run-candidate.sh
```

`--check`检查本地制品并输出SHA256，不初始化NPU；该输出用于版本追溯，不等于完整模型哈希或兼容性验收。芯片0须独占。GDN固定外部源码/构建方法见[候选说明](docs/gdn-112k-candidate.md)。模型完整性另用validate_checkpoint.py检查。prefix可选设ENABLE_PREFIX_CACHING=1；目前未宣称全面质量等价。此入口仍是实验候选，不支持TP2 C8、32G或多请求批处理。

最新速度、质量和上下文边界以[GDN候选说明](docs/gdn-112k-candidate.md)、[MTP对照](results/pr17640-mtp-20261013/README.md)和[prefix复用](results/pr17640-prefix-20261013/README.md)为准。下文原启动器保留用于最小复现及早期基线，不应把其1024/MTP1参数与新候选混用。

MindIEQ最新可选Prefill路径直接读取现有native W4权重，使用独立FP32算术A8量化；[服务门禁](results/mindieq-a8-service-20261014/README.md)和[输入规模分流](results/mindieq-input-dispatch-20261014/README.md)已完成固定64题及性能对照。阈值128消除短输入开销，短输入仍走原W4。需要区分早期 grouped 路径相对旧基线的历史提升与当前增量：在同一批输入、两颗同型号310P交替五次的复测中，MindIEQ 相对原始 W4A8 的 Prefill 中位提升为 8K **23.29%**、32K **20.69%**；Decode 变化为 **-0.06%/-0.48%**。64K 和近112K单次长请求仍分别提升 **17.59%/14.66%**，两边 Decode 差异小于0.1%。[投影数值定位](results/mindieq-projection-parity-20261014/README.md)确认一项差异来自scale累加顺序，尚不能解释全部模型差异。[Gate/Up量化共享](results/mindieq-shared-gate-20261014/README.md)通过单层数值/异步/Graph微测试、64题门禁、112K needle和32K多位置needle；因此当前建议是：将 MindIEQ 作为 **112K 单流长 Prefill 候选分流**，保留 `MIN_TOKENS=128` 阈值和原W4回退，现已作为长上下文业务的默认实验档；P95、完整业务质量集和TP2/C8并发门禁仍待完成，不宣称正式产品验收。早期“+2.5%～2.9%”仅指相对前一 grouped 版本的 E5 shared-gate 增量，不是相对原始 W4 的最终对比。

## 精度与计算路径

仅routed experts量化为signed INT4，使用ModelSlim对称per-channel SSZ。scale按G128 kernel布局重复存储，**不是独立group-wise量化**。activation为设备端per-token INT8，使用INT4 limbs分解组合Cube整数计算。非专家张量保留原始浮点；MTP保留Qwen3.6自身19个BF16张量。text-only生成已测试，图像能力未验收。来源与社区贡献见[provenance](docs/provenance.md)和[NOTICE](NOTICE)。

## 固定环境与构建

已测镜像`quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`，CANN9.1，Torch2.10，torch-npu2.10.0.post4，SOC=Ascend310P3。SDK/driver由使用者安装，仓库不附带这些二进制。

在该镜像内挂载本仓库与一个可写工作目录，设置Torch backend autoload=0，禁用镜像不兼容Triton，source CANN环境后：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
WORK_DIR=/work/qwen36-build bash /project/scripts/build-native.sh
```

工作目录必须是新的隔离目录。脚本取固定社区commit，应用本仓库算子快照，构建3个OPP家族与独立`qwen36_int4` Torch namespace，不覆盖官方`_C_ascend`。构建与独立启动的实际验证范围见release-validation；首次第三方依赖下载受网络影响。

权重下载后先做完整性检查：

```bash
ms-hub download ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P --local-dir ./weights
python3 tools/validate_checkpoint.py ./weights --verify-hashes
```

## 启动服务

宿主使用rootless Podman；TP1只暴露davinci0，TP2额外暴露davinci1。这里的默认4K是首次复现门禁，长上下文应按实测内存预算单独设置。

```bash
MODEL_DIR=/absolute/path/weights \
OPP_DIR=/absolute/path/qwen36-build/artifacts/opp/vendors/custom_transformer \
EXTENSION=/absolute/path/qwen36-build/artifacts/qwen36_int4_custom.so \
TP=1 MTP=1 MAX_MODEL_LEN=4096 MEMORY_UTILIZATION=0.90 PORT=18015 \
bash scripts/run-service.sh
```

`MTP=0`关闭MTP；TP2设置`TP=2`。标准启动器用于W4/FP16-KV基线；C8+Graph必须使用`experimental/run-c8-service.sh`。C8产品候选参数是`MAX_MODEL_LEN=114688 MEMORY_UTILIZATION=0.95 MAX_BATCHED_TOKENS=1024 QWEN36_C8_POOL_MODE=split_by_type QWEN36_C8_FUSED_WRITE=1 QWEN36_C8_DEVICE_LENGTH=1 ENFORCE_EAGER=0 QWEN36_C8_ALLOW_GRAPH=1 QWEN36_C8_FORCE_UPDATABLE_GRAPH=1 COMPILATION_MODE=0 GRAPH_CAPTURE_SIZES=1,2`；MTP2实验档额外设置`QWEN36_C8_ALLOW_MTP2=1 NUM_SPECULATIVE_TOKENS=2 GRAPH_CAPTURE_SIZES=1,2,3`。这些参数依赖当前实验镜像和自定义C8库，不能当作官方通用vLLM参数。容器只挂载自包含模型目录，不依赖原始BF16或sidecar父目录。

以下旧 C8 wrapper 用于内部诊断复现，示例显式选择 MTP1；正式实验候选统一使用上文 `scripts/run-candidate.sh` 的默认 `long-prefill`。旧的 `mtp1`/`mtp2` 名称仍可用作兼容别名：

```bash
MODEL_DIR=/absolute/path/weights \
OPP_DIR=/absolute/path/qwen36-build/artifacts/opp/vendors/custom_transformer \
EXTENSION=/absolute/path/qwen36-build/artifacts/qwen36_int4_custom.so \
C8_LIBRARY=/absolute/path/qwen36-build/c8-device-length-build/output/libqwen36_c8.so \
QWEN36_PROFILE=mtp1 PORT=18020 \
bash experimental/run-c8-112k-product.sh
```

`QWEN36_PROFILE=mtp2` 仅改变 draft 数和 Graph capture sizes，仍依赖实验性 updatable-graph 路径；该 wrapper 不声称 TP2 或 32G 支持。

```bash
python3 tools/bench_generation_sanity.py --base-url http://127.0.0.1:18015 --model qwen36-w4a8 --out /tmp/generation.json --max-tokens 128
python3 tools/bench_http_stream.py --base-url http://127.0.0.1:18015 --model qwen36-w4a8 --prompt-tokens 128 --output-tokens 256 --concurrency 1 --repeats 3 --out /tmp/stream.jsonl
```

sanity只检查服务可用性、非空输出与明显异常，不替代准确率评测。benchmark计时含HTTP/调度；设备kernel性能单独采集profile。代码API以脚本`--help`为准。

## C8与技术反馈

[C8独立复现](docs/c8.md)包含选页NZ转换、原生PA对照、Cube/Split-KV候选和混合Mmad编译探针。需要真实310P执行；FP16工作区额外内存必须计入容量收益。

[华为反馈报告](docs/huawei-feedback.md)整理已知结果与五项具体问题：C8接口、混合Matmul、NZ/UB/L1流水、W4小M/Prefill与hybrid/MTP cache状态。TP1/C8/MTP1/MTP2/Graph及112K长检索已测；正式backend、尾延迟、TP2 C8和32G实卡仍待验证。

[详细技术评审报告](docs/huawei-report.md)记录完整适配路径、初版TP1/TP2/MTP与C8测试数据、失败证据和待补门禁。最新Graph与长上下文数据见[C8记录](docs/c8.md)；此前已转发的DOCX/PDF为冻结版本，并未自动同步后续实验。

[Graph上游交接记录](docs/graph-upstream-patch.md)说明了310P Graph生命周期的版本差异、当前诊断开关和提交上游前必须核对的最小代码边界；当前不把实验monkeypatch冒充正式backend修复。

给硬件/算子评审的参数决策、C8/C16 和 MTP0/1/2 取舍见[参数选择与华为评审记录](docs/parameter-selection-20261012.md)。其中明确区分已验证结果、当前产品候选和待补实验，避免把容量估算或单次启动结果当作性能承诺。

最新同配置MTP1/2与长输入性能见[Prefill/Decode实验记录](docs/prefill.md)和[C8记录](docs/c8.md)。112K Graph、32K输入/256输出的同协议对照为：MTP1 Prefill/Decode `339.28/23.20 tok/s`，MTP2 `336.56/28.94 tok/s`，Decode约提升24.8%，Prefill约下降0.8%。并发2/4的MTP2总输出吞吐均约5.06 tok/s，说明并发瓶颈仍在Prefill/调度；固定协议结果不是P95或W8/W4严格等价对照。

## 从原始浮点重新量化

`tools/convert_modelslim_w4a8.py`与`modelslim_ssz_adapter.py`复用已安装ModelSlim SSZ，只输出expert sidecar；`prepare_hybrid_model.py`组合源模型与sidecar的实验目录，仍需导出自包含checkpoint。该组工具不会覆盖原始权重。ModelSlim版本应随重新转换单独锁定；当前公开制品的不可变标识为文件SHA256。

`python3 tools/export_checkpoint.py /path/to/hybrid /path/to/new-standalone`只复制index引用的tensor payload，逐tensor核对拷贝前后SHA256并生成新文件哈希；不重新量化、不保留源软链接。目标必须不存在。导出后运行`validate_checkpoint.py --verify-hashes`，再用隔离服务验证MTP和生成。
