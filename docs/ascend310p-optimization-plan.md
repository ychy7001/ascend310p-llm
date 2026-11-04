# Ascend 310P 优化：现状审计与执行计划

当前优先级（2026-11-04，用户新增模型/profile对照后更新）：P0继续正确性，9.2.1配套H/O在原Torch WY下已通过两卡算子门禁；35B结构化字段每栈26/26，但自由文本仍未接受。下一阶段保留期限差异的受控复核，并将27B Dense W8A8 TP2完整模型质量对照提前到正式性能/Decode扫描之前。跨模型对照不代替同模型高精度归因。之后优先35B W4A8 TP1预算档位与27B W8 TP1容量筛选，27B W4A8先审计权重/算子适配；八种组合不视作已支持。原生WY须在9.2.1重编译并走算子/模型门禁，质量过关后才正式性能。a1只读，a2实验后恢复原服务。详见docs/research/model-quality-and-deployment-profiles.md；下文按历史记录解释。

会话日期：2026-11-03。首次按用户“先梳理情况”完成只读审计，随后用户授权基线、Profile与优化。Git起点main / `9828073`；工作分支experiment/hyperqwen-310p-20261103，尚未提交。

## 用户明确后的主线（优先于下方首次审计记录）

- 生产事实源：**LAB_PRIMARY，Qwen3.6-35B-A3B，W8A8 TP2**。面向业务优化与上游反馈/PR。
- a1实际命令：MTP0、Graph[1,2]、maxlen262144、batch4096、seqs4、FP16 SSM、prefix on、async，量化后端ascend；完整脱敏manifest见results/production-a1-audit-20261103/。
- 单卡W4A8是未来预算节省方案。此前TP1四档/GSM64/Profile归档为这一支线，不作为生产基线或生产瓶颈结论。
- Prefix Cache已经在生产开启，仅记录能力与状态恢复；不作为本轮新增优化收益，不再扩展缓存扫描。
- a1不重启、不改参数、不采在线Profile、不发压测。a2已复制并校验W8权重，使用完全相同生产镜像digest完成隔离基线、业务回放、Profile和首项Graph初筛。驱动a1=26.0.rc1、a2=26.1.1，必须记录差异。
- 主线顺序：生产制品与业务形状冻结 → a2 W8 TP2基线/业务语义门禁 → W8 Profile → 最小性能/正确性改动及A/B → 筛选可提交上游的独立修复。27B Dense独立验证，不以Flash-Next替代。
- 历史Graph[1,2,3]失败原始日志明确含EL0008事件资源不足及IPC notify失败；不能只凭栈顶HcclAllGather就归因为Attention或GDN算法错误。先在隔离环境复现。

## 当前执行状态

- W8生产主线基线、GSM64及双rank两范围Profile已完成，原始数据与统计口径见baseline/profiling报告。
- Graph[1,4]重试成功运行，C1/C3/C4中位聚合吞吐分别+0.18%/+2.96%/+5.43%；C3未达到预设5%门槛，决定不选择，未执行候选长回放/GSM。
- a1生产保持不变；本会话实验服务已停止。下一项是根据W8 Prefill证据复用原生GDN/WY，先验证生产栈兼容/数值，再做单变量A/B。27B Dense留待35B主线形成可复用结论后独立测试。

## 首次审计结论

现有工程已经完成 W4A8、MTP1/2、GDN、Decode Graph、MindIEQ Prefill 和 C8 的多轮适配，不能把这些功能重新包装为本轮成果。当前最缺的是固定环境下可比较的 C16 性能/并发/正确性基线，以及 MTP verify 与状态管理的精确 Profile。

用户已确认：第二目标仍为 **Qwen3.8-27B Dense**。仓库和 a2 已有的是 **Qwen3.8-Flash-Next MoE**，不等价，不能用其结果替代 Dense 基线。此轮没有在已检查的模型位置找到 27B Dense 权重，也没有确认其 310P 配置。

## 实机情况

原始证据：[a2-environment.json](../results/audit-20261103/a2-environment.json)。

| 项目 | 实际情况 |
|---|---|
| 主机 | LAB_REPLICA，x86_64，Linux 6.8.0-31-generic |
| NPU | npu-smi 编号 8，两芯片/逻辑设备 0、1，均为 310P3、Health OK |
| 可见设备内存 | 44277 / 43693 MB；采样占用 1802 / 1163 MB；这是 npu-smi Memory-Usage，未将其另行认定为纯 HBM |
| 运行状态 | 两芯片 AICore 0%，无 NPU 进程；没有运行中的 Podman 容器或推理监听端口 |
| CPU 内存/磁盘 | 31 GiB RAM、约 29 GiB available，8 GiB swap；/home 所在盘约 1.3 TiB 空闲 |
| 驱动 | 26.1.1，宿主 version.info 与 npu-smi 一致 |
| 宿主 CANN | /usr/local/Ascend/cann → cann-9.1.1；compiler/version.info 为 9.1.1 |
| Qwen3.6 容器 CANN | **9.1.0**，V100R001C11SPC001B243，不能与宿主 9.1.1 混写 |
| 容器包 | torch 2.10.0+cpu，torch-npu 2.10.0.post4，vLLM 0.23.0+empty，vLLM-Ascend 0.23.0.post1，transformers 5.5.4；从停止容器 METADATA 读取，未导入 torch |
| Profiler | 宿主 /usr/local/Ascend/cann-9.1.1/tools/profiler/bin/msprof 存在；安装元数据 mindstudio-profiler 26.1.0；容器采集与导出需另验 |

保留的 `qwen36-image-managed` 已停止，inspect 记录 ExitCode137、OOMKilled=false、StoppedByUser=true；仅凭137不能判定 OOM。其镜像 ID 为 `b381d8e5ed45…`，manifest 来源仓库提交为 `c9fd1daf…`，与本地 HEAD 不同。基础源码固定为 vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`、Ascend `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`。manifest 记录不等于本轮重新验证了全部文件哈希和二进制来源。

Qwen3.6 模型位置：`/path/to/lab-runtime/qwen36-public-release-20261012/model`。配置为 qwen3_5_moe，40层、256 experts、hidden2048、vocab248320、max_position262144；模型名版本与内部架构名不同不自动构成错误。顶层 quantization_config 为空，应按本项目 sidecar/SSZ 格式核验，不能判断为浮点模型。此轮未做全权重 SHA256 重验。

Flash-Next 位置：`/path/to/models/Qwen3.8-Flash-Next-W4A16-G128-300i-d32978c`，qwen4_exp，48层、512 experts、hidden2560。已有16K TP2 Graph容器也已停止。

## 当前配置、历史推荐和适用边界

a2 停止容器保存的 Qwen3.6 配置是 TP1/W4A8、C16、MTP2、MindIEQ、Decode Graph、max_model_len114688、batch2048、max_num_seqs1、async调度、prefix off、利用率0.95。启动时还需要核对入口解析后的有效配置。容器名含C8不能证明实际用了C8。

2026-10-31 的[业务回放](qwen36/business-replay-20261031.md)则明确建议生产保持 W8A8 TP2、Graph[1,2]、MTP0；这是业务推荐，不是说 a2 目前有该服务运行。此轮没有在检查到的位置定位并校验可直接运行的 W8A8 权重和镜像组合，后续先追溯当次脚本/来源。不能为了复现而改动其他机器生产服务。

旧文档存在跨时间、跨机型结论，例如 RC/C8/MTP1 的“当前默认”和 a2/C16/MTP2 的默认并存；以后以设备+镜像digest+源码+有效参数+实验日期标识配置，不直接用“当前”或主机别名作基线身份。

## 功能矩阵

“已有代码/上游入口”“实机历史通过”“本轮运行通过”分开解释。此矩阵记录首次审计时点；后续本轮运行结果见baseline/profiling/results报告。

| 优化项 | 上游/参考项目情况 | 310P及本仓库证据 | a2保存配置 | 待验证 |
|---|---|---|---|---|
| W8A8 | 官方量化路线；业务记录使用官方后端 | 历史TP2跑通；TP1工作区OOM | 未运行，保存镜像为W4 | 权重/镜像定位，同协议TP2基线 |
| W4A8 | 社区310P实现，本仓库固定算子和SSZ ABI | TP1/TP2历史通过；不代表官方main通用兼容 | 启用 | 制品哈希、输出门禁、新基线 |
| GDN FP16 | 实际310P源码 `_310p_get_state_dtype` 返回float16 | **已实现**，不能只读模型config的float32就认定未启用 | 源码路径已存在 | 启动后记录真实tensor dtype；无收益假设前不改精度 |
| Prefix Cache | 上游具备混合模型缓存能力；HyperQwen依赖版本与模式 | 有检查脚本，部分旧配置开；新默认未全面验收 | 关闭 | GDN状态恢复、共享前缀/零命中分测、长后短与隔离 |
| ACLGraph | 现有310P兼容层；今日main非同一实现 | W4单流历史capture/replay通过 | FULL_DECODE_ONLY | 新C16复测；W8 TP2/并发组合不能继承结论 |
| MTP | 已有原生模型入口和310P适配 | W4 TP1草稿1/2历史通过；业务并发MTP2曾语义错配 | 2枚草稿 | MTP0/1/2固定语料、verify/rollback、采样行为 |
| N-gram/Suffix | 本地上游源码有Ascend proposer注册 | 未验证本固定0.23镜像/目标模型/310P组合 | 未启用 | 固定源码版本与拒绝采样/状态回滚 |
| MTP量化/裁词表 | HyperQwen有量化脚本和draft-vocab patch | 本项目MTP保留浮点权重；无此组合验收 | 未启用 | 先测draft/head占比；词表映射、拒绝/分布修正 |
| Context Lookup | HyperQwen与DFlash2联合实现；不可等同普通ngram | 未发现本仓库等效集成，310P未验证 | 未启用 | 先测现有ngram/suffix，再评估联合方案 |
| INT8 Prefill Attention | HyperQwen有Triton int8-QK路径 | 本仓库C8是KV存储，不是等效INT8 Attention | 未启用 | 310P算子支持/shape、长上下文数值与总收益 |
| DFlash2 | 本地上游已有DFlash2 proposer；HyperQwen有多项叠加补丁 | 不能据此推导310P支持，相关源码含Triton依赖 | 未启用 | 模型专用drafter、容量、310P算子、状态/verify |
| MindIEQ Prefill | 本仓库可选TP1分流 | E5历史8K/32K收益及63/64小样本 | 启用 | 实际专家分布、并发、语义质量和C16新Profile |

GDN证据：[停止容器实际源码](../results/audit-20261103/qwen36-installed-gdn.py)，149–151行为FP16状态选择；该文件也明确注释当时算子不支持FP32状态输入。模型config里的mamba_ssm_dtype=float32不能代表实际分配值。

## HyperQwen与上游现状

在线查询原文和来源地址存于 [audit目录](../results/audit-20261103/README.md)，避免后续main变化无法追溯。

- HyperQwen main：`f382342d749690e0a83a75348a48f355623e23d2`。已核对优化文档和patch目录，确有Embedding/head量化、MTP量化/裁词表、Marlin INT8修复、split-KV verify、Context Lookup、DFlash2、INT8 prefill attention及prefix相关补丁。本轮未逐行审阅全部补丁，文档性能是作者在3090上的自述，不能视为310P结果。
- 官方vLLM-Ascend main查询快照：`23e0e3764293f0c85ebf96c1383fbbf431b8fbb3`；vLLM：`190df0b90007eead441d97d3abf6270f84f2c63e`。这是查询时刻快照，不自动更新现有环境。
- Ascend #18730（ragged MTP GDN state rows）仍open、未合并；#17640（310P GDN Prefill）仍open、未合并。不能认为历史外部patch已经进入官方发布包。
- open PR线索包括 #20190（MRV2/310P sampler post-processing）、#20360（MRV2/310P structured outputs）、#20265（NPU top-k/top-p）、#20002（DFlash context K RoPE）。标题命中只是候选，尚未证明其修复本任务瓶颈或在310P可用。
- 本地另一个upstream checkout已注册ngram/suffix/DFlash2，但其SHA与在线main快照不同；使用前需在隔离分支固定精确版本、依赖和实现，不能与旧0.23镜像混合归因。

CUDA、Marlin、Triton GPU kernel不能直接用于310P。优先复用现有NPU算子和Ascend proposer；高级特性留到Profile证实价值之后。

## 下一轮顺序

1. 冻结a2镜像/源码/算子/模型/tokenizer/有效参数；使用独立容器和端口，不重启历史服务、不覆盖原模型。定位W8业务回放制品，但以已在a2保留的W4 TP1 C16配置作为最先可复现的工程基线。
2. 先跑短题、A/B/A、固定GSM64和长后短；再以C1、1K/4K/16K/32K输入、256输出、零prefix命中建立基线；固定输入token序列与哈希，多轮重复。资源允许再做512输出与C4；C16/C32须先估算状态/KV容量。
3. 对C16基线采集短/长Prefill、Decode以及MTP draft/verify、host/device时间线；旧C8/RC Profile仅作假设来源。
4. 第一个候选按Profile决定：优先隔离状态索引/metadata同步或现有graph/调度路径。GDN FP16已经存在，不重做；不在没有C16证据时改W4 kernel或直接上DFlash2。
5. MTP0/1/2每次只改变一个主变量，关闭MTP时配套合法capture size并如实记录；业务语料要检查职责/实体/期限错配，不能只看GSM分数。
6. 第二模型27B Dense单独做权重/容量/支持审计，再建独立基线。

首次交付为审计；后续已进入授权实验阶段，基线、Profile、结果分别维护于同名前缀的三个报告。

## 用户授权继续后的执行增量

已在experiment/hyperqwen-310p-20261103分支建立独立实验，TP1 C16四档C1性能及长前后短题门禁完成，详细数字见ascend310p-baseline.md。该组不是TP2；用户询问时已明确说明。GSM64完成后再切到独立Profile容器；原镜像、模型和历史容器保留。第二目标来源已由用户补充，三份元数据审计见results/qwen38-dense-metadata-20261103/。

实测32K请求Prefill约60秒而Decode约9秒，支持在重复长文任务上验证Prefix Cache，但不把命中收益当冷Prefill算力，不默认改变生产推荐。底层Decode优化仍由新C16 Profile决定。


## WY 单变量实验进展（2026-11-03）

PR17640 固定 head18163f6b 的 WY-only 接入已构建、数值门禁与模型初轮完成，FwdH/FwdO保持生产版本；详见[结果](ascend310p-results.md)。四档 Prefill 提升38%～54%，长输出业务回放耗时减少约3.1%，Decode单请求未改善。会议4曾出现严重负责人错配，目前未验收；重启后串行6次映射正确，正在复测原三并发序列，不能据此消除首次失败。

111-token 原路径偏差已在空闲NPU复现：原路径3/12在固定门限内、WY候选12/12。CPU两参考在同一失败输入上的输出最大差3.59e−6。当前尚未定位偏差是在WY helper还是后续FwdH/FwdO；计划同步截取helper输出并固定其输入重放，避免仅以候选不再出现症状认定已修复。候选Profile安排在普通计时与业务复测之后，当前尚无候选kernel trace结论。

用户已明确a2为实验机，授权暂停临时unit独占实验并在结束恢复。a1未修改。实验目录 production-native-wy、production-wy-control-recheck、production-wy-batching-recheck；控制器顺序执行，不并发压测。


### FwdO 最小补丁排除与 Dense 准备（本轮续）

原始算子链中，WY/FwdH/FwdO 输入逐位稳定时仍可出现 raw FwdO 输出误差；这只能定位接口范围，尚未证明根因。原始 FwdO 单独固定输入重复调用可以通过，因此不能用该单项通过代替完整链门禁。

两个独立 API 的最小补丁均失败：向量 mask 改写、保留原算法仅加 V/S fence，各在 48 次固定输入验证中为 0/48，而配对原始算子为 48/48。没有启动补丁模型，没有修改生产 OPP；补丁不具备合入资格。构建框架差异仍需通过“不改算法、仅独立重编译”对照排除。对照准备过程的符号链接、安装目录权限及历史日志/容器名称冲突均保留失败记录；准备失败不计为数值结果。

核对镜像真实旧路径 csrc/moe/chunk_fwd_o、chunk_gated_delta_rule_fwd_h 和 common，59 个文件与生产提交 Git blob 一致。此前误查带 _vllm 后缀路径造成“源码缺失”的判断已撤销。

Qwen3.8-27B Dense 采用用户提供的 Eco-Tech/Qwen3.8-27B-w8a8-310p：26 文件、36,447,005,796 字节、9 个权重分片；逐文件固定 revision/SHA256、safetensors header/index 校验通过。完成下载不代表模型启动或推理通过。新增 run-dense-baseline.py 在算子对照结束后独占 a2，先按生产镜像与完整上下文配置建立原始算子基线，结束恢复 a2 临时 unit。


### 原始重编译对照已完成；Dense 首次启动失败

原始 FwdO 算法独立重编译完成：固定输入 production/rebuild 各 48/48 在门限内且逐位稳定；原始完整链仍有 18/24 次越界（Torch WY 调用历史），本轮 frozen-WY-clone 为 0/24。故两补丁的 0/48 不可简单归咎于构建框架，而完整链异常尚未解决。证据 `results/fwd-o-rebuild-control-20261103/`。

Dense 完整生产配置启动已实际尝试：两个 rank 权重加载成功，各 19.2416 GB，随后首次 profiling/warmup 报 507015，设备日志指向 `QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21`。没有请求、性能或质量结果，不能记为 Dense 基线完成。第一次同步诊断因 ACL Graph 不允许 ASCEND_LAUNCH_BLOCKING=1 而被参数校验拒绝；改 eager/Graph NONE 后，设备异常仍复现，Python 同步栈定位 MLP down_proj 的 310P W8A8_DYNAMIC Linear。待合成矩阵形状/布局诊断，未证明具体根因。

上游反馈草稿 `docs/upstream/310p-chunk-gdn-repeatability.md` 已整理，不依赖权重的最小脚本可复现原路径；尚未发布，也没有宣称修复。


### 现有上游 FwdO 单独复用门禁

GruntovDima PR17640 固定 head18163f6b，算法不改、仅提取独立 FwdO 绑定，生产镜像构建成功。保留生产 FwdH 的组合，固定输入上游 FwdO 0/48 在容差内（48/48逐位稳定），配对生产48/48通过；maxabs约0.004327、relative RMSE约0.195875，异常token0～6、64～70。严格门禁停止，未运行模型、未测性能，也未测试该PR的完整FwdH+FwdO组合。不能外推为上游完整PR错误。证据 results/upstream-fwd-o-20261103/。


### Dense 实际输入捕获与缓存对照

同步 eager 捕获仍复现启动异常，首次 down_proj 为M4096/K8704/N5120：INT8激活ND、INT8权重NZ转置view、F32通道/逐token scale；激活及逐token scale全零且有限。真实数据独立重放6项全部通过CPU采样整数积检查（maxabs0），含4096行原/ND/NZ激活布局和1/4/64行子集。零scale不是已证实根因；独立重建改变分配、描述和调用历史。首次重放在weights_only序列化Format枚举处失败，未运行算子；v2显式allowlist安装库Format类后验证通过，原记录均保留。

GDN执行器缓存容量对照：默认21/48、ACLNN_CACHE_LIMIT=0为25/48在容差内；完整Torch-WY链两边均12/24越界。未消除原异常，不能算修复。控制器清理时恢复a2临时unit触发systemd启动频率限制，原status保留failed；reset-failed后交接Dense独立缓存启动screen，最后统一恢复。

Dense问题草稿见docs/upstream/310p-dense-w8a8-startup.md；无模型权重的GDN复现包results/upstream-gdn-repro-20261103.tar.gz及SHA清单已生成，均未发布。


### 本轮最终状态

Dense在原Graph配置下仅设ACLNN_CACHE_LIMIT=0，仍启动失败507015，故障kernel仍为QuantBatchMatmulV3_NZ_NZ_int8_int8_fp16_high_performance_21；没有HTTP请求或性能数据。记录results/qwen38-dense-no-executor-cache-screen-20261103/。至此，生产布局随机矩阵4项通过、真实输入独立重放6项通过，但模型启动故障未定位，不能提交猜测性数值修复。

a2本轮rootless实验容器全部停止，原临时unit已reset-failed后重新启动；最终采样unit=active，18081/health尚未ready（connection refused），不能称模型健康恢复。a1本轮无操作。此前清理失败及恢复过程均保留。

本轮仍无新增可上线改动或新发布PR。可复现的WY Prefill收益、失败候选、独立CPU参考和上游问题草稿均已归档。下一优先级为Dense完整调用上下文/运行时兼容性定位，以及独立的Decode瓶颈优化；保持35B-A3B为主线，禁止用新Dense启动参数冒充生产A/B。

## 2026-11-04 增量执行决定

Dense：原模型内权重重建不解决rank1故障；旧栈真实捕获和合成随机权重均能独立进程触发同kernel/hash507015。False/True不是因果开关，保留失败状态与未到算子的初始化超时分类。对外复现包只纳入生成数据、代码和元数据，不含模型/业务数据。

post4镜像的CANN安装与OPP均自报9.1.0，torch-npu2.10.0.post4，vLLM仍0fc695fc、Ascend改1cdb8c4d；启动需显式入口和关闭旧C8 bootstrap。配置0.90因KV7.66<8.09GiB拒绝，0.94保留完整上下文并初始化KV8.50GiB/274795tokens。strict7/8失败原样保留，不执行正式性能，业务/GSM/长请求仍待验证。

35B：两镜像chunk-GDN源码同为ab2b739b...，增加48次独立CPU参考+24/24完整链边界对照；这是整套运行栈筛选，不能归因单独torch-npu或作为性能A/B。历史W8 eager MTP1/2已有无收益/语义风险结果，不重复包装为新优化。


## 上游重复工作审计与 P0 实施（2026-11-04）

用户补充的PR17640/18057/11941/13122/17412/18410均在本次官方GitHub API快照中Open；MegaGDN8872单列。当前WY候选原本就直接复用PR17640固定head，保留作者归属，不另提重复WY功能PR。优先为现有PR贡献可复现测试/修复；只有独立缺陷和增量补丁才考虑新PR。状态、review及集成差异见results/upstream-gdn-pr-audit-20261104/README.md；#17640本次mergeable=true，不能沿用“当前有冲突”的旧说法。

当前build-validate-gdn-pair.py在a2构建固定PR17640成对H/O，保留binding内的NZ初始状态约定、独立OPP与namespace；按Torch/原生WY、device0/1各新进程做完整链重复，并在通过后做独立CPU参考多形状/状态门禁。首次准备失败日志保留，v2不覆盖旧结果。未加载模型、未改生产OPP。另已准备完整调用上下文的原路径/同步/克隆/保留接口诊断，是否运行由成对门禁结果决定；准备代码不计作实测。

## 2026-11-04：完整H/O候选本轮最终判定

配套H/O+原生WY算子验证通过（四组192次输出检查、两次严格10例），模型质量筛选已结束。前后短请求各8/8，事实16/16，业务自动28/28；人工会议4仍有负责人错配及完成状态编造，因此最终not_accepted_business_semantics，不进入性能或部署。GSM64=63/64，与冻结生产基线逐样本答案/正确性一致，唯一index205截断；文本19/64完全一致，不证明逐token等价。A′此前也有业务错误，不能据此将错误归因为候选独有回归。

结果：results/production-gdn-pair-quality-20261104/adjudication.json。上游路线和补充研究取舍：docs/research/gdn-upstream-pr-evaluation.md。此前段落中的“运行中”是过程状态，以本段最终判定为准。

## 2026-11-04：CANN9.2.1完整默认栈隔离验证

用户同意后在a2拉取官方v0.30.0-310p（IDd103c59b31c6…），核实Toolkit/OPP/NNAL9.2.1、torch-npu2.10post7、vLLM/Ascend0.30.0；a1和宿主驱动固件未改。初始GDN T111两卡96/96、Dense无权重Matmul10/10通过，但扩展门禁13/15，两组T512输出偏差。独立进程T512 HV16仅29/48在容差内、HV24为0/48，末state各48/48稳定。最终not_accepted_gdn_correctness；未加载模型、未测性能，不宣称CANN单独修复Dense或GDN。

完整评估docs/research/cann-upgrade-evaluation.md；原始results/cann921-upgrade-20261104；可公开合成复现results/upstream-gdn-cann921-repro-20261104.tar.gz（尚未发布）。本轮至此保留隔离新版环境，下一步优先GDN输出正确性，不扩大WY重复实现。


## 2026-11-04：P0新版配套H/O门禁已通过，继续原Torch WY模型质量

官方9.2.1固定镜像重新编译PR17640成对H/O，Torch WY保持原实现。两卡T512 seed318/HV16+HV24共192/192输出/末状态通过CPU参考和逐位重复性；两卡严格15组各15/15。修正并记录bootstrap后的同名API搜索顺序，验证进程实际仅映射paired API库。算法未改，无旧9.1扩展混入。

接下来先验证35B-A3B W8A8 TP2的9.2.1+配套H/O+原Torch WY：短题/ABA→事实→完整业务语义→前后复核。只有质量过关才重编译/验收原生WY、做性能。原服务恢复HTTP200后启动候选；启动脚本NNAL nounset错误的首轮保留，发生在权重加载前，不计模型失败。Dense与Decode优先级保持后续，不从小算子通过推断上线。

结果：results/upstream-gdn-pair921-v1-20261104/；反馈草稿docs/upstream/gdn-cann921-paired-validation-20261104.md，未发布。


## 2026-11-04：9.2.1配套H/O＋原Torch WY的35B质量筛选完成

该组合通过算子门禁后实际启动Qwen3.6-35B-A3B W8A8 TP2，保持maxlen262144、batch4096、seqs4、Graph[1,2]、SSM FP16、MTP0、prefix on，利用率0.90。各rank权重18.8545GB；完整上下文配置通过准入，不代表已发送262K请求。独立检查两个worker均映射paired API库，并记录原默认库仍服务其他算子。

前后短题各8/8、ABA输出一致无canary泄漏，事实16/16，业务自动28/28（20会议、3tool、5schema），但人工语义仍不通过。会议4本轮周明/刘洋/何丽映射正确，却把未标明完成状态的当天回复写成“已于当天完成回复”；会议5同样编造已完成停用，并把行动项截止时间从“当天”写成“本周内（已完成）”。原生产A也出现前两项完成状态编造，不能当作本候选独有的数值回归；本周内是本轮额外差异，仍需配对复核。20会议均已人工阅读。

28项原始请求与原生产A逐项严格相同；25条非tool文本7条相同，文字变化不能直接等同质量或数值退化。最终not_accepted_business_semantics，无新GSM、性能和生产采用。首次启动NNAL nounset错误发生在权重加载前，修正后v2实际完成本轮；原失败保留，不计入模型质量。

P0结论需分开：已测GDN数值异常可以由现有配套H/O在Torch WY不变时消除；自然语言完成状态问题也存在于原模型基线，尚未证明是GDN/CANN引起。下一步补齐负责人/截止时间/执行状态的同输入结构化配对回归及长上下文检查，再决定WY验收和正式性能；不靠继续换CANN或放宽门槛消除业务错误。

结果：results/cann921-upgrade-20261104/a3b-quality-pair921-v2/adjudication.json。新增无权重配套对照复现包results/upstream-gdn-cann921-pair-repro-20261104.tar.gz（10925B，脚本与实测冻结SHA一致、归档逐文件校验通过），可为现有PR17640补充测试；仍未公开发布。


## 2026-11-04：字段与长上下文配对回归完成

同a2串行测试原生产A与9.2.1+PR17640 paired H/O候选B，两者均35B-A3B W8A8 TP2、原Torch WY。13种独立预期用例×C1/C3，每栈26/26通过，共52请求；其中4095/4096/4097/16384/32768合成输入每栈10/10。两个栈13种实际token ID哈希一致，所有请求usage计数匹配；四次前后短题检查均8/8。仅行动项顺序可忽略，其他字段严格一致。

原会议4/5/18另做9个同payload请求/栈并人工审阅。会议4负责人映射A为2/3、B为3/3，样本量不足以证明修复；双方均3/3将当天回复补成已完成。会议5双方均3/3补成已执行停用；B的C1再次把当天改为本周内（已执行），A三次均遗漏该行动项行，不能因A未写错期限就说A完整正确。B的同一行还将所有部门细化为各部门负责人。会议18双方全文保留角色/时间关系，但行动项结构用未明确，按结构损失记录。完成状态采用事先冻结的保守标准，原短句本身存在时态歧义，不将该人工判定作为数值误差证据。

最终structured_gate=passed_for_tested_inputs，整体仍not_accepted_business_semantics。保留截止时间差异作为下一步受控复核样本；不宣称CANN/H/O单独导致差异。未进行新吞吐/GSM/logits验收，不把合成32K提取当作真实长会议或262K质量。此前38–54%收益不得迁移到新控制组。原生WY尚未在9.2.1验收，必须重新编译并通过算子及相同模型门禁；正式性能仍等待质量过关。Decode/Dense继续位于正确性工作之后，不扩展重复WY PR。

结果：results/action-state-paired-v1-20261104/，以adjudication.json为最终判定。a2原服务active且独立18081/health=200，无运行rootless实验容器，驱动仍26.1.1；未操作a1、宿主安装或公开发布。冻结工具/fixture SHA和新增三脚本编译校验通过。


## 2026-11-04：27B新版16K实测及排序/标点诊断

27B W8A8 TP2在固定9.2.1+PR17640配套H/O+原Torch WY下实际健康启动，maxlen16384，Graph[1,2]、MTP0、prefix on。两rank日志各19.2416GB权重，实际映射paired API；本轮未复现旧507015，但不宣称所有形状/长上下文均修复。

首轮短题7/8，排序期望2,7,9而输出279，和旧post4同题输出相同。ABA一致、无canary泄漏。按预定门禁停止事实/业务/性能；20条语义准则提前冻结但未执行，不能报告27B会议质量或与35B的业务排名。

首轮原服务独立HTTP200后再启动窄范围第二轮，保持同配置。6类×3重复：原排序3/3输出279，raw token IDs=[17,22,24,248046]，未生成逗号11；数字逗号复制、字母逗号复制、分号、数组复制和JSON schema排序五类15/15通过。原排序首个样本生成2后，下个token7的logprob约-0.257、逗号约-1.491，greedy选7。CPU tokenizer亦可完整往返2,7,9。该例定位到提示词相关的生成选择，不是已观察到的API解析删逗号；仍不能区分基础模型与量化影响。JSON schema改变约束，其通过不能覆盖原题失败。

两次均not_accepted_short_format_gate，未执行正式性能或业务比较。证据位于results/cann921-upgrade-20261104/dense-quality-pair921-16k-v1及dense-quality-format-pair921-16k-v1。新增短上下文容量选项、双worker库验证、独立诊断模式与原始token/logprobs留存。下一步保留失败，围绕同模型参考/精度与原始提示词对照定位，不为标点错误盲改GDN算子，也不扩大八种profile的可用声明。


### 窄范围35B对照与下一步

原服务恢复HTTP200后，a2原35B W8A8部署完成同样6类×3次请求，18/18符合预期；27B为15/18，差异仅原排序的3次重复（35B3/3、27B0/3）。请求除model字段外严格一致；完整18组配对见dense-quality-format-pair921-16k-v1/a3b-narrow-comparison.json。运行栈、模型/权重、上下文配置不同，不能将其称为单独模型能力或量化因果对照，亦不是会议业务评测。

原短题门禁保持7/8未通过；本轮没有绕过门禁跑业务/性能。下一阶段将格式遵循反例与算子数值验收分开，允许另立明确“不予验收”的业务诊断轮收集27B原会议/结构化结果，不能用诊断完成替代质量门禁通过。已有token证据不支持为此反例继续盲换CANN或改GDN；若要归因基础模型与量化，仍需同模型参考。正式性能和可部署profile均等待验收。

收尾：原服务active、18081/health=HTTP200，无运行rootless实验容器；驱动仍26.1.1，a1与宿主安装未变。两轮冻结工具SHA校验、第二轮源码与本地一致、两脚本编译和diff/新增源空白检查通过。上游未发布。
