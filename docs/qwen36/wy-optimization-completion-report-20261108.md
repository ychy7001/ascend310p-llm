# 310P native WY 生产优化执行与交付总结

日期：2026-11-08。仓库：`ascend310p-llm`，分支：`experiment/hyperqwen-310p-20261103`。

本报告按已归档实验、构建清单和提交历史整理；生成报告时不新增设备实验，也不调整交付配置。

## 1. 交付结论

本轮已完成 **TP2 W8A8 + native WY + MTP2 + 主模型 Graph、262144 上下文、调度并发上限4** 的工程优化、组合验收、镜像封装和 registry 发布。最终镜像默认入口也完成独立验收，能够脱离实验时挂载的宿主源码运行。

TP1 W4A8 沿用既有稳定 WY/MTP2 profile 和镜像，不为统一形式重复构建。两种变种均有资产登记，但本轮主要新增成果是 W8 的完整集成。

工程交付已经收尾；a1 生产切换与目标机验收尚未执行。公开 OCR 对照支持既有 WY 候选的小规模质量判断；最终满上下文 MTP2 镜像未完成真实中文票据的正式质量非劣效验证。

## 2. 完整执行路径

### 2.1 明确数值差异及验收边界

早期工作从 native WY 与 NPU Torch 的数值差异入手，补充 CPU FP32/FP64 参考、固定真实输入捕获及分阶段分析，分别检查 Gram 操作数舍入、累积 g 的舍入、scan 与三角求解。

这段工作的关键结论是：算子局部误差、模型 token 分叉、业务答案质量必须分别评价。CPU FP32 对照通过不自动证明所有模型路径等价；与 NPU Torch token 不同也不自动证明业务答案错误。

历史上尝试过 hybrid、guard、补偿 scan、refined solve 等路线。局部严格 G0 通过的候选仍曾在模型筛查失败；这些旧失败结论没有被后续性能或 PDF 检查覆盖。确定性的 native/Torch 差异与近似平局或非确定性现象也保留各自解释，未合并归因。

GPU 独立参考工作也有实际资产：本机3090完成 Qwen3.6-35B-A3B GPTQ Int4 独立模型筛查，记录固定制品、索引检查和运行配置。该路径具有量化、运行栈和计算精度差异，属于模型行为参照；不能冒充 BF16 真值或同输入 GPU Torch WY 的完整数值裁决。其会议语义问题亦不据此单独归因310P。详见 `results/local-gpu-business-20261104/README.md`。

### 2.2 收敛优化范围，保留有价值的 native WY

随后开展 Torch UT 脚本化、clone 移除、Graph replay 和 ReduceSum 路径诊断。部分方案能保持固定输入数值一致，但交替计时或长形状验证没有给出值得交付的收益，未继续扩展实现。

执行重点收敛为 PR17640 native WY 的一般 Prefill 优化：锁定候选源码、binding、OPP 和镜像身份，避免为追逐少数位置的 Torch 舍入结果继续叠加复杂补偿分支。

等效 PR17640 候选相对 Torch 臂，历史冷 Prefill 4K/8K/16K/32K 吞吐提升分别为 **30.54% / 28.96% / 28.64% / 27.87%**。这些是既有对应条件下的比较结果；最终 MTP2 满上下文镜像采用绝对指标验收，未重跑同条件 Torch/native 交替 A/B，不能把历史百分比直接移作最终组合的收益。

### 2.3 建立独立 WY 镜像并补充公开 OCR 证据

首先封装 TP2 W8A8、WY、无 MTP 的独立候选，完成固定镜像入口、冷文本、并发和 PDF 重复检查，锁定制品并发布 registry 版本。

公开标注数据用于补充质量判断，冻结来源、样本及评估字段：

| 数据集/规模 | Torch | WY | 解释 |
| --- | --- | --- | --- |
| FUNSD，20图，183个纳入评估的非空关联字段 | JSON 18/20；字段EM 62/183 | JSON 18/20；字段EM 64/183 | 表单 smoke，未观察到结构性退化 |
| SROIE，100图，400字段 | JSON 100/100；EM 347/400 | JSON 100/100；EM 345/400 | 字段EM少2个，即低0.5个百分点 |
| SROIE，日期与总额200字段 | 194/200 | 193/200 | 数字类差异需单独保留 |
| CORD v2，20图，63非空字段 | JSON 20/20；EM 56/63 | JSON 20/20；EM 55/63 | 少1字段；20张原图已人工核对标签 |

这些结果来自相应历史候选及其成对实验，不是最终262K/MTP2镜像的完整质量门禁。不能用“只稍差一点”概括所有业务，也不据此声称统计意义上的非劣效。中文发票金额、税号等仍需要对应业务样本验证。

### 2.4 复用 W4 稳定资产，将新增工作集中于 W8

核查确认 TP1 W4A8 已有 PR17640 GDN/native WY、MTP2、Graph 和 long-prefill 集成，不需要从头补做整套实验。

本轮补充核查确认实际 native eligibility、Graph 捕获和 MTP2。W4 的4K/8K/16K/32K冷文本 Prefill 中位数为 **635.552 / 622.833 / 595.862 / 544.377 tok/s**，4K/128 Decode 为 **35.637 tok/s**；PDF续跑8/8通过功能检查。随后按收敛范围停止多余 W4 对照，沿用稳定 profile。

W4 与 W8 在 TP、量化实现、CANN/torch_npu及兼容适配器上存在差别，不能只比较 Graph 模式名，也不能把这些吞吐数字作为量化优劣的受控比较。

### 2.5 修复 W8 Graph/MTP 的 SplitFuse 兼容问题

W8 初次增加 MTP 后，模型加载成功，但 Graph 捕获中执行 `query_start_loc.to("cpu")` 报107030/107027。关闭 Graph 的最小 smoke 成功，说明应先定位集成兼容性。

依赖对照表明：W4 较新栈具有压缩 SplitFuse v2 接口；W8 当前 post2/beta.1 栈走旧路径，在图捕获内产生 CPU 拷贝。修复将 mask 准备移到 capture/replay 外，写入地址固定的 NZ 缓冲区，图内读取；每次 replay 刷新内容。普通 eager/Prefill 仍调用原路径。

图内生成 mask 的尝试另遇整数 Sub 捕获限制，未作为最终方案。图外固定缓冲区小测20/20通过，随后整模型、镜像默认入口及早期 seq1 MTP2 候选通过验收并独立发布。此阶段证明组合可运行，尚不等于最终 seq4/262K 配置验收。

### 2.6 满上下文、多请求扩展及三个独立问题的定位

扩展至262144上下文、seq4时，将以下问题分开处理：

1. **recurrent GDN 状态行索引缺陷。** 原 W8 镜像 wrapper 展平状态表，配套 OPP 缺少 stateIndexStride。旧复现确认缺陷存在。按源码 SHA 核对后，复用2026-10-14 opp-review配套构建，移植匹配的 recurrent 函数；eager和Graph各42项输出/状态回归通过。本轮没有重新编译该kernel，也没有只替换Python而忽略host/kernel配套关系。
2. **MTP slot 准备中的 int32 Mul。** 上述状态修复后，多请求507015仍存在。同步诊断定位首次失败在 `attn_update_stack_num_spec_norm` 的 `block_ids * 128`，当时尚未进入当前draft前向。独立进程11项算术小测全部通过，无法宣布通用CANN Mul缺陷。最终仅将乘法前的block IDs提升为int64，保持slot地址公式；移除同步探针后的seq2、seq4 probe通过。
3. **draft Graph 的资源和性能。** 真正双Graph三档[3,6,12]在捕获2/3后耗尽event资源；收敛为[12]后短probe通过。但连续两窗Decode约29.317/28.792 tok/s，主Graph三档/draft eager约46.430 tok/s，4K Prefill基本相同。筛查采用顺序窗口，未包装成交替A/B结论；据此选择主Graph开启、draft eager。

因此最终配置仍包含图优化。draft eager是当前资源和性能证据下的选择，不是“draft永远不能Graph”。WY数学路径和CANN版本未因本轮兼容修复而改变。

### 2.7 固化最终镜像，完成两层验收后发布

最终先完成完整容量与组合窗口，再使用**同image ID、同入口SHA、无宿主源码挂载、关闭生命周期审计**的镜像默认入口验收。主模型两个rank的[3,6,12]均有实际capture/replay证据，实际PID1参数也与交付profile核对。

完整窗口耗时1324.247秒；默认入口窗口约438.875秒，均driver0且独立清场通过。之后发布独立registry tag，核验远端manifest及config SHA。临时单registry凭据已删除；实验结束时a2无残留实验容器或NPU进程，a1未变。

## 3. 最终交付配置与实测

| 配置项 | 最终值 |
| --- | --- |
| 模型 | Qwen3.6-35B-A3B，现有W8A8权重 |
| TP / 设备 | TP2，两张310P3 |
| WY / MTP | native WY / 2个speculative tokens |
| 主模型Graph / draft | FULL_DECODE_ONLY，[3,6,12] / eager |
| max_model_len / max_num_seqs | 262144 / 4 |
| max_num_batched_tokens / memory utilization | 2048 / 0.90 |
| prefix caching / async scheduling | 关闭 / 开启 |
| served model name / a1模板端口 | Qwen3.6-35B-A3B-w8a8 / 18081 |

最终完整窗口每档冷文本3次，prefix hits均0，下表为中位数：

| 输入tokens | Prefill tok/s | TTFT秒 |
| ---: | ---: | ---: |
| 4096 | 2049.608 | 2.002 |
| 8192 | 1937.866 | 4.233 |
| 16384 | 1746.774 | 9.390 |
| 32768 | 1456.722 | 22.512 |

以上是服务Prefill阶段吞吐，不是纯WY kernel吞吐。

- 4K输入/128输出：Decode **45.877 tok/s**，draft acceptance约**85.1%**。
- 四并发端到端测试：12/12完成，含Prefill的聚合输出吞吐中位数35.703 tok/s；TTFT P50/P95为6.755/9.316秒，P95仅为12样本经验分位数。
- 二/三/四请求确定JSON通过；混合长度4/4完成。
- 131072输入+128输出：完整返回，TTFT177.967秒，Prefill736.752 tok/s。
- 262016输入+128输出：总token262144，完整返回，TTFT595.558秒，Prefill440.041 tok/s；超限请求返回400。
- 四页PDF功能检查4/4；其中pdf-01达到512输出token上限，另三页正常stop；长请求后的JSON通过。
- 每10秒npu-smi采样，单卡观察到的最高使用41760MB，不能称为精确allocator峰值。

262144是输入与输出总上限。seq4是调度上限；启动KV pool约635026 tokens，约2.42条满上下文请求容量，不承诺四条262K同时驻留。容量检查不证明长上下文检索或推理质量。PDF未专门关闭多模态缓存，不能称为完全冷视觉性能测试。

## 4. 镜像身份与运行栈

最终不可变部署引用：

```text
registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend@sha256:a11b8aca640120deaaf5429d88b6a82fb4434db25c71cde5d5760d531553dbd5
```

版本tag：

```text
qwen36-tp2-w8a8-wy-mtp2-262k-maingraph-20261108-bb0fa66f
```

- image ID：`bb0fa66f1c715a94abf56afc17a6bd02e3f993c4804e200dfd229e22b0d5cab4`。
- 入口SHA256：`ca8391c85f1a89dae6f71eb61233c06eec0f849dedd0be0453b800cb4f60388a`。
- registry核验时间：2026-11-08 06:56:19 UTC。
- 元数据版本：torch2.10.0+cpu，torch_npu2.10.0.post2，vLLM0.23.0+empty，vllm-ascend0.23.0rc2.dev93+g88eac2713，CANN9.1.0-beta.1。本地overlay另有SHA清单，包版本不代表源码未经修改。
- a2驱动26.1.1，a1驱动26.0.rc1，故a2验收不替代目标机启动与切换验收。

既有W4稳定镜像digest为 `sha256:8e17ec9ae646b1e418cf119bbb347238c4505b8b0037782a1345914824ec8d35`，登记及其profile见 `wy-mtp-integration-20261108.md`。历史W8无MTP及早期seq1 MTP镜像保留独立身份，不用它们代替最终262K制品。

## 5. 仓库改动与可复核资产

### 5.1 生产入口和部署配置

| 路径 | 内容与作用 |
| --- | --- |
| `deploy/w8a8-tp2-wy-mtp/start-wy-mtp.sh` | WY/MTP镜像启动参数支持与交付入口维护 |
| `deploy/w8a8-tp2-wy-mtp/qwen36-wy-mtp-image.env.example` | 固定digest、262K/seq4、Graph/MTP2环境配置示例 |
| `deploy/w8a8-tp2-wy-mtp/qwen36-wy-mtp-production.manifest.json` | 面向a1的完整部署描述，保留模型目录、18081端口和服务名 |
| `deploy/w8a8-tp2-wy-mtp/qwen36-wy-mtp-production.container` | Quadlet模板、两卡设备、入口launcher、60秒冷却与失败重启 |

### 5.2 实现、诊断和自动收尾脚本

- `experimental/wy-mtp-20261108/`：早期组合运行器、direct Graph集成、SplitFuse固定缓冲区适配器及小测。
- `experimental/wy-mtp-production-20261108/pr18730/`：匹配状态表wrapper、制品来源、旧缺陷复现、42项回归、slot int64补丁、独立算术小测、同步定位探针与最终镜像入口/Containerfile。
- `run_profile.py`：a2重任务锁、空闲快照、实际CLI核验、超时、停止归档删除和冷却；packaged窗口从镜像读取入口SHA。
- `run_main_graph_closure.py`：完整窗口通过后，校验主图两rank生命周期，再串行启动同镜像默认入口验收；失败不继续后续步骤。
- `publish_main_graph.py`、`verify_registry.py`：验收通过后发布，校验远端manifest/config及image ID，临时凭据清理。
- `graph_lifecycle_audit.py`：按实际runnable识别主/draft，记录capture/replay；默认交付验收关闭审计，无额外同步探针。

实验脚本是可追踪的定位工具，不代表每个历史入口都应继续用于生产。当前收尾流程以 `run_main_graph_closure.py` 和最终构建清单为准。

### 5.3 文档、结果与来源记录

- `docs/qwen36/wy-release-candidate.md`：独立WY候选、历史收益和公开OCR证据。
- `docs/qwen36/wy-mtp-integration-20261108.md`：W4资产复用、初始W8组合失败和范围收敛。
- `docs/qwen36/wy-mtp-mask-fix-20261108.md`：SplitFuse兼容修复及早期MTP镜像验收。
- `docs/qwen36/wy-mtp-graph-localization-20261108.md`：状态表、Mul及draft Graph三个独立问题和无效标签更正。
- `docs/qwen36/wy-mtp-production-20261108.md`：最终配置、性能、镜像身份、部署回滚边界。
- `results/wy-mtp-production-20261108/build-manifest.json`：运行栈、全部关键制品SHA、profile和验收状态。
- 同目录 `main-graph-release-full-summary.json`、`main-graph-release-summary.json`：完整容量与默认入口结果。
- `graph-localize/main-graph-release-closure.json`：同镜像/同入口、主图两rank、无源码挂载、审计关闭等收尾条件。
- `graph-localize/main-graph-release-registry/`：远端manifest、config身份和digest。
- 公开OCR来源与人工核对文件位于 `docs/qwen36/`，数据manifest保存在工作区 `tasks/data/ocr-public/`，模型权重和原图不混入本次代码交付。
- `.gitignore` 对原始窗口日志、缓存和PDF响应做定向排除；结构化汇总、清场、CLI、生命周期及SHA证据入库。原始窗口仍保留在a2 runtime，没有因整理而删除。

### 5.4 本地提交组织

编号记录覆盖多个阶段，以下按交付逻辑分组；编号不是连续计数，也不代表所有提交都修改kernel：

| 编号/提交 | 成果 |
| --- | --- |
| [1]—[20] | 固定输入、Torch路径、FP64/舍入/补偿求解、Graph与ReduceSum诊断；记录无收益或失败边界 |
| [21]—[34] | 独立WY镜像、公开OCR来源及成对评测、锁定制品和候选交付 |
| [35]—[40] | W4既有资产核查、W8 MTP组合隔离、复用稳定profile及registry登记 |
| [41] `51b263a` | Graph外准备legacy SplitFuse mask |
| [42] `1b04943` | 发布早期通过验收的W8 WY/MTP2兼容镜像 |
| [43] `05f0a00` | Graph mask边界与满上下文MTP验收工具 |
| [44] `4a8ef93` | 配套recurrent状态修复审计与eager/Graph回归 |
| [45] `1a936bd` | slot Mul定位、int64兼容补丁与Graph策略筛查 |
| [46] `b6e6eb7` | 最终262K配置、完整/默认入口验收、镜像发布及部署资产 |

最终实现收尾三笔[44]—[46]相对其父版本共涉及 **73个文件，新增3796行、删除49行**，主要包含实验工具、结构化证据和文档，不应解读为生产kernel增加数千行。该统计不包括本报告，也不代表整个历史项目的净改动。

所有实现已本地提交，**Git尚未推送**；镜像registry发布已完成。本文是随后补充的报告文档，未改变上述实现或镜像身份。

## 6. 验收纪律及已纠正的证据问题

- a1始终未执行写操作或生产切换；不触碰3080。
- 最后兼容修复与验收未修改WY数学路径、冻结协议/规则/worker/R及CANN版本；recurrent OPP/wrapper、proposer和运行时mask适配器变更均单独登记。
- 历史名称含dual-graph但实际draft eager的窗口不算双Graph通过；生命周期首版角色识别错误已明确更正。
- 性能筛查后主动停止的双Graph完整窗口不算模型功能失败，未完成容量也不算通过。
- 旧eager `summary.json` / `full-v5/`不作为最终主Graph镜像的资格证据。
- 每个设备窗口均保留ExitCode、预算/实际耗时、独立清场和制品身份，任务文档追加六行记录。

## 7. 后续交接

本轮无需再扩展算子或参数搜索。下一阶段可直接使用固定digest和交付模板，在单独维护窗口完成a1切换：先保存旧manifest、Quadlet及镜像digest，确认llmctl部署描述同步更新；停旧服务、确认设备空闲、至少冷却60秒；启动后检查健康、模型名、实际CLI、短JSON、代表性冷Prefill/PDF及长请求。失败时使用保存的旧配置和旧digest，按同样冷却流程回滚。

若需正式业务放行，应另外冻结中文票据/字段答案集，关注金额、日期、税号、JSON合法率、截断和严重错误，并使用成对样本及置信区间评价。当前公开数据smoke、容量测试和PDF功能结果应作为已有证据复用，不应扩大为尚未完成的业务质量承诺。

主要原始窗口位置：a2 `/home/ychy/ascend-lab-runtime/hyperqwen-20261103/graph-localize/`。跨轮任务记录：工作区 `tasks/active/ascend310p-llm__310p-optimization.md`；总任务仍含其他独立事项，本报告仅关闭本轮WY组合配置和镜像工程交付。
