# 代码入口与维护边界

## 构建、模型与运行

| 目录/入口 | 职责 | 维护原则 |
| --- | --- | --- |
| scripts/build-native.sh | 固定社区基线、覆盖必要算子快照、构建独立OPP/扩展 | 不整体合并fork；固定SDK/镜像 |
| native-overlay/csrc/gmm/ | W4 pack、MatMul、SwiGLU及相关源码 | 数值/shape门禁后调整tile；候选不要自动默认启用 |
| tools/qwen36_int4_custom.cpp | 独立Torch算子namespace | 保持官方_C_ascend不冲突 |
| runtime/.../qwen36_w4_patch.py | routed MoE构造/加载、TP所有权、MTP排除 | 按量化metadata opt-in；后续替换临时类monkeypatch |
| runtime/.../qwen4_exp/ | 社区W4执行子集与ModelSlim ABI适配 | 保留来源；裁剪前先核实依赖 |
| bootstrap/ | 隔离进程加载适配层与fake contracts | 不修改宿主安装 |
| tools/convert_modelslim_w4a8.py等 | BF16→专家sidecar→自包含checkpoint | 不覆盖源权重；保留哈希/元数据 |
| scripts/run-service.sh | W4/FP16 KV基线启动 | 与C8实验启动器分开 |

## C8与测试

| 入口 | 职责 |
| --- | --- |
| experimental/c8_vllm_adapter.py | cache spec、预算、allocate及full-attention读取接入 |
| experimental/c8_cache_310.py、ascend_c8_kernel/ | INT8持久cache、FP16 scale、量化写入/读取算子 |
| experimental/direct_graph_310.py | 固定镜像Graph兼容诊断；尚非正式上游修复 |
| experimental/run-c8-112k-product.sh | 112K显式mtp1/mtp2候选档 |
| experimental/check-n128-product.sh | 复用、阶段计时和实际112K检索初步门禁 |
| experimental/run-w4-prefill-check.sh | 独占NPU的真实专家矩阵独立检查 |
| tools/bench_server_phases.py、compare_server_phases.py | 独占端点阶段计时/协议对比；记录输入哈希 |
| tools/eval_gsm8k_http.py、compare_gsm8k_runs.py | 固定子集逐题质量与截断比较 |
| results/ | 后续原始数据、关键日志与实验README |

默认服务入口收束为 `scripts/run-candidate.sh → experimental/run-c8-service.sh`，取消GDN/product wrapper的串联；旧wrapper仅用于显式历史基线。已删除四个旧批量sweep脚本，可在提交 `3ce8cb9` 中追溯。独立算子验证保留在experimental/，不加载到普通服务。下一步减小主线改动范围的具体清单见[会议评审报告](huawei-adaptation-review-20261013.md)。

## 源码与制品追踪

首发历史清单为`docs/runtime-source-sha256.json`；后续2026-10-13核心运行/构建快照为`docs/source-snapshot-20261013.json`。清单只校验源码，不证明二进制由该源码构建或通过NPU测试；二进制哈希与实际日志仍需逐实验核对。

```bash
python3 tools/check_source_snapshot.py docs/source-snapshot-20261013.json
```

新清单须随被追踪文件的修改同步维护，并通过上述命令；不要重写首发历史清单。Git记录说明每轮实际修改与测量依据。代码来源及许可证见[provenance](provenance.md)和NOTICE。


## 可选MindIEQ Prefill路径

`experimental/mindieq_prefill_adapter.py`仅替换TP1满足阈值的grouped routed experts，复用现有signed W4权重；Decode仍走native。`experimental/mindieq_grouped_kernel/`提供一次launch设备端专家调度、外部MindIEQ矩阵类及自有A8量化。`tools/prepare_mindieq_grouped_header.py`对固定外部构建树生成隔离头，不把外部代码复制进公开仓库。来源与CANN许可要求见NOTICE。

两个显式开关：QWEN36_MINDIEQ_ENABLED开启实验后端，QWEN36_MINDIEQ_CUSTOM_QUANT选择FP32算术A8准备；底层默认off；统一候选入口的long-prefill自动开启已测E5组合。新版量化需要含quantize接口的新库。仅已发布SSZ per-channel checkpoint、TP1、FP16输入和K512/2048经过测试；不支持任意独立group-wise权重、TP2 C8、多模态或并发批处理。`scripts/run-candidate.sh --check`拒绝无效开关组合/缺失库，并输出制品hash，不初始化NPU。

旧dynamic_quant版本B/A/B已证明Prefill收益但出现额外截断和8K接受率下降，保留实验选项；新A8版本的模型结果见results/mindieq-a8-service-20261014，不根据单算子测试宣称默认采用。

输入规模分流通过`QWEN36_MINDIEQ_MIN_TOKENS`配置：默认17保留历史行为，必须≥17以保留native Decode；long-prefill固定默认128。阈值比较的是进入MoE的实际block_input行数，可能含调度padding，不等于用户原始prompt长度，也不按上下文总长度选择。启动器记录阈值并传递容器。

`QWEN36_MINDIEQ_SHARED_GATE=1`可选在Gate/Up路由复制前量化，再用`mindieq_activation.py`和AscendC `gather`复制A8及scale/padding。需custom_quant与新版库，底层默认off，long-prefill自动开启；Down与Decode不改。单层微测试与E5整模型小样本门禁已通过，详见results/mindieq-shared-gate-20261014。

## 后端交付入口

见 [a2交付说明](backend-handoff.md)。`deploy/a2.env` 固定a2已测制品路径；`deploy/qwen36-a2.service` 提供用户systemd托管。后端只需消费OpenAI兼容HTTP接口，不需参与CANN构建或选择实验开关。历史源码清单必须在对应历史提交校验，清理后HEAD不再包含其全部文件。

## 与vLLM-Ascend主线的收束方式

部署参数先完成同协议对照，再删除无收益的host覆写。保留一套候选启动入口及明确的RC硬件适配；历史sweep结果由Git和results追溯，不再为每轮参数添加新启动器。

上游化按职责拆分，避免整体移植社区fork：

1. 独立W4内核与Torch schema、数值/真实shape/Graph微测试，保留社区作者和许可证。
2. 量化metadata、权重加载与routed expert实现，通过明确量化后端注册；让BF16 MTP继续走原生模型路径，替代运行时类替换。
3. GDN复用固定官方PR并在正式版本包含后移除外部source覆写；不重复维护官方实现。
4. MindIEQ仅作Prefill后端分流，固定输入/量化ABI；不把MTP、C8或RC行为耦合进矩阵内核。
5. C8保持可选；现有读取时转换FP16 workspace及单请求限制需明确。若不能同时证明容量价值和速度收益，不加入48G/112K默认依赖。
6. Graph兼容层与RC blocking multimodal H2D单独定位和提交，正式修复发布并通过复验后删除对应monkeypatch。

实验数据不构成主线兼容保证。当前固定镜像运行代码不能直接宣称可在今日main运行；迁移应另开固定main SHA的隔离分支，逐项验证后移除兼容层。

## 固定运行镜像入口

`tools/prepare_runtime_image.py → deploy/image/Containerfile → deploy/image/start.sh`通过指定提交与基础digest打包运行文件和已测算子，不包含实验脚本或权重。复用`configs/tp1-profile.sh`；MTP、调度和metadata顺序是显式参数，不按主机名选择。镜像内manifest、哈希验证、构建/启动/回滚与已验收范围见[runtime-image](runtime-image.md)。原bind-mounted入口保留用于回滚及源码实验；镜像候选门禁通过前不替换原管理单元。
