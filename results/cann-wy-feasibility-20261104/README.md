# CANN compute-WY 可行性核查（2026-11-04）

**结论：CPU构建路线可行；需要完成WY独立薄绑定和设备算子门禁后，再进入G1。** 现成镜像缺少目标算子。两个算子的自定义包及独立vendor包已实际构建成功；本轮没有设备调用，不给数值或性能结论。完整判定及证据SHA见 [adjudication.json](adjudication.json)。

1. **现有镜像没有目标ACLNN和Python绑定。** a2仅只读检查镜像元数据/层文件，本机再用同一image ID `d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a`的实际rootfs复核。扫描78,033份文本及11份opapi库，`nm -D --defined-only`未找到`aclnnChunkGatedDeltaRuleComputeWy`、`aclnnFusedGdnGating`及各自`GetWorkspaceSize`；相应头文件和310P op信息也不存在。检查范围含`libopapi.so`、`libopapi_transformer.so`及镜像自带`libcust_opapi.so`。通用`cann_ops_transformer`包存在，但没有这两个experimental绑定；框架中的同名Torch/Triton gating文件不能作为目标绑定证据。

2. **按固定提交实际CPU构建成功，无构建失败阶段。** 使用同一9.2.1镜像、官方ops-transformer `0340eae051ddf5ab956df0f7e0396dbd6314cfc5`，未修改算子源码。执行`bash build.sh --pkg --experimental --soc=ascend310p --vendor_name=custom --ops=fused_gdn_gating,chunk_gated_delta_rule_compute_wy -j16`，再按#18057构建experimental wheel，两步退出码均0；另实际构建`wy921_cann` vendor包及选定这两个算子的独立包名wheel，退出码也均0。提取包时使用`--noexec`，四个ACLNN导出、两个头文件、两个310P opinfo条目和两份310P kernel ELF均核验通过；两个wheel均含对应Python接口和C++桥接源码。两个本机容器均非privileged、未挂NPU设备或驱动，构建完成后已删除。原始首次构建约3分钟，独立vendor构建约1分钟；全轮含镜像下载约40分钟。wheel采用lazy JIT，桥接C++未单独JIT编译或加载。

3. **WY可以单独切换，gating保持原样；完整PR需要裁剪。** [#18057](https://github.com/vllm-project/vllm-ascend/pull/18057)核查head为`2620e77d7bdb3c54c6ea98b24b570269f6eed6b7`，状态open。完整PR12文件、+641/-4；相对镜像`c09d3e20`，WY文件仅+57/-1，另加23行加载器；gating包装另加25行。head的`gdn_310.py`相对镜像还带状态写回差异，不能整文件带入WY单变量候选。可沿用配套H/O worker，只替换`_compute_kernel_inputs_from_torch_wy`，保留`fused_gdn_gating_pytorch`及原dtype路径。独立`wy921_cann_transformer` vendor和`cann_ops_transformer_wy921_cann` Python包名已实际构建；**Torch dispatcher namespace仍固定为`cann_ops_transformer`，builder的部分utils导入也仍指向基础包，改vendor不能完成严格namespace隔离**，需薄绑定适配。#18057默认存在加载/shape失败时的Torch回退，后续门禁应显式固定绑定、断言库来源及shape，禁止静默Torch回退。WY契约与此前TP2的Hk8/Hv16/KV128形状静态相容，设备行为未验证。

4. **两MR均已合入；具体发行版收录未核实。** 官方[!11242 API](https://api.gitcode.com/api/v5/repos/cann/ops-transformer/pulls/11242)返回`merged`，合入时间2026-09-17 10:47:29+08:00，对应官方git提交`53172410248d3f4a934ed7a3ccea59812440264c`；[!12078 API](https://api.gitcode.com/api/v5/repos/cann/ops-transformer/pulls/12078)返回`merged`，合入时间同日17:20:01+08:00，对应提交`9fef367c5dc957418bf66db9089e0761b132bfc3`。官方仓库祖先关系确认固定`0340eae`包含两者及统一绑定重构!13630。检查18个公开tag的完整文件树（含v9.2.0），均未找到两个算子路径；[releases API](https://api.gitcode.com/api/v5/repos/cann/ops-transformer/releases?per_page=100)返回空列表。只确认`9.3.0-beta.1`分支包含源码，不能作为已发行证明，未按版本号推测二进制收录。

| 实际构建制品 | 大小（字节） | 核查 |
|---|---:|---|
| `cann-ops-transformer-custom_linux-x86_64.run` | 3540307 | 两算子的ACLNN、头文件、310P opinfo/kernel ELF齐全 |
| `cann-ops-transformer-wy921_cann_linux-x86_64.run` | 3539339 | 独立vendor，同样核验通过 |
| `cann_ops_transformer-1.0.0-py3-none-any.whl` | 251983 | 两目标接口及JIT桥接源码齐全 |
| `cann_ops_transformer_wy921_cann-1.0.0-py3-none-any.whl` | 134975 | 独立Python包名；Torch namespace仍共享 |

下一步仅需准备：WY薄绑定使用独立namespace和明确的vendor库来源，冻结原gating路径，沿用已通过H/O；待另一个thread释放NPU后先做设备算子门禁及桥接加载检查，再进入A/B/A′模型G1。FusedGdnGating继续作为独立变量。本轮未实现模型接入，不能直接把CPU构建成功当作G1就绪或数值通过。

制品完整SHA、导出符号、内核对象SHA、镜像检查库SHA及官方API证据SHA均在adjudication中。CPU构建控制器和静态核验脚本在`experimental/runs-20261104/`；原始日志、源码及二进制留在忽略目录，不提交。仅在容器内补齐Python构建依赖，无宿主安装、CANN/驱动变更、a1访问、a2容器启动或服务停启；未推送。此前线程的结果与工作文件未纳入本提交。

本地提交前缀`wy-1: 8`，最终号由交付消息提供。真实限制为CPU/静态验证范围、私有Torch namespace薄绑定未实现、发行版收录未核实；本轮未运行NPU、G1/G2或性能。
