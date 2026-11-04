# PR17640补充验证草稿：9.2.1保留Torch WY时配套H/O消除已测偏差

状态：本地草稿，尚未发送Issue/评论/PR。配套H/O归属PR17640原作者。

实际Atlas300IDUO / 310P3，官方vLLM-Ascend0.30.0-310p固定镜像，CANN9.2.1，torch-npu2.10post7。无权重T512/HK8/HV16或24/D128，seed318，非零FP16初始状态；独立CPU逐token递推参考。默认实现device0的输出通过率HV16为29/48、HV24为0/48；两组末state均48/48稳定且符合参考。

仅切换到PR17640 head18163f6b的配套H/O和原NZ binding，保留Torch WY、相同输入和门限，device0与device1各两组48次全部通过。输出maxabs6.54e−6/6.38e−6，相对RMSE约4.24e−4。两卡15形状严格门禁各15/15且每组3次逐位重复。候选不是单独O/旧H混合方案。

注意NPUBridge在库加载时捕获ASCEND_CUSTOM_OPP_PATH，vLLM bootstrap会将默认vendor前置；测试包装器在bootstrap后重新前置配套vendor，再dlopen，检查/proc/self/maps中的实际API库。需要一同提供该信息，避免同名API误绑定。

可贡献内容：为现有PR补充T512/HV24/非零状态/重复性回归与真实310P3验证；结果暂不足以证明H或O中某一处是独立根因。本草稿仅覆盖合成算子对照；模型判定见[结果](../ascend310p-results.md)。构建、默认/配套运行命令及原始证据见[配套对照复现说明](../../results/upstream-gdn-cann921-pair-repro-20261104/README.md)，其中repeat工具即使误差超限也可能exit0，须核对summary全部计数；strict另验边界/状态。

预期：固定门限atol2e-4、rtol0.04、relativeRMSE<=0.02，输出和末state符合独立参考且逐位重复。API搜索顺序、映射库及扩展SHA随结果一同提供。更多实测误差与版本依据见[CANN评估](../research/cann-upgrade-evaluation.md)。补充复现包10925B、逐文件SHA通过，尚未发布。
