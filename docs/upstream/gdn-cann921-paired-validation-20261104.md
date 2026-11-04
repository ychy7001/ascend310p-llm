# PR17640补充验证草稿：9.2.1保留Torch WY时配套H/O消除已测偏差

状态：本地草稿，尚未发送Issue/评论/PR。不将复用已有算子包装成新原创性能PR。

实际Atlas300IDUO / 310P3，官方vLLM-Ascend0.30.0-310p固定镜像，CANN9.2.1，torch-npu2.10post7。无权重T512/HK8/HV16或24/D128，seed318，非零FP16初始状态；独立CPU逐token递推参考。默认实现device0的输出通过率HV16为29/48、HV24为0/48；两组末state均48/48稳定且符合参考。

仅切换到PR17640 head18163f6b的配套H/O和原NZ binding，保留Torch WY、相同输入和门限，device0与device1各两组48次全部通过。输出maxabs6.54e−6/6.38e−6，相对RMSE约4.24e−4。两卡15形状严格门禁各15/15且每组3次逐位重复。候选不是单独O/旧H混合方案。

注意NPUBridge在库加载时捕获ASCEND_CUSTOM_OPP_PATH，vLLM bootstrap会将默认vendor前置；测试包装器在bootstrap后重新前置配套vendor，再dlopen，检查/proc/self/maps中的实际API库。需要一同提供该信息，避免同名API误绑定。

可贡献内容：为现有PR补充T512/HV24/非零状态/重复性回归与真实310P3验证；结果暂不足以证明H或O中某一处是独立根因。整模型语义、长上下文、token/logits和性能尚未验收。原始证据results/upstream-gdn-pair921-v1-20261104；默认栈最小无权重复现包upstream-gdn-cann921-repro-20261104.tar.gz。


补充复现包已生成：results/upstream-gdn-cann921-pair-repro-20261104.tar.gz，10925B，含已测原字节脚本、独立CPU参考、固定绑定提取器、对照数值和重新构建命令，逐文件SHA校验通过。无权重、业务请求或远端日志路径；未发布。后续35B TP2模型前后短题8/8、事实16/16、自动业务28/28，但人工语义不通过；原生产也有完成状态编造，不把算子门禁通过说成全模型质量修复。
