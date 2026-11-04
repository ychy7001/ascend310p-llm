# 9.2.1 原生 WY 重编译与两设备门禁

结论：两个逻辑 NPU 各15/15，合计30/30合成组合算子用例通过。仅切换 Torch/native WY，H/O保持同一配套版本；尚未验收真实模型输入、模型logits或端到端性能，不作为生产采用依据。最终判定见 adjudication.json。

固定官方镜像 d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a，CANN9.2.1，PR17640 18163f6b1d208ab3b4fb06e9752225611c37f40e。绑定代码只适配include路径；算子算法未修改，未使用9.1旧二进制。build/保留源核对、命令、52项制品SHA和构建日志；实际二进制仍在a2隔离目录，未复制到仓库。

覆盖T=1/63/64/65/111/127/128/129/512/1025、HV16/24/32、ragged边界、非零初始状态及不同衰减。两路径分别比较CPU FP32递推，再直接互比；容差保持atol=0.0002、rtol=0.04、relativeRMSE<=0.02，每路径每用例3次bitwise重复，每设备60次原生调用，无fallback。实际进程映射核对配套H/O与WY两套独立API。

失败尝试native-wy921-gate-v1因PYTHONPATH覆盖CANN环境而缺少tbe，0数值用例；恢复原服务健康后才开始v2。修正为在CANN环境上追加PYTHONPATH。失败记录保留，不计入数值通过率。

下一阶段：35B W8A8 TP2同权重、同token前缀、同H/O，只切WY。先采真实GDN输入再回放，并在模型层固定token序列比较top1及目标token logprob；如只采top-k，不报告完整分布KL。GPU现有GPTQ W4不作为W8A8逐token金标准。前缀缓存按生产保留，数值对照需记录冷/热与实际命中，避免把缓存跳过计算当验证。

独立恢复确认：a2原服务active、远端18081/health=200、无运行rootless实验容器；本机原GPU_SERVICE/18020健康200、3080两PID保留，见independent-closeout.json。a1、宿主CANN/驱动均未操作。
