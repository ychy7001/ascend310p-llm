# 310P / CANN9.2.1：配套H/O对照复现（未公开发布）

归属：配套H/O和提取出的绑定函数来自GruntovDima的vllm-project/vllm-ascend PR17640，固定commit18163f6b1d208ab3b4fb06e9752225611c37f40e。保留上游许可证。本包贡献为合成输入复现、独立CPU参考和回归验证；不是新原创算子。无模型权重或业务数据。

实际环境：Atlas300IDUO / 两个逻辑310P3；HDK26.1.1；官方quay.io/ascend/vllm-ascend:v0.30.0-310p，imageIDd103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a；CANN9.2.1、torch2.10+cpu、torch-npu2.10post7、vLLM/Ascend0.30.0。请使用匹配工具链重新编译，不加载旧9.1二进制。

默认栈T512/seed318/HK8/D128的device0：HV16为29/48在参考容差内，HV24为0/48，末state两组48/48逐位稳定且符合参考。配套H/O且Torch WY不变：两个device、两组HV各48/48输出/末state在容差内且逐位稳定；两卡严格15形状各15/15，每组再重复3次。阈值atol2e-4/rtol0.04，relativeRMSE<=0.02。详细数值见两个results.json。

将本包放到容器/pair，固定上游checkout放到/src/vllm-ascend；按现有310P构建文档准备子模块依赖。只有执行复现时需要独占暴露所选NPU；编译不需要设备。容器内构建：

```bash
source /usr/local/Ascend/cann/set_env.sh
cd /src/vllm-ascend/csrc
bash build.sh --pkg --ops=chunk_gated_delta_rule_fwd_h_vllm,chunk_fwd_o_vllm --soc=ascend310p --vendor_name=hyperqwen_gdn_pair921 -j4
bash build_out/*.run --quiet --install-path=/pair/opp
python3 /pair/build_pr17640_gdn_pair_extension.py --source-root /src/vllm-ascend --build /pair/extension
```

逐项使用新Python进程，避免将未通过的默认栈结果当作候选参考；两者均对照独立CPU逐token递推。切勿同时运行其他NPU模型。下面命令为device0/HV16，替换device为1、value-heads为24覆盖其余组合：

```bash
cd /pair
python3 repro_chunk_gdn_nondeterminism.py --out default-hv16-d0.jsonl --tokens 512 --seed 318 --value-heads 16 --device 0 --repeats 48
python3 check_cann921_gdn_pair.py --library /pair/extension/pr17640_gdn_pair.so --probe repeat --out pair-hv16-d0.jsonl --tokens 512 --seed 318 --value-heads 16 --device 0 --repeats 48
python3 check_cann921_gdn_pair.py --library /pair/extension/pr17640_gdn_pair.so --probe strict --out pair-strict-d0.jsonl --device 0
```

repeat脚本是诊断工具，即使出现误差也可exit0，必须检查summary的within_tolerance/output_equal_first/state_equal_first均等于repeats。strict额外检查非零初始state、Chunk边界、变长序列、HV24及衰减强度。

注意：通用probe的JSONL头仍保留default/original描述；候选实际范围由wrapper stdout记录的scope、扩展SHA、API SHA、搜索顺序和实际映射库确定。脚本保持已实际运行的原始字节，未为发布改写实验。wrapper在enable_custom_op之后、dlopen之前前置独立vendor，因为NPUBridge在dlopen时捕获路径，而9.2.1默认库也导出同名Vllm API。CPU参考文件导入native_wy仅为复用帮助函数，实际不调用install；Torch WY未改变。

这仅验证给定输入和范围内的配套组合。没有隔离H/O中单一根因，没有证明全模型逐token/logits等价，也没有性能或生产采用结论。
