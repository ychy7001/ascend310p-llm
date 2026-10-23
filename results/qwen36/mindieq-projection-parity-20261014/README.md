# 真实专家逐投影数值定位（2026-10-14）

使用空闲310P3芯片1，真实layer0专家0..7权重、合成FP16激活、固定分组。Gate/Up矩阵K2048/N1024，Down K512/N2048；M1/17/128/512/4096，共10案例。主模型服务在芯片0进行质量测试；本实验无测速结论。

## 结果

两后端的A8量化码及scale对CPU FP32 per-row参考完全一致。MindIEQ全部10案例输出与CPU INT32完整K点积、FP32 scale、FP16输出参考逐位一致。

较大M时原native有少量差异。例如M4096 Gate/Up 964/4194304元素不同，最大2.4414e-4、相对L2约6.87e-6；Down 1552/8388608元素不同，最大6.1035e-5、相对L2约4.32e-6。这不是模型精度评分。

进一步测试scale乘法先后次序、预乘scale和FP64参考，没有消除native较大M差异。按G128分别计算整数partial、每组应用scale再FP32累加，则Gate/Up M128/512/4096及Down M512/4096与native逐位一致。小M下native按专家count选择channel accumulation，故这类shape与完整K参考一致，未要求统一groupwise参考一致。

源码native_int4_schedule.h的Project根据CHANNEL_ACCUM及count<=N/BLOCK选择两种路径；当前多组重复scale checkpoint数学上等价，浮点舍入顺序不同。由CPU/NPU实验确认投影差异来源，而不是仅凭源码推测。不代表已证明整模型全部MTP接受率差异都来自此处；合成输入不能替代真实层间激活和路由诊断。

## 工程判断及复现

不通过故意改变MindIEQ结果来匹配旧后端。本checkpoint仍需更广质量及MTP接受率门禁，不能将完整K参考一致宣称未量化浮点模型精度等价。下一步可定位实际模型层间误差/路由，并复用Gate/Up token级A8量化减少top8复制后的重复量化。

run-parity.sh记录隔离容器、CANN/OPP、只挂载davinci1（容器逻辑编号0）、既有库路径。使用native-record-build-only库用于异步生命周期，该实验没有替换主服务native库。MindIEQ库沿用SHA246cc4656cf401cc01586628bf018c2132251d9e22db2fb93ad17ddba8685de0。

最终复现代码tools/check_mindieq_projection_parity_310.py；最终输出projection-parity-group128-20261014.jsonl，早期6案例/扩展10案例/乘法顺序诊断原始日志均保留。最终脚本同时生成所有参考，不需要逐轮重跑历史版本。无新增编译或权重转换。
