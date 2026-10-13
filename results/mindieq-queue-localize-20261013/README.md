# MindIEQ混合路径：异步生命周期门禁推进

2026-10-13，a2/Ascend310P3/CANN9.1/torch-npu2.10post4。原稳定服务未采用实验扩展。外部源码、OPP、模型沿用上一轮固定制品，补丁仅在新目录生成；不修改SDK或原适配头。

## 本轮改动

1. 原W4绑定`record-only`候选：对四个全Tensor参数算子记录输入/输出/workspace使用流，将workspace延长至调用函数末尾；不把Tensor捕获进任务handler。`all`/`keep-only`为消融选项，keep-only仅生成契约校验，未在NPU执行。本轮不验证跨stream或带标量属性的通用调用。
2. MindIEQ`caller-workspace`候选：调用线程创建ACL描述、GetWorkspaceSize、分配workspace并记录Tensor使用stream，任务handler只提交ACL计算并释放ACL描述；没有Torch分配和持有Tensor析构。改变绑定生命周期，不改变MindIEQ INT4解包/INT8 Cube kernel。
3. 验证阶段逐次提交进度与每60秒faulthandler栈，便于区分首次准备等待、实际停滞和退出问题。打印不在计时循环中；测试延迟含dispatch+synchronize，非纯device kernel latency。

## 已验证

record-only原W4 + 上轮all-record MindIEQ的完整MLP三counts分布、5后端、每后端8次连续提交，JSONL summary PASS；120输出均有限且同后端重复逐值相等。但外层180秒timeout命令退出124，最终结果已产生，不能视为完整进程成功。首次MindIEQ调用曾等待超过60秒后继续，故只凭等待和Python栈不能证明死锁。与上轮v3五分钟无结果不是同一配置，不撤销其失败事实。随后同p95/activation-draw2独立混合128/256各8次提交全部通过，退出0。

record-only原W4 + caller-workspace MindIEQ再次测试三counts分布、五后端、8次连续提交：summary PASS，退出0；120输出逐值一致。本轮混合128在三分布的相对当前W4 L2约0%/0.411%/0.596%，各自采用不同激活量化器，不是纯kernel整数计算误差或端到端质量。

完整MLP（ms，warmup5、重复20、后端交错）：

| counts样本 | W4 grouped | 全MindIEQ | 混合≥128行 | W8计算对照 |
|---|---:|---:|---:|---:|
| p05 |19.349|28.100|19.862|16.029|
| p50 |19.175|25.027|17.238|14.346|
| p95 |17.288|17.302|14.062|10.183|

混合耗时变化+2.7%/-10.1%/-18.7%，保持上一轮方向；全量替换不通过性能门禁。p05/p50/p95是按最大专家行数选counts，非延迟分位数。层0全部256真实专家权重、合成FP16激活、真实counts；不含router/共享专家/Attention/token route combine，不能外推模型tok/s。W8为同W4码展开的计算对照，不是原W8A8模型。

caller-workspace与record-only组合：gate_up/down × M1/2/4/8/128/512/1024/4096，共16shape的3后端eager/NPUGraph全通过，4组输入原地更新后对相同整数A8/scale的CPU参考，Graph errors为空。完整MLP Graph此前Cast仍未解决，本轮只验单矩阵Graph，不能证明完整vLLM动态图兼容。

## 复现与决定

在芯片0独占时顺序执行本目录run脚本；依赖外部固定镜像、OPP及权重。构建脚本使用全新目录，caller构建变量SOURCE_DIR/BUILD_DIR需指向work下新目录。CMake从外部固定版本生成，替换aarch64→x86_64和vendor目录；前两个caller编译尝试因include路径失败，第三次修正后编译完成。hash和进程退出情况见artifacts.json。

本轮完成异步组合的关键诊断和最小验证，仍不宣称全部生命周期根因已经证明。混合诊断仍预知CPU counts、逐专家Python调用，不能直接进入正式动态Prefill。下一步设备端grouped/选择，避免每层CPU同步与大量逐专家launch，再验证真实服务质量与Prefill收益。默认服务和模型制品保持不变。

相同p95/activation-draw2的caller-workspace+record-only独立进程压力测试：hybrid128/256分别连续32次提交，64输出全部有限、逐值一致，退出0。此轮未见原失败NaN，验证范围仍限定这些样本。
