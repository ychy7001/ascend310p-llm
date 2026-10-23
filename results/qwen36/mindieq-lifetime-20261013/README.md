# MindIEQ异步完整MLP与投入判断（2026-10-13）

结论：保留大专家候选，有边界继续验证；不全量替换默认W4。当前没有MindIEQ端到端加速数据。

Ascend310P3/CANN9.1/torch-npu2.10post4，layer0全部256真实专家权重，4096路由行，真实服务采集counts、合成FP16激活；包括激活量化、Gate/Up、SwiGLU、Down、Python逐专家调用和混合拆分合并，不包括router、共享专家、Attention、token combine。p05/p50/p95按每块最大专家行数选样，非延迟分位数。warmup5次、测量20次、后端交错顺序。权重驻留/打包不计时。

## 已完成的计时复测

绑定记录inputs/output/workspace的stream，固定调用者stream；两次独立完整MLP测试均通过。下表取第二轮中位数，单位ms：

| counts样本 | 原W4 grouped | 全部MindIEQ | 混合≥128行 | W8计算对照 |
|---|---:|---:|---:|---:|
| p05 | 19.343 | 26.197 | 19.843 | 15.823 |
| p50 | 19.168 | 23.309 | 17.371 | 14.141 |
| p95 | 17.324 | 16.104 | 14.093 | 10.135 |

混合≥128行耗时约+2.6%/-9.4%/-18.6%；第一次约+2.2%/-9.2%/-17.8%，见两个full-mlp JSONL原始20次采样。全量MindIEQ在前两个样本更慢。W8是同一W4整数展开后的计算对照，不是原W8A8模型。激活量化器不同，相对L2对照不能视为共享整数输入下的纯kernel误差，也不替代模型质量。第二轮3样本×5后端×8次连续异步提交=120输出均有限，且各后端同输入输出逐值一致；不能证明所有allocator历史稳定。

## 残余失败与定位

只固定stream、不recordStream仍出现NaN；单独inputs或workspace记录在最小尾样本通过，但workspace-only完整三样本混合128仍NaN。all-record两次完整测试通过后，单独p95/第三份相同随机输入的混合测试仍NaN，故不能宣称仅MindIEQ绑定修复已完成。

失败行定位均属于小于128行的原W4分支：专家0/60/62/63/64/101/102，各专家M44/42/43/1/66/74/91。本次未观察到这些NaN来自MindIEQ大专家输出；不能据此排除其全部问题。Python保留native输入仍失败。

隔离检查原W4的EXEC_NPU_CMD：原适配头workspace在if作用域销毁，handler持有raw workspace地址而不持有全部Tensor。新增`patch_native_queue_lifetime.py`仅为本项目四个全Tensor参数算子生成隔离候选：捕获Tensor和workspace、记录stream；不改SDK/公共头/稳定服务二进制，不用于带标量属性的泛用算子。编译前两个候选因相对头路径失败，v3修正两个相对include并完成编译。v3相同失败输入hybrid128/256数值通过，相对L2约0.596%/0.530%。这支持生命周期方向，单次PASS尚不证明完整根因或部署验收。随后三分布、五后端、连续8次提交/计时测试停滞约5分钟，JSONL只写入config，NPU AICore为0；手工停止测试（SIGTERM未在5秒内退出、Podman转SIGKILL）。根因未确定，不能称已通过或生成有效计时。见native-record-stall.txt；v3仅保留为失败诊断，不进入默认服务。

## Graph与复现边界

workspace-record绑定单矩阵16形状、4次输入更新Graph测试通过。完整MLP Graph探索在native gate.float的aclop Cast捕获失败，未完成全后端Graph；普通Prefill仍Eager，不能将该失败解释为现有Decode Graph失效。`run-record-npugraph.sh`是备用复现脚本，此轮未执行；其余实际执行结果以同名JSONL为准。

本目录脚本保留a2路径，依赖此前固定外部源码、镜像、OPP和权重。record绑定按build-record-extension.sh构建；scope选项与inputs/workspace/none脚本对应。原始外部代码不纳入本仓库Apache授权，补丁生成时保留原声明。依赖来源与二进制hash见artifacts.json。torch-npu源码取v2.10.0，未证明与运行wheel2.10post4完全一致。

后续门禁：异步混合数值、重复计时→设备端grouped/按M选择→真实服务同协议Prefill/质量回归。若无可重复端到端收益，停止全量迁移。不能把单矩阵大M加速倍数或本表MLP节省直接写成模型Prefill tok/s改善。默认服务参数和权重不变。

注意：部分失败JSONL末尾是前面样本的PASS，必须检查summary；没有summary不能当整组通过。完整MLP Graph Cast报错另见failure-full-mlp-graph-cast.txt。

实验结束恢复原候选端口18024，health HTTP200；英语/中文/代码/数学/重复文本5项64-token基本生成检查通过（candidate-restored-lifetime-sanity.json）。不是MindIEQ模型集成或完整质量复测。
