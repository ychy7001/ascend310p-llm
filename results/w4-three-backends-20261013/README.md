# 真实专家三后端对照与部署收束

2026-10-13，a2芯片0/310P3，CANN9.1/Torch2.10/torch-npu2.10，独占运行。使用公开checkpoint的layer0/expert0，gate/up K2048 N1024，down K512 N2048。native使用当前N128 OPP；MindIEQ固定d9c9871aa335f7b5638e9a2679a45bb3e7561d3a及已编译隔离镜像。逐张量断言offset0、scale为per-channel重复；无完整权重转换。

激活为合成FP16，由当前设备pack产生共同A8整数和per-token scale；CPU参考使用这些实际整数及真实权重整数。native W4、MindIEQ biased W4、NZ W8逐次校验，容差rtol/atol .002。M1/2/4/8/128/512/1024/4096×两projection，两次独立运行各16组全部通过，最大绝对误差约2.43e-4。每组4个独立输入、Graph固定地址原地更新全部校验通过。

测量预热5、重复30、三个后端轮换顺序，保留原始us样本。包含Python分发/Graph replay及同步，不是纯kernel；排除激活pack、权重打包、输入更新、路由及SwiGLU。不是实际模型Prefill/Decode吞吐。

第二轮Graph中位us（第一轮同方向，详见comparison.json）：

| projection | M | 当前W4 | MindIEQ W4 | W8 | 当前W4/MindIEQ |
|---|---:|---:|---:|---:|---:|
| gate/up | 1 | 85.37 | 117.43 | 89.42 | .73 |
| gate/up | 128 | 326.30 | 118.80 | 85.94 | 2.75 |
| gate/up | 512 | 773.64 | 137.59 | 116.49 | 5.62 |
| gate/up | 1024 | 1475.46 | 209.59 | 163.08 | 7.04 |
| down | 1 | 72.88 | 85.49 | 75.09 | .85 |
| down | 128 | 140.20 | 93.26 | 82.49 | 1.50 |
| down | 512 | 470.92 | 126.94 | 107.38 | 3.71 |
| down | 1024 | 845.23 | 193.93 | 157.79 | 4.36 |

结论：小M保留当前W4；MindIEQ值得继续作为较大M/Prefill候选，多数仍慢于W8。不能推算模型快数倍：MoE每expert实际token数、Python多次调用、分发与激活量化均未计入，也没有动态router。下一门禁应是八expert分组与完整gate/SwiGLU/down，对每expert小M覆盖；通过后才可选模型接入。

首次用原作者裸ACL capture接口，native/W8部分形状输入更新错误，日志three-backends-first-failure.txt。统一改成torch.npu.NPUGraph后全部通过；这是捕获入口差异证据，不宣称修复kernel。原始裸ACL诊断保留在a2，未作为有效性能。NPUGraph测试不证明vLLM动态图/路由兼容。

复现入口run-three-npugraph.sh包含实际隔离镜像、挂载和环境（需将远端路径换为本地制品路径）；bench_w4_backends_310.py只导入外部packing helper，不复制CANN许可代码。model_index hash在原始jsonl，制品hash见candidate-artifacts.json。real-shapes-native-w4.jsonl是已有工具的八expert数值检查，无三后端公平对照，不能直接与单expert结果比较。

部署第一步收束为configs/tp1-48g-112k.env与scripts/run-candidate.sh；README指向当前参数与实验边界。--check已在a2验证，不初始化NPU、输出实际哈希；缺失环境/制品直接拒绝。该入口仍调用原已测启动链，不增加新的kernel。尚未完成此新wrapper的完整服务重启回归。完成单算子实验后当前没有模型服务，芯片留空用于下一MoE实验。
