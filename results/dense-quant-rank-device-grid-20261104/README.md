# 两个TP分片 × 两张物理NPU独立重放

2026-11-04，生产固定镜像，真实down_proj捕获文件保持远端只读；不加载完整模型，逐容器运行。

rank0/device0、rank0/device1、rank1/device0、rank1/device1 均 PASS，4096 rows，CPU INT32采样参考96元素最大绝对差0，完整输出finite。输出零与原始激活及pertoken_scale均零一致，不能代替非零输入精度验证。

这补齐了此前6/6只覆盖rank0/device0的盲区；并非证明模型内运行安全。逻辑TP分片/物理设备单独不足以复现，完整调用历史、分配状态和软件栈仍待定位。

控制器complete，四个子进程returncode0；每项PASS已直接从结果JSON检查。结束恢复a2临时unit。
