# 第一次真实输入重放：准备失败

六个子进程均在torch.load(weights_only=True)时拒绝torch_npu Format枚举，不曾调用量化Matmul；不是六次算子失败。原日志保留。v2显式allowlist安装库的Format类后执行，见相邻v2目录。
