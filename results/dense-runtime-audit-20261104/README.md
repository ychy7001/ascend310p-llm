# Dense 后续兼容性镜像审计

2026-11-04，仅启动不带 NPU 的短命容器读取安装元数据，未升级或修改镜像。完整 inspect 和原始 stdout/stderr 见 audit.json。

- qwen36:c9fd1da-amd64：torch-npu 2.10.0.post4，vLLM 0.23.0+empty，vLLM-Ascend 0.23.0.post1，transformers 5.5.4。
- qwen38-runtime:ffa980c-3ab5dda-20261018：torch-npu 2.10.0.post4，vLLM/Ascend 0.30.0 私有版本，transformers 5.14.1。该镜像不能当作仅升级 torch-npu 的单变量。
- 两者 ascend_toolkit_install.info 均自报 CANN 9.1.0，内部 V100R001C11SPC001B243；历史文档的9.1.1标签需另查编译器/算子包版本，不能直接替代本次读数。
- 第一镜像不挂驱动库的 metadata 探针触发 sitecustomize 缺 libascend_hal.so；探针最终退出0并读到元数据。这不构成带设备启动是否成功的结论。

未进行 Dense 推理或性能测量；镜像整体差异只能用于单独的运行栈兼容性筛选。
