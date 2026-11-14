# WY 默认交付：eager mask 复用构建

本目录构建的 20261114 版本已经过性能、精度和封装门禁，并已发布到 registry，成为仓库双卡默认交付。镜像引用、配套权重、放行依据和回退见 [默认交付清单](../../../docs/qwen36/w8a8-wy-delivery.md)。

正式源码是 `runtime/wy_compat/eager_mask_reuse.py`，SHA256 固定 `04540ffa5eca866d7f3960dc987bc721757f90e76f62daa427aed70af97c145a`。基础 image ID 为 `bb0fa66f1c715a94abf56afc17a6bd02e3f993c4804e200dfd229e22b0d5cab4`，对应上一版 WY registry digest `sha256:a11b8aca640120deaaf5429d88b6a82fb4434db25c71cde5d5760d531553dbd5`。

在已有基础镜像的机器，从仓库根目录离线构建：

```bash
python3 deploy/w8a8-tp2-wy-mtp/mask-reuse/build.py --manifest image.json
```

独立暂存构建时，将正式源码与本目录一起复制，再显式指定 `--source /path/to/eager_mask_reuse.py`。构建器校验基础入口/sitecustomize、复用源码和挂钩 SHA，继承原入口、命令与环境，只增加复用代码和挂钩。构建产物默认 `QWEN36_EAGER_MASK_REUSE=1`，设为 `0` 可恢复原构造路径。

复用仅限同一 target eager forward。设备、布局及完整有序 query/context 长度一致才命中，forward 返回或异常均释放；Graph 和 SpecDecoding 保持原路径。

`smoke.sh` 为封装验证工具，要求本目录预备 `image.json`、冻结模型元数据 `frozen-model.sha256` 和 `tools/check_long_context_needle.py` 的副本，适用于 a2 原验收路径。它持有 NPU 锁，仅监听 localhost:18150，检查 32K 检索、双 rank 命中、Graph 捕获/重放、退出与清场。本版已经完成该验证；后续构建变更时按需再验收，不重复 WY/main 选型。
