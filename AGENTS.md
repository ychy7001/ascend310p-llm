# 交付约定

本仓库是内部 Ascend 310P 推理引擎的交付镜像与源码仓库，同时保留验证工具和历史证据。

- 双卡默认版本以 `deploy/variants/qwen36.json` 的 `default_variant` 为准。当前路线为 Qwen3.6 W8A8 + native WY + MTP2 + Decode Graph + eager mask 复用；面向全部重 Prefill 场景，不重复做 WY/main 选型。
- 新优化须有相同基线的性能与精度门禁，以及封装验证。通过后发布 registry 固定版本，核验 manifest/config 和 digest 拉取，并明确替换默认交付，不能长期以“实验候选”替代正式交付。
- 正式运行源码放 `runtime/`，`experimental/` 保留探针与试验。发布源码与镜像内对应文件的 SHA 必须一致。
- 发布更新须同步固定版本/default tag、variants 登记、部署 manifest、Quadlet、env 示例、中英文 README 与当前交付文档。保留上一版固定 digest 与权重 SHA256 清单。
- 运行 `python3 tools/check_default_delivery.py` 检查源码/证据/镜像引用一致性；针对复用运行 `python3 -m unittest discover -s experimental/prefill-mask-reuse -v`。
- 保留历史测试记录，当前交付指向 `docs/qwen36/w8a8-wy-delivery.md`；历史失败与修正如实记录。
- 仓库默认交付变更与各站点在线服务重启分别记录。registry 凭据、原始业务数据及模型权重不提交 Git。
