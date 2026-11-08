# 官方基础镜像替换评估（只读，2026-11-05）

评估对象为官方当前 `v0.30.0-310p`。已核对 registry tag 与本机已有缓存一致；没有拉取另一份大型镜像，没有用它构建候选，也没有修改生产基础锁文件或运行 NPU 验证。

- 镜像 tag：`${UPSTREAM_REGISTRY}/ascend/vllm-ascend:v0.30.0-310p`。
- 当前多架构索引 digest：`sha256:041cc7f7b9e9c51cd823da28dc5a6c8ea1ac06df2b203e69a82766fc1668eafa`。
- amd64 manifest digest：`sha256:ebea1420f0c31c3ef41344b64b8570326e78b9aee0ab3ba927646ce97131fcaf`。
- amd64 image config ID：`sha256:d103c59b31c61788998f91119e61aee6512db1177de2bfee345489c925a3e70a`。
- 镜像构建时间：2026-10-29 14:24:45 UTC。tag 可继续变化，后续若获准换底应锁 amd64 manifest digest。

| 项目 | 生产恢复基底 | 评估对象 |
| --- | --- | --- |
| vLLM HEAD | `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665` | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vllm-ascend HEAD | `88eac2713b14beedc7a48803274f8d6cee3e9f30` | `c09d3e20b9c45c80b34ae3781440d90904a44023` |
| vLLM 安装版本 | `0.23.0+empty` | `0.30.0+empty` |
| vllm-ascend 安装版本 | `0.23.0rc2.dev93+g88eac2713` | `0.30.0` |
| Python | 3.12.13 | 3.12.16 |
| torch | `2.10.0+cpu` | `2.10.0+cpu` |
| torch_npu | `2.10.0.post2` | `2.10.0.post7` |
| CANN | 安装信息确认 `9.1.0-beta.1` | 安装信息确认 `9.2.1` |
| 两份源码工作树 | 干净（生产 overlay 另有两个文件改动） | 干净 |

补丁适配仅检查文本及来源：

- 生产补丁在该固定官方镜像上执行 `git apply --check` 返回 0，两个文件均无需文本调整。
- 官方两个基线 blob 为 `5e68eb66d482260f01138873b04716f3717c67dd`（attention mask 实现）和 `dab84617d9f39dace3144af5ea97b4752bb122e0`（测试），与生产恢复基底一致。生产额外改动尚未出现在这两个上游文件中，当前不能删除该生产复现补丁。
- 上游已合并的 #10181 是基线文件来源，不是生产额外改动的 PR。没有对 attention mask 的数学、Ascend 算子/内核实现、数值或性能作分析或测试。

换底会同时改变 vLLM、vllm-ascend、torch_npu、Python 和 CANN。补丁文本可应用不证明新的执行路径、依赖或生产参数兼容。现有生产构建校验器会拒绝新镜像的两个 HEAD；不能只换 `BASE_IMAGE` 就把它当作已通过的生产重建。后续若 owner 决定升级，需要另外更新固定来源和依赖锁，并重新进行 CPU 文件与环境审计，再进入已审批的设备门禁。

本次建议先交付旧生产组合的可重复构建证据；官方升级作为独立决定。未修改 driver、CANN、模型或生产引用；未以新增本地补丁处理通用 vLLM 问题。
