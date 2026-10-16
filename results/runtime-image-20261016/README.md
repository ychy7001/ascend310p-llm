# 固定来源运行镜像验收（2026-10-16）

本轮只收束代码/制品交付，不叠加失败MTP2修复或升级依赖。实现和命令见 [runtime-image](../../docs/runtime-image.md)。源码提交 `70d71e9` 将原clt已验收的同步metadata提为显式开关；`c9fd1daf7f2b059684399473626dc0509fae2f7e` 增加固定基础、最小运行文件白名单和独立镜像入口。

## 基础镜像标识纠正

上一轮初次按 Podman/Docker 的 image ID 差异记录了疑点。本轮进一步核对，两个EP基础镜像共用registry digest `quay.io/ascend/vllm-ascend@sha256:16770d322d0e343bc5ff61f897acfc6fb1383eee56d7939554e1c9cf8b504773`，18层RootFS逐项相同。不同运行时显示的image ID不能作为镜像内容不同的证据；原始对照见 `base-comparison.json`。这不消除运行容器的bootstrap、调度、CPU线程、视觉、driver等差异，也不证明MTP2的硬件/驱动根因。

EP固定源：vLLM `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`，vLLM-Ascend `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`；构建时检查git HEAD及工作树。不是升级到main。

## 打包与完整性

- 实验目录：a2 `/home/ychy/ascend-lab-runtime/runtime-image-20261016`，独立checkout、context、日志和输出；原服务源码路径保持不变。
- 候选镜像：`localhost/qwen36:c9fd1da-amd64`，本机config ID `b381d8e5ed45c11633a7bd07a50a0f5fcf8cf71bc797f1b543d589ab07dd759d`。
- `image-manifest.json`锁定168个运行文件/算子/OPP/许可文件。已通过基础源码及逐文件哈希验证；临时容器篡改profile后校验器正确拒绝，见 `tamper-rejected.txt`。
- 只导出指定Git提交的运行必需文件；不把模型、测试脚本、完整fork或失败候选混入镜像。C8作为显式可选代码和库保留，默认C16，本轮不将其标为完成新镜像C8验收。
- 本轮使用已测二进制打包，逐文件hash和host ELF架构可校验；不是从源码重建CANN或全部自定义算子，不保证任意ABI兼容或镜像逐字节重现。

## 真实NPU协议

`run-a2.py`保留实际编排：原服务短题/A-B-A → 32768/128 warmup1+正式2 → 长后短题；独立镜像同短题/32K → 112000会议四字段 → 长后短题。结束或失败自动停止候选、恢复原systemd服务，再验收health/短题/A-B-A。只动device0，device1原服务保持。

独立镜像复用原环境与权重，只保留model及driver的5个bind，不挂载主机Python源码、动态库或OPP。参数为TP1/C16/W4A8/MTP2/Graph/GDN/MindIEQ E5、batch2048/chunk512、max114688、util.95、async、prefix off、视觉加载，候选端口18025，原18024。

Prefill/Decode使用独占端点Prometheus阶段计时，分别input/time、(output-1)/time；不是纯kernel时间。完整响应、输入hash、接受率、prefix hit均记录。基线已长时运行，候选是新进程，且主机出现约9.9GiB可用RAM及8GiB swap已用，因此本轮为打包可用性/协议对照，不能将速度差异单独归因镜像，非正式P95或业务质量验收。

## 其他环境边界

- clt随后恢复SSH，但主机已重启，新增310P7成为device0（Critical），原310P1变为device1，Docker服务未运行。仅做只读检查并停止本轮等待health的测试进程，未启动Docker或初始化NPU。用户确认该机正在维护，明确本轮只推进a2。`run-clt.py`是准备协议，不是已完成验收；原先device0映射不能在维护后直接复用。
- orangepi仅有118G持久分区，根分区剩约872MB；未发现额外挂载的持久空间。未自动清理文件或构建ARM镜像。ARM64 RC需要自己的基础/host制品和独立验收，不使用amd64镜像。
- 参数继续按 [profile](../../docs/profiles-20261014.md) 的已验证组合：a2 MTP2/2048，clt MTP1/2048，RC MTP1/4096。公共代码固定后也不根据hostname或SoC自动选取全部参数。

## 32K配对结果

| 模式 | Prefill tok/s（两次） | Decode tok/s（两次） |
| --- | --- | --- |
| 原bind-mounted服务 | 543.05 / 544.50 | 27.43 / 29.21 |
| 独立镜像 | 543.44 / 544.93 | 27.31 / 29.15 |

两个配对请求输入hash与完整输出逐字一致，正式样本均有效，prefix hit全0。差异很小，支持本轮未观察到明显打包回退；不是新的性能优化收益声明。新镜像短题8/8、A/B/A一致、无canary泄漏。

独立镜像112000-token会议提取四个字段全部正确，阶段计时Prefill **348.02 tok/s**、Decode **19.95 tok/s**，长后短题8/8、A/B/A一致且无泄漏。见`a2/image/`及`a2/image-gates.json`。会议输出长度和内容不同于32K/128，不能直接横比Decode；本轮未重测最大可用上下文。

## 发布与托管切换

镜像已发布到 `registry.cn-hangzhou.aliyuncs.com/meetai/llm-service-vllm-ascend`，固定 digest：

`sha256:8e17ec9ae646b1e418cf119bbb347238c4505b8b0037782a1345914824ec8d35`

a2 已按该 digest 拉取并再次通过镜像内 168 文件校验。`cutover-outcome.json` 显示新 `qwen36-image.service` 健康检查 200、质量门禁通过，实际镜像 ID 为 `b381d8e5...`，只挂载模型与驱动相关五条路径；旧 `qwen36-a2.service` 已禁用但保留用于回滚，device1 对照容器未改变。运行参数为 TP1/C16/W4A8/MTP2/Graph/GDN/MindIEQ、batch2048/chunk512、max114688、util.95。

公开 registry 当前按现有权限控制；部署者应先登录并按 digest 拉取，不能只依赖可变 tag。镜像上传、拉回、篡改拒绝、切换和回滚日志均保留在本目录。a2 当前服务已切换完成，后续可用 `systemctl --user status qwen36-image.service` 和 `curl -sf http://127.0.0.1:18024/health` 检查。
