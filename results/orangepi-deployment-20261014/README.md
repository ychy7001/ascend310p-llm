# OrangePi ARM64 / 310P1 部署验证

环境：openEuler 22.03 LTS SP3，16 核 ARM64，系统/NPU 均报告约 87.5 GiB 总内存；驱动 7.6.T7.0.B053，固件 7.6.7.0.b054。板载 RC 设备节点使用 `/dev/svm0`。原生 NNRT 为 CANN 8.0；已有 Docker 镜像为 `local/vllm-ascend:v0.23.0-310p-openeuler-orangepi-rcfix`，CANN 9.1、torch 2.10、torch-npu 2.10.0.post4。使用 Docker 是为了复用该机器现存 ARM64 镜像，避免重复下载占用有限根盘。

## 已验证

- 原生 ACL 初始化、设备计数、1 MiB NPU 分配和释放通过。
- 普通容器首次 PyTorch 探测报 E30006 / 507033；另有 ACL 初始化超时。共享主机 PID/IPC、挂载 `/var/driver`、`/var/dmp` 和设备管理配置后，CANN 9.1 ACL 及 PyTorch 32×32 矩阵计算通过。此处使用 privileged 作为部署探测配置，尚未收紧到最小权限。
- W4 MatMul、Pack、SwiGLU Pack 自定义 OPP 和独立 Torch 扩展在 ARM64 上编译成功。
- 使用真实 layer-0 expert0..7 权重，Gate/Up 与 Down 在 129/257/1024 行、均衡/单专家路由共 12 组单矩阵检查全部通过，误差阈值 rtol/atol 0.002。原始结果见 `w4-grouped.jsonl`；它不含模型端到端精度、路由/Attention 或服务性能。
- ModelScope 直接下载完成；52 个索引分片齐全，共 24,604,826,824 字节。索引 SHA256 与 a2 一致：`3f38112c8fc6e4f247e3c05b91df43e487e94ac7e8cf7787a01728cc87d47624`。

## 初始服务

机器目录 `/data/qwen36-deploy`，模型目录 `/data/models/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P`。
运行 `bash /data/qwen36-deploy/run-service.sh`，查看 `docker logs -f qwen36-orangepi`。
候选参数：TP1、FP16 KV、无 MTP、eager、4096 上下文、2048 token prefill chunk、单并发、内存利用率 0.45、显式 KV 1 GiB、保留视觉、开启前缀缓存、端口 18024。此文件为本机部署探测记录，尚非正式产品档。

`VLLM_ASCEND_FORCE_RC=1` 激活已有 rcfix 镜像的 RC 识别覆盖；没有该变量时可能因容器缺少 lspci 错选 EP 内存画像。`rc_kv_budget.py` 加载于本机 bootstrap 的 sitecustomize 末尾，为当前 310P worker 补上显式 KV 字节预算。另将 NPU `nn.init.normal_` 初始化移至 CPU：本机旧 AI CPU 的 StatelessRandomNormalV2 初始化曾报 0x2a/507018；不修改 checkpoint 或推理矩阵精度。

镜像 ID：`sha256:a9a8d356f62c7bbdf29fd7a44000d2bc930f124d4a48478e086ddb32a1e4c351`。
ARM64 扩展 SHA256：`e90b76dc8d3bc72b40e86d52a9fa158a911088e83c23665364ecadb35289e1af`。

## 共享内存与 48G 判断

模型加载期间观测：文件 Cached 约 26.3 GiB、服务相关进程 PSS 约 6.8 GiB、HugeTLB 约 23.2 GiB。RSS 包含共享页不能直接相加；Cached 通常可回收；大页应与设备分配记录对照，不能直接称为全部有效权重。NPU 可用内存检查不等同于 Linux MemAvailable：启动时前者只剩 51.31 GiB，而 utilization=0.75 要求 65.63 GiB，导致 vLLM 拒绝启动。

a2 的 W4+C8+MTP2+Graph 日志参考：权重 27.29 GiB、峰值激活 2.75 GiB、非 Torch 内存 0.19 GiB、Graph 0.30 GiB、分配 KV 4.30 GiB。a2 是 PCIe 独立设备，不能把这些值直接当作 RC 总内存预算；RC 还需 CPU/系统开销、加载临时量和工作区余量。

48G RC 值得验证，但 112K+MTP2+C8+Graph 尚未在限制为 48G 的同型硬件上验证。应先取得常驻/启动/长请求峰值，再确定规格。降低 utilization 只控制设备预算，不等同于验证总物理内存限制；页缓存回收也不代表权重可省去。不能声称当前 87.5 GiB 机器成功就证明 48G 可产品化。

视觉 checkpoint 张量 893,142,496 字节（0.832 GiB），MTP 张量 1,689,281,536 字节（1.573 GiB）；它们是存储量，不是完整运行时开销。保留视觉意味着还需计入图像激活与 encoder cache，前缀缓存只在命中时减少计算，不能免除未命中请求的 KV/峰值预算。

## 默认视觉与前缀缓存（用户确认）

视觉编码器默认保留，prefix cache 默认开启。视觉张量约占 checkpoint 3.63%；真实图片激活、embedding cache 和预处理缓存需另测。当前镜像源码 `MultiModalConfig.mm_processor_cache_gb=4`，LRU 缓存按 API / engine 进程复制，单 API + 单 engine 的理论总上限为 8 GiB；不是启动时固定占用 8 GiB。未来 48G RC 档应根据图片负载收紧该预算，并保留视觉功能。

开启 prefix cache 时，本机模型的 Mamba align 模式选择 block_size=1152，`max_num_batched_tokens=512` 因小于块尺寸而启动检查失败。已调整为 2048，失败日志见 `startup-prefix-align.log`。尚需实测冷/热请求、A/B/A 后缀状态复用和图片生成后才能称为默认档通过。前缀缓存命中只降低重复前缀计算，冷 Prefill 性能应继续单独报告。

本机隔离补丁复制到 `/data/qwen36-deploy/repo/bootstrap/rc_kv_budget.py`，并在同目录 `sitecustomize.py` 末尾追加一次 `import rc_kv_budget`。它不应直接移入通用官方主线。

实际服务曾完成启动和 `/health` 200，但首个文本请求触发 AI CPU `FloorDiv` 0x2a，引擎退出。定位到前缀缓存 align 的 `mamba_get_block_table_tensor` 块索引运算；隔离补丁将该小整数索引计算移到 CPU，表 gather 留在 NPU。该补丁增加一次 NPU→CPU 同步，尚需真实生成和热前缀收益测试。原始日志 `startup-first-request-floordiv.log`。截至本记录不能称为服务交付通过。

### 最新峰值采样

`memory.jsonl` 中 12:11:39–12:11:59 UTC（上一轮启动/多模态预热区间）观测 HugeTLB 47.99 GiB、AnonPages 12.69–12.81 GiB、Cached 23.77–23.98 GiB、MemAvailable 最低 24.82 GiB。整机 87.51 GiB，总量减 MemAvailable 约 62.69 GiB 是粗略不可用预算指标，并非精确的模型有效内存。它包含设备大页池与 CPU 进程、系统等；不能把整个峰值归因于视觉，也不能按 checkpoint 大小判定 48G 够用。当前默认最大图片 profiling/缓存参数需要收紧并复测，才可支持 48G RC 结论。

## 最终基础生成验证（12:25–12:26 UTC）

隔离块索引补丁后，服务 `/health` 200，端口 18024。实际运行参数仍为 TP1 / W4 / FP16 KV / 无 MTP / eager / 4096 上下文 / 2048 prefill chunk / prefix on / 视觉 on；不等同于 a2 的 112K+C8+MTP2+Graph。

- 文本 `What is 2 + 2?` 返回 `4`，请求 1.956s。
- 2304 token 相同前缀、32 token 输出：冷请求 12.519s，热请求 9.812s，prefix hit counter 增加 1152；输出逐字一致。单次冷/热结果，无置信区间，整请求时间含 Decode，不能当作纯 Prefill 吞吐。
- 合成 128×128 红色 PNG 返回 `Red`，请求 1.566s。只证明基本视觉输入和生成路径可用，不代表完整视觉质量或高分辨率稳定性。

原始输出 `service-smoke.jsonl`，成功日志 `startup-success.log`。复现探测：`python3 /data/qwen36-deploy/smoke_service.py`（输出追加模式，先运行服务）。下一步是限制默认最大图片画像/预处理缓存预算并重测 RC 常驻与峰值，随后再分阶段扩大上下文和引入 a2 已验证组合。
