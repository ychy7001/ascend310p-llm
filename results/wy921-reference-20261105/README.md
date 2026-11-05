# wy-5 高精度参考与 logprob 比较器可行性核查

结论：**待定，停止模型实验，未启动 a2 NPU 窗口**。

已修复并测试统一的 `tools/compare_token_logprobs.py`：

- top-k 截尾时目标 token 缺失，记录目标 logprob 差的下界，不中止比较。
- 两侧 top-k 没有共同 token 时，KL 记录为不可估计并列出位置，不抛异常。
- 噪声底本身含截尾时，不把下界当作精确阈值。候选实测值低于噪声下界可以判定非劣；候选也截尾或超过下界时保持待定。
- 判定倍率按本轮冻结协议为 `1.0`，无绝对阈值豁免；比较器同时保留原有 greedy 首分歧和共享 top-k 条件 KL 记录。

测试命令：

```bash
python3 -m unittest discover -s tools -p test_compare_token_logprobs.py
```

结果：**9/9**。

## 高精度参考 R

按用户规定的优先顺序核查：

1. 已有 GPU 35B 环境使用 GPTQ Int4/W4A16 权重，不能充当同模型 BF16 或更高精度参考；3090 核查时另有约 0.8 GiB 可用显存的容量限制。
2. 已有 GPU 27B 环境是另一模型、W4A16、FP8 KV、MTP3 和不同运行栈，不能充当 Qwen3.6-35B-A3B W8A8 TP2 的高精度 R。
3. 已检查的本机缓存及已有参考环境挂载中没有同模型 BF16/更高精度权重；发现的 35B 文件是 GPTQ 量化 shard。只读 safetensors header 审计见 `raw/gptq-header-audit.json`：7 个 shard、124611 个 tensor、约 24.42 GB，量化配置为 `bits=4`、`group_size=128`、`method=gptq`、`pack_dtype=int32`。
4. 已检查位置没有同模型高精度 checkpoint，这是 CPU 路线的直接阻塞。当前可用内存约 41 GiB、交换空间基本耗尽，还存在容量限制；未测试内存映射或 CPU offload，不能仅凭内存数字断言这些路线不可行。没有新下载权重，没有访问交付站，没有在本机 3080 分配资源或执行推理。

因此本轮不伪造 R，不把 GPU GPTQ 或旧模型输出当作高精度参考，也不启动 a2 NPU。

## wy-5 判定协议冻结内容

在查看任何 wy-5 B/B0 数据前，判定协议冻结为：

- 输入：复用 wy-2 冻结的 35 条 token ID 输入。
- R：同一输入逐位 teacher-forcing 高精度 logprob 分布；当前 R 不可取得，故模型门禁不执行。
- A、B、B0、R 各报 top-1 一致率、KL、最大目标 logprob 差和首分歧位置。
- 新采用条件：B→R 各指标不劣于 A→R，允许差值不超过 A′→R 与 A→R 的噪声底；保留旧 A/B/A′ 结果作为参考记录。
- B 使用 wy-4 修复版；B0 使用修复前原生 WY 制品。B/B0 均禁止回退到 Torch。
- G2 仍使用 GSM64、13 例结构化 C1/C3、8 短题前后，各项相对 P 和 A 的通过数差≤1。
- 性能仅在 G1、G2 均通过后运行，报告会议回放、RAG 8K/16K/32K、通用 1K/4K/16K/32K×C1/C4 和 WY 冷 Prefill。

## 真实限制

wy-4 已完成的 A/A′ 数据显示最大目标 logprob 差至少为 7.31，且 14/35 条 greedy 序列发生分歧；该噪声底必须由同轮数据重新完整记录，不能用绝对小阈值替代。由于本轮没有合格 R，未重跑 A/A′、未运行 B/B0、未运行 G2 和性能。后续首先需要已有或另行提供的同模型 BF16/更高精度 checkpoint，以及能容纳权重与 teacher-forcing 工作集的 GPU/CPU 资源；单纯释放显存仍不能补齐高精度权重。缩减 prompt 子集也不能省去模型权重。

比较器修复仅在规范工具及单元测试中完成；未部署或执行归档控制器，不宣称旧远端导入路径已在运行时验证。

冻结协议的精确定义、输入 SHA 和截尾规则见 adjudication.json。统一使用 A 的固定 256-token 续写作 teacher-forcing 条件；损失指标采用 top-1 不一致率、完整词表 KL(R∥臂) 的均值/最大值、目标 logprob 最大差，以及逐请求首分歧损失（256 减首分歧下标，无分歧为 0）。每项满足 `L(B,R) ≤ L(A,R) + |L(A′,R)−L(A,R)|`。无法取得完整 KL 时，条件 top-k KL 只作辅助记录，门禁保持待定；不以截尾下界代替精确阈值。尚未采集同轮数值阈值。

协议 SHA-256（adjudication.json 的 frozen_before_B 对象，UTF-8、排序键、紧凑 JSON）：`47a47f287a92ae049a9c2ec410972942ba24bcd4f0a1b0f8c9482ed3b5bc8d98`。

## wy-2 ≥13 的位置复核

在分歧前共享生成上下文（含首分歧位置）内逐位检查 35 条旧记录，≥13 的位置共 **1 个**：`replay-17`，零起始位置 **22**，也正是首个 greedy 分歧位置。A 的目标 token `98898` 的 logprob 为 **−0.0551017597**；旧 B（本轮称 B0）未把该 token 列入 top-20，其 logprob **≤−13.3086261749**，所以差值 **≥13.2535244152**。A/B0 的 top-1 分别为 `98898`/`96025`，各自 top-1 与 top-2 的 logprob 间隔约 **4.03125/6.17188**，这两侧观测并非接近平局。R 缺失，无法判断 A 或 B0 谁更接近参考；修复后 B 尚无该位置数据，也未继续作分层归因。原始逐位记录与 SHA 见 adjudication.json 和被忽略的 raw/wy2-gap-ge13.json。

本轮未操作实验站 NPU 或服务，没有恢复窗口；没有独立复核当前服务健康状态，也未启动附属图像服务。会议回放、RAG 8K–32K、通用 1K–32K×C1/C4 和 WY Prefill 收益均未测量。

## 用户指定的 CPU 子集方案（已冻结，权重核查后停止）

本节替代上一版要求完整词表 KL 的采用判据；完整词表 KL 不再是本轮前置要求。旧协议保留作为历史记录。

子集取全部 35 个用例的续写位置 `[0,8)`；其中 `replay-17` 改取 `[0,24)`，合计 **296** 个位置。逐用例名称、区间、prompt 长度完整列于 adjudication.json 的 cpu_subset_protocol.positions。完整 prompt 保留，同轮 A 的固定续写作为所有臂的 teacher-forcing 条件，不以独立生成后的不同上下文作比较。

R 与各臂固定 top-20；目标 token 缺失时记录 logprob 上界和误差区间。KL 使用五臂同一显式支持交集上的条件 KL，报告支持数与概率覆盖质量；不能解释为完整词表 KL。无交集或截尾区间无法确定大小时为待定。采用条件逐指标为 `L(B,R)−L(A,R) ≤ |L(A′,R)−L(A,R)|`，无绝对豁免，并保留同子集的 A′/A 噪声与 B/A 偏离。精确定义见 JSON。

CPU 子集协议 SHA-256（同前述规范 JSON 编码）：`761d1ebefe0f49d978dd99d8646d1dbe78b04b13fef6c0ee8ff240d8e0295fb9`。本次提交先于 R 和新 B/B0 读取；同轮 A 续写 SHA 仍须在获取后、读取 R/B 前补充冻结。实际核查未发现可用的原始 BF16 权重，因此 R 未启动。

## CPU 权重核查

在 CPU 重任务锁内核查了本机模型目录、缓存、容器卷及已有参考 fixture。Qwen3.6-35B-A3B 唯一完整候选仍是 GPTQ Int4：语言模型 shard 的 header 含 `I32` 打包 `qweight/qzeros/g_idx` 和 `F16` scale；BF16 部分包括视觉/MTP 及未量化的语言模型参数，但语言模型的 MoE 投影仍是打包 Int4，不能组成原始 BF16 全模型。它可以做反量化近似，但不是原始 BF16，无法作为本轮 R。已有 BF16 fixture 仅是另一模型的两块 float shard 加少量 expert tensor，缺少完整语言模型。

另有 Qwen3.6-35B-A3B APEX GGUF compact，header 显示 733 个 tensor、约 34.66B 参数，包含 Q6_K 151 个、Q4_K 191 个、Q3_K 90 个及 F32 301 个；按 BF16 物化约需 69.32 GB，仅读取 header 未读取 tensor data。它可按 GGUF 专用路径反量化，但会保留量化误差，且不能满足“原始权重、纯 torch eager”的 R 定义。转换 GPTQ/GGUF 还需要大规模 CPU/disk 临时空间；未在资源不足时启动。核查明细和 header SHA 见 `raw/cpu-weight-availability-audit.json`。

因此没有读取 R，也没有重跑同轮 A/A′ 或读取新的 B/B0；G1/G2/性能继续不执行。
