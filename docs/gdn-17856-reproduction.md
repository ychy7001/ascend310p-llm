# 310P GDN 上轮接受数大于当前 query 长度：最小复现

2026-10-14，a2 第二颗 Ascend 310P3，镜像
`quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`，CANN 9.1。
使用镜像内置完整 custom_transformer OPP；PR17640 的单独 OPP 包没有 recurrent 算子的 ACLNN API，不能替代完整 OPP。

## 已完成的验证

脚本：`tools/repro_gdn_17856.py`。原始日志：
`results/gdn-17856-20261014/operator-original.log`。

两请求当前长度 `[2,4]`，状态表原始两行各有四槽，当前压紧后的槽号为
`[0,1,4,5,6,7]`。NK=4，NV=8，K=V=128，FP16。
设置 k/v/beta=0、g=log(0.5)，使每个 token 的状态恰为上一步的一半。
各状态初始值 0.25，额外将 A 的历史第 4 槽置为 0.5，用于排除选错历史状态的假通过。

| 路径 | 上轮接受数 | A 当前两个槽期望 | A 当前两个槽实测 | 结果 |
| --- | --- | --- | --- | --- |
| 原始合法对照 | `[2,1]` | `[0.125,0.0625]` | `[0.125,0.0625]` | 通过 |
| 原始异常 | `[4,1]` | `[0.25,0.125]` | `[0.25,0.25]` | 未按递推更新 |
| 实验状态预取 | `[4,1]` | `[0.25,0.125]` | `[0.25,0.125]` | 最小算例通过 |

异常路径请求 B 的完整状态也与预期不符。输出缓冲有复用旧值的现象；不能单凭输出非零或生成接口未报错判定正确。
合法对照和状态预取路径的全部状态槽均与确定性预期一致，脚本有断言。

## 修复方向与边界

不能简单删除 `acceptedTokenNum > seqLen` 检查，也不能把接受数截断到当前长度。
压紧索引只保留当前 query 槽，上一轮第 4 个历史状态仍须从原始完整请求行中选择。

脚本的实验路径先把 A 第 4 个历史状态保存到当前基槽，再以 accepted=1 调用原算子。
这个写法只用于证明状态选择方向，**不是通用 wrapper 修复，也没有部署到服务**。
它尚未验证任意请求数量、状态别名、随机数据、Graph、padding、混合 batch 或真实生成。
后续实现应显式分离“每请求初始状态索引”和“当前 token 写回索引”，或在完整表仍可用时批量 gather 历史状态。
所有源状态须先 gather 再写，避免别名覆盖。

此次足以支持补充 Issue 的实机复现证据，但尚不足以宣称生产修复完成。
也未证明此前模型 MTP 接受率下降就是本问题导致。

## 复现环境

将脚本挂入独立容器，只暴露空闲的实验 NPU 与 manager/devmm/hisi_hdc 设备，
挂载主机 driver/lib64、driver/version.info、dcmi、ascend_install.info。
设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`、`ATEN_CPU_CAPABILITY=default`，
移开镜像内不兼容的 triton 目录后执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
python3 /tmp/repro.py
```

不要在模型服务容器里启动该诊断进程；本次采用独立容器并暂时停止第二芯片对照服务。

## 二维状态表补丁（2026-10-14 后续）

`patches/gdn-17856-state-table.patch` 修改 4 个文件：Python wrapper、host tiling、
kernel 与 tiling 数据结构。对应原文件与官方 main
`bc8115ed7fdc7e500320b5de70742ef049b4c1a2` 的文件内容相同，补丁 dry-run 通过。

- 二维索引不再按当前长度压紧，保留 `[B,W]` 完整请求行。
- kernel 从 `row_start + accepted_prev - 1` 读取初始状态，接受数上界为 W。
- 写回使用 `row_start + current_token_offset`，独立于 query 打包偏移。
- 一维 legacy 索引保留原协议；不更改 ACLNN/Torch 参数列表。
- 零长度请求在读取状态索引前跳过，支持该行填 `-1`。
- 无效接受数（非正数且请求非空、超过真实行宽）仍保留原错误防护。
  本补丁修复合法调度数据被当前长度错误拒绝，不声明可恢复损坏元数据。
- 修改了内部 tiling 结构，必须配套构建 host 和 kernel，不能混用新旧二进制。

已用独立容器编译为 310P OPP 包。第二版随机回归：

| 路径 | 通过数 | 输出最大绝对误差 | 状态最大绝对误差 |
| --- | --- | --- | --- |
| 直接算子 Eager | 42/42 | 7.63e-6 | 1.22e-4 |
| 直接算子 Graph | 42/42 | 7.63e-6 | 1.22e-4 |
| Python wrapper + Graph | 42/42 | 7.63e-6 | 1.22e-4 |

共 126 次断言；同组数据分别重放，不能当作 126 个独立随机样本。
覆盖 W=2/3/4、上轮接受数大于当前长度、随机且不同的请求状态、打乱物理槽号、
padding 行位于 batch 前/中/尾、变长 batch、query 缓冲尾部 padding、旧版一维索引。
参考实现比较有效输出及整个状态池，包含未使用槽。Graph 重放更新设备长度、
接受数和状态表，将输出预置 NaN 防止旧数据假通过。

第一版额外加入的一维索引长度校验过严：Graph 缓冲允许尾部 padding，
第二版移除该校验，并加入“3 个分配 token、1 个有效 token”回归。
但不能把模型启动失败全部归因于该检查：第二版仍出现 tiling 错误。

进一步的小张量实验确认了独立 OPP 叠加部署的加载顺序问题。
`enable_custom_op()` 会把镜像内置 vendor 插入 `ASCEND_CUSTOM_OPP_PATH` 前端，
从而选择旧版 rank-1 host tiling；此时新的二维 wrapper 无法运行。
先在环境路径尾端登记内置 vendor，再保留修复版在首位，可阻止重复前插。
`opp-order-check.log` 记录了错误顺序失败、正确顺序 21/21 通过。
这属于实验部署问题，未新增通用框架加载补丁；上游应配套重新编译整个算子包。

复现脚本：`tools/check_gdn_state_table_310.py`，启动器：
`tools/run-gdn-state-table-check.sh`。日志为 `eager-v2.log`、`graph-v2.log`、
`wrapper-graph-v2.log`，位于同一 results 目录。

```bash
# 独立源码副本应用 patch 后，在已安装 CANN 的容器中编译：
cd csrc
bash build.sh --pkg --soc=ascend310p --ops=recurrent_gated_delta_rule_v310 -j4
# 将包内 packages/vendors/custom_transformer 提取到独立 OPP 路径。
# 在空闲设备上从本实验仓库运行：
GDN_OPP_DIR=/path/to/patched/vendors/custom_transformer \
  bash tools/run-gdn-state-table-check.sh
GDN_OPP_DIR=/path/to/patched/vendors/custom_transformer \
  bash tools/run-gdn-state-table-check.sh --graph
GDN_OPP_DIR=/path/to/patched/vendors/custom_transformer \
GDN_WRAPPER_SOURCE=/path/to/patched/vllm_ascend/_310p/ops/fla/gdn_310.py \
  bash tools/run-gdn-state-table-check.sh --graph --wrapper-source /wrapper.py
```

## 真实权重集成验证

第一颗芯片 `18024` 主服务保持原版本。第二颗芯片以独立容器、`18029` 测试补丁，
复用 W4A8 权重、MTP2 和 Decode Graph。

- 原主服务：四客户端并发发起请求，由 `max_num_seqs=1` 排队执行；
  普通数字输出 8/8、严格 JSON Schema 8/8。不能称为四请求同时模型执行。
- 修复版 C8 候选：112K、`max_num_seqs=4`，模型加载与 Graph 捕获成功；
  第一批并发请求触发现有 `C8 requires real four-tensor storage and one request` 检查，
  engine 退出、请求失败。该检查发生在 C8 attention adapter，不能算作 GDN 数值回归通过。
- 为隔离 C8 的单请求限制，另用 C16 KV、32K、`max_num_seqs=4` 测试相同补丁。
  使用 `tools/launch_gdn_state_candidate.py --work /path/to/work --c16` 启动，
  HTTP 回归脚本为 `tools/check_gdn_candidate_http.py`。
  模型加载、MTP 和 Graph 捕获成功；四客户端变长并发请求共 16/16 通过：
  普通文本 8/8、严格 JSON Schema 8/8，答案和结构均有断言。
  捕获规模仍为单请求的 3 个 token，多请求超出范围会走 eager；
  不声称所有并发模型请求均通过 Graph 执行。
  记录见 `http-c16-candidate.log` 和 `model-candidate.log`。

测试完成后停掉独立候选、恢复第二颗芯片原对照容器，主服务未升级补丁。
本次不是吞吐 benchmark，日志中的时间窗口平均吞吐不能作为 prefill/decode 指标。
短算术与 schema 通过也不能替代长输出、长上下文或任务准确率评测。

## 上游提交边界

已有证据支持提交一个独立的 310P GDN 状态索引正确性修复：
保留四文件协议补丁，附旧版异常复现、FP32 参考比较与 Graph 回放测试。
不把 W4A8、C8 adapter、产品调参或实验 OPP 加载脚本混入该修复。
完整模型测试是辅助冒烟证据；C8 的单请求限制另行跟踪。
向现有 Issue 补充复现和提交 PR 时应清楚列明上述范围与尚未覆盖项。

当前证据尚不支持性能收益、长上下文精度或此前接受率下降的归因。

## 上游提交记录（2026-10-14）

- PR：[vllm-project/vllm-ascend#18730](https://github.com/vllm-project/vllm-ascend/pull/18730)。
- Fork 分支：`ychy7001/vllm-ascend:fix/310p-gdn-mtp-state-rows`。
- 修复提交：`df1c553`，基于官方 main `d27f8e3`，包含 DCO 签署。
- 五个文件：四个运行时文件和一份 nightly 310P pytest 回归。
- 提交版本使用真实 wrapper，实机 `3 passed`（126 次输出/状态断言）；日志见 `upstream-pytest.log`。
- 官方 `bash format.sh ci` 全部通过。
- 向现有 Issue #17856 补充状态更新异常与压紧索引风险证据，并关联 PR。
- 上游相关 #18612 修改 FLA 路径，没有覆盖本次 310P 独立算子。

PR 已提交，尚未合并；产品默认服务仍使用原版本。

上游 CI 首轮 mypy 检出测试直接导入无类型声明的 C++ 扩展，已以
`709a5b7` 改为 `importlib.import_module` 注册；没有修改算子计算逻辑。
修正后 DCO、pre-commit（含 mypy）通过，CPU UT 正在运行。
Issue 补充证据：[评论链接](https://github.com/vllm-project/vllm-ascend/issues/17856#issuecomment-6166963847)。

## 自动评审跟进

Gemini 自动评审指出状态写回索引与 NV 的首个乘法可能使用 32 位算术。
`70ccf5c85` 在乘法前显式转为 uint64_t；历史读取偏移原本已使用 uint64_t。
真实设备重新编译成功，三组 pytest 通过（126 次输出/状态断言），
日志为 `review-pytest.log`。普通大小状态池验证没有覆盖理论溢出门槛，不能声称实机复现了超大索引溢出。
已回复并解决该自动评审线程；人工评审仍待完成。
以 `50fbb5a95` 合入最新上游 main `8783fc6`，没有 GDN 运行时变更或冲突。

## 官方 W8A8 路径补充验证

复用 a2 已下载的 `Eco-Tech/Qwen3.6-35B-A3B-w8a8`；索引包含 785 项 MTP 参数，
对应量化描述均为 FLOAT。不借用 Qwen3.5 MTP，也未覆盖或转换原始权重。
使用官方 `v0.23.0.post1-310p` 镜像、C16 KV，只叠加待提交的 GDN wrapper/OPP，
不加载 W4/C8/PR17640 monkeypatch。这是官方镜像集成验证，不等于整套当前 main 的运行验证。

- TP1/8K/MTP2/max_num_seqs=4：NZ 格式转换 OOM，已分配 40.89 GiB，
  额外申请 972 MiB 时只有约 863 MiB 空闲，发生在 GDN 执行前。
- TP2 同配置：每卡模型占用约 20.59 GB，加载成功；Decode Graph 启动在
  `attention_v1.update_graph_params` 遇到 `graph_params=None`，没有完成 Graph 服务验收。
- 首次 eager 启动过早，失败候选 worker 尚未退出，导致 aclInit 507899；
  清理残留容器后重新启动，不把设备争用算作 GDN 数值失败。

TP2 eager 清理重启后成功：W8A8+C16+MTP2、8K、max_num_seqs=4，
四客户端变长普通文本 8/8、严格 JSON Schema 8/8，通过答案与结构断言。
日志 `http-w8-tp2.log`、`w8-tp2-eager.log`；完整启动命令为
`w8-tp2-eager-command.json`。测试完成后恢复原 W4A8 主服务及 chip1 对照。
这是短输出冒烟验证，不覆盖长上下文、任务准确率或完整模型 Graph。
不把输出成功视为强制命中了 accepted_prev > current_len 的证明；
该边界由确定性的单算子回归覆盖。

最新上游 CI：DCO、pre-commit 和 pinned vLLM main/v0.30.0 两组 CPU UT 均通过。
`ci-gate` 未通过原因是源码改动需维护者加 `ready-precise` 等标签触发后续测试；
已经在 PR 说明并请求维护者启用，不修改或绕过官方门禁。
a2 主服务 18024、chip1 对照 18028 已恢复健康。
