# MoE 少 token tiling 实验（4b-①）

**当前状态：按追加决定修复桥接库链接并继续原实验。** 首轮 B 因动态依赖装载失败，曾按停止规则收束；该故障现按构建链接遗漏处理，不作为实验结论。修复用显式 `LINKER:-rpath,$ORIGIN`，实际 ELF 已含 `RPATH=[$ORIGIN]`；原设备库 SHA 未变。新工具 `tools/check_moe_bridge_load.py` 启动全新 Python 进程，仅导入 Torch/Torch NPU 后直接加载桥接库，禁止依赖预加载或用桥接目录设置 `LD_LIBRARY_PATH`；微基准启动时也先执行此项。持锁生产环境复测及新同轮 G2 正在排队，后续仍依赖 G2 通过。


本项仅调整 `QuantGroupedMatmulDequant` 小 M 路径的调度，不叠加 QKVZ 量化，不改变专家权重、scale、路由、GEMV 算术或通信。基线提交 `74f7202`。

候选将各专家内部重复进行的动态 token scale 计算提前至专家循环前，计算所有输入行后仅同步一次；后面的逐专家 GEMV 继续使用原始源码，按原行号读取 scale。候选通过 Torch PrivateUse1 算子和 `OpCommand::RunOpApiV2` 发射，输出与 workspace 在调用内分配，并记录 stream 生命周期。仅面向总 M=8/16/24/32、每专家至多 8 行、K=256/2048、FP32 权重 scale 的实验。已准备隔离 worker 接入；B 未初始化成功。

`tools/prepare_moe_prescale_310.py` 从调用者提供的生产 SDK 源码生成隔离构建目录，记录输入头文件 SHA256；仓库不分发 SDK 源码。`tools/bench_moe_prescale_310.py` 先核验官方算子、自编译原调度、候选三方，再对照 CPU FP32 量化整数点积参考，并检查候选逐位重复；正式微基准使用真实激活、256 专家 bank 与累计分组。无 Profile 的 eager/Graph 计时分开，每档 3 对 AB/BA、每臂 256 次调用。CPU 参考容差在真实样本前固定为 `rtol=0.002, atol=0.002`，所有元素均须满足；自编译对照须与官方逐位一致，候选须逐位重复且为有限值；候选对官方若不逐位一致则要求完整 G1。

第一轮合成冒烟覆盖 8 案例（M 四档 × K 两类），真实 N 分别为 512、2048，稀疏非均匀分组为 7 个非空专家、每专家至多 2/4/6/8 行。全部自编译对照和候选均与官方算子逐位一致，候选各重复 3 次一致，A/B 最大绝对差为 0；参考与候选均为有限值。CPU FP32 最大绝对误差范围为 0.003858–0.008996，A/B 相同。这次冒烟尚未应用上述正式容差计数，不能直接标作完整 G0 通过。

`experimental/moe_small_m/capture.py` 是隔离副本的 eager 观测钩子，只采真实输入、权重和分组用于算子回放。观测执行方式不用于性能结论。原始张量、完整日志和机器信息仅放忽略的原始目录。

真实采集完成：双 rank、全部 40 层共 2,904 条路由观测，12 轮/30 请求、122,880 输入和 7,680 输出 token，所有输出均为 256 token。四档非空专家数中位数为 8/16/23/28，最大每专家行数为 1/2/3/4；全部累计边界等于 M。首层每 rank 各保留两类投影的真实 256 专家 bank，以及每档三请求、每请求前三步的输入。

首次真实微基准发现捕获前固定 stream 导致候选 Graph 为空；该批计时全部作废。已改为在调用时读取当前 stream，并要求修改输入后的 Graph replay 与该臂的新调用逐位一致。`tools/adjudicate_moe_prescale_310.py` 拒绝未经该核验的数据，并检查双 rank bank、每档每投影至少三请求×三稳定输入。有效重测完成：144/144 案例均与官方和自编译对照逐位一致，432/432 次重复通过，144/144 Graph 输入变更核验通过，CPU FP32 容差外元素为 0。本轮算子收益折算为会议回放纯 Decode 转移降幅 **6.4965%**，达到 3% 理想进入门槛；随后进入整模型 G2；A 完成，B 启动失败，因此没有进入正式回放。按逐位一致规则免做 G1。尚未测会议回放、RAG 8K–32K Prefill、通用 1K–32K×C1/C4，WY Prefill 不在本项中改动。未报告实测收益或非劣结论。

生成和构建命令（在匹配的生产 SDK 镜像内；CPU 构建不挂载设备）：

```sh
python3 tools/prepare_moe_prescale_310.py --source "$CANN_OP_SOURCE" --output "$BUILD_SOURCE"
cmake -S "$BUILD_SOURCE" -B "$BUILD_DIR" -DSOC_VERSION=Ascend310P3 -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j1
```

算子命令须在已持 `flock /tmp/a310-npu.lock` 的独占窗口内执行，退出前恢复原服务并用新 SSH 独立确认 health=200：

```sh
python3 tools/bench_moe_prescale_310.py --library "$BUILD_DIR/libmoe_prescale.so" --snapshots "$CAPTURE_DIR" --output results/moe-small-m-20261105/raw/micro.json
python3 tools/adjudicate_moe_prescale_310.py --micro results/moe-small-m-20261105/raw/micro.json --attribution "$ATTRIBUTION_DIR" --output results/moe-small-m-20261105/adjudication.json
python3 -m unittest discover -s tools -p test_adjudicate_moe_prescale_310.py -v
```

| 档位 | 采用计时 | 两投影合计加速（跨 rank 成本比均值） | 全回放降幅贡献（推算） |
|---|---|---:|---:|
| B1 | graph | 1.7139× | 3.8291% |
| B2 | graph | 1.6048× | 1.2426% |
| B3 | eager | 1.6064× | 0.2657% |
| B4 | eager | 1.6401× | 1.1590% |

上述 6.4965% 为冻结归因池的理想转移估计，未包含 Prefill/混合阶段、固定到达与模型集成开销；不是端到端验收结果。真实单算子只重放每 rank 首个专家层，其余 39 层仅留路由分布。


整模型 A 同轮结果：GSM64 **63/64**（1 题截断且缺严格答案），16 事实 **16/16**，结构字段 C1/C3 各 **13/13**，前后短题各 **8/8**。26 条结构输出 usage/tokenization 全部匹配，覆盖 4095/4096/4097 边界、16K/32K；前后 A/B/A 复用文本一致且无 canary 泄漏。B 在两个 rank 的 worker 初始化时均因上述动态依赖装载失败，实际完成门禁请求 **0**。这是模型接入失败，不能从中判定 B 的任务精度。

| 负载画像 | 实测模型收益 | 当前判定 |
|---|---|---|
| 会议回放 | 未运行；5 对 AB/BA 完成 0 对 | 单算子转移预算 6.4965%，不是端到端收益 |
| RAG 8K–32K Prefill | 未运行 | 未验证非劣 |
| 通用 1K–32K × C1/C4 | 未运行 | 未验证非劣 |
| WY Prefill | 未运行；本项不改 WY | 无新增收益结论 |

`experimental/moe_small_m/worker.py` 保留隔离接入草案；`tools/replay_frozen_arrivals.py`、`tools/adjudicate_moe_model.py`、`tools/bench_moe_load_profiles.py` 保留已冻结的后续测量工具，尚未在本候选的完整模型上完成性能验证。固定到达 HTTP stub 证明慢请求不会延后下个冻结到达；回放裁决硬校验冻结计划 SHA、11 请求、200,895 输入、17,045 输出、prefix 零命中和阶段闭合。模型收益及 TTFT/TPOT 采用判定测试 4/4、单算子裁决测试 7/7、HTTP 联调测试 1/1，共 **12/12** 通过：

```sh
python3 -m unittest discover -s tools -p 'test_*moe*.py' -v
python3 -m unittest discover -s tools -p test_replay_frozen_arrivals.py -v
```

本次未运行候选模型双 rank 的 64–128 稳定步 Profile，也未做采用前人工会议语义审阅。冻结 5 对回放和 14 个画像的工具与门槛已经保留，但没有任何模型性能采用结论。原始 G2 输出、错误日志、ELF 依赖和所有窗口恢复证据只保留在忽略目录。

G2 独占窗口于 2026-11-05 01:30:15 UTC 开始，原配置服务于 01:50:46 UTC 恢复；新 SSH 于 **01:51:00 UTC 独立确认 health=200**，01:51:01 UTC 放锁，共 **20 分 46 秒**。随后其他实验线程获得共享锁；未干预其窗口。全程未启动被禁用的图片服务。无推送、无合并、无其他仓库改动。

追加只读 journal 核查：首轮原服务由本实验恢复控制器于 2026-11-05 **01:47:21 UTC** 发起启动，01:50:45 应用启动完成，01:50:46 健康，01:51:00 新 SSH 确认健康。01:51:01 journal 先记录外部 `systemctl stop` 和 systemd `Stopping`，随后 EngineCore 收到 SIGTERM；01:51:05 为 `Deactivated successfully`。与下一 WY 持锁窗口及其控制器停服步骤相符，没有自行崩溃证据。WY 窗口随后于 **02:39:12 UTC** 执行启动，**02:42:46 UTC** 记录应用启动完成；接续图片实验取得锁后于 **02:43:00 UTC** 再次主动停服。本任务没有干预其他窗口。

执行偏差记录：追加修复时一次挂载设备的纯装载探针未包共享 flock，未生成模型请求；已向用户说明，该次结果不计门禁。新增冷进程检查在 G2 持锁窗口重做。
