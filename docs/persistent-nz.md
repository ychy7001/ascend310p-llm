# Persistent NZ 增量转换实验

## 目的

C8 attention cache 的 K/V 以 INT8 存储，原有适配路径在每次 Decode/SpecDecoding 前把当前有效页全部转换为 FP16 NZ。这个实验为每个 attention 层保留一个 NZ 工作区：首次请求或页表变化时执行 full conversion，后续新增 token 仍落在同一 64-token 子页时只执行 `c8_cache_to_nz_range`。

实验开关默认关闭：

```bash
C8_PERSISTENT_NZ=1 \
C8_PERSISTENT_NZ_ASSUME_STABLE=1 \
C8_LIBRARY=/work/artifacts/c8-transdata-build/output/libqwen36_c8.so
```

`C8_PERSISTENT_NZ_ASSUME_STABLE` 是显式实验开关。Graph 路径不使用该增量分支；跨页、chunked prefill、页表变化和非 compact attention state 回退到 full conversion。

## 为什么当前没有接入 Decode Graph

Graph capture 会固定 Python 侧的 `pages` 和 `dst_block`。因此 eager 中可以判断
“本次新增 token 仍在同一页”并调用 range kernel，Graph replay 却不能安全地把这个
host 分支改成动态分支。把 Graph 简单改成只更新末页会留下三个正确性问题：

1. 首次 capture 时 NZ 工作区的历史页还没有初始化；
2. 新请求、prefix-cache 命中后的不同物理页表或页表重排会使旧 NZ 页失效；
3. MTP verify 可能一次写入并跨越两个页，单页 range 更新会读到旧数据。

要在 Graph 中使用增量 NZ，算子需要额外的设备端状态协议：首次执行的 full/range
状态、请求或页表 generation、动态末页索引，以及跨页多页更新。状态必须在请求切换
时由调度器显式失效，并经过 prefix cache、MTP1/2、跨页和 Graph replay 的组合门禁。
在此协议实现前，Graph 保持每步 device-length full conversion 是正确但较慢的路径，
eager persistent NZ 只作为研究开关。

## 算子级证据

在真实 Ascend 310P3/CANN 9.1 上，114688 slots、GQA 2、head dim 256、block 64 的交替同步 benchmark：

| 路径 | 中位耗时 |
| --- | ---: |
| full C8 -> NZ | 3384.3 us |
| one-page range update | 162.2 us |

range kernel 只改目标 `dst_block`，输入仍使用物理 C8 cache；host wrapper 要求输出工作区覆盖目标页即可，因此 persistent workspace 仍按模型上下文申请，不复制整套物理 cache。

## 真实模型门禁

TP1、W4A8、C8、MTP1、eager、max model length 4096 的服务验证结果：

- 首次 `SpecDecoding`（2 token）执行 full update，随后同页执行 range update。
- 连续生成 128 token 返回 HTTP 200，usage 正常；日志确认 41 个 full-attention/MTP 层均出现 `mode=range`。
- 63-token prompt、128-token output 的跨页请求返回 256-token HTTP 200，无 EngineCore 崩溃；跨页由 `same_page` 条件自动回到 full。
- 覆盖的 8 个质量题和请求复用路径未出现错误。

## 端到端对照

这是 eager、TP1、MTP1、C8、短上下文的探索性数据，不是产品基准。`bench_server_phases.py` 使用独占 endpoint、unique prompt 和服务端阶段计时。

| 配置 | prompt/output | Decode |
| --- | --- | ---: |
| persistent NZ | 63/128，单次 | 6.995 tok/s |
| full conversion | 63/128，两次 | 6.964 / 6.508 tok/s |

样本量不足以证明服务级加速；当前结论是 persistent NZ 已经稳定命中并且没有观察到明显回退，但整体 Decode 仍受 MTP、attention、MoE 和 host 调度共同限制。算子级收益不能直接换算成端到端收益。

## 当前决策

默认服务继续使用 C8 device-length + Decode Graph + MTP1，persistent NZ 保持关闭。后续如要产品化，需要在同一长上下文、相同接受率和多轮请求协议下补充 P50/P95、跨页比例、页表重排和数值回归；只有端到端 TPOT 稳定改善后才应打开默认开关。

## 稳定产品基线

恢复后的 OrangePi/310P 稳定服务为 TP1、W4A8、C8、MTP1、114688 max model length、Decode Graph、6 GiB KV budget，persistent NZ=0。独占 8192/128 unique-prompt 请求的两条有效样本为：

| 请求 | Prefill | Decode | MTP acceptance |
| --- | ---: | ---: | ---: |
| 1 | 567.11 tok/s | 24.15 tok/s | 100% |
| 2 | 566.22 tok/s | 24.17 tok/s | 100% |

日志确认实际出现 `Replaying aclgraph`，没有加载 persistent NZ range 路径。原始阶段数据保存在 `results/persistent-nz-20261015/stable-114k-p8192-o128.jsonl`；该文件只有两条有效测量，不将中断的第三次请求计入统计。
