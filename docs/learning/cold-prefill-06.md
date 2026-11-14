# 310P eager Prefill：一次 forward 内复用 SplitFuse mask

2026-11-14，Atlas 300I DUO `ychy-a2`，同一验收镜像 `bb0fa66f…` 和只读 W8A8 模型。两臂用 TP2、batch 2048、prefix caching、MTP2、FULL_DECODE_ONLY，使用相同入口和实际 CLI，普通计时不采样。先跑候选 B，再停容器、冷却 60 秒、启动基线 A。两臂挂同样的实验文件，只用 `QWEN36_EAGER_MASK_REUSE=1/0` 切换。实际计时窗口是 a2 的 `~/ascend-lab-runtime/prefill-mask-reuse-310p-20261114`。

[上一轮拆测](cold-prefill-05.md)看到同一个 1920-token 分段重复构造 mask。此次将复用作用域放在 `NPUModelRunner310._model_forward`，按设备、FP16 FRACTAL_NZ、完整有序 query 起点和 context 长度匹配，最多保留一份。Graph 捕获、Graph 重放和 `SpecDecoding` 仍走原路径；eager prefill 沿用原函数构造首次 mask。退出 forward 后清理，包含异常路径。第一次挂到通用 `NPUModelRunner`，被 310P 的覆写绕过；该窗口没有复用命中，保存日志但不计入本轮性能。

仍从 NPU 读取 context 长度作为匹配依据：异步 MTP 模式下这些长度由设备校正，宿主乐观长度可能包含之后被拒绝的 draft token。复用 mask 节省的是重复的矩阵构造和布局转换，候选仍有逐层长度读取，分页算子本身不变。

冷单流：输入 token ID 固定，每次用独立 cache salt，先做一次预热，再测三次，每次只生成 1 个 token。两档、两臂均确认实际 prefix hit 为 0，且同档输出文本一致。

| 冷输入 | 首 token 延迟中位数 A → B | 服务端 Prefill 吞吐中位数 A → B |
|---|---:|---:|
| 4096 | 2.142 → 2.134 s（−0.37%） | 1916.6 → 1923.8 tok/s（+0.37%） |
| 32768 | 22.521 → 20.942 s（−7.01%） | 1456.1 → 1566.1 tok/s（+7.55%） |

混合回放使用与[上一轮](cold-prefill-04.md)相同的 4 路输入和固定到达偏移，各跑三轮。每轮 priming 为冷；两路暖请求分别实际命中 11520 token，两路冷请求命中 0。聚合吞吐中位数 **17.70 → 18.72 tok/s（+5.78%）**。

| 请求 | 首 token 延迟中位数 A → B | 每请求最长输出间隔的中位数 A → B | 同轮完整文本哈希相同 |
|---|---:|---:|---:|
| 暖跟进 14K | 1.945 → 1.838 s | 1.869 → 1.694 s | 2/3 |
| 冷长输入 32K | 29.367 → 27.071 s | 1.518 → 1.328 s | 2/3 |
| 冷短输入 1K | 13.842 → 12.887 s | 1.869 → 1.694 s | 1/3 |
| 暖提取 15K | 30.359 → 27.815 s | 0.261 → 0.265 s | 0/3 |

两臂 32K 针检索均返回 `ASCEND732941`，短 JSON 在计时前后均通过。两臂均实际捕获并重放 decode Graph；候选两 rank 均记录 `starts=(0,1280), lengths=(3200,)` 的真实复用，基线没有。报告 KV 容量为 632375 → 631754 token（−0.10%，单次进程窗口，未归因）。

混合回放只有 5/12 对完整文本哈希一致，且基线、候选各自内部也出现跨轮变化。固定的是 HTTP 到达顺序，实际 scheduler batch/shape 未锁定，因此这些数据不能区分批次变化与候选影响，也不能证明混合生成逐字等价。下一层质量验证应锁定真实调度步形状，捕获第一处分歧的 logits 或逐 token logprobs，做同一批次的 fresh/reused 对照。单机每臂一个窗口、三轮合成负载不代表线上 P95，也未测 262K 上限、多模态或正式质量集。

独立 NPU 数值门禁比较 4 个布局：长历史、前缀命中形状、三路混合及三路换序；每个布局换两组随机 query。使用相同 KV 页表、相同实际 legacy SplitFuse 算子，fresh/reused mask 的完整 FP16 输出 8/8 逐位一致，与 CPU FP32 参考抽样的最大绝对误差为 `2.36e-4`。因此保留[可选实验补丁](../../experimental/prefill-mask-reuse/README.md)，生产默认仍为 2048 和原 mask 路径，等固定调度质量对照后再评估启用。

[结构化结果](../../results/prefill-mask-reuse-20261114/summary.json)。原始协议、HTTP 记录、设备日志和数值输出保存在 a2 实验目录；本机副本位于 `artifacts/prefill-mask-reuse-20261114/raw/`。服务与数值窗口都退出 0；独立 SSH 核对 rootless/rootful 无运行容器、测试端口和设备空闲、锁可重新获取。a1 未操作。

后续[精度门禁](cold-prefill-07.md)通过后，已将复用封装发布为[内部双卡默认交付](../qwen36/w8a8-wy-delivery.md)。
