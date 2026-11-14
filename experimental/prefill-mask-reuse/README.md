# 310P mask 复用：历史验证工具

复用实现已正式交付，唯一源码为 `runtime/wy_compat/eager_mask_reuse.py`；本目录保留历史数值探针、回放与测试，`eager_mask_reuse.py` 仅转接正式源码。[当前交付](../../docs/qwen36/w8a8-wy-delivery.md)。

这些探针针对 300I DUO 上的 Qwen3.6-35B-A3B-w8a8 验收镜像 `bb0fa66f…`。由 `QWEN36_EAGER_MASK_REUSE=1` 开启，只复用同一次 target `NPUModelRunner310._model_forward` 中、同设备、同 FP16 FRACTAL_NZ 布局、且有序 query 起点与 context 长度完全一致的最后一份 legacy SplitFuse mask。作用域在 forward 返回或抛异常时释放；Graph capture/replay 和 `SpecDecoding` 继续走原路径。

本地检查：`python3 -m unittest discover -s experimental/prefill-mask-reuse -v`。

在实验镜像中，将 `eager_mask_reuse.py` 只读挂到 `/opt/qwen36/wy-compat/`，再只读覆盖该目录下的 `sitecustomize.py`。从实际镜像提取原 `sitecustomize.py`，只在 `graph_safe_splitfuse.install()` 后、现有 `try` 内加入：

```python
if os.environ.get('QWEN36_EAGER_MASK_REUSE') == '1':
    import eager_mask_reuse
    eager_mask_reuse.install()
```

对照臂使用同样的挂载，设置 `QWEN36_EAGER_MASK_REUSE=0`。保持镜像原启动入口、实际 CLI、模型和 NPU 不变。候选服务日志会在每个 rank 首次复用较长 query 的 mask 时输出长度序列。`numeric_gate.py` 在相同镜像中用实际 legacy SplitFuse 算子比较 fresh/reused mask 的完整 FP16 输出，并抽样与 CPU FP32 对照；它使用一颗空闲 310P，需加载镜像自带 CANN 环境。

`mask_same_batch_audit.py` 是服务侧质量探针：从镜像的 `sitecustomize.py` 调用其 `install()`（代替直接调用候选的 `install()`），仍设置 `QWEN36_EAGER_MASK_REUSE=1`。将该文件挂到 `/opt/qwen36/wy-compat/`，并在 `/mask-quality-control` 只读挂载控制目录，启动前写入 `mode` 为 `fresh`。`quality_replay.py` 与仓库 `tools/bench_http_stream.py`、`tools/bench_server_phases.py` 一起运行，通过控制目录交替切换 `fresh/reuse`，最后用 `audit` 在同一个实际 SplitFuse 调用上比较完整输出。审计会重复执行算子并同步，仅用于数值验证。`summarize_quality.py` 汇总服务日志中的实际调度形状、双 rank 对照，以及 HTTP 首个 token 分歧和候选 logprob。

质量门禁通过后的内部镜像封装见 [`deploy/w8a8-tp2-wy-mtp/mask-reuse`](../../deploy/w8a8-tp2-wy-mtp/mask-reuse/README.md)。WY 为内部默认路线；后续收益以同一 WY 基线评估，不重复进行 WY/main 选型。
