# C8 + MTP2 + Graph：4096/640 对照（2026-10-13）

a2 单颗 310P3，TP1/W4A8/C8/MTP2，`max_model_len=114688`，`.95`，Decode Graph，device-length，prefix cache关闭；仅把产品候选的 `max_num_batched_tokens/grouped_chunk_tokens` 从 `1024/512` 改为 `4096/640`。使用正确的 `c8-device-length-build/output/libqwen36_c8.so`（需要第9个 `context_length` 参数）。

| 输入/输出 | Prefill 中位 tok/s | Decode 中位 tok/s |
| --- | ---: | ---: |
| 8192/128 | 370.58 | 36.89 |
| 32256/256 | 339.51 | 28.80 |

每项 warmup 后三轮，unique prompt，计数有效。服务启动和 Graph capture 通过，Graph 约 0.31 GiB。原始日志中包含真实启动参数和 ABI 相关信息。

结论：`4096/640` 在当前产品配置下没有相对 `1024/512` 的稳定、可归因收益，保留为实验覆盖项，不改变产品默认。它也没有改变 TP1 Prefill 约 340～371 tok/s 的量级。

注意：`fused-write-build-v2` 只有 8 参数 `c8_cache_to_nz` ABI，不能与 device-length Graph adapter 一起使用；误配会在 Graph capture 报 `expected at most 8 ... received 9`。device-length 路径必须绑定 `c8-device-length-build` 产物。
