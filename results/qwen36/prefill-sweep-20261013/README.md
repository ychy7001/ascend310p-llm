# TP1 Prefill 参数对照（2026-10-13）

a2 单颗310P3，TP1/W4A8，FP16 KV，MTP关闭，eager，prefix cache关闭，maxlen4096，memory=.90，max-seqs1。3072输入/1输出，warmup1后三轮，阶段Prometheus计数差分；每组独立3072-token needle通过。顺序扫描而非交替复测；结果不是C8/MTP/112K产品档性能。

| batched tokens / expert chunk | Prefill中位 tok/s | 相对1024/512 |
| --- | ---: | ---: |
| 1024/512 | 379.83 | 基线 |
| 1024/640 | 379.12 | −0.19% |
| 4096/512 | 375.00 | −1.27% |
| 4096/640 | 384.10 | +1.12% |

结论：这轮参数调整未产生明显收益，维持产品档1024/512。优先优化W4 grouped MatMul及路由，再在真实112K/C8/MTP/Graph配置验证。

源码：275dab9516e2fc3c3b672f14e65337d51a3a6548。
镜像ID：30724037ccd4509c1aaea06a11334b26b58534303f6a6f5d68fefc6316f683be。
W4 extension SHA256：a8ac855968ddc3a56db676c1ac8dd9a8657dbae2fffb28c4276ddb7fef4b5950。
C8 library SHA256：ee91aac12582a05f1581580d2e1499a2446bf698d202a62c24641c6edb084755（本轮未启用C8）。

复现：配置MODEL_DIR/OPP_DIR/EXTENSION/C8_LIBRARY/RESULT_DIR，执行 `PAIRS="1024:512 1024:640 4096:512 4096:640" bash experimental/run-prefill-sweep.sh`。OPP_DIR指向含bin/set_env.bash的vendors/custom_transformer目录。结果必须写新目录以保留旧证据。原始a2路径：`/home/ychy/ascend-lab-runtime/optimization-sweeps/prefill-20261013-final/`。
