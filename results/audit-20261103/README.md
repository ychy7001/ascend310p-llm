# 只读审计原始证据

会话日期2026-11-03。详细结论见 ../../docs/ascend310p-optimization-plan.md。

- a2-environment.json：ssh只读查询，NPU/内存/端口/容器inspect/宿主版本/模型config。
- qwen36-image-manifest.json：从停止的qwen36-image-managed容器通过podman cp读出，未启动容器。
- qwen36-installed-gdn.py：同一容器实际310P GDN源码快照，保留原版权；用于核对FP16 state和状态索引分支。
- qwen36-container-cann.info、qwen36-package-*.txt、qwen36-installed-packages.txt：实际容器CANN和包METADATA头，未导入torch。
- hyperqwen-*.json/md：GitHub API/原始文档快照；head和文档main读取为相近时刻的独立请求，未保证多请求原子快照。
- ascend-head.json / vllm-head.json：在线main查询时刻SHA。
- ascend-pr-*.json：PR状态与完整API响应；open列表按310P关键词搜索，非完整支持清单，也非全部未合并PR的源码审阅。
- sources.json：外部查询来源URL。

没有推理请求、Benchmark、Profile或本轮收益数据。宿主时间命令可能显示UTC；用户面向日期采用会话America/Toronto日期，API时间戳保留原值。
