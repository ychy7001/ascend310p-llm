# PDF OCR Torch基线冒烟（2026-11-07）

用户暂未提供票据图片和字段答案，授权使用现有5份PDF。本次每份取1页转PNG，原Torch WY + 既有paired H/O，未加载任何WY候选。输入manifest SHA `120bd8af0f6edb0fd866d55e69ad3f0e4a04a6a30481b03df2e5532816baeaf7`；原Torch模块SHA `4f10c88e13739df03544c594f0cf49b9762c3d3d1f567348767b52e73a49126f`；完整image/worker/template SHA见freeze和launch-manifest-identity。

3000秒预算，实际315.03秒（含模型启动与清场），ExitCode 0、cleanup_passed=true；5/5请求完成，4次正常stop、1次length。五次prompt cached_tokens均为0。视觉入口可用，不代表OCR质量通过。

| case | 输入token/图像token | 首内容秒 | 输出token | 结束 |
| --- | --- | --- | --- | --- |
| pdf-00 | 1941/1900 | 3.084 | 3072 | length |
| pdf-01 | 1941/1900 | 2.819 | 598 | stop |
| pdf-02 | 1166/1125 | 1.724 | 84 | stop |
| pdf-03 | 1166/1125 | 1.641 | 84 | stop |
| pdf-04 | 1166/1125 | 1.638 | 55 | stop |

每页仅一次、无AB/重复，也包含视觉编码和解码开销；这些时延只能描述本次请求，不能用于WY加速或生产性能结论。

## 真实质量缺口和参考限制

目录页pdf-00在第一行反复生成引导点并耗尽3072 token，后续内容未识别；不以单纯“输出上限不足”解释。目视核对SSD页pdf-03，其正文大体识别，但右侧图中的Category prediction、Bounding box prediction、Anchor box等英文标签漏识。该基线没有加载native WY，不能把这类缺口归因于本轮WY候选。

PDF文本层本身有字体编码/阅读顺序问题：部分中文在文本层变为“光/桌”等字，图中文字与正文的顺序也不同。auxiliary-text-layer-metrics中的CER仅是两段字符串的距离，不能作为OCR真实错误率，不能按此排序页面质量。相对干净的代码正文页pdf-01辅助CER约2.09%，仍不是人工字段正确率。

本次未进行重试、提示词调参、阈值修改或票据字段验收。原始图片、模型文本、SSE和日志仅保留在私有任务证据。后续必须把基线退化与WY增量误差分开评估；真实票据字段质量仍需样本与人工答案。本次不扩大OCR设备窗口。
