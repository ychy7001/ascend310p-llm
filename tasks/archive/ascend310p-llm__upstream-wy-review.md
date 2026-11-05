# PR17640 复现包独立审查与发布

- 完成：vllm-project/vllm-ascend#17640 发布前 OPEN、未合并、head 精确 18163f6b1d208ab3b4fb06e9752225611c37f40e。
- 修复：归档身份元数据、文本数值输入与输出、独立CPU校验、固定PR源码加载/构建、英文README和观察措辞。逐项审查记录位于 results/wy921-localize-20261104/review-report.md。
- 验证：原/新manifest SHA；12文件脱敏与tar元数据；3项CPU测试；输入无损回建与回读；数值张量SHA；记录输出三方指标重算；新解压包验证；固定PR helper AST一致且CPU可执行。
- 发布：公开gist https://gist.github.com/ychy7001/b94596c3949f19012d635f97b203dd23 ，12文件远端SHA一致；唯一评论 https://github.com/vllm-project/vllm-ascend/pull/17640#issuecomment-6646514137 ，正文回读一致。
- 限制：本机验证torch2.5.1+cpu；设备观测为前一日310P/CANN9.2.1证据，本次未重复设备构建/运行，原驱动精确版本未保留；未证明模型logprob因果链。
- 本地upstream-1提交，不推送；数值数据和tar继续遵循现有忽略策略，公开gist保存完整文本包，清单锚定SHA。
