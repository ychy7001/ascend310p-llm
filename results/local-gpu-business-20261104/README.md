# GPU27B业务与字段诊断（2026-11-04）

最终not_accepted_business_semantics，见adjudication.json。现有3090测试实例，W4A16/FP8 KV/MTP3，未在本轮重启或改参数。

首/末短题各8/8，ABA一致且无canary泄漏；事实16/16；原业务自动28/28（20会议格式/完成、3tool、5schema）。20会议已全部按预先冻结准则人工阅读，会议4/5仍补写已完成，但角色与期限正确；会议18全文保留关系，行动项顶栏有结构损失。其他推断单列review_required，不凭自动通过或轻微措辞差异判全部语义合格/不合格。

v1已完成短题/事实/业务，用户中断导致字段阶段仅11条，无汇总；原始文件与过时status保留，interruption-record.json解释实际状态。resumed-v1单独完成26/26字段（C1/C3、4095/4096/4097/16K/32K），13种chat token ID哈希全部与原35B冻结fixture一致，usage计数无失配，结束短题8/8。复用缓存仅做质量诊断，不报告性能。

28条业务请求除model字段外与原35B生产回放完全相同。跨模型/精度/模板/运行栈的行为对照不等价于高精度参考或模型能力排名。修改字段工具仅增加评估端点model别名；预期、字段schema、token一致性和评分未改变。工具/fixture SHA核对通过。

GPU35B已完成隔离筛选，原27B已恢复并独立健康确认。量化parser预检与下载清单在本目录根部；无权重进入仓库。第一次35B下载准备因索引引用被省略的MTP文件安全停止，v2完整保留索引所需MTP文件，执行仍计划MTP0。不能将parser成功或下载完成写成模型启动成功。


## GPU35B实测与GPU27B对照

GPU35B：固定GPTQ Int4/g128制品（完整来源摘要及7分片124611张量索引检查通过），3090 TP1、FP16计算/KV/SSM、MTP0、4K eager。实际Marlin MoE、Triton/FLA GDN，权重20.27GiB、KV池1.8GiB。前后短题8/8，事实16/16，业务自动28/28，原8短字段×C1/C3为15/16。8种token哈希匹配、usage0失配；C3 short-4将无人确认进度的三状态写待执行而非未明确，同输入C1正确，无重复因果验证。没有长字段或正式性能请求。

20会议均已人工审阅。会议4回复/修复的周明和刘洋错配、补已回复；会议5补已执行停用且遗漏对应行动项；会议18全文关系保留但行动项结构丢失。最终not_accepted_business_semantics_and_structured_field，见gpu35b-v1/adjudication.json。28请求除model外与原NPU35生产严格相同，文字完全一致比例不当作数值指标。

两种GPU部署都在原自由文本出现完成状态问题；不能将其认定为310P独有，仍不能单独归因基础模型或量化。GPU27B为W4A16/FP8 KV/MTP3、GPU35B为GPTQ W4/FP16 KV/MTP0，模型/制品/模板/TP/运行栈不同，无高精度权重参考，不生成模型能力排名或直接可部署的310P W4 profile。

小范围提示词策略对照（3会议×2提示×2栈）部分改善，但GPU27B会议5出现长规则分析，NPU35B会议5仍补写已停用，策略整体不采用，原失败不覆盖，见prompt-policy-adjudication.json。

收尾：实验35B容器停止；原GPU_SERVICE同ID/镜像恢复，18020健康200且served model为qwen3.8-27b；3080原两个PID仍运行，a2原服务active/18081健康200，a1未操作。详见gpu35b-v1/independent-closeout.json。冻结脚本/工具/fixture SHA复核与Python编译检查通过。无提交/发布/PR/宿主CANN安装。
