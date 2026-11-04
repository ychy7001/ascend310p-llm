# short-4有限复核

GPU35B固定前轮镜像与权重、FP16计算/KV/SSM、MTP0、4K eager。复用前轮完整校验清单，仅复核文件大小，未重做25GB哈希。增加raw_logprobs/top20、返回tokenIDs和CPU侧调度器观测；模型/kernel未补丁，观测环境与fresh启动/盐值不同于原轮。

C1/C3各3轮，C3固定short3/4/5组成，共12请求、6目标。目标0/6满足原oracle，均三项状态待执行（预期未明确）；每轮C1/C3生成token逐项相同，原C1正确/C3错误翻转未复现。第一状态token位置54，调度记录对应C1实际1请求、C3实际3请求。raw logprob(待)-logprob(未)为0.078125～0.46875，全偏待；top20不是全分布KL，不作数值根因结案。

参考v1/adjudication.json、v1/probe/responses.jsonl和v1/scheduler.jsonl。原27B已恢复、18020健康200（v1/service-restoration.json）。没有正式性能、外部API调用或生产采用。
