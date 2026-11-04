# 本机GPU现状与27B现有服务对照

2026-11-04，只读核对本机GPU_HOST，复用已运行接口做18条短诊断，未停止/重启服务或另起GPU模型。

- RTX3090报告24GiB，已用约22.73GiB，可用显存约0.83GiB；已有27B vLLM服务，18020健康200。
- RTX3080本机报告20GiB，已用约12.60GiB，可用显存约6.98GiB；已有Guard0.6B和4B服务。
- 两卡采样瞬时利用率0%不代表显存空闲。详细现场读数、PID和配置见audit.json，尾部快照见health-after.json。
- 现有27B是W4A16 AutoRound制品（主Linear/lm_head4bit、embedding8bit），FP8 KV、MTP3，非FP16/BF16参考，不能与310P W8A8/MTP0作单精度因果或吞吐结论。

同一诊断18条除model字段外严格一致：GPU18/18、310P15/18。原排序GPU3/3为2,7,9，310P3/3为279。其他五类均15/15。不是会议业务质量排名，也不证明W4比W8质量好。

两个制品的tokenizer/模板文件SHA不同；但原排序请求经GPU/tokenize和NPU原制品离线AutoTokenizer核对，实际37个chat token ID逐项完全相同。该反例不能继续笼统归因模板输入不同；生成侧量化制品、运行栈、KV/MTP及基础checkpoint来源仍不同或待核对。原短题门禁未通过的结论保持。

CPU离线tokenizer在无NPU容器、USE_TORCH=0下完成；首次序列化BatchEncoding失败后显式提取input_ids，原问题摘要在comparison.json。没有模型推理或宿主修改。

下一步先复用现有GPU端点补原会议语义诊断，保留完整量化/模板/推理参数身份；严格归因需要统一MTP/KV并核对底模与量化参考，目前不为此打断本机现有服务。a1未访问，a2原服务保持运行。
