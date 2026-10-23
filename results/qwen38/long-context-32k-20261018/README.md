# Qwen3.8 TP2 / 32K 长上下文边界（2026-10-18）

本实验仅把已通过的16K TP2/EP2、W4A8 G128、C16、MTP0、Decode Graph、batch256配置的 `max_model_len` 改为32768，其他参数、权重和三个基线补丁不变。

## 启动结果

32K服务可以成功加载完整权重、完成Graph capture和3条基本生成；启动日志显示两rank物理attention blocks仍为578/584，Graph内存约0.18/0.19GiB。配置启动成功不等于32K请求能调度。

16K输入+128输出在该服务完成3次计时，TTFT约77.56–77.78s，Decode约16.45–16.50 tok/s，和16K `max_model_len`基线一致。

## 32K真实请求结果

`32256+128=32384`请求通过 `/tokenize` 后提交，服务长时间保持 `Waiting: 1 reqs`，`Running: 0 reqs`，`NPU KV cache usage: 0.0%`，未获得调度；没有输出usage。脚本已终止，避免保留悬挂请求。服务已恢复到已通过的 `qwen38-tp2-text-16k-graph` 容器。

该结果是物理容量/调度边界证据：当前 C16、batch256和compact状态布局在两颗310P上能实测16K，但不能把模型配置的262144位置或框架逻辑KV估算解释为32K可用上下文。32K服务的启动成功也不能作为产品规格。

原始证据：`long32k-bench-16384-128.jsonl`、`long32k-bench-32256-128.jsonl`、`long32k-gates.log`、`long32k-resources.jsonl`。失败请求的jsonl保留config和warmup/之前结果，但没有伪造usage。

## 当前决定

产品候选继续使用16K / batch256 / C16 / MTP0 / Decode Graph。若需超过16K，应优先研究C8或减少非专家/PLE内存、物理KV分页和更合适的长Prefill调度；简单提高 `max_model_len` 不会增加可用物理KV。
