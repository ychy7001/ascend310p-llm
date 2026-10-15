# RC MTP2 block128：设备端精确slot算术

基于上一轮CPU精确算术候选，改为设备端FP32缩放/floor后转回int32，
余数使用`position - quotient * 128`。不使用整数FloorDiv/FloorMod或CPU复制。
仅支持block128、int32、max_model_len<=1048576；代码显式拒绝其他配置。
该位置域所有整数均可被FP32精确表示，除以128为二进制精确缩放，
floor后转int32与有符号整数向下取整除法一致，余数仍为整数运算。

## 真实RC独立探针

`probe_device_indices.py`穷举[-1048576,1048576]共2097153个位置，
商/余数与CPU整数//和%逐项完全匹配；batch1/2/4/8的Graph改变输入replay均通过。
见`probe.log`。未加载权重，原服务停止后独占NPU执行，探针正常退出。
首次探针因清空PYTHONPATH晚于CANN set_env导致tbe缺失；修正环境加载顺序后
通过，首次日志保留`probe-env-failed.log`，不视为算术失败。

## 完整模型候选

`qwen36-mtp2-slot-device-32k-20261015`仅挂载替换proposer文件，保留
accepted-token快照补丁，target Graph/draft eager、C16、GDN/MindIEQ、同步调度、
max32768、KV3GiB、batch2048。差异见`proposer-slot-device.patch`。
先短质量检查，再32K冷请求。通过后才尝试draft Graph及112K。
本轮不升级驱动、不替换默认源码，结束恢复MTP1并复验。

设备版target Graph/draft eager短题8/8、A/B/A及32K后质量均通过。
32256/128冷请求warmup一次+计时两次，prefix hit0：

| 轮次 | Prefill tok/s | Decode tok/s | draft接受率 |
|---|---:|---:|---:|
| 1 | 506.52 | 20.97 | 78.0% |
| 2 | 508.07 | 21.73 | 83.3% |

两次输入哈希和完整输出文本均与上一轮CPU版本一致，接受数量也相同。
性能接近CPU版，不能宣称去掉复制带来明确提速。日志确认target Graph replay。
原始结果见`bench-p32256.jsonl`、`quality-before.json`、`quality-after-32k.json`。

后续独立候选`qwen36-mtp2-slot-device-fullgraph-32k-20261015`只改
`MTP_DRAFT_EAGER=0`，先重新短质量检查，再决定是否测试吞吐。
