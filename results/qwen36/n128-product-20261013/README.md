# N=128 专家 Prefill：112K C8/MTP2/Graph 组合门禁

日期：2026-10-13。源代码：`ac5f191`，另加本轮验收脚本。
硬件：a2 第一颗 Ascend 310P3，TP1，48GB 级内存。
隔离服务：`qwen36-n128-product`，端口18023，第二颗芯片未使用。

## 固定条件与制品

使用 `experimental/run-c8-112k-product.sh`，`QWEN36_PROFILE=mtp2`：
W4A8 routed experts、浮点 MTP、C8 FP16 scales、fused-write、split-by-type、
device-length、MTP2、FULL_DECODE_ONLY、capture `[1,2,3]`、mode=0、
诊断性 updatable-graph 修复、max length114688、利用率.95、maxseq1、
batch1024、专家chunk512、prefix off。镜像为
`quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`。

- N=128 内核源 SHA256：`8727515fba385b37604faa35715d59d2374017d3105a05d71b99250f0f4e7350`。
- C8 库 SHA256：`85f383cfab4e1f5595394bb1317febfcc65b7b04815c55e8a39ab8b94c3f05e1`。
- W4 扩展 SHA256：`a8ac855968ddc3a56db676c1ac8dd9a8657dbae2fffb28c4276ddb7fef4b5950`。

服务加载27.2880GiB权重，C8分配116 blocks、4336091136 bytes，
预留scratch234881024 bytes，Graph约.31GiB。vLLM的186236-token KV
报告是调度估算，不是通过请求验证的上下文长度。

## 已完成结果

`check_cache_request_reuse.py`：严格答案8/8，A/B/A文本一致，无标记泄漏。
阶段计时：每档warmup1、unique prompt、三轮、独占counter，所有usage有效。

| 输入/输出 | Prefill中位tok/s | Decode中位tok/s |
| --- | ---: | ---: |
| 8192/128 | 391.5909 | 37.0124 |
| 32256/128 | 357.5768 | 29.0453 |

这些是服务端请求阶段计时，不是纯内核时间或可靠P95。重复文本有利于
MTP接受，不能外推业务平均吞吐。历史同档N=64配置不是本轮匹配A/B，
不能把两者差值认定为N=128的确定收益。

112K检索通过：输入114560、实际输出14、总长度114574，415.3063秒，
正确返回`ASCEND732941`，finish reason为stop。该结果证明本候选组合
完成了这一实际长请求；单个检索样本不代表正式长上下文精度评测。
总耗时包含Prefill和生成，不能以输入数除总时间当作阶段Prefill吞吐。

## 复现

设置独立制品路径后以 `QWEN36_PROFILE=mtp2` 启动上述产品脚本，健康
检查通过后，在独占端点运行：

```bash
BASE_URL=http://127.0.0.1:18023 \
RESULT_DIR=/absolute/path/new-results \
bash experimental/check-n128-product.sh
```

该脚本失败会保留服务与证据；测试结束后显式停止自己启动的容器。
原始JSONL含prompt IDs、响应、usage和阶段counter；metrics包含MTP统计。
