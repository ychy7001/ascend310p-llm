# orangepi RC：MTP 对照与 C8 元数据修复（2026-10-15）

本目录仅对应 `orangepi` 的单颗 310P1/RC，不对应 `ychy-a2` 的 EP。

## 固定条件

TP1、W4A8、C8 device-length/fused-write/split-by-type、主模型 Decode Graph、GDN、MindIEQ、Prefix Cache 开启、同步调度、max_model_len=114688、KV6GiB、batch2048、max_num_seqs=1。RC 为共享内存环境，当前 gpu_memory_utilization=0.45，不能照搬 EP 的 0.90/0.95。

MTP1 对照为 [stage1](../orangepi-rc-baseline-20261015/README.md)：8K/128 Prefill/Decode 中位数563.62/23.87 tok/s；32K/128为511.15/19.29 tok/s。

## MTP2：无有效性能结果

MTP2 + Graph 能启动并通过 capture，但首个8192/32预热请求没有返回任何文本或最终usage。约913秒后人工停止候选容器，HTTP流关闭；原始JSONL中空白warmup不代表成功。随后读取metrics连接失败，因此没有任何有效计时样本。

在停止前，两次抓取均定位到 `take_draft_token_ids -> _get_draft_token_ids_cpu -> Event.synchronize -> aclrtSynchronizeEventImpl`。这是观测到的等待位置，不能据此认定event本身、D2H拷贝或某个上游kernel是根因。日志和调用栈保留在本目录。

复核10月14日记录确认，MTP2 draft eager历史上也曾在相同等待处伴随设备TS/AI CPU异常。本轮重复启动的draft-eager候选在服务就绪前主动终止，没有提交新的推理结论。历史证据见 [RC部署记录](../orangepi-deployment-20261014/README.md)。

## MTP0：修复后通过capture、最小正确性与重复计时

原始MTP0 Graph capture失败信息：`device context length requires selected pages and one contiguous INT32 value`。该次容器日志未单独保存，不视为完整故障快照。

当前镜像源码表明：父metadata builder优先使用CPU长度；310P子builder只对SpecDecoding/ChunkedPrefill绑定 `common_attn_metadata.seq_lens` 设备view，DecodeOnly提前返回。因此C8转换在原生PA执行设备拷贝之前收到CPU长度。

修复限制在显式C8 Graph + device-length路径：在builder的DecodeOnly分支绑定common metadata的设备view，保持更新发生在Graph forward外，并让C8转换/原生PA读取同一块长度buffer。不在capture中新增CPU到NPU拷贝，不更改MTP1/2的SpecDecoding路径。

实机Decode Graph capture通过（日志记录约2秒/0.37GiB）；固定答案8/8、A/B/A文本一致且无canary泄漏。8K/32K各3轮计时均有效，prompt SHA与MTP1基线逐项一致，Prefix Cache命中均为零。这些是最小正确性门禁，不是正式精度评测。`mtp0-device-source-snapshot.txt`记录驱动、CANN、源码SHA和库SHA256。

## 基准完整性

本次中断暴露了HTTP流被关闭时预热可返回空结果的问题。基准工具现要求最终usage与请求的prompt/output token数一致；断流/不足输出直接失败。`python3 -m unittest discover -s tools -p test_bench_http_stream.py`覆盖正常usage但无DONE、中途断流和输出不足。

## MTP0 / MTP1 同协议结果

|输入/输出|MTP0 Prefill|MTP1 Prefill|MTP0 Decode|MTP1 Decode|MTP1 Decode变化|
|---|---:|---:|---:|---:|---:|
|8192/128|555.78|563.62|17.66|23.87|+35.18%|
|32768/128|522.51|511.15|14.39|19.29|+34.04%|

单位tok/s，均为warmup后3次中位数。MTP1的Prefill相对MTP0在8K约+1.41%、32K约-2.17%；Decode约+34%–35%。这些是顺序跨服务测量，不是交替A/B/A性能或正式P95。MTP1基线来自stage1且接受率100%，该重复文本负载有利于MTP，不代表所有业务文本都有相同收益。

当前RC默认继续选MTP1；MTP0作为可运行的诊断对照。MTP2未通过当前RC推理门禁，不能套用A2 EP上的MTP2收益。MTP0的112K长检索尚未在本轮验证。下一项为MTP1的MindIEQ关闭/开启对照。
