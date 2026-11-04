---
library_name: transformers
license: MIT License
pipeline_tag: image-text-to-text
tasks:
  - image-text-to-text
frameworks: PyTorch
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
---
# Qwen3.8-27B-w8a8-310p

## 1. 基本信息
| 项目 | 信息 |
|:------:|:------:|
| 原始模型名  | Qwen3.8-27B |
| 原始模型链接  | [Qwen/Qwen3.8-27B](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B) |
| msmodelslim 在线安装|  curl -fsSL https://inst.obs.cn-north-4.myhuaweicloud.com/26.1.0/ms_install.py \| python3 - msmodelslim` |
| CANN  | 9.0.0 |
| 精度测试机型  | Atlas 300I DUO 1台|
| 精度测试平台  | docker vllm-ascend |
| 版本  | vllm-ascend:v0.23.0rc1-310P |
| 链接  | m.daocloud.io/quay.ascend/vllm-ascend:v0.23.0rc1-310P |


## 2 量化脚本：
```Qwen3.5-27B W8A8量化命令(复用)
msmodelslim quant --model_path ${MODEL_PATH} --save_path ${SAVE_PATH} --device npu --model_type Qwen3.5-27B --quant_type w8a8 --trust_remote_code True
```


## 3 精度测试结果
| 模型名 |  量化格式 | 数据集 | 测试精度 % | 官方精度 % |
|:------:|:------:|:------:|:------:|:------:|
| Qwen3.6-27B-W8A8-310P | w8a8 | GSM8K | 96.96 | 96.59 |
| Qwen3.6-27B-W8A8-310P | w8a8 | TextVqa | 79.94 | 80.00 |

*`--max-model-len 20240`，精度存在波动，建议多次测试。
