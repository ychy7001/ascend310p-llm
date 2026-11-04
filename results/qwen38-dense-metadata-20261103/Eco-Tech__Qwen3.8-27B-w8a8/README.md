---
license: Apache License 2.0
tasks:
  - image-text-to-text
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
---
# Qwen3.8-27B-w8a8

## 1. 基本信息

|         项目          |                             信息                             |
| :-------------------: | :----------------------------------------------------------: |
|      原始模型名       |                       Qwen3.6-27B                      |
|     原始模型链接      | [Qwen3.8-27B](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B) |
| msmodelslim commit id | 7952a43968ca6c30a920f8d1d1dc652b486ab129 |
|     精度测试机型      |   Atlas 800T A2 1台   |
|     精度测试平台      |   docker vllm-ascend  |
|         版本          |  vllm-ascend:qwen3.8-a2  |
|         链接          |  docker pull quay.nju.edu.cn/ascend/vllm-ascend:qwen3.8-a2  |


## 2 量化脚本：

```shell
msmodelslim quant \
  --model_path weights/Qwen3.8-27B/ \
  --save_path weights/Qwen3.8-27B-W8A8/ \
  --device npu \
  --model_type Qwen3.6-27B \
  --device_id 0 1 2 3 4 5 6 7 \
  --config  .yaml \
  --trust_remote_code true
```

## 3 精度测试结果

|        模型名        | 量化格式 | 数据集 | 测试精度 % | 官方浮点精度 % |
| :------------------: | :------: | :----: | :--------: | :------------: |
| Qwen3.8-27B-w8a8 |  w8a8  |  GPQA-D  |   90.91   |   89.2   |

* 精度存在波动，建议多次测试。
