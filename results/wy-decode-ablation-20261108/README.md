# W8 WY Decode受控对照

2026-11-08在a2完成最终镜像MTP0/1/2与归档a1现役profile的受控对照。a1只读核验，没有发送请求。

最终镜像固定image ID `bb0fa66f1c715a94abf56afc17a6bd02e3f993c4804e200dfd229e22b0d5cab4`，入口SHA `ca8391c85f1a89dae6f71eb61233c06eec0f849dedd0be0453b800cb4f60388a`。固定4K输入、512输出，c1/c4各2轮，每臂两次启动；唯一cache salt，prefix hits均为0。`summary.json`是结构化汇总，原始窗口保留在a2 runtime和工作区任务数据目录，不把日志/cache/完整输出提交到仓库。

结果摘要：

| 臂 | 单流Decode | 四并发端到端 | 备注 |
| --- | ---: | ---: | --- |
| 最终MTP0 | 33.10 tok/s | 61.94 tok/s | WY/最终运行时，关闭MTP |
| 最终MTP1 | 40.58 tok/s | 71.10 tok/s | 单流约比MTP0快22.6% |
| 最终MTP2 | 43.69 tok/s | 70.04 tok/s | 单流约比MTP0快32.0%，四并发略低于MTP1 |
| a1旧profile复现 | 32.28 tok/s | 51.60 tok/s | a2复现，不是a1实机压测 |

单流MTP0相对旧profile只有约2.6% Decode差异；最终镜像相对旧profile的主要优势是Prefill和整栈配置。详情见生产配置文档。
