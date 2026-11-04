# Qwen3.8-27B Dense 权重元数据审计

用户提供三个ModelScope来源，已读取config、模型卡、文件清单与可用的quant_model_description，尚未下载权重或运行模型。各已下载文件的SHA256均与仓库文件清单匹配；来源、权重文件大小/哈希/逐文件revision见sources-and-integrity.json。master请求非原子快照，下载时应使用验证过的revision/文件哈希。

| 来源 | 文件清单总量 | 配置 | 量化区别 |
|---|---:|---|---|
| Qwen/Qwen3.8-27B | 51.77 GiB | qwen3_5_text、64层、hidden5120、BF16、GDN config FP32 | 官方浮点来源，无quant_model_description |
| Eco-Tech/Qwen3.8-27B-w8a8 | 28.30 GiB | 同一Dense架构 | 量化包含GDN投影和部分视觉层 |
| Eco-Tech/Qwen3.8-27B-w8a8-310p | 33.94 GiB | 同一Dense架构 | GDN线性注意力336个weight条目均FLOAT，视觉167个weight条目均FLOAT；MLP192和full-attention64个weight条目为W8A8_DYNAMIC |

两份量化描述共同key中有351项类型不同。普通版在GDN投影中240项为W8A8_DYNAMIC，310P版相应保留FLOAT，因此不能仅按同为W8A8互换。描述文件包含scale等条目，条目数不等于独立线性层数。

310P模型卡声称用Atlas300I DUO、CANN9.0.0与v0.23.0rc1-310P测试；但精度表仍写Qwen3.6-27B，普通版卡也有旧模型名。这些是来源自述和元数据一致性问题，不是本项目实测，也不能证明当前a2 CANN9.1.0镜像已兼容Qwen3.8。

后续优先310P专用版，先核对ModelSlim ABI/反量化dtype、MTP张量、Tokenizer及实际weight header，估算TP1/TP2的NZ、工作区、状态与KV预算。33.94GiB检查点体积不等于设备驻留量，不能据此断言TP1一定能装入。官方BF16仅权重文件总量已超单芯片容量，是否TP2可行仍须按实际驻留和运行开销核算。

本任务保持Qwen3.6第一优先级，此处不启动第二模型下载/推理，不以Flash-Next结果替代。
