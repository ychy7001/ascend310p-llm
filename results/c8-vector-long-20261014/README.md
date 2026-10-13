# C8 vector Attention长上下文对照

复用稳定device-length C8库，direct vector路径仅支持splits=1；芯片1、Q1 query/16 heads/2 KV heads/dim256，5次预热+20次launch+synchronize。数值和Graph可变页表检查通过。

|路径|8K|32K|112K|
|---|---:|---:|---:|
|native FP16 PA|约0.31ms|约1.12ms|**2.954ms**|
|direct C8 vector|见原始日志|见原始日志|**59.167ms**|
|C8转换+native PA|见原始日志|1.891ms|**6.161ms**|

112K direct vector约为native PA **20倍**，比Cube直接Attention约24.23ms更慢；C8转换+native PA仍略慢于约6.03ms基线。误差相对FP16 reference约1.9e-4以内，代表正确性不是性能。该路径不接入服务，当前直接C8 vector/Cube原型均只能作为研究对照。

失败的splits=4 vector调用记录在早期命令错误中，vector实现本身要求Cube和splits1；没有修改服务。若继续直接C8，需要重写块级数据流/硬件流水线，不能通过简单调参获得产品收益。
