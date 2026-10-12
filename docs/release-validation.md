# 公开制品验证记录

2026-10-12，北京时间。首版为研究预览，不等同于完整项目验收。

## 自包含checkpoint

真实a2只读结构与SHA256验证PASS：93,125 indexed tensors，92,160 expert weight/scale/offset tensors，19个BF16 MTP tensors，52 shards，tensor bytes=24,591,896,288；目录无软链接。逐文件SHA256与`deployment-sha256.json`全部匹配。重导出仅移除未被index引用的张量，并核验所有保留tensor的payload字节；源权重未覆盖。

## 独立启动与生成

使用本仓库bootstrap、裁剪后的W4 runtime模块、通用启动器与自包含checkpoint，a2 Ascend310P3单芯片、TP1、MTP=1、eager、4096 context、memory utilization=.90、FP16 KV。仅挂载该checkpoint，不挂载原始BF16/sidecar父目录。初轮仍复用先前已构建的同路径OPP与extension，不能将其当成干净构建产物验证。

启动通过，target/draft共享embedding，日志权重加载约27.288GB。英文、中文、代码、数学、重复输入五类HTTP生成sanity全部通过，completion tokens分别67/91/35/128/128；未发现空文本、非有限字面量或明显重复。数学等样例可能被长度截断，不能当成题目正确率。

p128/o64、单流、1次warmup后3轮streaming：aggregate output吞吐为21.9236/21.9136/21.9567 tok/s；TTFT为0.6659/0.6632/0.6627s，mean TPOT约35.69/35.82/35.74ms。该文本是重复固定token工作负载，不能用来证明生成质量；三轮也不是正式尾延迟统计。

初次打包将包含overlay namespace的runtime目录直接置于PYTHONPATH，导致遮蔽官方包、platform plugin缺失；现已将bootstrap与runtime分离，正式启动日志不再出现该失败。本记录保留失败原因帮助复现，发布启动器已修正。

## 源码构建

三个W4算子家族的公开snapshot已投入新的隔离CANN构建目录；基础社区源固定`0637d3baa6bc2000ba6433b301f233c6516847b4`。SDK/依赖沿用既有镜像。测试准备阶段最初递归忽略`build`目录，误删`cmake/third_party/build/modules/patch/`内受版本管理的补丁，host构建失败；补回后重试。该过滤方式不是公开build-native脚本的行为，正式脚本从固定Git checkout获取源文件。

三个OPP家族干净编译、打包、安装验证已成功；独立Torch extension编译和链接也成功。构建容器未挂NPU，CANN的“无法取得设备/console log level”诊断不等于设备执行失败。extension最初因镜像Triton不兼容失败，禁用Triton后成功；公开说明要求执行同样准备。替换新产物后的生成门禁正在验证，完成前不声称“从公开脚本全新下载并构建已验收”。完整ModelScope上传与远端文件检查也将在完成后追加。

## 发布状态

GitHub公开仓库`ychy7001/ascend310p-qwen36`已创建，首个源码commit为`3b6022c`。ModelScope公开仓库`ychy7001/Qwen3.6-35B-A3B-W4A8-SSZ-MTP-310P`已创建，64个文件正在上传；完整权重尚需以远端清单校验确认，不能把空仓创建当作发布完成。

## 尚未完成的门禁

C8完整模型、W4A8+MTP+C8组合、公开制品TP2复测、长上下文真实请求、多模态、正式质量集合、同条件W8A8比较与全部吞吐/尾延迟矩阵仍未完成。
