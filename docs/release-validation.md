# 公开制品验证记录

2026-10-12，北京时间。首版为研究预览，不等同于完整项目验收。

## 自包含checkpoint

真实a2只读结构与SHA256验证PASS：93,125 indexed tensors，92,160 expert weight/scale/offset tensors，19个BF16 MTP tensors，52 shards，tensor bytes=24,591,896,288；目录无软链接。逐文件SHA256与`deployment-sha256.json`全部匹配。重导出仅移除未被index引用的张量，并核验所有保留tensor的payload字节；源权重未覆盖。

## 独立启动与生成

使用本仓库bootstrap、裁剪后的W4 runtime模块、通用启动器与自包含checkpoint，a2 Ascend310P3单芯片、TP1、MTP=1、eager、4096 context、memory utilization=.90、FP16 KV。仅挂载该checkpoint，不挂载原始BF16/sidecar父目录。初轮仍复用先前已构建的同路径OPP与extension，不能将其当成干净构建产物验证。

启动通过，target/draft共享embedding，日志权重加载约27.288GB。英文、中文、代码、数学、重复输入五类HTTP生成sanity全部通过，completion tokens分别67/91/35/128/128；未发现空文本、非有限字面量或明显重复。数学等样例可能被长度截断，不能当成题目正确率。

初次打包将包含overlay namespace的runtime目录直接置于PYTHONPATH，导致遮蔽官方包、platform plugin缺失；现已将bootstrap与runtime分离，正式启动日志不再出现该失败。本记录保留失败原因帮助复现，发布启动器已修正。

## 源码构建

三个W4算子家族的公开snapshot已投入新的隔离CANN构建目录；基础社区源固定`0637d3baa6bc2000ba6433b301f233c6516847b4`。SDK/依赖沿用既有镜像。测试准备阶段最初递归忽略`build`目录，误删`cmake/third_party/build/modules/patch/`内受版本管理的补丁，host构建失败；补回后重试。该过滤方式不是公开build-native脚本的行为，正式脚本从固定Git checkout获取源文件。

完整OPP/extension新构建结果、替换后的生成门禁正在验证，完成前不声称“从公开脚本全新下载并构建已验收”。完整ModelScope上传与远端文件检查也将在完成后追加；本文件不把计划写成结果。

## 尚未完成的门禁

C8完整模型、W4A8+MTP+C8组合、公开制品TP2复测、长上下文真实请求、多模态、正式质量集合、同条件W8A8比较与全部吞吐/尾延迟矩阵仍未完成。
