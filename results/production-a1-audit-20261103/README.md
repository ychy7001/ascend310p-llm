# 生产基线只读审计与隔离复现

会话日期2026-11-03，用户明确生产为a1 TP2 W8A8 Qwen3.6-35B-A3B。a1只读取manifest、容器身份/版本、启动日志与一份metrics；没有重启、改参数、线上Profile或发送Benchmark。模型文件只读复制到a2实验目录，服务继续运行。

- `manifest.filtered.json`：生产命令与非凭证env；TP2/MTP0/Graph[1,2]/max262144/batch4096/seqs4/prefix on/FP16 SSM/async。
- `container.json`：镜像、模型/脚本挂载，过滤env；未归档完整含其他服务的inspect。
- `source-and-cann.txt`：实际源码提交。vLLM0fc695fc，Ascend88eac271；容器torch-npu2.10.0.post2，transformers5.5.4，CANN9.1.0-beta.1。a1驱动26.0.rc1，与a2的26.1.1不同。
- `startup-selected.log`：只保留初始化/有效配置/内存/Graph捕获条目。block自动调整到640，mamba align；两rank权重各18.85GiB，Graph约0.09GiB。日志内UTC/宿主时区原样保留，不当作Toronto本地时间。
- `metrics.prom`：一次累计快照，prefix queries3768/hits0，success342；不是本轮Benchmark或命中率实验。
- `source/`：从a2同digest镜像直接导出的相关源文件，未修改。
- `historical-graph123-resource-error.txt`：a2保留10月31日记录的错误片段；HcclAllGather异步报错同时有EL0008/IPC notify事件资源不足，不代表根因已定位。
- `weight-transfer.json`：复制状态。先经工作机中转，后改为a1→a2直连LAN的rsync，限速64MiB/s；使用本地已核验host key和临时转发agent，私钥未复制到远端，结束即清理agent。SSH/rsync传输完整性与a2完整checkpoint SHA指纹分开记录。

W4 TP1测试属于预算方案，见../baseline-20261103与../profile-c16-20261103；不与本生产配置直接归因比较。现有ragged GDN PR#18730已含算子和TP2/eager小样本证据，新的业务/长上下文证据须独立取得，不重复宣称既有工作。
