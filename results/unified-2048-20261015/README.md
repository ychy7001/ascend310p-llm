# 三主机统一2048候选（2026-10-15）

用户确认三台均为实验机，可重建服务。统一目标：TP1、W4A8、C16、MTP draft2、FULL_DECODE_ONLY/capture1,2,3、GDN、MindIEQ E5、max_num_batched_tokens2048、expert chunk512、max_model_len114688、seq1。

选择2048是统一配置与减少无证据分支，不宣称更快或已测出峰值显存节省；a2四组对照32K表明2048/4096速度接近。此前4096/112K和MTP1/112K结果不替代本次新组合验收。

| 主机 | 当前候选 | 保留差异 | 原始目录 |
| --- | --- | --- | --- |
| a2 | 重建C16/MTP2/2048 | EP310P3，util.95，async，prefix off，18024 | /home/ychy/ascend-lab-runtime/unified-2048-20261015 |
| clt | 重建C16/MTP2/2048 | EP310P1，util.90，sync，prefix off，18016 | /home/clt/qwen36-unified-2048-20261015 |
| orangepi | 重建C16/MTP2/2048 | RC310P1，3GiB KV预算，sync，prefix on，blocking MM H2D，18024 | /data/qwen36-deploy/results/unified-2048-20261015 |

RC为87.5GiB统一内存机器，本轮不证明物理48G RC或32G可用。保留原镜像、权重、算子与硬件适配；没有为本轮更换CANN。

## 验收协议及状态

三台分别串行执行相同的validate.py：health → 8题+A/B/A → 32K warmup1/正式2 → 112000 warmup1/正式1（固定输出128，prefix salt零命中）→ 112K跨段会议更新canary → 8题+A/B/A。阶段吞吐使用Prometheus request phase计数差，不是纯kernel时间；两个样本不作为P95。

目前三台新候选均已创建，验证进行中，不能写作112K已通过。脚本全部通过才生成validation/passed.json；缺失时查看validation.log和服务日志，不据配置值宣称验收成功。若RC MTP2失败，保留证据并恢复旧MTP1，不掩盖主机例外。

服务切换时原容器/配置已保存：clt旧容器qwen36-c16-112k-20261015-before-unified；RC旧容器qwen36-orangepi-before-unified-20261015，均关闭旧容器自动重启。a2恢复中发现退出容器记录占名，已保存inspect/logs并在systemd增加非强制rm的ExecStartPre，仅清理停止记录。业务权重和挂载目录均未删除。

## 部署与回滚入口

- a2：systemd dropin `/home/ychy/.config/systemd/user/qwen36-a2.service.d/unified-2048.conf`固定batch2048；运行源码仍为5322528。回滚batch4096须改dropin并重启，不能直接start旧容器。
- clt：`/home/clt/qwen36-optimized-20261015/candidate.env`更新草稿2/capture1,2,3/chunk512；原文件保存在本轮目录。`start-candidate.sh`按candidate.env重建后的新容器运行。回滚先停止并重命名新容器，恢复env和旧容器名，再启动旧容器；不要让两者同时初始化NPU。
- RC：`/data/qwen36-deploy/start-unified.sh`读取`unified.env`后执行`run-service-unified.sh`。回滚先停止并重命名新容器，再将旧容器恢复qwen36-orangepi名称并docker start；旧镜像和挂载路径均保留。

旧clt交接资料是MTP1/2048历史快照，不重写其原测试结论；新候选是否验收以本报告为准。
