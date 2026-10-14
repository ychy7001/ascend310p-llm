# a2 后端交付验证（2026-10-14）

部署代码固定07a3d6c，目录/home/ychy/services/qwen36/repo。部署清单deployment.json记录代码commit、容器镜像ID与实际使用制品SHA256。a2配置在deploy/a2.env；权重和库均复用已测文件，本次未重新构建二进制。

已实际停止旧18024服务，安装并enable用户systemd qwen36-a2.service，再从固定代码目录启动。日志确认GDN、MindIEQ custom A8/shared Gate阈值128，MTP2，Decode Graph捕获完成，max_model_len114688。HTTP health=200，Tailscale接口可达，NRestarts=0。

五类生成sanity为5/5 PASS（非质量分数，部分触及128输出上限），见sanity.json。8K/128经过脚本warmup后单次phase冒烟：Prefill625.88 tok/s、Decode35.79 tok/s，draft接受84/88=95.45%，mean acceptance length2.909；与整理前625/35.9量级一致，未见明显回退。单轮不是P50/P95或长上下文重新验收。记录见8k-smoke.jsonl；服务最终状态与日志见service-final.txt。

代码整理移除四个旧sweep脚本259行量级，保留实际运行适配、kernel、构建、验证工具与历史结果。后端调用/重启说明见docs/backend-handoff.md。
