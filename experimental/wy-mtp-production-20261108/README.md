# W8A8 + WY + MTP2 满上下文配置验收

目标：参照a1正在运行的262144上下文、TP2、seq4、batch4096和memory0.9配置，在a2完成可部署profile。a1只读，不切生产，不修改WY kernel数学路径。

`run_profile.py`每次只运行一个隔离窗口，持有a2 `/tmp/a310-npu.lock`，在finally中保存服务日志与退出状态并独立清场。`qualify`包含四并发确定JSON、4K/8K/16K/32K各三次冷Prefill、128输出单并发/四并发、128K和262016+128满上下文、超限400、PDF及长请求后JSON。`tune`只进行4K/32K冷Prefill和128输出，用于有限batch调优、同配置无MTP对照；每请求独立cache salt，验证prefix hits为0。

对比时保持同镜像、容量、并发上限与缓存策略；记录Graph差异，不把W4/W8直接比较作为WY或MTP增益。性能计数器是服务请求阶段时间，不是纯kernel时间。满上下文容量测试使用合成token输入，不证明262K语义检索质量。PDF只作功能验收。

运行前把仓库入口复制到a2的 `wy-mtp-production-20261108/start-wy-mtp.sh`。最终还须对封装后的镜像默认入口单独验收，不能以host入口挂载代替制品验证。

2026-11-08收尾：正式结果见docs/qwen36/wy-mtp-production-20261108.md。当前串行器为run_main_graph_closure.py；run_followups.py、run_closure.py和run_graph_closure.py保留历史配置复现，不是当前交付流程。publish_main_graph.py只在同镜像完整与封装门禁通过后发布独立tag，临时单域registry凭据在finally删除，不推送Git。
