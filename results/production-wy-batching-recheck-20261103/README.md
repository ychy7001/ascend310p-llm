# WY并发、数值拆分与候选Profile

候选原三并发整套业务重放3轮，84/84自动完成/工具/schema通过；这些自动项不等于纪要语义通过。会议4的负责人映射第0轮错误，第1/2轮正确，第1轮遗漏“当天”期限；原配置A′也出现同类映射错误。当前不推荐部署，不能据此认定候选独有回归。

stage-isolation-stages.jsonl：两种WY helper各12轮输出逐位稳定，但固定helper输出重新执行后续链路，torch路径11/12、native路径9/12在门限内；候选WY不等于数值修复。standalone复现脚本不加载native扩展，原路径5/12在门限内。

intermediate-isolation原始链路24次有21次超限、冻结WY链路1次超限，所有WY/FwdH/FwdO输入接口各自逐位稳定。intermediate-isolation-raw进一步保留FwdO原始输出，原链路24次12次超限，原始输出与最终转置输出完全一致，排除后续转置生成该偏差。raw supervisor最初引用上一轮summary的路径错误已修正，旧status单独保留；新轮原始JSONL/summary从未更改。固定输入FwdO-only的24/24通过记录位于results/fwd-o-isolation-20261103，说明调用历史影响仍需排查；不能单凭该通过消除完整链路异常。

4份kernel CSV gzip、摘要与profile-artifacts.json对应2范围×2rank。index0/2是4K Prefill，1/3是32K首输出后Decode；rank来自目录tp0_rank0/tp1_rank1，先后范围由两个采集时间及请求日志确定。4K每rank8492 kernels（基线38072），原生ComputeWy60次，FwdH/O各60次，Tril0、ReduceSum160、ViewCopy8。捕获跨度约2.056s（基线约3.162s），诊断计时不替代常规吞吐；Decode仍380000 kernels，窗口约4.506s。普通计时已在Profile前完成。

a2原临时unit按各阶段恢复/暂停；a1未修改。后续mask修复在独立namespace构建验证，尚无模型/业务验收。
