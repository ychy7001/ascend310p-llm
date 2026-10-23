# MTP2 greedy sampler隔离（2026-10-15）

目的：区分通用MTP2 greedy sampler中的动态计数/索引问题与Graph/GDN状态问题，不将310P1或RC/EP作为已证明的故障原因。当前仅在x86_64/EP310P1、driver26.0.t1、CANN9.1环境验证，追溯别名clt。

## 实现边界

bounded_greedy.py基于已知CPU draft counts，逐行处理至多2个草稿，使用固定形状where/copy更新输出，保留“首次不匹配输出target、后续不接受；全部匹配追加bonus”的规则。只替换通用greedy函数，不修改random sampler、模型权重、GDN或Graph实现。短草稿小batch的诊断实现，不承诺吞吐更优，也不支持max_spec_len>2。参数cu_num_draft_tokens由CPU counts的累计值替代用于定位；生产路径尚未启用。

原CPU参考和候选CPU/NPU对照40组，包含零草稿、混合count、全部接受、各位置拒绝和greedy mask；全部输出相同。NPU探针在打印40组passed后退出清理滞留，出现TBE子进程main process disappeared，180秒超时；probe-outcome.json保留为失败，不将数值通过伪装成进程生命周期通过。随后明确分开启动完整模型候选，未重复相同数值测试。

check_sampler.py断言reference不是候选函数，避免bootstrap覆盖参考造成自证。独立bootstrap放在candidate_bootstrap目录，只用于完整模型候选；探针不把该目录加入PYTHONPATH。

## 执行与回滚

远端实验目录/home/clt/qwen36-mtp2-sampler-20261015。run_clt.py复制原容器已配置的镜像、设备/制品挂载和环境，仅对模型候选增加独立只读bootstrap、NUM_SPECULATIVE_TOKENS=2、capture1,2,3。原容器先停止，独占同颗芯片；所有情况finally停止候选并启动原MTP1容器。原容器源码和candidate.env未改。

CPU参考在原容器用check_sampler.py --device cpu运行。完整重做可执行python3 run_clt.py；已存在同名实验容器时拒绝覆盖。当前完整模型候选使用--model-only-after-probe，要求原始数值通过标记，同时保留探针生命周期失败记录，不将其标为成功。自动验收短题+A/B/A，通过后32K warmup1+正式2；完成恢复原MTP1并另查health。Graph完整模型结果为5/8，错误输出与仅blocking-count补丁完全相同：3848、7636、NPU7PU732329。A/B/A一致且无canary泄漏，未发生原巨量分配；但正确性门禁失败，不进行32K吞吐测试，不改变默认MTP1推荐。原始graph-short/quality-before.json及outcome.json保留。随后保持sampler与其他设置，仅关Graph启动eager候选，结果也为5/8，三类错误一致；说明该环境下问题在关闭Graph后仍复现，不能只归因Graph replay。两个候选均未进入32K吞吐测试。下一步应定位sync调度下MTP accepted-token计数、验证后状态推进及GDN state位置；本轮尚未证明根因。

## 复现入口

```bash
# 在当前实验环境的Python/vLLM环境内，原CPU参考；不得加载candidate_bootstrap
python3 check_sampler.py --device cpu
# NPU须独占，且source CANN并保留原worker兼容补丁
python3 check_sampler.py --device npu
# 现场脚本使用固定追溯路径；换主机需先修改ROOT/BASE/TOOLS及设备挂载
python3 run_clt.py --model-only-after-probe
python3 run_clt.py --model-only-after-probe --eager
```

上述为诊断复现命令，同名容器已存在时会拒绝覆盖。新环境不能只凭310P1名称直接运行；应先比对../unified-2048-20261015/clt的runtime/source指纹。原始sampler源码和CPU参考均来自已安装vLLM-Ascend，没有复制上游函数实现；候选为独立短草稿诊断实现。
