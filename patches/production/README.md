# 生产镜像补丁队列

顺序：`attention-mask.patch`。

- `attention-mask.patch`：来源为锁定生产镜像第 20、21 层及其 vllm-ascend Git 工作树差异（镜像 overlay 标签 revision `f8f58951e43054b2da7f7e200939d01f933197cb`，未验证为源码提交）；对应上游 PR/issue：无；需要它是为了在固定上游 `88eac2713b14beedc7a48803274f8d6cee3e9f30` 上复现生产 attention mask 文件与配套测试，本轮不评估其数值、性能或算子实现。

补丁只包含生产已有的两个文件差异。基线文件来自上游 commit `fc0b9e3540bc85dfe355ebb244c80965ab39a2b2`（PR #10181，已经合并）；该上游实现本身不在补丁队列中。生产增量未找到对应的公开上游 PR 或 commit，不能把 #10181 当作该增量的上游来源。

来源核查：生产镜像记录的原始覆盖层分别为 `sha256:2299544ecf91092d8fc2d67ed7ab6b205c20de686b5a55c3979947fca08e8b33` 和 `sha256:d8158c2e08f17b9e749d5d91c58a0d43b70ded591d4f1297971366c6d8af74b4`；两者与 Git diff 的结果文件哈希共同锁定补丁来源。公开 vllm-project/vllm-ascend 和 opensensor/vllm-ascend 的 commit 查询均未找到标签中的 revision，公开 commit 搜索也无匹配，故不把它作为已证实的上游提交。

当前官方 `v0.30.0-310p` 的两个基线 blob 仍分别为 `5e68eb66d482260f01138873b04716f3717c67dd`（实现）和 `dab84617d9f39dace3144af5ea97b4752bb122e0`（测试），与生产基线一致；补丁文本检查可应用。此检查不代表新运行时通过门禁。以后升级上游时，如已包含该增量，应从队列中删除补丁，而非重复应用。
