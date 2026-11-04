# 27B W8A8 TP2：9.2.1配套H/O，16K质量筛选

最终状态：`not_accepted_short_format_gate`，见adjudication.json。

固定官方9.2.1镜像d103c59b…、PR17640 head18163f6b配套H/O，原Torch WY。max_model_len=16384，其余沿用生产TP2/MTP0/Graph[1,2]/prefix on控制参数。此容量筛选不能与旧262K/eager配置作单变量性能对照。

实际健康启动，两rank日志各19.2416GB权重，worker-library-maps.log确认两进程均映射paired API库。8题7题严格匹配，排序应输出2,7,9却输出279。ABA前后相同，无canary泄漏。旧post4同题也输出279，不证明根因相同或基础模型能力上限。事实、业务、性能均未执行。

cpu-tokenizer-roundtrip.json确认2,7,9可以完整往返，逗号token ID=11。仅为CPU词表/解码检查，不代替实际生成token验证。首次AutoTokenizer CPU import触发torch_npu缺HAL，记录在cpu-probe-attempt1.json；随后使用纯tokenizers，无设备暴露。

20条原业务的semantic-review-oracle.json在查看候选输出前冻结，但本轮未执行业务，不能把这些预期写成模型测试结果。工具和fixture/元数据、镜像/worker库、原始短题、完整日志及原服务HTTP200恢复记录均保留。无权重/缓存归档，未改a1或宿主安装。
