# SPDX-License-Identifier: Apache-2.0
"""独立 SSZ 专家量化适配；不修改上游模型注册或生产环境。

G128 模式将每个输出通道的每组作为独立行交给上游 SSZ。
只复用权重量化器，不声称为官方已验证的 group-wise 导出策略。
返回 logical INT4、FP16 scale 和 zero-point，供 native_bank 打包。
"""
import torch


def quantize_modelslim_ssz(weight, groupwise=False, steps=None):
    """调用上游量化器；per-channel scale 展开为 kernel 的 G128 布局。

    激活仍为社区 per-G128 INT8，不声称等同于 ModelSlim per-token W4A8。
    """
    from msmodelslim.core.quantizer.base import QConfig
    from msmodelslim.core.quantizer.impl.ssz import WeightPerChannelSsz, get_ext_scale
    from msmodelslim.ir.qal.qbase import QStorage, QDType
    # Preserve the upstream search by default. Reduced iteration counts are
    # explicit experiment parameters, never an implicit precision tradeoff.
    quantizer = WeightPerChannelSsz(QConfig(dtype='int4', scope='per_channel',
                                          symmetric=True, method='ssz',
                                          ext={} if steps is None else {'step': steps}))
    source = weight.reshape(-1, 128) if groupwise else weight
    quantizer.init_weight(QStorage(QDType.FLOAT, source.float()))
    quantizer()
    q = quantizer.get_q_storage().value.to(torch.int8).reshape_as(weight)
    scale = get_ext_scale(quantizer.get_q_param()).flatten().half()
    underflow = scale == 0
    if underflow.any():
        # SSZ computes FP32 scales. Tiny nonzero channels can round to zero
        # in the FP16 checkpoint; requantize only those channels against the
        # smallest representable positive FP16 scale.
        minimum = torch.nextafter(torch.zeros((), dtype=torch.float16),
                                  torch.ones((), dtype=torch.float16)).item()
        scale[underflow] = minimum
        codes = q.reshape_as(source)
        codes[underflow] = torch.round(source[underflow].float() / minimum).clamp(-8, 7).to(torch.int8)
    scales = (scale.reshape(weight.shape[0], -1) if groupwise else
              scale[:, None].expand(-1, weight.shape[1] // 128).contiguous())
    return q, scales, torch.zeros_like(scales)
