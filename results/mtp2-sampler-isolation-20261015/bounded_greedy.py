# SPDX-License-Identifier: Apache-2.0
"""实验用短草稿greedy rejection；仅替换通用greedy函数，不改变随机采样。"""
import inspect
import logging


def bounded_greedy(output_token_ids, cu_num_draft_tokens, draft_token_ids,
                   target_argmax, bonus_token_ids, draft_tokens_per_req,
                   max_spec_len, is_greedy=None):
    import torch
    if max_spec_len not in (0, 1, 2):
        raise ValueError('probe only supports max_spec_len 0/1/2')
    if len(draft_tokens_per_req) != output_token_ids.shape[0]:
        raise ValueError('request count mismatch')
    start = 0
    for row, count in enumerate(draft_tokens_per_req):
        if not isinstance(count, int) or not 0 <= count <= max_spec_len:
            raise ValueError('draft count outside probe bounds')
        # 每行只处理CPU已知的草稿数；不将计数上传后做动态shape算术。
        active = (torch.ones(1, dtype=torch.bool, device=output_token_ids.device)
                  if is_greedy is None else is_greedy[row:row+1])
        for pos in range(count):
            target = target_argmax[start+pos:start+pos+1]
            draft = draft_token_ids[start+pos:start+pos+1]
            current = output_token_ids[row, pos:pos+1]
            current.copy_(torch.where(active, target.to(current.dtype), current))
            active = active & (draft == target)
        current = output_token_ids[row, count:count+1]
        bonus = bonus_token_ids[row].reshape(1).to(current.dtype)
        current.copy_(torch.where(active, bonus, current))
        start += count


def install():
    import vllm_ascend.sample.rejection_sampler as sampler
    current = sampler.rejection_greedy_sample_pytorch
    if current is bounded_greedy:
        return
    if tuple(inspect.signature(current).parameters) != tuple(inspect.signature(bounded_greedy).parameters):
        raise RuntimeError('unsupported rejection sampler signature')
    sampler.rejection_greedy_sample_pytorch = bounded_greedy
    logging.warning('EXPERIMENT: bounded greedy rejection max_spec_len<=2; no random sampler replacement')
