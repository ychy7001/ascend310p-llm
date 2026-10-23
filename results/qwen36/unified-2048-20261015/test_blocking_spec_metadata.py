"""在安装的vLLM-Ascend环境比较原版/阻塞计数上传版采样结果（CPU参考）。"""
import json
import torch
import vllm_ascend.sample.rejection_sampler as sampler
from speculative_metadata_310 import install

original = sampler.rejection_greedy_sample_pytorch
install()
patched = sampler.rejection_greedy_sample_pytorch
install()
assert sampler.rejection_greedy_sample_pytorch is patched
cases = []
for counts in ([2], [1, 2], [0, 2]):
    for mismatch in (-1, 0, 1):
        batch, size = len(counts), sum(counts)
        drafts = torch.arange(size, dtype=torch.int64) + 10
        target = drafts.clone()
        if 0 <= mismatch < size:
            target[mismatch] += 100
        cumulative = torch.tensor(counts, dtype=torch.int64).cumsum(0)
        bonus = torch.arange(batch, dtype=torch.int32).reshape(batch, 1) + 700
        outputs = []
        for function in (original, patched):
            out = torch.full((batch, 3), -1, dtype=torch.int32)
            function(out, cumulative.clone(), drafts.clone(), target.clone(),
                     bonus.clone(), list(counts), 2)
            outputs.append(out)
        assert torch.equal(*outputs), (counts, mismatch, outputs)
        cases.append({'counts': list(counts), 'mismatch': mismatch,
                      'output': outputs[0].tolist()})
print(json.dumps({'passed': True, 'scope': 'CPU sampler parity; not NPU stream correctness',
                  'cases': cases}, indent=2))
