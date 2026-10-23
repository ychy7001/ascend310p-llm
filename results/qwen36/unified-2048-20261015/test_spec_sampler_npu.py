"""比较CPU参考与310P阻塞计数版rejection sampler；不加载模型。"""
import json
import torch
import torch_npu
import vllm_ascend.sample.rejection_sampler as sampler
from speculative_metadata_310 import install

original = sampler.rejection_greedy_sample_pytorch
install()
patched = sampler.rejection_greedy_sample_pytorch
torch.npu.set_device(0)
rows = []
for counts in ([2], [1, 2], [0, 2]):
    for mismatch in (-1, 0, 1):
        batch, size = len(counts), sum(counts)
        drafts = torch.arange(size, dtype=torch.int64) + 10
        target = drafts.clone()
        if 0 <= mismatch < size:
            target[mismatch] += 100
        cumulative = torch.tensor(counts, dtype=torch.int64).cumsum(0)
        bonus = torch.arange(batch, dtype=torch.int32).reshape(batch, 1) + 700
        reference = torch.full((batch, 3), -1, dtype=torch.int32)
        original(reference, cumulative, drafts, target, bonus, list(counts), 2)
        output = torch.full((batch, 3), -1, dtype=torch.int32).to('npu')
        patched(output, cumulative.to('npu'), drafts.to('npu'), target.to('npu'),
                bonus.to('npu'), list(counts), 2)
        torch.npu.synchronize()
        actual = output.cpu()
        row = {'counts': list(counts), 'mismatch': mismatch, 'expected':reference.tolist(),
               'actual':actual.tolist(), 'passed':torch.equal(reference, actual)}
        rows.append(row)
        print(json.dumps(row), flush=True)
assert all(row['passed'] for row in rows)
print('ALL_NPU_SAMPLER_CASES_PASSED',flush=True)
