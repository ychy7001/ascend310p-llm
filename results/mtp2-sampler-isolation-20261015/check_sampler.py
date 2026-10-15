"""原CPU函数与固定形状CPU/NPU实现对比；保留每例原始输出。"""
import argparse,json
import torch
import vllm_ascend.sample.rejection_sampler as sampler
from bounded_greedy import bounded_greedy

p=argparse.ArgumentParser()
p.add_argument('--device',choices=('cpu','npu'),default='cpu')
a=p.parse_args()
if a.device=='npu':
    import torch_npu
    torch.npu.set_device(0)
original=sampler.rejection_greedy_sample_pytorch
assert original is not bounded_greedy, "reference must be the unmodified upstream CPU function"
cases=0
for counts in ([0],[1],[2],[0,2],[2,0],[1,2],[2,1,0]):
    size=sum(counts)
    for mismatch in (-1,*range(size)):
        for mask in (None, [i%2==0 for i in range(len(counts))]):
            drafts=torch.arange(size,dtype=torch.int64)+10
            target=drafts.clone()
            if mismatch>=0: target[mismatch]+=100
            cumulative=torch.tensor(counts,dtype=torch.int64).cumsum(0)
            bonus=torch.arange(len(counts),dtype=torch.int32).reshape(-1,1)+700
            expected=torch.full((len(counts),3),-1,dtype=torch.int32)
            greedy=None if mask is None else torch.tensor(mask,dtype=torch.bool)
            original(expected,cumulative,drafts,target,bonus,list(counts),2,greedy)
            actual=torch.full_like(expected,-1,device=a.device)
            bounded_greedy(actual,cumulative.to(a.device),drafts.to(a.device),target.to(a.device),
                           bonus.to(a.device),list(counts),2,None if greedy is None else greedy.to(a.device))
            if a.device=='npu':torch.npu.synchronize()
            actual=actual.cpu()
            passed=torch.equal(expected,actual)
            print(json.dumps({'device':a.device,'counts':counts,'mismatch':mismatch,'greedy':mask,
                              'expected':expected.tolist(),'actual':actual.tolist(),'passed':passed}),flush=True)
            assert passed
            cases+=1
print(json.dumps({'passed':True,'cases':cases,'scope':'greedy sampler parity, not model quality'}),flush=True)
