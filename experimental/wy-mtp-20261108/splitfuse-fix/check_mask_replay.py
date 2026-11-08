"""NPU graph replay audit: changing device lengths must change causal visibility."""
import json
import torch
import torch_npu
from graph_safe_splitfuse import causal_mask

torch.npu.set_device(0)
rows=[]
for query_lens in ([1], [3], [3,3], [1,3]):
    width=128
    lengths=torch.tensor([7]*len(query_lens),dtype=torch.int32,device='npu')
    for _ in range(3): causal_mask(lengths,query_lens,width)
    torch.npu.synchronize()
    buffer=causal_mask(lengths,query_lens,width)
    graph=torch.npu.NPUGraph()
    with torch.npu.graph(graph): result=buffer.clone()
    for step in (0,1,9,64,120):
        contexts=[max(query_lens[i],step+i+3) for i in range(len(query_lens))]
        lengths.copy_(torch.tensor(contexts,dtype=torch.int32))
        buffer.copy_(causal_mask(lengths,query_lens,width))
        graph.replay()
        actual=result.cpu()
        offset=0
        for q,c in zip(query_lens,contexts):
            for j in range(q):
                visible=c-q+j+1
                assert torch.all(actual[offset,:visible]==0)
                assert torch.all(torch.isneginf(actual[offset,visible:]))
                offset+=1
        rows.append({'query_lens':query_lens,'contexts':contexts,'passed':True})
print('MASK_REPLAY_RESULT='+json.dumps({'passed':True,'cases':rows}))
