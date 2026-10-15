"""RC block128 exact slot-index probe, including signed positions and Graph replay."""
import json
import torch
import torch_npu

torch.npu.set_device(0)
torch.npu.set_compile_mode(jit_compile=False)

def indices(x):
    quotient = torch.floor(x.to(torch.float32) * (1.0 / 128)).to(x.dtype)
    return quotient, x - quotient * 128

cpu = torch.arange(-1048576, 1048577, dtype=torch.int32)
x = cpu.npu()
q, r = indices(x)
torch.testing.assert_close(q.cpu(), cpu // 128, rtol=0, atol=0)
torch.testing.assert_close(r.cpu(), cpu % 128, rtol=0, atol=0)
print(json.dumps({'stage':'eager','positions_checked':cpu.numel(),'min':-1048576,'max':1048576,'status':'PASS'}),flush=True)
for size in (1, 2, 4, 8):
    source = torch.zeros(size,dtype=torch.int32,device='npu')
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        q,r=indices(source)
    for values in ([27,127,128,129,-1,-128,-129,1048576],
                   [114688,262144,0,1048575,-1048576,255,256,-255]):
        cpu_values=torch.tensor(values[:size],dtype=torch.int32)
        source.copy_(cpu_values.npu());graph.replay();torch.npu.synchronize()
        torch.testing.assert_close(q.cpu(),cpu_values//128,rtol=0,atol=0)
        torch.testing.assert_close(r.cpu(),cpu_values%128,rtol=0,atol=0)
    print(json.dumps({'stage':'graph','size':size,'changed_inputs':2,'status':'PASS'}),flush=True)
print('ALL_SLOT_INDEX_CASES_PASSED',flush=True)
