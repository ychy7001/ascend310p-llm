"""穷举已部署长度域的无FloorDiv设备索引，并检查动态Graph replay。"""
import json
import torch
import torch_npu

torch.npu.set_compile_mode(jit_compile=False)
cpu = torch.arange(114689, dtype=torch.int32)
device = cpu.npu()
for block in (64, 576, 1152, 1728):
    def indices(x):
        return ((x.float()-1).clamp_min(0)/block).to(torch.int32)
    expected = ((cpu-1)//block).clamp_min(0)
    torch.testing.assert_close(indices(device).cpu(), expected, rtol=0, atol=0)
    x = torch.tensor([0,1,block,block+1,114688],dtype=torch.int32).npu()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = indices(x)
    alternate = torch.tensor([114687,block*2,block*2+1,63,0],dtype=torch.int32)
    x.copy_(alternate.npu());graph.replay();torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(),((alternate-1)//block).clamp_min(0),rtol=0,atol=0)
    print(json.dumps({'block_size':block,'lengths_checked':114689,
                      'graph_changed_input':'PASS','status':'PASS'}),flush=True)
