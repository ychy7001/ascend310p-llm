#!/usr/bin/env python3
"""Compare frozen CANN grouped GEMV with a prescale scheduling experiment."""
import argparse
import ctypes
import json
from pathlib import Path
import statistics
import time
import torch
import torch_npu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--snapshots', type=Path)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.smoke == bool(a.snapshots):
        p.error('select exactly one of --smoke or --snapshots')
    torch.set_num_threads(1)
    torch.manual_seed(20261105)
    torch.npu.set_device(0)
    lib = ctypes.CDLL(str(a.library.resolve()))
    launch = lib.launch_moe_prescale
    launch.argtypes = [ctypes.c_void_p]*7 + [ctypes.c_uint32]*5
    launch.restype = ctypes.c_int
    results = []

    def run(xcpu, wcpu, scpu, ends, label):
        m, k = xcpu.shape
        e, n, wk = wcpu.shape
        assert xcpu.dtype == torch.float16 and wcpu.dtype == torch.int8 and scpu.dtype == torch.float32
        assert scpu.shape == (e,n)
        assert k == wk and m in (8,16,24,32) and k in (256,2048)
        assert ends.shape == (e,) and e == 256 and ends[-1] == m
        counts = torch.diff(ends, prepend=torch.zeros(1,dtype=torch.int64))
        assert int(counts.min()) >= 0 and int(counts.max()) <= 8
        x = xcpu.npu()
        w = torch_npu.npu_format_cast(wcpu.npu(),29)
        scale, group = scpu.npu(), ends.npu()
        def custom(prescale):
            stream = torch.npu.current_stream().npu_stream
            workspace = torch.empty(((m+15)//16*16*k + m*32+256+4096,),dtype=torch.uint8,device='npu')
            out = torch.empty((m,n),dtype=torch.float16,device='npu')
            code = launch(stream,x.data_ptr(),w.data_ptr(),scale.data_ptr(),group.data_ptr(),out.data_ptr(),workspace.data_ptr(),m,k,n,e,prescale)
            if code: raise RuntimeError(f'launch returned {code}')
            return out
        def official():
            return torch_npu.npu_quant_grouped_matmul_dequant(x,w,scale,group,quant_mode='pertoken')
        torch.npu.synchronize()
        reference = official().cpu()
        compiled = custom(0).cpu()
        candidate = custom(1).cpu()
        bits = lambda t: t.view(torch.int16)
        parity = {'compiled_baseline':torch.equal(bits(reference),bits(compiled)),
                  'candidate':torch.equal(bits(reference),bits(candidate))}
        repeat = []
        for _ in range(3):
            repeat.append(torch.equal(bits(candidate), bits(custom(1).cpu())))
        cpu = torch.empty((m,n),dtype=torch.float32)
        start = 0
        for expert,end in enumerate(ends.tolist()):
            if end == start: continue
            activation = xcpu[start:end].float()
            xs = activation.abs().amax(-1,keepdim=True)/127
            q = torch.round(activation * (1/xs)).clamp(-128,127)
            accum = q @ wcpu[expert].float().t()
            cpu[start:end] = (accum * scpu[expert]) * xs
            start = end
        errors = {}
        for name,tensor in [('A',reference),('B',candidate)]:
            delta = tensor.float()-cpu
            errors[name] = {'max_abs':delta.abs().max().item(),
                            'rmse':delta.square().mean().sqrt().item(),
                            'finite':bool(tensor.isfinite().all()),
                            'outside_tolerance':int((~torch.isclose(tensor.float(),cpu,rtol=.002,atol=.002)).sum())}
        row = {'label':label,'m':m,'k':k,'n':n,'active_experts':int((counts>0).sum()),
               'max_expert_rows':int(counts.max()),'parity':parity,'repeat':repeat,'cpu_fp32':errors,
               'AB_max_abs':(reference.float()-candidate.float()).abs().max().item()}
        # Incorrect control reconstruction is a hard failure, never a fallback.
        if not parity['compiled_baseline'] or not all(repeat) or not all(z['finite'] and z['outside_tolerance']==0 for z in errors.values()):
            results.append(row)
            a.output.write_text(json.dumps(results,indent=2)+'\n')
            raise RuntimeError(f'G0 bitwise/finite failure: {row}')
        timings = {'A':[], 'B':[]}
        if not a.smoke:
            # Keep 256 operator calls on the captured stream to avoid timing
            # Python scheduling. Eager includes launch and allocation overhead.
            for mode in ('eager','graph'):
                funcs={'A':official,'B':lambda:custom(1)}
                graphs={}
                graph_outputs={}
                if mode == 'graph':
                    for name,fn in funcs.items():
                        for _ in range(5):fn()
                        torch.npu.synchronize()
                        graph=torch.npu.NPUGraph()
                        with torch.npu.graph(graph):
                            for _ in range(256):last_output=fn()
                        graphs[name]=graph
                        graph_outputs[name]=last_output
                    # A launch on a stale stream can leave an empty graph.
                    # Change a live input and compare replay to a fresh call.
                    x.neg_()
                    torch.npu.synchronize()
                    for name,fn in funcs.items():
                        expected=fn().cpu()
                        graphs[name].replay()
                        assert torch.equal(bits(graph_outputs[name].cpu()),bits(expected)), f'{name} graph failed input-mutation replay verification'
                    x.neg_()
                    torch.npu.synchronize()
                    row['graph_replay_verified']=True
                    funcs={name:graph.replay for name,graph in graphs.items()}
                for pair in range(3):
                    for name in (('A','B') if pair%2==0 else ('B','A')):
                        fn=funcs[name]
                        for _ in range(5):fn()
                        torch.npu.synchronize()
                        start_time=time.perf_counter()
                        if mode=='graph':fn()
                        else:
                            for _ in range(256):fn()
                        torch.npu.synchronize()
                        timings[name].append({'mode':mode,'pair':pair,'us':(time.perf_counter()-start_time)*1e6/256})
                selected={name:statistics.median(t['us'] for t in ts if t['mode']==mode) for name,ts in timings.items()}
                row[mode]={'median_us':selected,'speedup':selected['A']/selected['B']}
            row['timings']=timings
        results.append(row)
        a.output.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(row),flush=True)

    if a.smoke:
        for k,n in ((2048,512),(256,2048)):
            w=torch.randint(-127,128,(256,n,k),dtype=torch.int8)
            scales=torch.rand((256,n),dtype=torch.float32)*.002+.001
            for m in (8,16,24,32):
                x=torch.randn(m,k,dtype=torch.float16)
                # Nonuniform, sparse expert IDs; rows per expert reflect
                # distinct per-token top-k routing rather than uniform banks.
                ids=torch.tensor(([0,1,7,7,31,80,128,255]*4)[:m])
                ends=torch.bincount(ids,minlength=256).cumsum(0)
                run(x,w,scales,ends,f'smoke-M{m}-K{k}')
    else:
        snapshots = sorted(a.snapshots.glob('input-*.pt'))
        assert snapshots, 'No real operator snapshots captured'
        for snapshot in snapshots:
            data=torch.load(snapshot,map_location='cpu',weights_only=True)
            bank=torch.load(a.snapshots/data['bank'],map_location='cpu',weights_only=True)
            run(data['x'],bank['weight'],bank['scale'],data['ends'],snapshot.stem)
            del bank, data


if __name__=='__main__':main()
