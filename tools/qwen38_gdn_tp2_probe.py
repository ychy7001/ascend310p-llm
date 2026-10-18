# SPDX-License-Identifier: Apache-2.0
"""Check Qwen3.8 TP2-local GDN geometry against independent CPU recurrence.
Run with torchrun --standalone --nproc-per-node=2, in the pinned runtime.
This is a component gate; it is not a whole-model quality/performance test.
"""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch_npu  # noqa: F401
import vllm_ascend.vllm_ascend_C  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, nargs='+', default=[1,2,3,4,8,32,64,70,128,256])
    args = parser.parse_args()
    if any(tokens < 1 or tokens > 4096 for tokens in args.tokens):
        parser.error('Token counts must be between 1 and 4096')
    torch.set_num_threads(1)
    rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(rank)
    torch.npu.set_compile_mode(jit_compile=False)
    dist.init_process_group('hccl')
    marker = torch.tensor([rank+1.0],device=f'npu:{rank}')
    dist.all_reduce(marker)
    assert marker.item() == 3, marker
    records = []
    for tokens in args.tokens:
        torch.manual_seed(108+tokens+rank)
        hk, hv, dk, dv = 8,24,128,128
        q = torch.nn.functional.normalize(torch.randn(tokens,hk,dk),dim=-1).half()
        k = torch.nn.functional.normalize(torch.randn(tokens,hk,dk),dim=-1).half()
        v = torch.randn(tokens,hv,dv).half()
        beta = torch.sigmoid(torch.randn(tokens,hv)).half()
        g = -torch.rand(tokens,hv)*.2
        state = torch.randn(tokens+4,hv,dv,dk)*.1
        indices = torch.arange(tokens,dtype=torch.int32)
        seq = torch.tensor([tokens],dtype=torch.int32)
        ref_state = state.clone()
        s = ref_state[0].clone()
        qf = q.float().repeat_interleave(hv//hk,dim=1)*(dk**-.5)
        kf = k.float().repeat_interleave(hv//hk,dim=1)
        ref_out = []
        for t in range(tokens):
            s *= g[t].exp()[:,None,None]
            delta = (v[t].float()-(s*kf[t][:,None,:]).sum(-1))*beta[t].float()[:,None]
            s += delta[:,:,None]*kf[t][:,None,:]
            ref_state[t] = s
            ref_out.append((s*qf[t][:,None,:]).sum(-1))
        ref_out = torch.stack(ref_out)
        device = f'npu:{rank}'
        qn,kn,vn,bn,sn,ln,idx,gn = [t.to(device) for t in (q,k,v,beta,state,seq,indices,g)]
        if tokens <= 8:
            out = torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(qn,kn,vn,bn,sn,ln,idx,gn,None,None,dk**-.5)
            torch.npu.synchronize()
            got=out.float().cpu(); got_state=sn.cpu()
            torch.testing.assert_close(got_state[:tokens],ref_state[:tokens],rtol=3e-3,atol=2e-3)
            torch.testing.assert_close(got_state[tokens:],state[tokens:],rtol=0,atol=0)
            state_error=(got_state[:tokens]-ref_state[:tokens]).abs().max().item()
            path='recurrent_decode'
        else:
            from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import chunk_gated_delta_rule_310
            out, final_state = chunk_gated_delta_rule_310(
                q=qn.unsqueeze(0), k=kn.unsqueeze(0), v=vn.unsqueeze(0),
                g=gn.unsqueeze(0), beta=bn.unsqueeze(0),
                initial_state=sn[:1].contiguous(), output_final_state=True,
                cu_seqlens=torch.tensor([0,tokens],dtype=torch.int32,device=device),
                head_first=False, use_qk_l2norm_in_kernel=False)
            torch.npu.synchronize()
            got=out.squeeze(0).float().cpu()
            got_final=final_state.cpu()[0]
            # Both output and state errors are recorded before enforcing the gate.
            state_error=(got_final-ref_state[tokens-1]).abs().max().item()
            path='chunk_prefill'
        diff=(got-ref_out).abs()
        bad=diff > (2e-3 + 3e-3*ref_out.abs())
        records.append({'tokens':tokens,'path':path,'state_dtype':'float32','hk':hk,'hv':hv,'relative_l2':((got-ref_out).norm()/ref_out.norm()).item(),'state_max_abs_error':state_error,'max_abs_error':diff.max().item(),'bad_elements':bad.sum().item(),'bad_token_indices':bad.any(-1).any(-1).nonzero().flatten().tolist(),'output_pass':not bad.any().item()})
        if tokens == 64:
            from vllm_ascend._310p.ops.fla.chunk_gated_delta_rule import _compute_kernel_inputs_from_torch_wy
            cpu_wy = _compute_kernel_inputs_from_torch_wy(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),g.unsqueeze(0),beta.unsqueeze(0),64)
            npu_wy = _compute_kernel_inputs_from_torch_wy(qn.unsqueeze(0),kn.unsqueeze(0),vn.unsqueeze(0),gn.unsqueeze(0),bn.unsqueeze(0),64)
            torch.npu.synchronize()
            records[-1]['wy_inputs'] = {name:{'relative_l2':((a.float()-b.float().cpu()).norm()/a.float().norm().clamp_min(1e-30)).item(),'max_abs':(a.float()-b.float().cpu()).abs().max().item()} for name,a,b in zip(('q','k','w','u','g'),cpu_wy,npu_wy)}
        print(json.dumps({'rank':rank,**records[-1]}),flush=True)
    from qwen38_chunk_causal_probe import run as check_causal_mask
    causal_pass=check_causal_mask(rank,os.environ.get('QWEN38_PROBE_OUTPUT','/experiment'))
    # Barriers also exercise synchronization after the actual kernels.
    dist.barrier(); dist.destroy_process_group()
    output = Path(os.environ.get('QWEN38_PROBE_OUTPUT','/experiment'))
    (output/f'gdn-tp2-rank{rank}.json').write_text(json.dumps({'rank':rank,'hccl_sum':3,'blocked_inverse':os.environ.get('VLLM_ASCEND_GDN_UT_BLOCKED','1'),'cases':records},indent=2))
    if not causal_pass or not all(r['output_pass'] for r in records):
        raise AssertionError('GDN numerical gate failed; see recorded metrics')

if __name__ == '__main__': main()
