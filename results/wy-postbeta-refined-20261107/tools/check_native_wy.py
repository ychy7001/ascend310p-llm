# SPDX-License-Identifier: Apache-2.0
"""生产栈 WY 与完整 GDN 数值门禁；CPU 逐 token 递推独立参考。"""
import argparse
import importlib
import json
from pathlib import Path

import torch
import torch_npu
from native_wy import install, supported


def recurrent_reference(q, k, v, g, beta, state, cu):
    """直接递推，不复用 chunk/WY/padding 实现。state 为 [N,Hv,V,K]。"""
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    ratio = v.shape[2] // q.shape[2]
    q, k = (x.repeat_interleave(ratio, dim=2) for x in (q, k))
    outputs = torch.empty_like(v)
    final = []
    for n, (start, end) in enumerate(zip(cu[:-1], cu[1:])):
        h = state[n].float().clone()
        for t in range(start, end):
            h = h * g[0, t].exp()[:, None, None]
            delta = (v[0, t] - (h * k[0, t, :, None, :]).sum(-1)) * beta[0, t, :, None]
            h = h + delta[:, :, None] * k[0, t, :, None, :]
            outputs[0, t] = (h * q[0, t, :, None, :]).sum(-1) * q.shape[-1] ** -.5
        final.append(h)
    return outputs, torch.stack(final)


def compare(actual, expected, *, atol=.0002, rtol=.04):
    records = []
    for a, e in zip(actual, expected):
        a, e = a.detach().cpu().float(), e.detach().cpu().float()
        if not torch.isfinite(a).all() or not torch.isfinite(e).all():
            raise AssertionError('non-finite outputs')
        torch.testing.assert_close(a, e, atol=atol, rtol=rtol)
        rmse = (a-e).square().mean().sqrt().item()
        scale = e.square().mean().sqrt().item()
        relative_rmse = rmse / max(scale, 1e-30)
        if relative_rmse > .02:
            raise AssertionError(f'relative RMSE {relative_rmse} > 0.02')
        records.append({'max_abs': (a-e).abs().max().item(), 'rmse': rmse,
                        'relative_rmse': relative_rmse, 'atol': atol, 'rtol': rtol})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    from vllm_ascend.utils import enable_custom_op
    if not enable_custom_op():
        raise RuntimeError('Production custom ops bootstrap failed')
    torch.npu.set_device(0)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    module = importlib.import_module('vllm_ascend._310p.ops.fla.chunk_gated_delta_rule')
    original = module._compute_kernel_inputs_from_torch_wy
    install(a.library)
    native = module._compute_kernel_inputs_from_torch_wy
    cases = [(64,8,16,.01,None), (111,8,16,1.,None), (128,16,32,1.,None),
             (512,8,16,8.,None), (177,8,16,1.,[0,1,65,177]),
             (128,8,24,1.,None)]
    with a.out.open('x') as f:
        for idx, (tokens,hk,hv,decay,cu) in enumerate(cases):
            torch.manual_seed(310+idx)
            q = torch.nn.functional.normalize(torch.randn(1,tokens,hk,128),dim=-1).half()
            k = torch.nn.functional.normalize(torch.randn_like(q).float(),dim=-1).half()
            v = (torch.randn(1,tokens,hv,128)*.1).half()
            g = -torch.rand(1,tokens,hv)*decay
            beta = (.05+.9*torch.rand(1,tokens,hv)).half()
            cu = cu or [0,tokens]
            state = (torch.randn(len(cu)-1,hv,128,128)*.01).half()
            expected = recurrent_reference(q,k,v,g,beta,state,cu)
            device = [x.npu() for x in (q,k,v,g,beta)]
            kw = dict(initial_state=state.npu(), output_final_state=True,
                      head_first=False, use_qk_l2norm_in_kernel=False)
            if len(cu)>2:
                kw['cu_seqlens'] = torch.tensor(cu,dtype=torch.int64,device='npu')
            module._compute_kernel_inputs_from_torch_wy = original
            baseline = module.chunk_gated_delta_rule_310(*device,**kw)
            module._compute_kernel_inputs_from_torch_wy = native
            native_calls = []
            def counted(*args):
                native_calls.append(supported(*args))
                return native(*args)
            module._compute_kernel_inputs_from_torch_wy = counted
            candidate = module.chunk_gated_delta_rule_310(*device,**kw)
            if not native_calls or not all(native_calls):
                raise AssertionError(f'native WY not exercised: {native_calls}')
            # 已复现原 Torch WY 路径的偶发偏差。保留其失败，不能以错误基线
            # 要求候选复现偏差；候选仍须严格满足独立参考与逐位重复检查。
            try:
                baseline_metrics = compare(baseline,expected)
                baseline_error = None
            except AssertionError as error:
                baseline_metrics = None
                baseline_error = str(error)
            if baseline_error is None:
                comparison = compare(candidate,baseline)
            else:
                comparison = None
            row = dict(case=idx,tokens=tokens,hk=hk,hv=hv,decay=decay,cu=cu,
                       nonzero_initial_state=True,native_calls=len(native_calls),
                       baseline_vs_reference=baseline_metrics, baseline_reference_error=baseline_error,
                       candidate_vs_reference=compare(candidate,expected),
                       candidate_vs_baseline=comparison,
                       candidate_baseline_comparison='strict' if baseline_error is None else 'baseline_deviation_recorded_use_independent_reference')
            for _ in range(3):
                repeated = module.chunk_gated_delta_rule_310(*device,**kw)
                compare(repeated,candidate,atol=0,rtol=0)
            row['deterministic_repeats']=3
            row['status']='PASS'
            f.write(json.dumps(row)+'\n');f.flush()
            print(json.dumps(row),flush=True)
            module._compute_kernel_inputs_from_torch_wy = native
        # Standalone WY exact q/k layout and numerical accuracy, including D=64.
        for kd,vd in ((64,64),(64,128),(128,128)):
            torch.manual_seed(kd+vd)
            inputs = [torch.nn.functional.normalize(torch.randn(1,128,8,kd),dim=-1).half() for _ in range(2)]
            inputs += [(torch.randn(1,128,16,vd)*.1).half(),-torch.rand(1,128,16),torch.rand(1,128,16).half()]
            expected = original(*inputs,64)
            candidate = native(*(x.npu() for x in inputs),64)
            compare(candidate[:2],expected[:2],atol=0,rtol=0)
            errors=compare(candidate[2:],expected[2:],atol=.0005,rtol=.04)
            row=dict(kind='wy_layout',kd=kd,vd=vd,errors=errors,status='PASS')
            f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
        # Unsupported CPU inputs must retain original behavior without calling custom op.
        actual=native(*inputs,64)
        compare(actual,expected,atol=0,rtol=0)
        row=dict(kind='cpu_fallback',status='PASS')
        f.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
