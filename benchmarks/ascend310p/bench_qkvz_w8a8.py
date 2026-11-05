#!/usr/bin/env python3
"""QKVZ-only TP2 weight preparation and full-cost operator screening (no model B)."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def prepare(model, output):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=False)
    index_path = model / 'quant_model_weights.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map']
    config = json.loads((model / 'config.json').read_text())['text_config']
    assert config['hidden_size'] == 2048
    qk = config['linear_key_head_dim'] * config['linear_num_key_heads']
    v = config['linear_value_head_dim'] * config['linear_num_value_heads']
    assert (qk, v) == (2048, 4096)
    rows = []
    keys = sorted(k for k in index if k.endswith('.linear_attn.in_proj_qkv.weight')
                  and '.mtp.' not in k)
    assert len(keys) == 30, len(keys)
    for key in keys:
        zkey = key.replace('in_proj_qkv.', 'in_proj_z.')
        with safe_open(model / index[key], framework='pt', device='cpu') as src:
            qkv_source = src.get_tensor(key)
            qkv = qkv_source.to(torch.float16)
        with safe_open(model / index[zkey], framework='pt', device='cpu') as src:
            z_source = src.get_tensor(zkey)
            z = z_source.to(torch.float16)
        assert qkv.shape == (8192, 2048) and z.shape == (4096, 2048)
        for rank in (0, 1):
            pieces = [part.chunk(2, dim=0)[rank]
                      for part in (*qkv.split((qk, qk, v)), z)]
            weight = torch.cat(pieces).contiguous()
            assert weight.shape == (6144, 2048)
            absmax = weight.float().abs().amax(dim=1)
            zero = absmax == 0
            scale = absmax / 127
            # Exact representation of production FP16 zero rows; no floor on other rows.
            scale[zero] = 1.0
            quant = (weight.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
            assert (quant[zero] == 0).all() and (scale[zero] == 1).all()
            source_parts = [part.chunk(2, dim=0)[rank]
                            for part in (*qkv_source.split((qk, qk, v)), z_source)]
            source_absmax = torch.cat([part.float().abs().amax(dim=1) for part in source_parts])
            underflow = zero & (source_absmax > 0)
            name = f'layer-{key.split(".layers.")[1].split(".")[0]}-rank-{rank}.safetensors'
            save_file({'fp16': weight, 'int8': quant, 'scale': scale}, output / name)
            rows.append({'module': key.removesuffix('.weight').replace('in_proj_qkv', 'in_proj_qkvz'),
                         'rank': rank, 'file': name, 'fp16_sha256': digest(weight),
                         'int8_sha256': digest(quant), 'scale_sha256': digest(scale),
                         'fp16_zero_channels': int(zero.sum()),
                         'bf16_to_fp16_underflow_channels': int(underflow.sum()),
                         'bf16_underflow_absmax_max': float(source_absmax[underflow].max()) if underflow.any() else 0.,
                         'zero_int8_exact': bool((quant[zero] == 0).all()),
                         'zero_scale_exact': bool((scale[zero] == 1).all())})
    write(output / 'manifest.json', {
        'bf16_to_fp16_underflow_channels_total': sum(r['bf16_to_fp16_underflow_channels'] for r in rows),
        'bf16_underflow_absmax_max': max(r['bf16_underflow_absmax_max'] for r in rows),
        'index_sha256': hashlib.sha256(index_path.read_bytes()).hexdigest(),
        'policy': 'production FP16 source; symmetric per-output-channel absmax/127; exactly zero rows: INT8=0, scale=1; no floor on nonzero rows; nearest-even; no clipping tuning; dynamic per-token activation absmax',
        'calibration': 'none: weight-only scale selection; synthetic seeded Gaussian operator inputs',
        'tp_sharding': 'split Q/K/V/Z separately across two ranks, then concatenate',
        'rows': rows})


def benchmark(weights, output, rank, gate=False):
    import torch
    import torch_npu
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    torch.manual_seed(310)
    torch.npu.set_device(rank)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    manifest = json.loads((weights / 'manifest.json').read_text())
    entries = [r for r in manifest['rows'] if r['rank'] == rank]
    assert len(entries) == 30
    cpu = [load_file(weights / row['file']) for row in entries]
    # The validated 310P layout is NZ([N,K]).T for INT8, NZ([N,K]) for FP16.
    fp = [torch_npu.npu_format_cast(w['fp16'].npu(), 29) for w in cpu]
    qw = [torch_npu.npu_format_cast(w['int8'].npu(), 29).T for w in cpu]
    scales = [w['scale'].npu() for w in cpu]
    result = {'rank': rank, 'scope': '30 real projection weights; synthetic inputs; isolated operator screen',
              'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
              'manifest_sha256': hashlib.sha256((weights / 'manifest.json').read_bytes()).hexdigest(),
              'action': 'g0' if gate else 'bench',
              'g0_rule': {'quant_reference_atol': .002, 'quant_reference_rtol': .002,
                          'fp32_original_nrmse_max': .03, 'fp32_original_cosine_min': .999,
                          'repeat': 'bitwise FP16 outputs, 3 calls'}, 'g0': [], 'timings': []}
    write(output, result)
    for batch in (1, 2, 3, 4):
        xc = torch.randn(batch, 2048, dtype=torch.float16)
        x = xc.npu()
        if gate:
            # Validate every layer and every output element; no sampling or silent truncation.
            for i, w in enumerate(cpu):
                qx, xs = torch_npu.npu_dynamic_quant(x)
                y = torch_npu.npu_quant_matmul(qx, qw[i], scales[i], pertoken_scale=xs,
                                             output_dtype=torch.float16)
                yc = y.cpu().float()
                quant_reference = ((qx.cpu().float() @ w['int8'].float().T)
                                   * xs.cpu()[:, None] * w['scale'][None, :])
                torch.testing.assert_close(yc, quant_reference.half().float(), atol=.002, rtol=.002)
                original = xc.float() @ w['fp16'].float().T
                nrmse = ((yc-original).square().mean() / original.square().mean()).sqrt().item()
                cosine = torch.nn.functional.cosine_similarity(yc.flatten(), original.flatten(), dim=0).item()
                repeated = all(torch.equal(y.cpu(), torch_npu.npu_quant_matmul(
                    *[qx, qw[i], scales[i]], pertoken_scale=xs, output_dtype=torch.float16).cpu())
                    for _ in range(2))
                # Repeat the entire changed path too, including dynamic quantization.
                for _ in range(2):
                    q2, s2 = torch_npu.npu_dynamic_quant(x)
                    y2 = torch_npu.npu_quant_matmul(q2, qw[i], scales[i], pertoken_scale=s2,
                                                  output_dtype=torch.float16)
                    repeated = repeated and torch.equal(y.cpu(), y2.cpu())
                fp_y = torch.nn.functional.linear(x, fp[i]).cpu().float()
                torch.testing.assert_close(fp_y, original.half().float(), atol=.002, rtol=.002)
                row = {'batch': batch, 'module': entries[i]['module'], 'nrmse': nrmse,
                       'cosine': cosine, 'finite': bool(torch.isfinite(yc).all()),
                       'bitwise_repeat': repeated,
                       'quant_reference_max_abs': (yc-quant_reference).abs().max().item()}
                result['g0'].append(row)
                write(output, result)
                assert nrmse <= .03 and cosine >= .999 and row['finite'] and repeated, row
            continue
        def a():
            return [torch.nn.functional.linear(x, weight) for weight in fp]
        def b():
            outputs = []
            for weight, scale in zip(qw, scales, strict=True):
                quant, token_scale = torch_npu.npu_dynamic_quant(x)
                outputs.append(torch_npu.npu_quant_matmul(
                    quant, weight, scale, pertoken_scale=token_scale, output_dtype=torch.float16))
            return outputs
        calls = {'A': a, 'B': b}
        # Production graph sizes [1,2]; B3/B4 are measured eagerly as separate strata.
        graphs = {}
        for arm in ('A', 'B'):
            for _ in range(8): calls[arm]()
            torch.npu.synchronize()
            if batch in (1, 2):
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    retained = calls[arm]()
                graphs[arm] = (graph, retained)
        for round_id, order in enumerate(('AB', 'BA', 'AB')):
            times = {}
            for arm in order:
                fn = graphs[arm][0].replay if graphs else calls[arm]
                for _ in range(8): fn()
                torch.npu.synchronize()
                start = time.perf_counter()
                for _ in range(256): fn()
                torch.npu.synchronize()
                times[arm] = (time.perf_counter() - start) * 1000 / 256
            result['timings'].append({'batch': batch, 'round': round_id, 'order': order,
                                      'mode': 'graph' if graphs else 'eager',
                                      'steps': 256, 'ms_per_30_projections': times,
                                      'speedup': times['A']/times['B']})
            write(output, result)
        del graphs
    result['g0_passed'] = True if gate else None
    result['median_speedup_by_batch'] = {} if gate else {
        str(b): statistics.median(t['speedup'] for t in result['timings'] if t['batch'] == b)
        for b in (1, 2, 3, 4)}
    write(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--model', type=Path, required=True)
    prep.add_argument('--weights', type=Path, required=True)
    for action in ('bench', 'g0'):
        bench = sub.add_parser(action)
        bench.add_argument('--weights', type=Path, required=True)
        bench.add_argument('--output', type=Path, required=True)
        bench.add_argument('--rank', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if args.action == 'prepare': prepare(args.model.resolve(), args.weights.resolve())
    else: benchmark(args.weights, args.output, args.rank, gate=args.action == 'g0')


if __name__ == '__main__':
    main()
