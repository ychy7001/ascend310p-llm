# SPDX-License-Identifier: Apache-2.0
"""External MindIEQ QMM check: full A8 range, real expert shapes, host-sync timing.
Imports the pinned external packing helper; does not redistribute its code.
"""
import argparse
import importlib.util
import json
import time
from statistics import median
import numpy as np
import torch
import torch_npu

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--upstream-test', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
spec = importlib.util.spec_from_file_location('external_qmm', a.upstream_test)
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)
torch.set_num_threads(4)
rng = np.random.RandomState(20261013)

def bench(fn):
    for _ in range(5): fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(30):
        start = time.perf_counter()
        fn(); torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    return samples

with open(a.out, 'x') as out:
    for projection, k, n in [('gate_up', 2048, 1024), ('down', 512, 2048)]:
        w = rng.randint(-8, 8, (k, n)).astype(np.int8)
        ws = rng.uniform(.001, .01, n).astype(np.float32)
        packed = torch.from_numpy(ext.pack_biased_nz_int32_vec(w, k, n).copy()).npu()
        w8 = torch_npu.npu_format_cast(torch.from_numpy(w.T.copy()).npu(), 29).T
        scales = torch.from_numpy(ws).npu()
        for m in (1, 2, 4, 8, 128, 512, 1024, 4096):
            x = rng.randint(-128, 128, (m, k)).astype(np.int8)
            x[0, 0:4] = [-128, -127, 126, 127]
            xs = rng.uniform(.001, .01, m).astype(np.float32)
            golden = torch.from_numpy((x.astype(np.int32) @ w.astype(np.int32)).astype(np.float32) * xs[:, None] * ws[None, :])
            tx, ts = torch.from_numpy(x).npu(), torch.from_numpy(xs).npu()
            def q4(): return ext.OP(tx, ts, packed, scales)
            actual = q4().cpu().float()
            torch.testing.assert_close(actual, golden, rtol=.002, atol=.002)
            row = dict(projection=projection, m=m, k=k, n=n, status='PASS',
                       max_abs=(actual-golden).abs().max().item(),
                       weight_bytes_w4=packed.numel()*4, weight_bytes_w8=w8.numel(),
                       timing_scope='host dispatch + launch + synchronize; packing excluded', warmup=5, repeats=30)
            row['w4_us'] = bench(q4)
            row['w4_median_us'] = median(row['w4_us'])
            # Same integer codes/scales, resident W8 storage; includes layout conversion if any.
            try:
                def q8(): return torch_npu.npu_quant_matmul(tx, w8, scales, pertoken_scale=ts, output_dtype=torch.float16)
                torch.testing.assert_close(q8().cpu().float(), golden, rtol=.002, atol=.002)
                row['w8_us'] = bench(q8)
                row['w8_median_us'] = median(row['w8_us'])
                row['w8_over_w4'] = row['w8_median_us']/row['w4_median_us']
            except RuntimeError as error:
                row['w8_error'] = str(error)
            out.write(json.dumps(row)+'\n'); out.flush()
            print(json.dumps({key:value for key,value in row.items() if key not in ('w4_us','w8_us')}), flush=True)
