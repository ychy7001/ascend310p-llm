# SPDX-License-Identifier: Apache-2.0
"""Probe 310P Cube QK packing and CO1-to-UB readback against FP32 reference."""
import argparse
import json
import torch
import torch_npu  # noqa: F401

p = argparse.ArgumentParser()
p.add_argument('--library', required=True)
p.add_argument('--profile-dir')
a = p.parse_args()
torch.ops.load_library(a.library)
torch.manual_seed(113)
chosen = None
for dim in (16, 128, 256):
    q = torch.randn(16, dim).half().npu()
    k = torch.randn(16, dim).half().npu()
    reference = q.cpu().float() @ k.cpu().float().T
    candidates = []
    for transpose_b in (False, True):
        actual = torch.ops.npu.c8_qk_cube(q, k, transpose_b).cpu()
        for transpose_out in (False, True):
            output = actual.T if transpose_out else actual
            error = (output-reference).abs().max().item()
            candidates.append((error, transpose_b, transpose_out))
            print(json.dumps({'dim': dim, 'transpose_b': transpose_b,
                              'transpose_out': transpose_out, 'max_abs': error,
                              'finite': bool(torch.isfinite(output).all())}), flush=True)
    passing = [item for item in candidates if item[0] < .003]
    assert passing, ('no Cube packing candidate passed', dim, candidates)
    best = min(passing)
    assert chosen is None or chosen == best[1:], ('layout changed with dim', chosen, best)
    chosen = best[1:]
    print(json.dumps({'status': 'PASS', 'dim': dim, 'layout': chosen,
                      'max_abs': best[0]}), flush=True)

transpose_b, transpose_out = chosen
for _ in range(3):
    torch.ops.npu.c8_qk_cube(q, k, transpose_b)
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(graph):
    captured = torch.ops.npu.c8_qk_cube(q, k, transpose_b)
for iteration in range(3):
    q.add_(.01)
    k.add_(.02)
    graph.replay()
    torch.npu.synchronize()
    actual = captured.cpu()
    if transpose_out:
        actual = actual.T
    error = (actual - q.cpu().float() @ k.cpu().float().T).abs().max().item()
    assert error < .003, error
    print(json.dumps({'status': 'PASS', 'graph_replay': True,
                      'iteration': iteration, 'max_abs': error}), flush=True)

# PV consumes a [query rows, token] probability tile and [token, output dim]
# V tile; for K=16, transposing B's basic block gives P @ V directly.
probability = torch.randn(16, 16).softmax(-1).half().npu()
value = torch.randn(16, 16).half().npu()
reference = probability.cpu().float() @ value.cpu().float()
actual = torch.ops.npu.c8_qk_cube(probability, value, True).cpu()
error = (actual - reference).abs().max().item()
assert error < .003, ('PV Cube tile failed', error)
print(json.dumps({'status': 'PASS', 'path': 'PV Cube tile', 'transpose_b': True,
                  'max_abs': error}), flush=True)

if a.profile_dir:
    for _ in range(3):
        torch.ops.npu.c8_qk_cube(probability, value, True)
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(a.profile_dir),
    ) as profiler:
        for _ in range(5):
            torch.ops.npu.c8_qk_cube(q, k, False)
            torch.ops.npu.c8_qk_cube(probability, value, True)
        torch.npu.synchronize()
        profiler.step()
    print(json.dumps({'profile_dir': a.profile_dir, 'QK_PV_pairs': 5}), flush=True)
