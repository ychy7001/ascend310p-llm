# SPDX-License-Identifier: Apache-2.0
"""按需记录真实 grouped Prefill 路由；CPU 同步使该运行不能用于测速。"""
import json
import os
from pathlib import Path

import torch

from vllm_ascend.models.qwen4_exp.w4_moe import W4SparseMoE


trace = Path(os.environ["QWEN36_ROUTE_TRACE_PATH"])
marker = Path(str(trace) + ".enabled")
limit = int(os.environ.get("QWEN36_ROUTE_TRACE_CALLS", "256"))
original = W4SparseMoE._forward_grouped
modules = {}
calls = 0


def traced(self, block_input, weights, ids):
    global calls
    # Marker stays absent during model profiling / startup graph capture.
    if marker.exists() and calls < limit:
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("路由诊断不允许在图捕获内同步")
        module = modules.setdefault(id(self), len(modules))
        for start in range(0, block_input.shape[0], self.grouped_chunk_tokens):
            selected = ids[start:start + self.grouped_chunk_tokens].flatten()
            counts = torch.bincount(selected.long(), minlength=self.num_experts).cpu().tolist()
            row = {
                "kind": "route_occupancy", "pid": os.getpid(),
                "module_ordinal": module, "call": calls,
                "call_tokens": block_input.shape[0], "chunk_start": start,
                "chunk_tokens": min(self.grouped_chunk_tokens, block_input.shape[0] - start),
                "top_k": self.top_k, "num_experts": self.num_experts,
                "expert_offset": self.expert_offset,
                "num_local_experts": self.num_local_experts,
                "counts": counts, "timing_valid": False,
            }
            with trace.open("a") as output:
                output.write(json.dumps(row) + "\n")
        calls += 1
    return original(self, block_input, weights, ids)


W4SparseMoE._forward_grouped = traced
