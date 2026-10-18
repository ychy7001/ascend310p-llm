# SPDX-License-Identifier: Apache-2.0
"""Host-only probe of physical-to-planner KV budgets, including single live state."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec
from vllm_ascend.worker.worker import NPUWorker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    mamba = MambaSpec(block_size=2, shapes=((20, 4),), dtypes=(torch.float32,))
    attention = FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=4,
                                  head_size_v=4, dtype=torch.float16,
                                  page_size_padded=mamba.page_size_bytes)
    groups = [KVCacheGroupSpec(['attn'], attention), KVCacheGroupSpec(['gdn'], mamba)]
    results = []
    for prefix, requests, slots, live_mapping in [(False, 1, 1, False), (False, 4, 3, True), (True, 2, 64, False)]:
        worker = NPUWorker.__new__(NPUWorker)
        worker.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(
            enable_prefix_caching=prefix, get_resolved_kv_cache_layout=lambda:
                SimpleNamespace(is_layer_compact=True, is_block_compact=True)))
        worker.model_runner = SimpleNamespace(supports_compact_mamba_state=True,
            supports_prefix_mamba_state_tier=prefix, num_compact_mamba_blocks=slots,
            max_num_reqs=requests, _live_mamba_slots=object() if live_mapping else None)
        worker.get_kv_cache_spec = lambda: {'attn': attention, 'gdn': mamba}
        physical_budget = 123450
        fixed = slots * (1 if prefix else requests) * mamba.page_size_bytes
        expected_blocks = (physical_budget - fixed) // attention.real_page_size_bytes
        with patch('vllm_ascend.worker.worker.get_kv_cache_groups', return_value=groups):
            actual = worker._scale_kv_cache_memory_for_multi_group(physical_budget)
        record = dict(prefix=prefix, requests=requests, slots=slots,
                      physical_budget=physical_budget, fixed_state_bytes=fixed,
                      actual_planner_budget=actual,
                      expected_planner_budget=expected_blocks*mamba.page_size_bytes,
                      passed=actual == expected_blocks*mamba.page_size_bytes)
        results.append(record)
        print(json.dumps(record), flush=True)
    args.output.write_text(json.dumps({'scope': 'budget arithmetic, not NPU scheduling', 'cases': results}, indent=2))
    if not all(r['passed'] for r in results):
        raise AssertionError('Compact state planner budget mismatch')


if __name__ == '__main__':
    main()
