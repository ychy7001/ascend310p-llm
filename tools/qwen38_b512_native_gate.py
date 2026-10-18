# SPDX-License-Identifier: Apache-2.0
"""Two-rank native W4 grouped arithmetic gate at 512 tokens x top10 routes.

Uses the pinned community test's independent CPU reference and synthetic weights.
This is not a full-model accuracy or performance benchmark.
"""
import importlib.util
import json
import os
from pathlib import Path
import torch
import torch_npu  # noqa: F401
from vllm_ascend.utils import enable_custom_op


def main():
    torch.set_num_threads(1)
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.npu.set_device(rank)
    torch.npu.set_compile_mode(jit_compile=False)
    enable_custom_op()
    path = Path('/experiment/community/tests/e2e/nightly/310p/single_node/ops/'
                'test_qwen_w4_grouped_native_310.py')
    spec = importlib.util.spec_from_file_location('pinned_native_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = []
    for inputs, outputs in ((2560, 1280), (640, 2560)):
        for ends_list in ([5, 17, 32], [1024, 1537, 2560]):
            data = module.values(5120, inputs=inputs, outputs=outputs)
            ends = torch.tensor(ends_list, dtype=torch.int64)
            actual = module.invoke(module.device_arguments(data, ends, True), True).cpu()
            expected = module.reference(data, ends, True)
            torch.testing.assert_close(actual, expected, rtol=.005, atol=.003)
            assert torch.isfinite(actual).all()
            assert torch.count_nonzero(actual[ends_list[-1]:]) == 0
            records.append({'rank': rank, 'routes': 5120, 'inputs': inputs,
                            'outputs': outputs, 'group_ends': ends_list,
                            'max_abs_error': (actual.float() - expected.float()).abs().max().item(),
                            'relative_l2': ((actual.float() - expected.float()).norm() /
                                            expected.float().norm()).item(), 'pass': True})
            print(json.dumps(records[-1]), flush=True)
    output = Path('/experiment/b512-native')
    output.mkdir(exist_ok=True)
    (output / f'native-rank{rank}.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
