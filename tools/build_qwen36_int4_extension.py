# SPDX-License-Identifier: Apache-2.0
"""Build the independent qwen36_int4 Torch extension inside the vLLM image.

The OpenSensor checkout supplies the adapter headers and CANN bridge; the
official vLLM-Ascend extension must remain loaded under its original namespace.
"""
import argparse
from pathlib import Path

from torch.utils.cpp_extension import load
import torch_npu

parser = argparse.ArgumentParser()
parser.add_argument('--community-root', type=Path, required=True)
parser.add_argument('--source', type=Path, default=Path(__file__).with_name('qwen36_int4_custom.cpp'))
parser.add_argument('--build', type=Path, default=Path('/tmp/qwen36-int4-build'))
args = parser.parse_args()
npu_root = Path(torch_npu.__file__).resolve().parent
load(
    name='qwen36_int4_custom',
    sources=[str(args.source), str(args.community_root / 'csrc/aclnn_torch_adapter/NPUBridge.cpp')],
    extra_include_paths=[str(args.community_root / 'csrc'), str(args.community_root / 'csrc/common/include'),
                         '/usr/local/Ascend/ascend-toolkit/latest/include', str(npu_root / 'include')],
    extra_cflags=['-O2', '-fvisibility=hidden'],
    extra_ldflags=[f'-L{npu_root / "lib"}', '-ltorch_npu'],
    is_python_module=False, verbose=True, build_directory=str(args.build),
)
