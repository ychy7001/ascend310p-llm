# SPDX-License-Identifier: Apache-2.0
"""Load W4 without placing a shadow vllm_ascend package on PYTHONPATH."""
import os
from pathlib import Path
import vllm_ascend
import vllm_ascend.models
import torch
import torch_npu  # noqa: F401

torch.ops.load_library(os.environ.get('QWEN36_EXTENSION', '/custom/qwen36_int4_custom.so'))
import w4_fake  # noqa: E402,F401
overlay = Path(os.environ.get('QWEN36_RUNTIME', '/w4only'))
vllm_ascend.models.__path__ = [
    str(overlay / 'vllm_ascend/models'), *list(vllm_ascend.models.__path__),
]
import vllm_ascend.models.qwen36_w4_patch  # noqa: E402,F401

if os.environ.get('RC_BLOCKING_MM_H2D', '0') == '1':
    import rc_blocking_mm_h2d  # noqa: E402,F401
