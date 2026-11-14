# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for historical probes; delivery source lives in runtime/."""
import importlib.util
from pathlib import Path
import sys

_spec = importlib.util.spec_from_file_location(
    __name__, Path(__file__).resolve().parents[2] / 'runtime/wy_compat/eager_mask_reuse.py'
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _module
_spec.loader.exec_module(_module)
globals().update(vars(_module))
