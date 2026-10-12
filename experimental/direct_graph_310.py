# SPDX-License-Identifier: Apache-2.0
"""Align direct FULL capture with attention graph-parameter initialization."""
import vllm_ascend.ops  # noqa: F401
from vllm.config.compilation import CompilationMode
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

_original_use_aclgraph = NPUModelRunner._use_aclgraph


def use_aclgraph(self):
    if (self.compilation_config.mode == CompilationMode.NONE
            and self.compilation_config.cudagraph_mode.has_full_cudagraphs()
            and not self.model_config.enforce_eager):
        return True
    return _original_use_aclgraph(self)


NPUModelRunner._use_aclgraph = use_aclgraph
