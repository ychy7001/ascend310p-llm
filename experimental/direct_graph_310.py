# SPDX-License-Identifier: Apache-2.0
"""Align direct FULL capture with attention graph-parameter initialization.

The optional force switch is a diagnostic for images where the 310P attention
backend is an updatable graph but the generic parameter-update path is still
selected. It is disabled by default and must not be used as a production fix
without capture/replay and numerical evidence.
"""
import os
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

if os.environ.get('QWEN36_C8_FORCE_UPDATABLE_GRAPH', '0') == '1':
    import vllm_ascend.compilation.acl_graph as acl_graph

    # update_full_graph_params consults this module-global function. Keep the
    # switch explicit so ordinary Graph and eager runs retain upstream behavior.
    acl_graph.use_updatable_graph = lambda _attn_backend: True
