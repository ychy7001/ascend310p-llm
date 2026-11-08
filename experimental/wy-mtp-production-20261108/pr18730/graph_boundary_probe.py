"""Diagnostic only: find the first failed boundary outside ACL capture."""
import functools
import json
import torch
from vllm.forward_context import get_forward_context
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
from vllm_ascend.spec_decode.llm_base_proposer import AscendSpecDecodeBaseProposer


def sync(label, **info):
    try:
        torch.npu.synchronize()
    except Exception:
        print('[boundary-failed] ' + json.dumps({'boundary': label, **info}), flush=True)
        raise


_graph_call = ACLGraphWrapper.__call__


@functools.wraps(_graph_call)
def graph_call(self, *args, **kwargs):
    context = get_forward_context()
    active = not context.capturing and context.cudagraph_runtime_mode == self.runtime_mode
    info = {'mode': str(self.runtime_mode), 'batch': str(getattr(context, 'batch_descriptor', None))}
    if active:
        sync('graph.before', **info)
    result = _graph_call(self, *args, **kwargs)
    if active and not context.capturing:
        sync('graph.after', **info)
    return result


ACLGraphWrapper.__call__ = graph_call


def wrap_entry(cls, name, label):
    original = getattr(cls, name)

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        sync(label)
        return original(self, *args, **kwargs)

    setattr(cls, name, wrapped)


wrap_entry(NPUModelRunner, 'sample_tokens', 'sample_tokens.before')
wrap_entry(NPUModelRunner, 'propose_draft_token_ids', 'runner.propose.before')
wrap_entry(AscendSpecDecodeBaseProposer, '_propose', 'proposer.prepare.before')
print('[boundary-installed] graph, sample, proposal boundaries; diagnostic only', flush=True)
