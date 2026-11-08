"""Opt-in capture/replay evidence; no device reads or synchronization added."""
import functools
import json
from vllm.forward_context import get_forward_context
from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

_original = ACLGraphWrapper.__call__
_seen = set()


@functools.wraps(_original)
def audited_call(self, *args, **kwargs):
    context = get_forward_context()
    descriptor = context.batch_descriptor
    active = context.cudagraph_runtime_mode == self.runtime_mode
    previous = self.concrete_aclgraph_entries.get(descriptor) if active else None
    had_graph = previous is not None and previous.aclgraph is not None
    output = _original(self, *args, **kwargs)
    entry = self.concrete_aclgraph_entries.get(descriptor) if active else None
    if entry is not None and entry.aclgraph is not None:
        runnable = self.runnable
        while isinstance(runnable, functools.partial):
            runnable = runnable.func
        callable_name = getattr(runnable, '__qualname__', type(runnable).__qualname__)
        role = 'draft' if getattr(runnable, '__name__', None) == '_run_merged_draft' else 'main'
        event = 'replay' if had_graph else 'capture'
        key = (id(self), event, descriptor)
        if key not in _seen:
            _seen.add(key)
            print('[graph-lifecycle] ' + json.dumps({
                'role': role, 'callable': callable_name, 'event': event, 'batch': str(descriptor),
                'runtime_mode': str(self.runtime_mode),
            }), flush=True)
    return output


ACLGraphWrapper.__call__ = audited_call
