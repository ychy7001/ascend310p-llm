#!/usr/bin/env python3
"""Read-only vLLM scheduler/runner telemetry. No token values, synchronization or tensors copied.

Enable through install_import_hook() before importing vLLM. JSONL is per process.
CPU timestamps measure dispatch/completion observation, not device kernel duration.
"""
import functools
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time

_STREAM = None

def request_key(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:20]


def tensor_meta(value):
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return {'shape': [x if isinstance(x, int) else str(x) for x in value.shape], 'dtype': str(value.dtype),
                'device': str(getattr(value, 'device', '')), 'layout': str(getattr(value, 'layout', ''))}
    if isinstance(value, (tuple, list)):
        return [tensor_meta(x) for x in value[:8]]
    if isinstance(value, dict):
        return {str(k): tensor_meta(v) for k, v in value.items() if hasattr(v, 'shape')}
    return None


def emit(kind, **fields):
    global _STREAM
    directory = os.environ.get('DECODE_OBSERVE_DIR')
    if not directory:
        return
    if _STREAM is None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        _STREAM = open(Path(directory) / ('observe-%s.jsonl' % os.getpid()), 'a', buffering=1)
    _STREAM.write(json.dumps({'kind': kind, 'pid': os.getpid(), 'wall': time.time(),
                             'mono': time.monotonic(), **fields}, separators=(',', ':')) + '\n')


def request_meta(request, scheduled=0, after_schedule=False):
    computed = int(request.num_computed_tokens) - (scheduled if after_schedule else 0)
    return {'id': request_key(request.request_id), 'prompt': int(request.num_prompt_tokens),
            'computed': computed, 'scheduled': int(scheduled),
            'output': int(getattr(request, 'num_output_tokens', len(getattr(request, '_output_token_ids', [])))),
            'arrival': float(request.arrival_time),
            'phase': 'prefill' if computed < request.num_prompt_tokens else 'decode'}


def patch_scheduler(module):
    cls = module.Scheduler
    if getattr(cls, '_decode_observed', False):
        return
    cls._decode_observed = True
    original_add, original_schedule, original_update = cls.add_request, cls.schedule, cls.update_from_output

    @functools.wraps(original_add)
    def add(self, request, *args, **kwargs):
        emit('arrival', **request_meta(request))
        return original_add(self, request, *args, **kwargs)

    @functools.wraps(original_schedule)
    def schedule(self, *args, **kwargs):
        start = time.monotonic()
        result = original_schedule(self, *args, **kwargs)
        if not hasattr(self, '_decode_observe_steps'):
            self._decode_observe_steps = {}
        self._decode_observe_steps[id(result)] = int(self.current_step)
        rows = [request_meta(self.requests[rid], n, True)
                for rid, n in result.num_scheduled_tokens.items() if rid in self.requests]
        emit('schedule', step=int(self.current_step), start=start, end=time.monotonic(),
             requests=rows, waiting=len(self.waiting), running=len(self.running),
             total_tokens=int(result.total_num_scheduled_tokens))
        return result

    @functools.wraps(original_update)
    def update(self, scheduler_output, model_runner_output, *args, **kwargs):
        references = {rid: self.requests[rid] for rid in scheduler_output.num_scheduled_tokens if rid in self.requests}
        start = time.monotonic()
        result = original_update(self, scheduler_output, model_runner_output, *args, **kwargs)
        emit('completion', step=getattr(self, '_decode_observe_steps', {}).pop(id(scheduler_output), None),
             start=start, end=time.monotonic(),
             ids=[request_key(rid) for rid in scheduler_output.num_scheduled_tokens],
             requests=[{**request_meta(req), 'finished': bool(req.is_finished())} for req in references.values()])
        return result
    cls.add_request, cls.schedule, cls.update_from_output = add, schedule, update
    emit('installed', target='scheduler')


def shape_dispatch_mode(module_stack):
    """Metadata-only mode, including compiled higher-order operator passthrough."""
    from torch.utils._python_dispatch import TorchDispatchMode
    class Shapes(TorchDispatchMode):
        supports_higher_order_operators = True
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            output = func(*args, **kwargs)
            if any(k in str(func).lower() for k in ('mm', 'linear', 'matmul', 'reduce', 'gather')):
                emit('torch_op', op=str(func), inputs=tensor_meta(args), outputs=tensor_meta(output),
                     module=module_stack[-1][0] if module_stack else None)
            return output
    return Shapes()


def dump_fx_graphs():
    """Read compiled FX metadata so inlined modules remain attributable."""
    import gc
    import torch
    count = 0
    for obj in gc.get_objects():
        if not issubclass(type(obj), torch.fx.GraphModule):
            continue
        nodes = []
        for node in obj.graph.nodes:
            meta = node.meta
            nodes.append({'name': node.name, 'op': node.op, 'target': str(node.target),
                          'args': str(node.args)[:4000], 'kwargs': str(node.kwargs)[:2000],
                          'module_stack': {str(k): str(v) for k, v in meta.get('nn_module_stack', {}).items()},
                          'value': tensor_meta(meta.get('val', meta.get('example_value'))),
                          'source_fn_stack': str(meta.get('source_fn_stack', ''))[:1000]})
        emit('fx_graph', graph=count, nodes=nodes, code=obj.code)
        count += 1
    emit('fx_graph_count', count=count)


def patch_runner(module):
    cls = module.NPUModelRunner310
    if getattr(cls, '_decode_observed', False):
        return
    cls._decode_observed = True
    original_execute = cls.execute_model
    original_forward = cls._model_forward
    original_load = cls.load_model
    original_sample = cls.sample_tokens

    @functools.wraps(original_execute)
    def execute(self, scheduler_output, *args, **kwargs):
        self._decode_observe_seq = getattr(self, '_decode_observe_seq', 0) + 1
        sequence = self._decode_observe_seq
        start = time.monotonic()
        self._decode_observe_requests = {request_key(k): int(v) for k, v in scheduler_output.num_scheduled_tokens.items()}
        emit('execute_start', seq=sequence, rank=int(self.device.index if self.device.index is not None else -1),
             requests=self._decode_observe_requests)
        try:
            return original_execute(self, scheduler_output, *args, **kwargs)
        finally:
            emit('execute_return', seq=sequence, start=start, end=time.monotonic())

    @functools.wraps(original_sample)
    def sample(self, *args, **kwargs):
        start = time.monotonic()
        try:
            return original_sample(self, *args, **kwargs)
        finally:
            emit('sample_return', seq=getattr(self, '_decode_observe_seq', 0), start=start, end=time.monotonic())

    @functools.wraps(original_load)
    def load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        model = self.get_model()
        self._decode_map_enabled = False
        self._decode_map_count = 0
        self._decode_module_stack = []
        for name, child in model.named_modules():
            if not any(x in name for x in ('proj', 'shared_expert', 'lm_head', 'experts')):
                continue
            if list(child.children()):
                continue
            def before(mod, inputs, module_name=name):
                if not self._decode_map_enabled:
                    return
                import torch
                scope = torch.profiler.record_function('decode_module::' + module_name)
                scope.__enter__()
                self._decode_module_stack.append((module_name, scope))
                emit('module_enter', seq=getattr(self, '_decode_observe_seq', 0), module=module_name,
                     module_type=type(mod).__name__, inputs=tensor_meta(inputs),
                     parameters={n: tensor_meta(p) for n, p in mod.named_parameters(recurse=False)})
            def after(mod, inputs, output, module_name=name):
                if not self._decode_map_enabled:
                    return
                emit('module_exit', module=module_name, outputs=tensor_meta(output))
                if self._decode_module_stack:
                    _, scope = self._decode_module_stack.pop()
                    scope.__exit__(None, None, None)
            child.register_forward_pre_hook(before)
            child.register_forward_hook(after)
        emit('installed', target='runner_modules')
        return result

    @functools.wraps(original_forward)
    def forward(self, num_tokens_padded, *args, **kwargs):
        from vllm.forward_context import get_forward_context
        context = get_forward_context()
        batch = self.input_batch
        num_reqs = int(batch.num_reqs)
        mode = str(context.cudagraph_runtime_mode)
        capture = bool(getattr(context, 'capturing', False))
        ids = list(batch.req_ids)[:num_reqs]
        computed = [int(x) for x in batch.num_computed_tokens_cpu[:num_reqs]]
        marker = Path(os.environ['DECODE_OBSERVE_DIR']) / 'map.enabled'
        # Only annotate real eager forwards; graph replay executes no Python modules.
        mapping = (not capture and 'NONE' in mode and marker.exists()
                   and getattr(self, '_decode_map_count', 0) < 3)
        if mapping:
            self._decode_map_count += 1
        self._decode_map_enabled = mapping
        emit('forward_start', seq=getattr(self, '_decode_observe_seq', 0), batch=num_reqs,
             rank=int(self.device.index if self.device.index is not None else -1),
             padded_tokens=int(num_tokens_padded), mode=mode, capturing=capture,
             graph_hit=not capture and 'FULL' in mode, context=computed,
             ids=[request_key(x) for x in ids], mapping=mapping)
        try:
            if mapping:
                if self._decode_map_count == 1:
                    dump_fx_graphs()
                with shape_dispatch_mode(self._decode_module_stack):
                    return original_forward(self, num_tokens_padded, *args, **kwargs)
            return original_forward(self, num_tokens_padded, *args, **kwargs)
        finally:
            self._decode_map_enabled = False
            emit('forward_return', seq=getattr(self, '_decode_observe_seq', 0))
    cls.execute_model, cls.load_model, cls._model_forward, cls.sample_tokens = execute, load, forward, sample
    emit('installed', target='runner')


def patch_profiler(module):
    """Enable task/shape observation within explicit profiler windows only."""
    import torch_npu
    def create(config, trace_name):
        experimental = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            msprof_tx=False, aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False, op_attr=False, data_simplification=True,
            record_op_args=False, gc_detect_threshold=None)
        return torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False, with_stack=False, profile_memory=False, with_modules=False,
            experimental_config=experimental,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(config.torch_profiler_dir,worker_name=trace_name))
    module.TorchNPUProfilerWrapper._create_profiler = staticmethod(create)
    emit('installed',target='profiler_level1_lean')


TARGETS = {'vllm.v1.core.sched.scheduler': patch_scheduler,
           'vllm_ascend._310p.model_runner_310p': patch_runner,
           'vllm_ascend.profiler.torch_npu_profiler': patch_profiler}

class ObserverFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        original = loader.exec_module
        def execute(module):
            original(module)
            TARGETS[fullname](module)
        loader.exec_module = execute
        return spec


def install_import_hook():
    if os.environ.get('DECODE_OBSERVE_DIR'):
        sys.meta_path.insert(0, ObserverFinder())
        emit('bootstrap')
