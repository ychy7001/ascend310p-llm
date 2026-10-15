# SPDX-License-Identifier: Apache-2.0
"""显式导入的启动期KV池诊断；不改变分配、路由或逐token执行。"""
import json
import math
import torch

from vllm_ascend._310p.model_runner_310p import NPUModelRunner310


_allocate = NPUModelRunner310._allocate_kv_cache_tensors


def allocate_with_report(self, config):
    groups = []
    for group in config.kv_cache_groups:
        spec = group.kv_cache_spec
        maximum = spec.max_memory_usage_bytes(self.vllm_config)
        groups.append({
            'spec': type(spec).__name__,
            'layers': group.layer_names,
            'block_size': spec.block_size,
            'page_size_bytes': spec.page_size_bytes,
            'max_memory_usage_bytes': maximum,
            'required_blocks': math.ceil(maximum / spec.page_size_bytes),
        })
    allocated = sum(pool.size for pool in config.kv_cache_tensors)
    required = 1 + sum(group['required_blocks'] for group in groups)
    report = {
        'scope': 'storage bytes exclude allocator rounding, model, activations and external workspaces; '
                 'minimum is a scheduler-contract estimate, not a tested minimum budget',
        'max_model_len': self.vllm_config.model_config.max_model_len,
        'num_blocks': config.num_blocks,
        'logical_pool_bytes': allocated,
        'logical_bytes_per_block': allocated // config.num_blocks,
        'scheduler_required_blocks': required,
        'estimated_min_logical_pool_bytes': required * (allocated // config.num_blocks),
        'groups': groups,
        'pools': [{'size': pool.size, 'shared_by': pool.shared_by}
                  for pool in config.kv_cache_tensors],
    }
    result = _allocate(self, config)
    storages = {}

    def visit(value, layer):
        if torch.is_tensor(value):
            storage = value.untyped_storage()
            identity = (str(value.device), storage.data_ptr())
            item = storages.setdefault(identity, {'bytes': storage.nbytes(), 'layers': set()})
            item['layers'].add(layer)
        elif isinstance(value, (tuple, list)):
            for tensor in value:
                visit(tensor, layer)

    for name, value in result.items():
        visit(value, name)
    actual = sum(item['bytes'] for item in storages.values())
    report.update({
        'actual_unique_storage_bytes': actual,
        'actual_storage_bytes_per_block': actual / config.num_blocks,
        'estimated_min_physical_cache_bytes': math.ceil(required * actual / config.num_blocks),
        'storages': [{'device': device, 'address': hex(address), 'bytes': item['bytes'],
                      'layers': sorted(item['layers'])}
                     for (device, address), item in storages.items()],
    })
    print('KV_MEMORY_REPORT ' + json.dumps(report), flush=True)
    return result


NPUModelRunner310._allocate_kv_cache_tensors = allocate_with_report
