import json, math
import torch
from vllm_ascend._310p.model_runner_310p import NPUModelRunner310
original = NPUModelRunner310._allocate_kv_cache_tensors

def allocate(self, config):
    groups=[]
    for group in config.kv_cache_groups:
        spec=group.kv_cache_spec
        groups.append({'spec':repr(spec),'layers':group.layer_names,'page_size_bytes':spec.page_size_bytes,'required_bytes':spec.max_memory_usage_bytes(self.vllm_config),'block_size':spec.block_size})
    print('KV_AUDIT_CONFIG '+json.dumps({'num_blocks':config.num_blocks,'tensor_budget_bytes':sum(t.size for t in config.kv_cache_tensors),'tensors':[{'size':t.size,'shared_by':t.shared_by} for t in config.kv_cache_tensors],'groups':groups,'scheduler_required_blocks':1+sum(math.ceil(g['required_bytes']/g['page_size_bytes']) for g in groups)}),flush=True)
    before=torch.npu.memory_allocated()
    result=original(self,config)
    storages={}
    def walk(value):
        if isinstance(value,torch.Tensor):
            storage=value.untyped_storage()
            storages[storage.data_ptr()]=storage.nbytes()
        elif isinstance(value,dict):
            for v in value.values():walk(v)
        elif isinstance(value,(list,tuple)):
            for v in value:walk(v)
    walk(result)
    print('KV_AUDIT_STORAGE '+json.dumps({'num_unique_storages':len(storages),'unique_storage_bytes':sum(storages.values()),'memory_allocated_delta':torch.npu.memory_allocated()-before,'storage_bytes':sorted(storages.values())}),flush=True)
    return result
NPUModelRunner310._allocate_kv_cache_tensors=allocate
