"""310P worker 忽略 kv_cache_memory_bytes 时的隔离部署补丁。"""
from vllm_ascend._310p.worker_310p import NPUWorker310
from vllm_ascend.utils import is_rc_device
import torch
import os

# 旧板载 AI CPU 的 StatelessRandomNormalV2 在初始化时出现 0x2a。
# 只把 normal_ 随机初始化搬到 CPU；实际权重仍由 checkpoint 加载。
normal = torch.nn.init.normal_


def cpu_normal_for_npu(tensor, mean=0.0, std=1.0, generator=None):
    if tensor.device.type != "npu":
        return normal(tensor, mean, std, generator)
    temporary = torch.empty(tensor.shape, dtype=tensor.dtype, device="cpu")
    normal(temporary, mean, std, generator)
    with torch.no_grad():
        tensor.copy_(temporary)
    return tensor


torch.nn.init.normal_ = cpu_normal_for_npu

original = NPUWorker310.determine_available_memory


def determine_available_memory(self):
    budget = self.cache_config.kv_cache_memory_bytes
    if not is_rc_device() or budget is None:
        return original(self)
    self.model_runner.profile_run()
    self.available_kv_cache_memory_bytes = budget
    print(f"[orangepi] explicit KV budget: {budget} bytes", flush=True)
    return budget


NPUWorker310.determine_available_memory = determine_available_memory

# Prefix align computes a small block index vector, not model activations.
# This board's old AI CPU fails on FloorDiv. Keep only index arithmetic on CPU;
# perform the actual state-table gather on NPU as before.
from vllm.v1.attention.backends import utils as attention_utils
from vllm_ascend.ops import gdn_attn_builder

original_mamba_block_table = attention_utils.mamba_get_block_table_tensor


def mamba_block_table_cpu_indices(block_table, seq_lens, kv_cache_spec, mamba_cache_mode):
    if mamba_cache_mode != "align" or block_table.device.type != "npu":
        return original_mamba_block_table(block_table, seq_lens, kv_cache_spec, mamba_cache_mode)
    if os.environ.get('RC_DEVICE_BLOCK_INDICES', '0') == '1':
        # 仅在已穷举验证的长度域内使用；非负数float32除法再转整数等价floor。
        # 避免旧AI CPU FloorDiv，也避免seq_lens.cpu()的D2H同步。
        if kv_cache_spec.block_size not in (64, 576, 1152, 1728):
            raise ValueError('RC device block-index experiment received an unvalidated block size')
        start = ((seq_lens.float() - 1).clamp_min(0) / kv_cache_spec.block_size).to(torch.int32)
        offsets = torch.arange(1 + kv_cache_spec.num_speculative_blocks,
                               device=block_table.device, dtype=torch.int32)
    else:
        start = torch.clamp((seq_lens.cpu() - 1) // kv_cache_spec.block_size, min=0)
        offsets = torch.arange(1 + kv_cache_spec.num_speculative_blocks, dtype=torch.int32)
    indices = (start.unsqueeze(1) + offsets).to(device=block_table.device, dtype=torch.int64)
    return torch.gather(block_table, 1, indices)


attention_utils.mamba_get_block_table_tensor = mamba_block_table_cpu_indices
gdn_attn_builder.mamba_get_block_table_tensor = mamba_block_table_cpu_indices

# Optional experiment adapters. Default remains the proven C16/eager path.
import sys
sys.path.insert(0, '/work/repo/experimental')
if os.environ.get('QWEN36_C8_ENABLED', '0') == '1':
    import c8_vllm_adapter
elif os.environ.get('QWEN36_C8_ALLOW_GRAPH', '0') == '1':
    import direct_graph_310

if os.environ.get('GDN', '0') == '1':
    import pr17640_gdn_adapter

if os.environ.get('QWEN36_MINDIEQ_ENABLED', '0') == '1':
    import mindieq_prefill_adapter
